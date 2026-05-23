"""FastAPI application: API routes, static frontend, ingestion worker.

Run with:  uv run uvicorn app.server:app --host 127.0.0.1 --port 8000
Bind to 127.0.0.1 only — remote access is via `tailscale serve` (see TECH_SPEC).
"""
import asyncio
import json
import logging
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import ollama
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.auth import login, require_auth
from app.jobs import JOBS, MODEL_LOCK, create_job, start_worker
from app.schemas import (
    ChatRequest,
    HistoryRequest,
    HistoryResponse,
    LoginRequest,
    LoginResponse,
)
from core.analysis import SUPPORTED_EXTENSIONS
from core.config import (
    CHAT_MODEL,
    MAX_HISTORY_MESSAGES,
    MAX_UPLOAD_MB,
    SOURCE_FILES_DIR,
)
from core.qa import (
    build_messages,
    expand_query,
    retrieve,
    sources_from_hits,
    stream_answer,
)
from core.rag import delete_document, ingest_raw_text, list_documents
from core.transcribe import transcribe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
log = logging.getLogger('medical_history')

WEB_DIR = Path(__file__).resolve().parent.parent / 'web'

# Ephemeral, in-memory chat history keyed by conversation id (lost on restart).
CONVERSATIONS: dict[str, list[dict]] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_worker()
    yield


app = FastAPI(title='Medical History Assistant', lifespan=lifespan)


# --- Auth & health ---------------------------------------------------------

@app.get('/api/health')
def health() -> dict:
    try:
        ollama.list()
        reachable = True
    except Exception:
        reachable = False
    return {'status': 'ok', 'ollama_reachable': reachable}


@app.post('/api/login', response_model=LoginResponse)
def login_route(req: LoginRequest) -> LoginResponse:
    token, ttl = login(req.pin)
    return LoginResponse(token=token, expires_in=ttl)


# --- Ingestion -------------------------------------------------------------

@app.post('/api/history', response_model=HistoryResponse)
def history_route(
    req: HistoryRequest,
    _token: str = Depends(require_auth),
) -> HistoryResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(422, 'History text is empty')
    now = datetime.now(timezone.utc)
    doc_id = 'history-' + now.strftime('%Y-%m-%dT%H-%M-%SZ')
    # Stamp the date into the stored text itself: chunk metadata is not shown
    # in the chat prompt, so an in-text date is what lets the model answer
    # "when did I report this".
    dated_text = f"Medical history entry — {now.strftime('%Y-%m-%d')}\n\n{text}"
    chunks = ingest_raw_text(doc_id, 'Medical history', dated_text, 'history')
    log.info('history: stored %s (%d chunk(s))', doc_id, chunks)
    return HistoryResponse(doc_id=doc_id, chunks=chunks)


@app.post('/api/transcribe')
async def transcribe_route(
    file: UploadFile = File(...),
    _token: str = Depends(require_auth),
) -> dict:
    """Transcribe an uploaded voice note to text (local Whisper).

    Returns the transcript only — it does NOT save anything. The phone shows
    the text for the user to review and edit, then a separate POST /api/history
    saves it. Audio reaches the laptop over the same Tailscale link as file
    uploads; no third-party speech service is involved.
    """
    data = await file.read()
    if not data:
        raise HTTPException(422, 'Empty audio')
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f'Audio exceeds the {MAX_UPLOAD_MB} MB limit')

    suffix = Path(file.filename or 'audio.webm').suffix.lower() or '.webm'
    log.info('transcribe: received %d byte(s)', len(data))
    try:
        with tempfile.TemporaryDirectory(prefix='mh_audio_') as tmp_dir:
            audio_path = Path(tmp_dir) / f'audio{suffix}'
            audio_path.write_bytes(data)
            text = await asyncio.to_thread(transcribe, audio_path)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client
        log.exception('transcribe: failed')
        raise HTTPException(500, f'Transcription failed: {exc}')

    log.info('transcribe: produced %d char(s)', len(text))
    return {'text': text}


@app.post('/api/documents', status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form(alias='type'),
    _token: str = Depends(require_auth),
) -> dict:
    if doc_type not in ('report', 'prescription'):
        raise HTTPException(422, "type must be 'report' or 'prescription'")

    safe_name = Path(file.filename or '').name
    if not safe_name or safe_name in ('.', '..'):
        raise HTTPException(422, 'Invalid file name')
    if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            415, f'Unsupported file type. Allowed: {sorted(SUPPORTED_EXTENSIONS)}'
        )

    data = await file.read()
    if not data:
        raise HTTPException(422, 'Empty file')
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f'File exceeds the {MAX_UPLOAD_MB} MB limit')

    tmp_dir = tempfile.mkdtemp(prefix='mh_upload_')
    temp_path = Path(tmp_dir) / safe_name
    temp_path.write_bytes(data)

    job = create_job(safe_name, doc_type, str(temp_path))
    log.info('upload: %s (%s) queued as job %s', safe_name, doc_type, job.job_id)
    return {'job_id': job.job_id, 'status': job.status}


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str, _token: str = Depends(require_auth)) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, 'Job not found')
    return job.public()


# --- Chat (SSE) ------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f'event: {event}\ndata: {json.dumps(data)}\n\n'


@app.post('/api/chat')
async def chat(req: ChatRequest, _token: str = Depends(require_auth)) -> StreamingResponse:
    conversation_id = req.conversation_id or uuid.uuid4().hex
    history = list(CONVERSATIONS.get(conversation_id, []))
    log.info('chat: start conversation=%s message=%r', conversation_id, req.message)

    async def event_stream():
        # MODEL_LOCK is a threading.Lock shared with the ingest worker. Acquire
        # it by polling a non-blocking acquire: a cancelled request can then
        # never leak the lock. (`await to_thread(MODEL_LOCK.acquire)` is unsafe
        # — if cancelled, the worker thread still acquires the lock and nothing
        # is left to release it, deadlocking every later request.)
        lock_held = False
        try:
            log.info('chat: waiting for model lock…')
            while not lock_held:
                lock_held = MODEL_LOCK.acquire(blocking=False)
                if not lock_held:
                    await asyncio.sleep(0.1)
            log.info('chat: model lock acquired')

            # Query expansion calls the chat model, so it must run inside the
            # lock. Retrieval (embeddings + a keyword scan) is light and simply
            # shares the lock span rather than juggling the lock twice.
            vector_text, keywords, is_value_query, date_range = (
                await asyncio.to_thread(expand_query, req.message))
            hits = await asyncio.to_thread(
                retrieve, vector_text, keywords, is_value_query, date_range)
            log.info(
                'chat: retrieved %d chunk(s) [value_query=%s dates=%s]: %s',
                len(hits), is_value_query, date_range,
                [meta.get('section_title', '?') for _doc, meta, _dist in hits],
            )
            messages, user_msg = build_messages(req.message, hits, history)

            log.info('chat: generating with %s', CHAT_MODEL)
            answer_parts: list[str] = []
            gen = stream_answer(messages)
            sentinel = object()

            def _next():
                try:
                    return next(gen)
                except StopIteration:
                    return sentinel

            while True:
                chunk = await asyncio.to_thread(_next)
                if chunk is sentinel:
                    break
                if chunk:
                    answer_parts.append(chunk)
                    yield _sse('token', {'text': chunk})

            answer = ''.join(answer_parts)
            log.info('chat: response complete, %d char(s)', len(answer))

            if not answer.strip():
                log.warning('chat: model returned an empty response')
                yield _sse('error', {
                    'message': 'The model returned an empty response. '
                               'Please try asking again.',
                })
                return

            yield _sse('sources', {'sources': sources_from_hits(hits)})

            conv = CONVERSATIONS.setdefault(conversation_id, [])
            conv.append({'role': 'user', 'content': user_msg})
            conv.append({'role': 'assistant', 'content': answer})
            del conv[:-MAX_HISTORY_MESSAGES]

            yield _sse('done', {'conversation_id': conversation_id})
        except asyncio.CancelledError:
            log.warning('chat: cancelled — client disconnected (conversation=%s)',
                        conversation_id)
            raise
        except Exception as exc:  # noqa: BLE001 - report any failure to the client
            log.exception('chat: failed')
            yield _sse('error', {'message': str(exc)})
        finally:
            if lock_held:
                MODEL_LOCK.release()
                log.info('chat: model lock released')

    return StreamingResponse(
        event_stream(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


# --- Documents -------------------------------------------------------------

@app.get('/api/documents')
def documents(_token: str = Depends(require_auth)) -> dict:
    return {'documents': list_documents()}


@app.get('/api/documents/{doc_id}/file')
def document_file(doc_id: str, _token: str = Depends(require_auth)) -> FileResponse:
    base = SOURCE_FILES_DIR.resolve()
    path = (base / Path(doc_id).name).resolve()
    if base not in path.parents or not path.is_file():
        raise HTTPException(404, 'File not available')
    return FileResponse(path)


@app.delete('/api/documents/{doc_id}')
def remove_document(doc_id: str, _token: str = Depends(require_auth)) -> dict:
    if not delete_document(doc_id):
        raise HTTPException(404, 'Document not found')
    return {'deleted': True}


# --- Static frontend (mounted last so /api/* routes take precedence) -------

class _NoCacheStaticFiles(StaticFiles):
    """Serve frontend assets with revalidation, so an updated file is picked
    up instead of a stale copy being served from the browser cache."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers['Cache-Control'] = 'no-cache'
        return response


app.mount('/', _NoCacheStaticFiles(directory=WEB_DIR, html=True), name='web')
