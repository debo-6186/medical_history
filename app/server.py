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
    ReminderConfirmRequest,
)
from core.analysis import SUPPORTED_EXTENSIONS
from core.config import (
    CHAT_MODEL,
    MAX_HISTORY_MESSAGES,
    MAX_UPLOAD_MB,
    SOURCE_FILES_DIR,
)
from core import calendar as gcal
from core import reminders as reminders_store
from core.agent import run_turn
from core.followups import create_pending_followups
from core.qa import stream_answer
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

    # Scan the saved text for future follow-up actions and draft them as pending
    # reminders. The model call must run under MODEL_LOCK; extraction failure
    # must never fail the save itself.
    drafted: list[dict] = []
    with MODEL_LOCK:
        try:
            drafted = create_pending_followups(text)
        except Exception:  # noqa: BLE001
            log.exception('history: follow-up extraction failed')
    if drafted:
        log.info('history: drafted %d follow-up reminder(s)', len(drafted))
    return HistoryResponse(doc_id=doc_id, chunks=chunks, reminders=drafted)


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

            # The agent graph (router -> records/general/mixed, with a bounded
            # retry cycle) makes blocking model calls, so it runs in a worker
            # thread under the same lock. It is a synchronous generator of
            # ('status', ...) progress events followed by one ('result', state);
            # surface each step to the client and capture the final state.
            agen = run_turn(req.message, history)
            agen_sentinel = object()

            def _next_event():
                try:
                    return next(agen)
                except StopIteration:
                    return agen_sentinel

            state: dict | None = None
            while True:
                event = await asyncio.to_thread(_next_event)
                if event is agen_sentinel:
                    break
                kind, payload = event
                if kind == 'status':
                    yield _sse('status', payload)
                elif kind == 'result':
                    state = payload
            if state is None:
                raise RuntimeError('agent produced no result')

            messages = state['messages']
            user_msg = state['user_msg']
            sources = state.get('sources', [])
            log.info(
                'chat: intent=%s attempts=%s sources=%d',
                state.get('intent'), state.get('retrieval_attempts'),
                len(sources),
            )

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

            yield _sse('sources', {'sources': sources})

            # A scheduling turn drafted a reminder — surface it for the user to
            # review and confirm (nothing has been sent to Google yet).
            if state.get('proposal'):
                yield _sse('proposal', {'reminder': state['proposal']})

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


# --- Reminders -------------------------------------------------------------
# Reminders are drafted locally (by the chat scheduling node, and later by
# medication capture / follow-up extraction) and only pushed to Google Calendar
# when the user confirms one here.

@app.get('/api/reminders')
def list_reminders(
    status: str | None = None, _token: str = Depends(require_auth),
) -> dict:
    return {'reminders': reminders_store.list_reminders(status)}


@app.post('/api/reminders/{reminder_id}/confirm')
async def confirm_reminder(
    reminder_id: int,
    req: ReminderConfirmRequest,
    _token: str = Depends(require_auth),
) -> dict:
    reminder = reminders_store.get(reminder_id)
    if reminder is None:
        raise HTTPException(404, 'Reminder not found')
    if reminder['status'] == 'scheduled':
        raise HTTPException(409, 'Reminder is already scheduled')

    # Apply any edits the user made in the confirmation panel.
    edits = req.model_dump(exclude_none=True, exclude={'timezone'})
    if edits:
        reminder = reminders_store.update_fields(reminder_id, **edits)

    if not reminder['start']:
        raise HTTPException(422, 'Set a date/time before confirming')
    try:
        start_dt = datetime.fromisoformat(reminder['start'])
    except ValueError:
        raise HTTPException(422, f"Invalid start datetime: {reminder['start']!r}")

    # Medication reminders send only the truncated medicine name (privacy);
    # manual/follow-up reminders send the user-approved text.
    title = (reminder['title'] if reminder['kind'] == 'medication'
             else reminder['proposed_text'])
    try:
        event_id = await asyncio.to_thread(
            gcal.create_reminder,
            title, start_dt, reminder['rrule'], reminder['kind'], req.timezone,
        )
    except gcal.NotAuthorizedError as exc:
        raise HTTPException(409, str(exc))
    except Exception as exc:  # noqa: BLE001 - surface Google API errors
        log.exception('reminder confirm: Google Calendar insert failed')
        raise HTTPException(502, f'Google Calendar error: {exc}')

    updated = reminders_store.mark_scheduled(reminder_id, event_id)
    log.info('reminder %s scheduled as event %s', reminder_id, event_id)
    return {'reminder': updated}


@app.post('/api/reminders/{reminder_id}/dismiss')
def dismiss_reminder(
    reminder_id: int, _token: str = Depends(require_auth),
) -> dict:
    reminder = reminders_store.get(reminder_id)
    if reminder is None:
        raise HTTPException(404, 'Reminder not found')
    return {'reminder': reminders_store.mark_dismissed(reminder_id)}


# --- Google Calendar connection -------------------------------------------

@app.get('/api/google/status')
def google_status(_token: str = Depends(require_auth)) -> dict:
    from pathlib import Path as _Path

    from core.config import GOOGLE_CREDENTIALS_PATH
    return {
        'authorized': gcal.is_authorized(),
        'credentials_present': _Path(GOOGLE_CREDENTIALS_PATH).exists(),
    }


@app.post('/api/google/authorize')
async def google_authorize(_token: str = Depends(require_auth)) -> dict:
    """Run the one-time OAuth consent (opens a browser on the laptop)."""
    try:
        await asyncio.to_thread(gcal.authorize)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception('google authorize failed')
        raise HTTPException(500, f'Authorization failed: {exc}')
    return {'authorized': True}


# --- Static frontend (mounted last so /api/* routes take precedence) -------

class _NoCacheStaticFiles(StaticFiles):
    """Serve frontend assets with revalidation, so an updated file is picked
    up instead of a stale copy being served from the browser cache."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers['Cache-Control'] = 'no-cache'
        return response


app.mount('/', _NoCacheStaticFiles(directory=WEB_DIR, html=True), name='web')
