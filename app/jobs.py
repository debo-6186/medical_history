"""Background ingestion jobs for uploaded files.

A single worker thread processes uploads one at a time (OCR + analysis are
heavy and a laptop runs one model inference at a time anyway). MODEL_LOCK
serializes every heavy model call so an ingest job and a chat request never
hit the model simultaneously — see core/ callers and the chat route.
"""
import logging
import queue
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from core.analysis import analyze_extracted, extract_file
from core.rag import ingest_analysis, store_source_file

# Held around every gemma4 / qwen3-vl call (ingest worker + chat route).
MODEL_LOCK = threading.Lock()

log = logging.getLogger('medical_history.jobs')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    job_id: str
    filename: str
    doc_type: str          # "report" | "prescription"
    temp_path: str
    status: str = 'queued'  # queued | processing | done | error
    progress: list[str] = field(default_factory=list)
    doc_id: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def public(self) -> dict:
        data = asdict(self)
        data.pop('temp_path', None)  # never expose server-side paths
        return data


JOBS: dict[str, Job] = {}
_QUEUE: "queue.Queue[str]" = queue.Queue()


def create_job(filename: str, doc_type: str, temp_path: str) -> Job:
    job = Job(
        job_id=uuid.uuid4().hex,
        filename=filename,
        doc_type=doc_type,
        temp_path=temp_path,
    )
    JOBS[job.job_id] = job
    _QUEUE.put(job.job_id)
    return job


def _update(job: Job, *, status: str | None = None, progress: str | None = None) -> None:
    if status is not None:
        job.status = status
    if progress is not None:
        job.progress.append(progress)
        log.info('job %s: %s', job.job_id, progress)
    job.updated_at = _now()


def _process(job: Job) -> None:
    src = Path(job.temp_path)
    _update(job, status='processing')

    stored_path = store_source_file(src)
    _update(job, progress='Saved source file')

    with MODEL_LOCK:
        _update(job, progress='Extracting text…')
        text = extract_file(src)
        _update(job, progress='Analyzing with the local model…')
        analysis = analyze_extracted(job.filename, text, job.doc_type)

    _update(job, progress='Embedding & storing…')
    chunks = ingest_analysis(
        doc_id=job.filename,
        filename=job.filename,
        file_path=stored_path,
        doc_type=job.doc_type,
        analysis=analysis,
    )
    job.doc_id = job.filename
    _update(job, status='done', progress=f'Added to your record ({chunks} section(s))')


def _worker() -> None:
    while True:
        job_id = _QUEUE.get()
        try:
            job = JOBS.get(job_id)
            if job is None:
                continue
            log.info('job %s: started — %s (%s)', job.job_id, job.filename, job.doc_type)
            try:
                _process(job)
            except Exception as exc:  # noqa: BLE001 - surface any failure to the client
                _update(job, status='error')
                job.error = str(exc)
                log.exception('job %s: failed', job_id)
            finally:
                shutil.rmtree(Path(job.temp_path).parent, ignore_errors=True)
        finally:
            _QUEUE.task_done()


def start_worker() -> None:
    threading.Thread(target=_worker, daemon=True, name='ingest-worker').start()
