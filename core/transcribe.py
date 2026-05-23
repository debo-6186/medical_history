"""Local speech-to-text via faster-whisper.

Audio recorded in the phone browser is uploaded and transcribed here on the
laptop — no third-party speech service is used, consistent with TECH_SPEC
constraints C1 (data stays on the user's own devices) and C2 (local
inference).

The model is loaded once on first use — that first call also downloads the
model weights from Hugging Face — and is then cached for the process.
"""
import logging
import threading
from pathlib import Path

from faster_whisper import WhisperModel

from core.config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL

log = logging.getLogger('medical_history.transcribe')

_model: WhisperModel | None = None
_load_lock = threading.Lock()


def _get_model() -> WhisperModel:
    """Return the cached Whisper model, loading it on first use."""
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:  # re-check inside the lock
                log.info(
                    'loading whisper model %r (device=%s, compute=%s)…',
                    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
                )
                _model = WhisperModel(
                    WHISPER_MODEL,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                )
                log.info('whisper model ready')
    return _model


def transcribe(audio_path: Path) -> str:
    """Transcribe one audio file to plain text.

    `model.transcribe` returns a lazy generator of segments; iterating it is
    what actually runs the inference.
    """
    model = _get_model()
    segments, _info = model.transcribe(str(audio_path))
    return ' '.join(seg.text.strip() for seg in segments).strip()
