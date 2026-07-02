"""Environment-driven configuration shared by the core logic and the web app.

Lives in core/ (not app/) because the core modules are the lower layer and must
not import from app/.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Ollama models ---------------------------------------------------------
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
CHAT_MODEL = os.getenv('CHAT_MODEL', 'gemma4:26b')
OCR_MODEL = os.getenv('OCR_MODEL', 'qwen3-vl:8b')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')
# nomic-embed-text is 768-dimensional. The ObjectBox HNSW index is
# fixed-dimension, so this must match the embedding model and the Android entity.
EMBED_DIM = int(os.getenv('EMBED_DIM', '768'))

# The ollama client library reads OLLAMA_HOST from the environment; make sure
# it is set even when only CHAT_MODEL/etc. were provided via .env.
os.environ.setdefault('OLLAMA_HOST', OLLAMA_HOST)

# --- Ollama call tuning ----------------------------------------------------
# Keep the chat model resident between requests so a query after an idle gap
# does not pay the multi-GB reload from disk. Ollama duration string: '30m',
# '-1' for forever, '0' to unload immediately.
MODEL_KEEP_ALIVE = os.getenv('MODEL_KEEP_ALIVE', '30m')
# gemma4:26b is a "thinking" model. Reasoning before every answer is slow on
# laptop hardware and unnecessary for grounded record lookups — off by
# default. Set MODEL_THINKING=true to re-enable it for analysis and chat.
MODEL_THINKING = os.getenv('MODEL_THINKING', 'false').lower() == 'true'

# --- Context windows -------------------------------------------------------
ANALYSIS_NUM_CTX = int(os.getenv('ANALYSIS_NUM_CTX', '32768'))
CHAT_NUM_CTX = int(os.getenv('CHAT_NUM_CTX', '8192'))

# --- Speech-to-text (faster-whisper, runs locally) -------------------------
# Voice notes recorded on the phone are transcribed on the laptop — no
# third-party speech service (TECH_SPEC C1/C2). Model size trades accuracy
# for speed: tiny / base / small / medium / large-v3. The model is downloaded
# once on first use. device 'auto' picks GPU when available, else CPU.
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'small')
WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'auto')
WHISPER_COMPUTE_TYPE = os.getenv('WHISPER_COMPUTE_TYPE', 'int8')

# --- Storage ---------------------------------------------------------------
DB_PATH = os.getenv('DB_PATH', './rag_db')
SOURCE_FILES_DIR = Path(DB_PATH) / 'source_files'

# --- Reminders / Google Calendar -------------------------------------------
# Reminders are drafted and persisted locally (sqlite) and only pushed to
# Google Calendar after the user confirms. Drafting needs no Google auth; only
# the confirm -> create step does, so the feature works offline until then.
REMINDERS_DB_PATH = os.getenv('REMINDERS_DB_PATH', str(Path(DB_PATH) / 'reminders.db'))

# OAuth client secrets ("Desktop app" credentials downloaded from Google Cloud
# Console) and the stored user token (written after the one-time consent). Both
# live under rag_db/ by default and must never be committed.
GOOGLE_CREDENTIALS_PATH = os.getenv(
    'GOOGLE_CREDENTIALS_PATH', str(Path(DB_PATH) / 'google_credentials.json'))
GOOGLE_TOKEN_PATH = os.getenv(
    'GOOGLE_TOKEN_PATH', str(Path(DB_PATH) / 'google_token.json'))
GOOGLE_CALENDAR_ID = os.getenv('GOOGLE_CALENDAR_ID', 'primary')

# Privacy: a medication reminder's title is truncated to this many leading
# characters before it is sent to Google (TECH_SPEC privacy boundary). Follow-up
# and manual reminders send the user-approved text as-is.
MED_TITLE_PREFIX_LEN = int(os.getenv('MED_TITLE_PREFIX_LEN', '4'))

# --- Vector store (ObjectBox) ----------------------------------------------
# The vector store lives in its own directory (data.mdb) so the whole store can
# be copied to Android and opened by the native ObjectBox binding. The distance
# type is baked into the HNSW index and must match the Android entity. Cosine
# suits nomic-embed-text. See docs/objectbox_android_schema.md.
OBJECTBOX_DIR = str(Path(DB_PATH) / 'objectbox')
VECTOR_DISTANCE = os.getenv('VECTOR_DISTANCE', 'cosine')

# --- Retrieval / chat ------------------------------------------------------
# Chunks are now fine-grained (a few bullets each), so more are fetched to
# keep each answer's context coverage comparable to the old coarse chunks.
TOP_K = int(os.getenv('TOP_K', '10'))
MAX_HISTORY_MESSAGES = 6

# --- Auth / server ---------------------------------------------------------
APP_PIN = os.getenv('APP_PIN', '')
SESSION_TTL_HOURS = int(os.getenv('SESSION_TTL_HOURS', '24'))
MAX_UPLOAD_MB = int(os.getenv('MAX_UPLOAD_MB', '25'))
HOST = os.getenv('HOST', '127.0.0.1')
PORT = int(os.getenv('PORT', '8000'))
