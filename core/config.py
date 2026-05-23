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
COLLECTION_NAME = 'medical_records'
SOURCE_FILES_DIR = Path(DB_PATH) / 'source_files'

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
