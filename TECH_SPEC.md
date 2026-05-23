# Medical History Assistant — Technical Specification

Status: Draft for implementation
Owner: debo-6186
Last updated: 2026-05-20

---

## 1. Overview

A personal medical-records assistant. The user ingests their medical history
(free text), prescriptions, and lab/scan reports (files) from a phone's web
browser. A backend running on the user's own laptop extracts, analyzes, and
stores everything in a local vector store (RAG). The user can then chat with an
assistant that answers questions grounded in those records.

Everything — OCR, analysis, embeddings, and chat inference — runs locally on
the laptop via [Ollama](https://ollama.com). No medical data or prompt is ever
sent to a commercial/cloud LLM.

### 1.1 Goals

- Phone web browser is a thin client; all compute happens on the laptop.
- Works when the phone is **not** on the home network.
- Single stable HTTPS link the user opens in the phone browser.
- Ingest three input kinds: free-text history, prescription files, report files.
- Grounded Q&A over everything ingested, with source citations.

### 1.2 Non-negotiable constraints

| # | Constraint |
|---|------------|
| C1 | No medical data or derived prompt leaves the laptop except to the user's own phone. No commercial LLM APIs. |
| C2 | All inference (OCR, analysis, embeddings, chat) runs locally through Ollama. |
| C3 | The API is never exposed on a public/open port. Remote access is via Tailscale only. |
| C4 | Medical files and the vector DB are never committed to git. |
| C5 | Every API route except login and health requires authentication. |

### 1.3 Locked design decisions (from spec review)

- **Remote access:** Tailscale (free Personal plan). End-to-end encrypted; no
  third party sees the data.
- **Domain:** none owned — use the Tailscale MagicDNS name (`*.ts.net`).
- **Authentication:** single shared PIN.

---

## 2. Current state (what already exists)

Working CLI scripts in the repo root:

| File | Role | Reused as |
|------|------|-----------|
| `analyze_reports.py` | PDF/image text extraction + OCR + LLM analysis | → `core/analysis.py` |
| `rag_store.py` | Embeddings, ChromaDB, section splitting, file storage | → `core/rag.py` |
| `ask.py` | Interactive retrieval + chat loop | → `core/qa.py` |
| `ingest_reports.py` | Analyze a file then ingest its analysis | logic moves into the job worker |
| `main.py` | Legacy one-off prescription OCR script | **delete** (superseded) |

Models in use (Ollama tags — must match what is pulled locally):

- OCR / vision: `qwen3-vl:8b`
- Analysis & chat: `gemma4:26b`
- Embeddings: `nomic-embed-text`

Storage: ChromaDB persistent client at `./rag_db`; source files copied to
`./rag_db/source_files/`.

> Note: `analyze_reports.py` uses `gemma4:26b` while `ask.py` uses `gemma4:e4b`.
> This spec **standardizes on `gemma4:26b`** for both analysis and chat, exposed
> as a single config value. Adjust the tag if a different model is pulled.

---

## 3. System architecture

```
 ┌──────────────────────┐         Tailscale (WireGuard, end-to-end encrypted)
 │  Android phone        │◀───────────────────────────────────────────────┐
 │  - Browser (the app)  │                                                 │
 │  - Tailscale app ON   │   https://<laptop>.<tailnet>.ts.net              │
 └──────────────────────┘                                                  │
                                                                            │
 ┌──────────────────────────────────────────────────────────────────────┐ │
 │  Laptop                                                                │ │
 │                                                                        │ │
 │   tailscale serve  (HTTPS :443 on ts.net)  ──▶  uvicorn 127.0.0.1:8000 │◀┘
 │                                                        │               │
 │                                              ┌─────────▼──────────┐    │
 │                                              │  FastAPI app        │    │
 │                                              │  - auth (PIN)       │    │
 │                                              │  - ingest routes    │    │
 │                                              │  - chat route (SSE) │    │
 │                                              │  - static web/      │    │
 │                                              │  - job queue+worker │    │
 │                                              └────┬──────────┬─────┘    │
 │                                                   │          │          │
 │                                       ┌───────────▼──┐  ┌────▼───────┐  │
 │                                       │ Ollama        │  │ ChromaDB   │  │
 │                                       │ qwen3-vl:8b   │  │ ./rag_db   │  │
 │                                       │ gemma4:26b    │  │ + source_  │  │
 │                                       │ nomic-embed   │  │   files/   │  │
 │                                       └───────────────┘  └────────────┘  │
 └──────────────────────────────────────────────────────────────────────┘
```

Key points:

- uvicorn binds **`127.0.0.1:8000` only** — never `0.0.0.0`. Nothing is exposed
  on the LAN or the public internet.
- `tailscale serve` is the only thing in front of the API; it is reachable only
  by devices in the user's tailnet.
- The frontend is static files served by the same FastAPI app → **same origin**,
  so no CORS configuration is needed.

---

## 4. Technology stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Backend framework | FastAPI + uvicorn | async, SSE-friendly |
| File uploads | `python-multipart` | required for `UploadFile` |
| Vector DB | ChromaDB (persistent) | already in use |
| PDF parsing | PyMuPDF (`pymupdf`) | already in use |
| LLM runtime | Ollama (`ollama` py client) | local only |
| Frontend | Static HTML + vanilla JS + CSS | mobile-first; no build step |
| Remote access | Tailscale + `tailscale serve` | free Personal plan |
| Auth | Shared PIN → server-side session token | `secrets`, `hmac` (stdlib) |

### 4.1 Dependency changes (`pyproject.toml`)

Add:

```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
python-multipart>=0.0.12
```

Remove (privacy + unused):

- `google-genai` — commercial LLM client, violates C1; the Gemini code path is
  to be deleted.
- `paddleocr`, `paddlepaddle`, `opencv-python` — OCR is done by `qwen3-vl:8b`
  via Ollama; these are unused heavyweight deps. Confirm nothing imports them,
  then remove.

Keep: `ollama`, `chromadb`, `pymupdf`, `python-dotenv`.

### 4.2 Hardware note

`gemma4:26b` is a large model. With a big context window it needs substantial
memory. Recommended: Apple Silicon Mac, **32 GB RAM minimum, 64 GB preferred**.
Ollama swaps between `qwen3-vl:8b` and `gemma4:26b` within a single ingest job,
which adds load latency — acceptable for a single-user tool. See §10.3 for the
context-size tuning recommendation.

---

## 5. Remote access design (Tailscale)

### 5.1 How it works

1. Install Tailscale on the **laptop** and the **phone**; sign both into the
   same account → they join the same private network ("tailnet").
2. MagicDNS gives the laptop a permanent name: `<laptop-name>.<tailnet>.ts.net`.
   This name never changes and resolves **only** on the user's own devices.
3. `tailscale serve` puts an HTTPS front door (valid auto-provisioned TLS cert)
   on the laptop's `ts.net` name, proxying to the local API.
4. The phone (with Tailscale running) opens `https://<laptop>.<tailnet>.ts.net`
   from anywhere — mobile data, other Wi-Fi — because both devices stay in the
   tailnet regardless of physical network.

`tailscale funnel` (public exposure) is **not** used — that would defeat C3.

### 5.2 One-time setup

On the laptop:

1. Install Tailscale (macOS app). Sign in.
2. In the Tailscale admin console: enable **MagicDNS** and **HTTPS
   Certificates** (DNS settings).
3. Note the laptop's MagicDNS name: `tailscale status` shows it.
4. Ensure the Tailscale CLI is available (macOS app → "Install CLI…", or use
   `/Applications/Tailscale.app/Contents/MacOS/Tailscale`).

On the phone:

1. Install the Tailscale app, sign into the same account, leave it connected.

### 5.3 Running the front door

With the API running on `127.0.0.1:8000`:

```
tailscale serve --bg 8000
tailscale serve status      # verify: shows https://<laptop>.<tailnet>.ts.net
```

The phone then opens that `https://…ts.net` URL. Optionally "Add to Home
Screen" so it launches like an app (see §11.5 PWA manifest).

---

## 6. Data model & storage

### 6.1 Concept: Document

Every ingested item — a history text blob, a prescription, or a report — is a
**Document**. A Document is split into one or more **chunks**; each chunk is one
row in ChromaDB with its embedding.

### 6.2 ChromaDB

- Path: `./rag_db`, collection `medical_records` (unchanged).
- Chunk id format: `{doc_id}::section_{NN}`.
- Chunk metadata:

```json
{
  "doc_id":        "report_2026-05-10.pdf",
  "filename":      "report_2026-05-10.pdf",
  "doc_type":      "report | prescription | history",
  "section_title": "3. Parameter values",
  "file_path":     "/abs/path/in/source_files or empty",
  "created_at":    "2026-05-20T14:30:00Z"
}
```

`doc_id` equals the stored filename for file uploads; for free-text history it
is `history-<UTC-timestamp>`.

### 6.3 Chunking rules

- **File documents** (report/prescription): the LLM analysis is split by `## `
  headings — existing `split_analysis_sections()` behavior.
- **History documents** (free text): stored as a single chunk if ≤ ~4000
  chars; otherwise split on blank lines into ≤4000-char chunks. New function
  `ingest_raw_text()` in `core/rag.py`.

### 6.4 Source files

Uploaded files are copied to `./rag_db/source_files/` (existing
`store_source_file()`). Free-text history has no source file (`file_path` empty).

### 6.5 Re-ingest / delete

Ingesting a `doc_id` again first deletes existing chunks for that `doc_id`
(`collection.delete(where={"doc_id": ...})`) — existing upsert-after-delete
pattern, keyed on `doc_id` instead of `filename`.

---

## 7. Backend API specification

Base path `/api`. All responses JSON unless noted. All routes except
`POST /api/login` and `GET /api/health` require header
`Authorization: Bearer <token>`; missing/invalid → `401`.

### 7.1 `GET /api/health`  — no auth

Response `200`:
```json
{ "status": "ok", "ollama_reachable": true }
```

### 7.2 `POST /api/login`  — no auth

Request:
```json
{ "pin": "1234" }
```
Response `200`:
```json
{ "token": "<opaque-session-token>", "expires_in": 86400 }
```
Wrong PIN → `401 {"detail": "Invalid PIN"}`. See §8 for throttling.

### 7.3 `POST /api/history`  — auth

Store free-text medical history. Fast (embed + upsert only) → synchronous.
Triggered by the composer's `Add to record` toggle (§11.1).

Request:
```json
{ "text": "Diagnosed hypertension 2021. Allergic to penicillin. ..." }
```
Response `200`:
```json
{ "doc_id": "history-2026-05-20T14-30-00Z", "chunks": 1 }
```
Empty/whitespace text → `422`.

### 7.4 `POST /api/documents`  — auth — multipart/form-data

Upload one prescription or report file. Triggered by the composer's attach
button (§11.1). Slow (OCR + analysis) → returns a job id immediately; the work
runs in the background queue (§9).

Form fields:
- `file`: the upload (`.pdf .jpg .jpeg .png .webp`).
- `type`: `report` | `prescription`.

Response `202`:
```json
{ "job_id": "8f3c…", "status": "queued" }
```
Unsupported extension → `415`. Over size limit (`MAX_UPLOAD_MB`) → `413`.

### 7.5 `GET /api/jobs/{job_id}`  — auth

Response `200`:
```json
{
  "job_id": "8f3c…",
  "status": "queued | processing | done | error",
  "filename": "report_2026-05-10.pdf",
  "doc_type": "report",
  "progress": [
    "Saved source file",
    "Extracting text (qwen3-vl:8b OCR)…",
    "Analyzing (gemma4:26b)…",
    "Ingested 6 section(s)"
  ],
  "doc_id": "report_2026-05-10.pdf",
  "error": null,
  "created_at": "2026-05-20T14:30:00Z",
  "updated_at": "2026-05-20T14:34:10Z"
}
```
Unknown id → `404`.

### 7.6 `POST /api/chat`  — auth — Server-Sent Events

Request:
```json
{ "message": "What was my last cholesterol reading?",
  "conversation_id": "optional-uuid" }
```
Response `200`, `Content-Type: text/event-stream`. Event sequence:

```
event: token    data: {"text": "Your"}
event: token    data: {"text": " LDL"}
 …
event: sources  data: {"sources": [
                  {"doc_id": "lipid_panel.pdf", "filename": "lipid_panel.pdf"}]}
event: done     data: {"conversation_id": "…"}
```
On failure: `event: error  data: {"message": "…"}`.

If `conversation_id` is omitted, the server creates one and returns it in the
`done` event. See §10 for the chat pipeline.

### 7.7 `GET /api/documents`  — auth

List ingested documents (deduplicated from Chroma metadata).

Response `200`:
```json
{ "documents": [
  { "doc_id": "report_2026-05-10.pdf", "doc_type": "report",
    "filename": "report_2026-05-10.pdf", "created_at": "…",
    "has_file": true, "chunks": 6 },
  { "doc_id": "history-2026-05-20T14-30-00Z", "doc_type": "history",
    "filename": "Medical history", "created_at": "…",
    "has_file": false, "chunks": 1 }
]}
```

### 7.8 `GET /api/documents/{doc_id}/file`  — auth

Returns the original uploaded file (`FileResponse`). `404` if the document has
no stored file (e.g. history text).

### 7.9 `DELETE /api/documents/{doc_id}`  — auth — optional (Phase 6)

Deletes the document's chunks and its source file. Response `200 {"deleted": true}`.

---

## 8. Authentication design

Single shared PIN, configured as `APP_PIN` (env). Flow:

1. `POST /api/login` compares the submitted PIN to `APP_PIN` with
   `hmac.compare_digest` (constant-time).
2. On success: generate `secrets.token_urlsafe(32)`, store it in an in-memory
   dict `SESSIONS: dict[token, expiry_epoch]` with TTL `SESSION_TTL_HOURS`.
   Return the token.
3. `require_auth` FastAPI dependency: read the bearer token, check it exists in
   `SESSIONS` and is unexpired → else `401`. Expired tokens are dropped lazily.
4. Frontend stores the token in `localStorage`; any `401` clears it and shows
   the PIN screen.

Throttling (brute-force defense, since the PIN space is small):

- Track failed attempts in-memory. After 5 consecutive failures, reject logins
  for 60 seconds (`429 {"detail": "Too many attempts, wait 60s"}`).
- Add a fixed 500 ms delay to every failed login response.

Notes:

- Sessions are in-memory → lost on backend restart; the user simply re-enters
  the PIN. Acceptable for single-user.
- This is a second layer; the primary boundary is Tailscale (C3). The PIN
  protects against another device already in the tailnet.
- Choose a PIN of at least 6 digits.

---

## 9. Ingestion design

### 9.1 Free-text history (synchronous)

`POST /api/history` → `core/rag.ingest_raw_text(doc_id, text, doc_type="history")`
→ chunk per §6.3 → embed each chunk (`nomic-embed-text`) → `collection.upsert`.
Fast enough to answer within the request.

### 9.2 File documents (asynchronous job)

`POST /api/documents` saves the upload to a temp path, creates a `Job`, enqueues
it, and returns the `job_id`. A **single background worker thread** processes the
queue one job at a time (a laptop can run only one heavy model inference at a
time anyway).

`Job` object (in `app/jobs.py`):

```python
@dataclass
class Job:
    job_id: str
    filename: str
    doc_type: str          # "report" | "prescription"
    temp_path: str
    status: str = "queued" # queued|processing|done|error
    progress: list[str] = field(default_factory=list)
    doc_id: str | None = None
    error: str | None = None
    created_at: str = ...
    updated_at: str = ...
```

`JOBS: dict[str, Job]` holds all jobs; `queue.Queue` feeds the worker.

Worker steps per job (each step appends to `progress` and updates `updated_at`):

1. `status = "processing"`.
2. `store_source_file()` → copy into `source_files/`. → "Saved source file".
3. `extract_file()` (PyMuPDF text-layer or `qwen3-vl:8b` OCR). → "Extracting text…".
4. `analyze()` with `gemma4:26b` using the report/prescription prompt. →
   "Analyzing (gemma4:26b)…".
5. `split_analysis_sections()` → embed each → `upsert` with metadata (§6.2). →
   "Ingested N section(s)".
6. `status = "done"`, set `doc_id`. On any exception: `status = "error"`,
   `error = str(e)`.
7. Delete the temp upload file.

### 9.3 Model serialization

A module-level `threading.Lock` named `MODEL_LOCK` wraps **every** call to
`gemma4:26b` and `qwen3-vl:8b` (embeddings are light and may run without it).
Both the ingest worker and the chat route acquire it, so a chat request and an
ingest job never hit the model simultaneously. The chat route runs its blocking
Ollama call inside `asyncio.to_thread` so the event loop stays free while it
waits for (and holds) the lock.

Consequence: a chat sent during an ingest job waits until the job's current
model step finishes. Acceptable and expected on a single laptop; the frontend
shows a "working…" state.

### 9.4 Ingestion sequence (file)

```
phone ──POST /api/documents (file,type)──▶ API
API   ──save temp, create Job, enqueue──▶ returns 202 {job_id}
worker: store file ▸ extract ▸ analyze ▸ chunk ▸ embed ▸ upsert ▸ done
phone ──GET /api/jobs/{id} (poll every 2s)──▶ progress[] / status
phone shows progress; on "done" → success; on "error" → show error
```

---

## 10. Q&A / chat design

### 10.1 Pipeline (`core/qa.py`, refactored from `ask.py`)

1. Embed the user message (`nomic-embed-text`).
2. `collection.query(n_results=TOP_K)` → top chunks with metadata + distance.
3. `format_context()` → labelled excerpt blocks (existing behavior).
4. Build messages: `SYSTEM_PROMPT` + last `MAX_HISTORY_MESSAGES` of this
   conversation + the new user message wrapped with the retrieved excerpts.
5. `ollama.chat(model=gemma4:26b, messages=…, stream=True)` under `MODEL_LOCK`.
6. Stream tokens to the client as SSE `token` events; after the stream, send a
   `sources` event built from the retrieved chunks' `doc_id`/`filename`, then
   `done`.

The existing `SYSTEM_PROMPT` (answer only from excerpts, cite filenames, say
"Not found in the available records" otherwise, educational-only disclaimer) is
kept.

### 10.2 Conversation state

In-memory `CONVERSATIONS: dict[conversation_id, list[message]]`, trimmed to the
last `MAX_HISTORY_MESSAGES`. Ephemeral — cleared on restart. The frontend keeps
one `conversation_id` per chat session in memory.

### 10.3 Context-window tuning (recommendation)

`ask.py` currently passes `num_ctx=131072` for chat. Retrieved chat context is
small (top-5 chunks + short history), so a 128K window wastes memory and slows
load. Recommend two configurable values:

- `ANALYSIS_NUM_CTX` — large, for whole-document analysis (e.g. `32768`, raise
  toward `131072` only if a document needs it and RAM allows).
- `CHAT_NUM_CTX` — small, for chat (e.g. `8192`).

Both live in `app/config.py`; not a hard requirement, but a meaningful speed and
memory win.

---

## 11. Frontend specification

A single static page served at `/`, mobile-first, no build step.
Files: `web/index.html`, `web/app.js`, `web/style.css`.

### 11.1 Layout — a single chat thread

1. **PIN gate** — shown when there is no valid token. PIN input → `POST
   /api/login` → store token → show the chat screen.
2. **Chat screen** — one scrolling message thread plus a composer bar at the
   bottom. There are no separate forms or tabs; history updates, document
   uploads, and Q&A all happen inline in the one thread.

**Composer bar** — three controls:

- **Attach button (📎)** — opens the file picker
  (`accept="application/pdf,image/*"`, multi-select enabled). Chosen files are
  staged in a list, each row with its own `report` / `prescription` selector;
  sending uploads each file as a separate `POST /api/documents` job.
- **Text input** — free text.
- **Intent toggle** — a two-state segmented control, `Ask` (default) /
  `Add to record`, that routes a text message:
  - `Ask` → `POST /api/chat` (streamed answer).
  - `Add to record` → `POST /api/history` (stored as history text).

**Message rendering** — every action becomes a visible message in the thread so
the user always sees what happened. This visible confirmation is the safeguard
against misfiled medical info (why the explicit toggle was chosen over
auto-detection):

- `Add to record` send → the user's text, then an assistant bubble
  "Saved to your record."
- `Ask` send → the user's question, then the streamed answer with a sources line.
- File send → each staged file produces its own filename user message and its
  own status bubble that updates from `GET /api/jobs/{id}` polling (renders
  `progress[]`), ending in "Added to your record" or an error.

### 11.2 Token handling

- Token in `localStorage` under `mh_token`.
- A `fetch` wrapper adds `Authorization: Bearer`; on `401` it clears the token
  and returns to the PIN gate.

### 11.3 Streaming chat in the browser

`EventSource` only supports GET, so the chat uses `fetch` with a streamed body:

```js
const res = await fetch('/api/chat', { method:'POST', headers, body });
const reader = res.body.getReader();
// decode chunks, parse SSE "event:"/"data:" lines, append token text live
```

### 11.4 File upload from the phone

`<input type="file" accept="application/pdf,image/*">` — on Android this offers
both the camera and the file picker, covering photographed prescriptions and
saved PDFs. Show the selected filename and an upload progress/working state.

### 11.5 Optional PWA manifest

Add `web/manifest.webmanifest` (name, icons, `display: standalone`) and link it
from `index.html` so "Add to Home Screen" launches the app full-screen. Pure
polish — not required for function.

### 11.6 No CORS

Frontend and API share the same `ts.net` origin → no CORS headers needed.

---

## 12. Project structure

```
medical_history/
├── core/                  # no web deps — pure logic, importable & testable
│   ├── __init__.py
│   ├── analysis.py        # ← analyze_reports.py: extract_file, analyze, prompts
│   ├── rag.py             # ← rag_store.py: embed, collection, ingest_*, list, delete
│   └── qa.py              # ← ask.py: retrieve, format_context, stream_answer
├── app/                   # web layer
│   ├── __init__.py
│   ├── server.py          # FastAPI app: startup/shutdown, static mount, all routes
│   ├── config.py          # env-driven settings
│   ├── auth.py            # login, SESSIONS, require_auth dependency, throttling
│   ├── jobs.py            # Job dataclass, JOBS, queue, worker thread, MODEL_LOCK
│   └── schemas.py         # Pydantic request/response models
├── web/                   # static frontend
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── manifest.webmanifest   # optional
├── rag_db/                # gitignored: chroma db + source_files/
├── .env                   # gitignored: real secrets
├── .env.example
├── pyproject.toml
└── TECH_SPEC.md
```

`main.py` is deleted. `ingest_reports.py` may be kept as a thin CLI wrapper over
`core/` for offline bulk ingest, or deleted — not used by the web app.

### 12.1 `.gitignore` (must include)

```
.env
rag_db/
prescriptions/
__pycache__/
*.pyc
```

`rag_db/` and `prescriptions/` hold medical data and must never be committed (C4).

---

## 13. Configuration

All config via environment / `.env`, loaded in `app/config.py`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `APP_PIN` | *(required)* | Shared login PIN (≥6 digits) |
| `SESSION_TTL_HOURS` | `24` | Session token lifetime |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama endpoint |
| `CHAT_MODEL` | `gemma4:26b` | Analysis + chat model |
| `OCR_MODEL` | `qwen3-vl:8b` | Vision/OCR model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `ANALYSIS_NUM_CTX` | `32768` | Context window for document analysis |
| `CHAT_NUM_CTX` | `8192` | Context window for chat |
| `DB_PATH` | `./rag_db` | ChromaDB + source files location |
| `MAX_UPLOAD_MB` | `25` | Upload size cap |
| `TOP_K` | `5` | Retrieved chunks per query |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | uvicorn bind (keep loopback) |

Updated `.env.example`:

```
APP_PIN=change-me-6-digits
SESSION_TTL_HOURS=24
OLLAMA_HOST=http://127.0.0.1:11434
CHAT_MODEL=gemma4:26b
OCR_MODEL=qwen3-vl:8b
EMBED_MODEL=nomic-embed-text
ANALYSIS_NUM_CTX=32768
CHAT_NUM_CTX=8192
MAX_UPLOAD_MB=25
```

The old `GEMINI_API_KEY` is removed (no commercial LLM).

---

## 14. Setup & run

### 14.1 First-time laptop setup

```
# 1. Models
ollama pull gemma4:26b
ollama pull qwen3-vl:8b
ollama pull nomic-embed-text

# 2. Python deps
uv sync

# 3. Config
cp .env.example .env        # then edit APP_PIN

# 4. Tailscale: install app, sign in, enable MagicDNS + HTTPS certs
```

### 14.2 Each run

```
# terminal 1 — API (loopback only)
uv run uvicorn app.server:app --host 127.0.0.1 --port 8000

# terminal 2 — Tailscale front door
tailscale serve --bg 8000
tailscale serve status      # copy the https://<laptop>.<tailnet>.ts.net URL
```

### 14.3 On the phone

1. Tailscale app installed, signed into the same account, connected.
2. Open `https://<laptop>.<tailnet>.ts.net` in the browser.
3. Enter the PIN. Optionally "Add to Home Screen".

---

## 15. Security considerations

- **Layered access:** Tailscale (private network) is the outer boundary; the
  PIN session is the inner one. The API never binds beyond `127.0.0.1`.
- **TLS:** provided end-to-end — WireGuard between devices, plus a real cert on
  `tailscale serve`. No cert warnings.
- **Secrets:** `APP_PIN` only in `.env` (gitignored). Session tokens are random
  256-bit values, in memory only.
- **Uploads:** validate extension and size before saving; analysis prompts
  already instruct the model not to invent data.
- **Data at rest:** medical files and the vector DB live only under `rag_db/`
  on the laptop; never committed (C4). Consider laptop full-disk encryption
  (FileVault).
- **Logging:** do not log file contents, extracted text, prompts, or answers.
  Log only job ids, statuses, and timings.

---

## 16. Implementation plan (phased)

Each phase is independently testable.

**Phase 0 — Refactor & cleanup**
- Move `analyze_reports.py` → `core/analysis.py`, `rag_store.py` → `core/rag.py`,
  `ask.py` → `core/qa.py`. Delete `main.py`.
- Delete the Gemini code path; remove `google-genai` and the unused paddle/opencv
  deps from `pyproject.toml`.
- `core/rag.py`: add `doc_id`/`doc_type`/`created_at` metadata, `ingest_raw_text()`,
  `list_documents()`, `delete_document()`; key delete on `doc_id`.
- `core/qa.py`: expose a `stream_answer()` generator.
- *Done when:* CLI ingest/ask still work via the refactored modules.

**Phase 1 — API skeleton + auth**
- `app/config.py`, `app/server.py`, `app/auth.py`, `app/schemas.py`.
- `GET /api/health`, `POST /api/login`, `require_auth`, static mount of `web/`.
- *Done when:* health works; login returns a token; a protected stub route
  rejects a missing/bad token with 401.

**Phase 2 — Ingestion**
- `app/jobs.py`: `Job`, `JOBS`, queue, worker thread, `MODEL_LOCK`.
- `POST /api/history`, `POST /api/documents`, `GET /api/jobs/{id}`.
- *Done when:* a posted history string and an uploaded PDF/image both end up as
  queryable chunks in Chroma; job progress is observable.

**Phase 3 — Chat**
- `POST /api/chat` SSE streaming via `core/qa.stream_answer()`; conversation
  store; `sources` event.
- *Done when:* `curl -N` against `/api/chat` streams tokens then sources.

**Phase 4 — Frontend**
- `web/index.html`, `app.js`, `style.css`: PIN gate, single chat thread with
  composer bar (attach button, intent toggle), streaming chat, and inline
  job-polling status for uploads.
- *Done when:* full flow works in a desktop browser against `127.0.0.1:8000`.

**Phase 5 — Remote access**
- Configure Tailscale; run `tailscale serve`; test the whole flow from the
  phone over mobile data.
- *Done when:* the phone completes ingest + chat over the `ts.net` URL while
  off the home Wi-Fi.

**Phase 6 — Polish (optional)**
- `GET /api/documents`, `GET /api/documents/{id}/file`,
  `DELETE /api/documents/{id}`; document list UI; PWA manifest.

---

## 17. Out of scope

- Multi-user accounts / per-user data isolation (single shared PIN only).
- Editing or versioning of stored analyses.
- Mobile-native app (browser only).
- Public internet exposure (`tailscale funnel`).
- Cloud backup of `rag_db/` (local responsibility; consider a manual encrypted
  backup).

## 18. Open risks

| Risk | Mitigation |
|------|------------|
| `gemma4:26b` too slow / too large on the laptop | Tune `CHAT_NUM_CTX`/`ANALYSIS_NUM_CTX`; fall back to a smaller chat model if needed (config-only change). |
| Long ingest blocks chat (shared `MODEL_LOCK`) | Expected on one laptop; surface a "working…" state in the UI. |
| Laptop asleep / Ollama down → phone can't connect | `GET /api/health` check on app load; show a clear "backend offline" message. |
| OCR quality on handwritten prescriptions | Existing prompts already forbid guessing; flag low-confidence fields as `NA`. |
| Model tag mismatch (`gemma4:26b` not pulled) | Validate models on startup; fail fast with a clear message. |
```
