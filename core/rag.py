"""Vector store: embeddings, ChromaDB access, ingestion, listing.

Refactored from rag_store.py, extended with raw-text ingestion, document
listing, and deletion. Every chunk carries doc_id / doc_type / created_at
metadata so the web app can list and manage documents.
"""
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import ollama

from core.config import COLLECTION_NAME, DB_PATH, EMBED_MODEL, SOURCE_FILES_DIR

_H2_RE = re.compile(r'^##\s+(.+?)\s*$')
# A '### ' sub-heading or a whole-line **bold** label (e.g. a parameter group).
_SUBHEAD_RE = re.compile(r'^(?:###\s+(.+?)|\*\*(.+?)\*\*)\s*$')
_BULLET_RE = re.compile(r'^\s*(?:[-*+]|\d+[.)])\s+\S')
_HISTORY_CHUNK_CHARS = 4000
# Cap for a single analysis chunk — a few bullets, not a whole section.
_MAX_CHUNK_CHARS = 700
_MAX_BULLETS_PER_CHUNK = 6

# nomic-embed-text is asymmetric: it needs a task prefix on every input.
_NOMIC_EMBED = 'nomic' in EMBED_MODEL.lower()

# The analysis prompt emits a leading `DOCUMENT_DATE: YYYY-MM-DD` line.
_DOC_DATE_RE = re.compile(r'DOCUMENT_DATE:\s*(\d{4}-\d{2}-\d{2})', re.IGNORECASE)
_DOC_DATE_LINE_RE = re.compile(r'(?im)^.*DOCUMENT_DATE:.*$\n?')

_collection = None  # cached; one PersistentClient per process


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def embed_text(text: str, *, kind: str = 'document') -> list[float]:
    """Embed text for the vector store.

    nomic-embed-text needs a task instruction prefix: stored passages must be
    prefixed 'search_document:' and queries 'search_query:'. Omitting it
    measurably degrades retrieval — worst for short queries like 'crp'. Pass
    kind='query' when embedding a user question, 'document' (default) for text
    being stored. Non-nomic models are embedded verbatim.
    """
    if _NOMIC_EMBED:
        prefix = 'search_query: ' if kind == 'query' else 'search_document: '
        text = prefix + text
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)['embedding']


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_PATH)
        _collection = client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def store_source_file(path: Path) -> str:
    """Copy an uploaded file into the managed store; return its absolute path."""
    SOURCE_FILES_DIR.mkdir(parents=True, exist_ok=True)
    dest = SOURCE_FILES_DIR / path.name
    shutil.copy2(path, dest)
    return str(dest.resolve())


def split_analysis_sections(analysis: str) -> list[dict]:
    """Split an analysis into small, self-contained chunks for retrieval.

    First by '## ' headings, then within each section long bullet lists are
    packed into chunks of at most a few bullets each, splitting further at
    '### ' or whole-line bold sub-headings. Every chunk repeats its heading
    breadcrumb so it still makes sense when retrieved on its own.

    The previous splitter emitted exactly one chunk per '## ' heading, so the
    'Parameter values' section — dozens of unrelated parameters — became a
    single chunk whose embedding was a blurry average that matched no specific
    parameter query well. Fine-grained chunks are the fix.
    """
    chunks: list[dict] = []
    for h2_title, body in _split_h2(analysis):
        chunks.extend(_chunk_section(h2_title, body))
    return [c for c in chunks if c['text'].strip()]


def _split_h2(analysis: str) -> list[tuple[str, list[str]]]:
    """Split into [(heading_title, body_lines), ...] on '## ' headings."""
    sections: list[tuple[str, list[str]]] = []
    title = 'preamble'
    buf: list[str] = []
    for line in analysis.splitlines():
        match = _H2_RE.match(line.strip())
        if match:
            if buf:
                sections.append((title, buf))
            title = match.group(1).strip()
            buf = []
        else:
            buf.append(line)
    if buf:
        sections.append((title, buf))
    return sections


def _chunk_section(h2_title: str, lines: list[str]) -> list[dict]:
    """Break one '## ' section into small chunks, packing bullet runs."""
    out: list[dict] = []
    subhead = ''
    buf: list[str] = []
    bullets = 0
    chars = 0

    def flush() -> None:
        nonlocal buf, bullets, chars
        body = '\n'.join(buf).strip()
        if body:
            header = f'## {h2_title}'
            if subhead:
                header += f'\n**{subhead}**'
            breadcrumb = f'{h2_title} › {subhead}' if subhead else h2_title
            out.append({'title': breadcrumb, 'text': f'{header}\n{body}'})
        buf, bullets, chars = [], 0, 0

    for line in lines:
        stripped = line.strip()
        match = _SUBHEAD_RE.match(stripped)
        if match:
            flush()
            subhead = (match.group(1) or match.group(2) or '').strip()
            continue
        if not stripped:
            if not buf:
                continue                      # drop leading blank lines
            if chars > _MAX_CHUNK_CHARS:      # paragraph break in long prose
                flush()
                continue
            buf.append(line)
            continue
        is_bullet = bool(_BULLET_RE.match(line))
        if is_bullet and buf and (
            bullets >= _MAX_BULLETS_PER_CHUNK
            or chars + len(line) > _MAX_CHUNK_CHARS
        ):
            flush()
        buf.append(line)
        chars += len(line)
        if is_bullet:
            bullets += 1
    flush()
    return out


def _chunk_raw_text(text: str, size: int = _HISTORY_CHUNK_CHARS) -> list[str]:
    """Chunk free text on blank lines, packing paragraphs up to `size` chars."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks: list[str] = []
    buf = ''
    for para in paragraphs:
        if buf and len(buf) + len(para) + 2 > size:
            chunks.append(buf)
            buf = para
        else:
            buf = f'{buf}\n\n{para}' if buf else para
    if buf:
        chunks.append(buf)
    return chunks


def _upsert_chunks(
    doc_id: str,
    filename: str,
    doc_type: str,
    file_path: str,
    chunks: list[dict],
    doc_date: int = 0,
) -> int:
    """chunks: list of {title, text}. Replaces any existing chunks for doc_id.

    `doc_date` is the document's own date as a YYYYMMDD integer (the test/
    report/entry date, not the ingestion time) — 0 when unknown. Stored on
    every chunk so date-scoped queries can range-filter on it.
    """
    if not chunks:
        return 0
    created = _now()
    ids, documents, embeddings, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(f'{doc_id}::section_{i:02d}')
        documents.append(chunk['text'])
        embeddings.append(embed_text(chunk['text']))
        metadatas.append({
            'doc_id': doc_id,
            'filename': filename,
            'doc_type': doc_type,
            'section_title': chunk['title'],
            'file_path': file_path,
            'created_at': created,
            'doc_date': doc_date,
        })
    collection = get_collection()
    collection.delete(where={'doc_id': doc_id})  # drop stale chunks for this doc
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(ids)


def _extract_doc_date(analysis: str) -> tuple[int, str, str]:
    """Pull the leading `DOCUMENT_DATE:` line the analysis prompt emits.

    Returns (yyyymmdd_int, iso_string, analysis_without_that_line). Yields
    (0, '', analysis) when no valid date is present.
    """
    doc_date, iso = 0, ''
    match = _DOC_DATE_RE.search(analysis)
    if match:
        candidate = match.group(1)
        try:
            datetime.strptime(candidate, '%Y-%m-%d')
            iso = candidate
            doc_date = int(candidate.replace('-', ''))
        except ValueError:
            pass
    cleaned = _DOC_DATE_LINE_RE.sub('', analysis).lstrip()
    return doc_date, iso, cleaned


def ingest_analysis(
    doc_id: str,
    filename: str,
    file_path: str,
    doc_type: str,
    analysis: str,
) -> int:
    """Store a report/prescription analysis, split into fine-grained chunks.

    The document's own date is read from the analysis (the `DOCUMENT_DATE:`
    line the model emits), stored as `doc_date` metadata on every chunk, and
    stamped into each chunk's text so the model sees it in context.
    """
    doc_date, iso_date, analysis = _extract_doc_date(analysis)
    sections = split_analysis_sections(analysis)
    if iso_date:
        marker = f'(Document dated {iso_date})'
        for section in sections:
            section['text'] = f"{marker}\n{section['text']}"
    return _upsert_chunks(
        doc_id, filename, doc_type, file_path, sections, doc_date,
    )


def ingest_raw_text(
    doc_id: str,
    filename: str,
    text: str,
    doc_type: str = 'history',
) -> int:
    """Store free-text medical history as-is, chunked if long.

    A history entry's `doc_date` is today — date-scoped queries then span both
    reports and history from the same period.
    """
    chunks = [{'title': 'history', 'text': c} for c in _chunk_raw_text(text)]
    today = int(datetime.now(timezone.utc).strftime('%Y%m%d'))
    return _upsert_chunks(doc_id, filename, doc_type, '', chunks, today)


def list_documents() -> list[dict]:
    """Return one entry per ingested document, newest first."""
    collection = get_collection()
    data = collection.get(include=['metadatas'])
    docs: dict[str, dict] = {}
    for meta in data['metadatas']:
        doc_id = meta.get('doc_id') or meta.get('filename') or 'unknown'
        if doc_id not in docs:
            docs[doc_id] = {
                'doc_id': doc_id,
                'filename': meta.get('filename', doc_id),
                'doc_type': meta.get('doc_type', 'report'),
                'created_at': meta.get('created_at', ''),
                'has_file': bool(meta.get('file_path')),
                'chunks': 0,
            }
        docs[doc_id]['chunks'] += 1
    return sorted(docs.values(), key=lambda d: d['created_at'], reverse=True)


def delete_document(doc_id: str) -> bool:
    """Delete a document's chunks and its stored source file."""
    collection = get_collection()
    data = collection.get(where={'doc_id': doc_id}, include=['metadatas'])
    if not data['ids']:
        return False
    file_path = ''
    for meta in data['metadatas']:
        if meta.get('file_path'):
            file_path = meta['file_path']
            break
    collection.delete(where={'doc_id': doc_id})
    if file_path:
        try:
            Path(file_path).unlink(missing_ok=True)
        except OSError:
            pass
    return True
