"""Retrieval-augmented Q&A over the ingested medical records.

Refactored from ask.py: the interactive loop is gone; this module exposes the
query-expansion + retrieval + prompt-building + streaming pieces the web chat
route consumes.

Retrieval is hybrid: the question is first sent to the chat model, which
expands it (so 'crp' also searches 'C-Reactive Protein'), classifies its
intent, and extracts any date range it asks for. A vector search and a literal
keyword search are then fused with reciprocal rank fusion. Value questions
exclude prescription chunks; date-scoped questions ('… for May 2026') filter
on each chunk's document date.
"""
import re
from collections.abc import Iterable, Iterator
from datetime import date, datetime

import ollama

from core.config import (
    CHAT_MODEL,
    CHAT_NUM_CTX,
    MODEL_KEEP_ALIVE,
    MODEL_THINKING,
    TOP_K,
)
from core import vector_store
from core.rag import embed_text

SYSTEM_PROMPT = (
    'You are a medical records assistant. You answer questions about the user\'s '
    'past medical reports.\n\n'
    'Each turn you are given record excerpts, every excerpt labelled '
    '[file: <filename> | type: <report/prescription/history>]. Rules:\n'
    '- Answer using ONLY the provided excerpts and the earlier conversation.\n'
    '- Quote values with units and reference ranges when present.\n'
    '- Measured values, lab readings and test results come from report '
    'excerpts. Do not quote a number from a prescription (e.g. a medication '
    'dose) as if it were a lab result.\n'
    '- An excerpt may begin with "(Document dated YYYY-MM-DD)" — use that date '
    'when the question is about when a test was done or to order results in '
    'time.\n'
    '- Refer to a source document by its filename only. Never output file '
    'system paths. The documents your answer draws from are listed as '
    'downloadable links below your reply, so do not tell the user a file is '
    'unavailable or ask them to locate it.\n'
    '- If a fact is not in the excerpts, say "Not found in the available '
    'records." Do not guess or invent.\n'
    '- End each answer with a one-line disclaimer that this is educational '
    'information, not medical advice.'
)

# --- Query expansion + intent / date classification -----------------------

EXPANSION_PROMPT = (
    'You prepare a question about a patient\'s own medical records for a '
    'search engine. Output EXACTLY three lines and nothing else:\n'
    '  Intent: values | general\n'
    '  Dates: <YYYY-MM-DD to YYYY-MM-DD> | none\n'
    '  Terms: <comma-separated search terms>\n\n'
    'Intent — "values" if the question asks for specific measured numbers, '
    'lab readings, test results, levels or counts; otherwise "general".\n\n'
    'Dates — if the question restricts to a time period, give the inclusive '
    'date range, resolving months, years and relative terms against the given '
    '"Today" date (e.g. "May 2026" -> 2026-05-01 to 2026-05-31; "in 2025" -> '
    '2025-01-01 to 2025-12-31; "last month"/"this year" relative to Today). '
    'If the question names no time period, write "none".\n\n'
    'Terms — expand every medical abbreviation to its full name AND keep the '
    'abbreviation; add the standard lab/test parameter names and common '
    'spelling variants. Do NOT answer the question or invent values.\n\n'
    'Example (Today is 2026-06-10):\n'
    'Question: pull my glucose fasting data for may 2026\n'
    'Intent: values\n'
    'Dates: 2026-05-01 to 2026-05-31\n'
    'Terms: Glucose Fasting, Fasting Glucose, Fasting Blood Sugar, FBS, '
    'Plasma Glucose Fasting\n\n'
    'Example (Today is 2026-06-10):\n'
    'Question: what was my crp\n'
    'Intent: values\n'
    'Dates: none\n'
    'Terms: CRP, C-Reactive Protein, C Reactive Protein, inflammation marker\n\n'
    'Example (Today is 2026-06-10):\n'
    'Question: what tablets has the doctor prescribed this year\n'
    'Intent: general\n'
    'Dates: 2026-01-01 to 2026-12-31\n'
    'Terms: medication, prescription, tablet, drug, dose\n'
)

# Expansion is deterministic per (day, question); cache so a repeat is free.
_EXPANSION_CACHE: dict[str, tuple] = {}
_TERM_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9\-]{2,}')
_ISO_DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}')
_RRF_K = 60  # reciprocal-rank-fusion damping constant


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _parse_date_range(line: str) -> tuple[int, int] | None:
    """Pull a date range out of the model's `Dates:` line.

    Returns (start, end) as YYYYMMDD integers, or None when no valid date is
    present. A single date becomes a one-day range.
    """
    nums: list[int] = []
    for token in _ISO_DATE_RE.findall(line):
        try:
            datetime.strptime(token, '%Y-%m-%d')
        except ValueError:
            continue
        nums.append(int(token.replace('-', '')))
    if not nums:
        return None
    return min(nums), max(nums)


def expand_query(query: str) -> tuple[str, list[str], bool, tuple[int, int] | None]:
    """Expand a question for retrieval and classify intent + date scope.

    Returns (vector_text, keywords, is_value_query, date_range):
    - vector_text / keywords feed the vector and keyword searches.
    - is_value_query True -> retrieve() drops prescription chunks.
    - date_range (start, end) as YYYYMMDD ints, or None -> retrieve() keeps
      only chunks whose document date falls in that range.

    MUST be called with MODEL_LOCK held — it invokes the chat model. On any
    failure it degrades gracefully to the raw query so retrieval still runs.
    """
    today = date.today().isoformat()
    key = f'{today}|{query.strip().lower()}'
    if key in _EXPANSION_CACHE:
        return _EXPANSION_CACHE[key]

    intent = 'general'
    date_range: tuple[int, int] | None = None
    terms = ''
    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {'role': 'system', 'content': EXPANSION_PROMPT},
                {'role': 'user', 'content': f'Today is {today}.\nQuestion: {query}'},
            ],
            think=False,        # expansion never needs chain-of-thought
            keep_alive=MODEL_KEEP_ALIVE,
            # num_predict caps output: three short lines — without a cap a
            # rambling model can make this step take tens of seconds.
            options={'num_ctx': CHAT_NUM_CTX, 'temperature': 0.0,
                     'num_predict': 160},
        )
        content = response['message']['content'].strip()
        leftover: list[str] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith('intent:'):
                intent = 'values' if 'value' in low else 'general'
            elif low.startswith('dates:'):
                date_range = _parse_date_range(line)
            elif low.startswith('terms:'):
                terms = line.split(':', 1)[1].strip()
            else:
                leftover.append(line)
        if not terms and leftover:        # model skipped the 'Terms:' label
            terms = leftover[0]
    except Exception:  # noqa: BLE001 - never let expansion break retrieval
        terms = ''

    is_value_query = intent == 'values'
    if terms:
        vector_text = f'{query}\n{terms}'
        keywords = _dedupe(t.lower() for t in terms.split(','))
    else:
        vector_text = query
        keywords = _dedupe(m.group(0).lower() for m in _TERM_RE.finditer(query))

    result = (vector_text, keywords, is_value_query, date_range)
    _EXPANSION_CACHE[key] = result
    return result


# --- Retrieval -------------------------------------------------------------
# Filters are two plain params threaded through to the vector store:
# `exclude_prescriptions` (value questions) and `date_range` ('… for May 2026').


def _vector_hits(
    vector_text: str,
    k: int,
    exclude_prescriptions: bool,
    date_range: tuple[int, int] | None,
) -> list[tuple]:
    """Semantic search; returns [(chunk_id, document, metadata, distance), ...]."""
    return vector_store.vector_search(
        embed_text(vector_text, kind='query'),
        k,
        exclude_prescriptions=exclude_prescriptions,
        date_range=date_range,
    )


def _keyword_hits(
    keywords: list[str],
    k: int,
    exclude_prescriptions: bool,
    date_range: tuple[int, int] | None,
) -> list[tuple]:
    """Literal search: score every chunk by how many keywords it contains.

    Returns [(chunk_id, document, metadata), ...]. The store is one person's
    medical history (a few hundred chunks at most), so scanning every chunk is
    cheap and needs no embedding call.
    """
    if not keywords:
        return []
    scored: list[tuple] = []
    for cid, doc, meta in vector_store.all_chunks(
        exclude_prescriptions=exclude_prescriptions, date_range=date_range,
    ):
        haystack = f"{doc}\n{meta.get('section_title', '')}".lower()
        score = sum(1 for kw in keywords if kw in haystack)
        if score:
            scored.append((score, cid, doc, meta))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [(cid, doc, meta) for _score, cid, doc, meta in scored[:k]]


def _fuse(vec_hits: list[tuple], kw_hits: list[tuple], k: int) -> list[tuple]:
    """Reciprocal rank fusion of the vector and keyword result lists.

    Returns [(document, metadata, distance), ...]. A chunk found by both
    retrievers is ranked highest; either retriever alone can still surface it.
    """
    scores: dict[str, float] = {}
    payload: dict[str, tuple] = {}
    for rank, (cid, doc, meta, dist) in enumerate(vec_hits):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
        payload[cid] = (doc, meta, dist)
    for rank, (cid, doc, meta) in enumerate(kw_hits):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
        payload.setdefault(cid, (doc, meta, None))
    ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [payload[cid] for cid in ranked[:k]]


def retrieve(
    vector_text: str,
    keywords: list[str],
    is_value_query: bool = False,
    date_range: tuple[int, int] | None = None,
    k: int = TOP_K,
) -> list[tuple]:
    """Hybrid retrieval: vector search + keyword search, fused with RRF.

    Returns [(document, metadata, distance), ...]. Pass the tuple from
    `expand_query`. Value questions exclude prescription chunks; a date_range
    keeps only chunks whose document date falls inside it (a chunk with an
    unknown date is excluded — "for May 2026" means only May 2026).
    """
    if vector_store.count() == 0:
        return []
    vec_hits = _vector_hits(vector_text, k, is_value_query, date_range)
    kw_hits = _keyword_hits(keywords, k, is_value_query, date_range)
    return _fuse(vec_hits, kw_hits, k)


# --- Prompt building -------------------------------------------------------

def format_context(hits: list[tuple]) -> str:
    blocks = []
    for doc, meta, _dist in hits:
        fname = meta.get('filename', '?')
        dtype = meta.get('doc_type', 'report')
        blocks.append(f'[file: {fname} | type: {dtype}]\n{doc}')
    return '\n\n---\n\n'.join(blocks)


def sources_from_hits(hits: list[tuple]) -> list[dict]:
    """Deduplicated source list for the SSE `sources` event."""
    seen: dict[str, dict] = {}
    for _doc, meta, _dist in hits:
        doc_id = meta.get('doc_id') or meta.get('filename')
        if doc_id and doc_id not in seen:
            seen[doc_id] = {
                'doc_id': doc_id,
                'filename': meta.get('filename', doc_id),
                'has_file': bool(meta.get('file_path')),
            }
    return list(seen.values())


def build_messages(
    query: str,
    hits: list[tuple],
    history: list[dict],
) -> tuple[list[dict], str]:
    """Return (messages for ollama.chat, the user message stored in history).

    Today's date is injected into the system prompt every turn so the model has
    an absolute year anchor when computing future dates from a document date —
    without it, follow-up arithmetic ("next HRCT after 6 months" from a
    prescription dated 2026-05-16) can drift to the model's training-era year.
    """
    context = format_context(hits) if hits else '(no records ingested yet)'
    user_msg = f'Record excerpts:\n{context}\n\nQuestion: {query}'
    system_content = f'Today is {date.today().isoformat()}.\n\n{SYSTEM_PROMPT}'
    messages = [
        {'role': 'system', 'content': system_content},
        *history,
        {'role': 'user', 'content': user_msg},
    ]
    return messages, user_msg


def stream_answer(messages: list[dict]) -> Iterator[str]:
    """Yield answer text fragments from the chat model as they are produced.

    Thinking is off by default (see MODEL_THINKING): a thinking model emits
    reasoning into `message.thinking`, not `message.content`, so leaving it on
    can stream a long, empty-`content` response. temperature is held low for
    consistent, grounded extraction of values from the excerpts.
    """
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
        think=MODEL_THINKING,
        keep_alive=MODEL_KEEP_ALIVE,
        options={'num_ctx': CHAT_NUM_CTX, 'temperature': 0.0},
        stream=True,
    )
    for part in stream:
        yield part['message'].get('content') or ''
