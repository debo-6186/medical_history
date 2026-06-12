"""Vector store backed by ObjectBox (on-device HNSW vector search).

Replaces the previous ChromaDB store. ObjectBox keeps everything in a single
portable database file (`<OBJECTBOX_DIR>/data.mdb`) so the finished RAG DB can be
copied to Android and opened by the native ObjectBox binding for on-device
inference. The Android app must define a matching entity (same property names,
types, and UIDs) — see docs/objectbox_android_schema.md.

This module owns the entity schema and exposes a small, explicit API
(`count`, `replace_doc`, `delete_doc`, `vector_search`, `all_chunks`) that the
ingestion (`core.rag`) and retrieval (`core.qa`) layers call. Each stored chunk
carries its document metadata so the app can rank, list, and manage documents.
"""
from pathlib import Path

from objectbox import (
    Entity,
    Float32Vector,
    HnswIndex,
    Id,
    Index,
    Int64,
    Store,
    String,
    VectorDistanceType,
)

from core.config import EMBED_DIM, OBJECTBOX_DIR, VECTOR_DISTANCE

# Pinned next to this module so the entity/property UIDs are deterministic
# regardless of cwd or which module first opens the Store. These UIDs are the
# contract the Android entity must match — keep this file in version control.
_MODEL_JSON = str(Path(__file__).with_name('objectbox-model.json'))

_DISTANCE_TYPES = {
    'cosine': VectorDistanceType.COSINE,
    'euclidean': VectorDistanceType.EUCLIDEAN,
    'dot_product': VectorDistanceType.DOT_PRODUCT,
}
_DISTANCE_TYPE = _DISTANCE_TYPES.get(VECTOR_DISTANCE.lower(), VectorDistanceType.COSINE)


@Entity()
class Chunk:
    """One retrievable chunk of a medical document.

    `chunk_id` is the business key (`'<doc_id>::section_xx'`); `id` is
    ObjectBox's own 64-bit row id. `embedding` carries the HNSW vector index —
    its dimension and distance type are fixed here and must match on Android.
    """
    id = Id
    chunk_id = String(index=Index())
    doc_id = String(index=Index())
    filename = String
    doc_type = String
    section_title = String
    file_path = String
    created_at = String          # ISO-8601 ingestion time
    doc_date = Int64             # the document's own date, yyyymmdd (0 = unknown)
    text = String
    embedding = Float32Vector(
        index=HnswIndex(dimensions=EMBED_DIM, distance_type=_DISTANCE_TYPE)
    )


_META_FIELDS = (
    'doc_id', 'filename', 'doc_type', 'section_title',
    'file_path', 'created_at', 'doc_date',
)

_store = None  # one Store per process


def _box():
    global _store
    if _store is None:
        _store = Store(directory=OBJECTBOX_DIR, model_json_file=_MODEL_JSON)
    return _store.box(Chunk)


def _meta(chunk: Chunk) -> dict:
    """Rebuild the metadata dict the rag/qa layers expect from a Chunk row."""
    return {field: getattr(chunk, field) for field in _META_FIELDS}


def count() -> int:
    return _box().count()


def replace_doc(doc_id: str, items: list[dict]) -> int:
    """Replace all chunks for `doc_id` with `items` (delete-then-insert).

    Each item: chunk_id, filename, doc_type, section_title, file_path,
    created_at, doc_date, text, embedding. Returns the number inserted.
    """
    box = _box()
    _remove_doc(box, doc_id)
    rows = [
        Chunk(
            chunk_id=item['chunk_id'],
            doc_id=doc_id,
            filename=item['filename'],
            doc_type=item['doc_type'],
            section_title=item['section_title'],
            file_path=item['file_path'],
            created_at=item['created_at'],
            doc_date=item['doc_date'],
            text=item['text'],
            embedding=item['embedding'],
        )
        for item in items
    ]
    for row in rows:
        box.put(row)
    return len(rows)


def delete_doc(doc_id: str) -> tuple[int, str | None]:
    """Remove all chunks for `doc_id`.

    Returns (number_removed, one stored file_path or None). A zero count means
    the document was not present.
    """
    box = _box()
    existing = box.query(Chunk.doc_id.equals(doc_id)).build().find()
    file_path = next((c.file_path for c in existing if c.file_path), None)
    for chunk in existing:
        box.remove(chunk.id)
    return len(existing), file_path


def _remove_doc(box, doc_id: str) -> None:
    for chunk in box.query(Chunk.doc_id.equals(doc_id)).build().find():
        box.remove(chunk.id)


def _filter_condition(exclude_prescriptions: bool, date_range: tuple[int, int] | None):
    """Build the combined property-filter condition, or None when unfiltered."""
    cond = None
    if exclude_prescriptions:
        cond = Chunk.doc_type.not_equals('prescription')
    if date_range is not None:
        start, end = date_range
        date_cond = Chunk.doc_date.between(start, end)
        cond = date_cond if cond is None else (cond & date_cond)
    return cond


def vector_search(
    embedding: list[float],
    k: int,
    *,
    exclude_prescriptions: bool = False,
    date_range: tuple[int, int] | None = None,
) -> list[tuple]:
    """Semantic search; returns [(chunk_id, text, meta, distance), ...].

    When a metadata filter is active we over-fetch nearest neighbours so the
    post-filter still leaves a full page of results, then trim back to `k`.
    """
    cond = _filter_condition(exclude_prescriptions, date_range)
    nn_count = k if cond is None else max(k, k * 5)
    nn = Chunk.embedding.nearest_neighbor(embedding, nn_count)
    query = (nn if cond is None else (nn & cond))
    results = _box().query(query).build().find_with_scores()
    return [
        (chunk.chunk_id, chunk.text, _meta(chunk), score)
        for chunk, score in results
    ][:k]


def all_chunks(
    *,
    exclude_prescriptions: bool = False,
    date_range: tuple[int, int] | None = None,
) -> list[tuple]:
    """Every chunk (optionally filtered) as [(chunk_id, text, meta), ...].

    Used by keyword search and document listing. The store is one person's
    history (a few hundred chunks), so a full scan is cheap.
    """
    box = _box()
    cond = _filter_condition(exclude_prescriptions, date_range)
    rows = box.get_all() if cond is None else box.query(cond).build().find()
    return [(c.chunk_id, c.text, _meta(c)) for c in rows]
