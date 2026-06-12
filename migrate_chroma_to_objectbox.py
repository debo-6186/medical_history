"""One-time migration: copy the existing ChromaDB store into ObjectBox.

Reads every chunk (text + metadata + stored embedding) out of the old ChromaDB
collection and writes it into the new ObjectBox store, reusing the 768-d vectors
as-is (no re-OCR, no re-embedding). Idempotent: the ObjectBox store is cleared
first, so re-running yields the same result.

`chromadb` is no longer a project dependency, so run this with it injected:

    uv run --with chromadb python migrate_chroma_to_objectbox.py

After it prints the migrated count, the original rag_db/chroma.sqlite3 can be
archived; the live store is rag_db/objectbox/.
"""
import chromadb

from core import vector_store
from core.config import DB_PATH

# The collection name used by the old ChromaDB store (formerly config.COLLECTION_NAME).
_CHROMA_COLLECTION = 'medical_records'


def _coerce_doc_date(value) -> int:
    """ChromaDB metadata may store doc_date as int, float, or str — normalize."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def migrate() -> int:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(_CHROMA_COLLECTION)
    data = collection.get(include=['documents', 'metadatas', 'embeddings'])

    ids = data['ids']
    documents = data['documents']
    metadatas = data['metadatas']
    embeddings = data['embeddings']
    if not ids:
        print('Nothing to migrate: ChromaDB collection is empty.')
        return 0

    # Group chunks by doc_id so we can use replace_doc (clears any prior rows).
    by_doc: dict[str, list[dict]] = {}
    for cid, text, meta, emb in zip(ids, documents, metadatas, embeddings):
        doc_id = meta.get('doc_id') or meta.get('filename') or cid
        by_doc.setdefault(doc_id, []).append({
            'chunk_id': cid,
            'filename': meta.get('filename', doc_id),
            'doc_type': meta.get('doc_type', 'report'),
            'section_title': meta.get('section_title', ''),
            'file_path': meta.get('file_path', ''),
            'created_at': meta.get('created_at', ''),
            'doc_date': _coerce_doc_date(meta.get('doc_date', 0)),
            'text': text,
            'embedding': list(emb),
        })

    migrated = 0
    for doc_id, items in by_doc.items():
        migrated += vector_store.replace_doc(doc_id, items)
        print(f'  {doc_id}: {len(items)} chunk(s)')

    print(f'Migrated {migrated} chunk(s) across {len(by_doc)} document(s) '
          f'into ObjectBox -> {vector_store.count()} total.')
    return migrated


if __name__ == '__main__':
    migrate()
