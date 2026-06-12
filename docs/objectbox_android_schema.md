# ObjectBox schema — Android handoff

The RAG DB is built on the laptop by the Python pipeline (`core/vector_store.py`)
and stored as a single portable ObjectBox file. To use it on Android for on-device
inference, copy that file onto the device and open it with the **native ObjectBox
Kotlin/Java (or Flutter) binding** — *not* `objectbox-python`, which has no
Android/bionic native library and cannot run under Termux.

For ObjectBox to open a file written elsewhere, the Android entity must match the
Python entity **exactly**: same entity name, same property names and types, the
same HNSW vector dimensions + distance type, and — critically — the same **UIDs**.

## Source of truth: `core/objectbox-model.json`
The Python side persists every entity/property/index UID in
`core/objectbox-model.json` (path pinned in `core/vector_store.py`). Keep it in
version control; never hand-edit it. The Android model must reproduce these UIDs.

## Entity: `Chunk`
One row per retrievable chunk of a medical document. Property types below are the
ObjectBox type codes from the model JSON (`9` = String, `6` = Long/Int64,
`28` = Float32Vector).

| Property        | Type           | Notes |
|-----------------|----------------|-------|
| `id`            | Long (6)       | ObjectBox row id (`@Id`) |
| `chunk_id`      | String (9)     | indexed. Business key `"<doc_id>::section_xx"` |
| `doc_id`        | String (9)     | indexed. Groups chunks of one document |
| `filename`      | String (9)     | |
| `doc_type`      | String (9)     | `report` / `prescription` / `history` |
| `section_title` | String (9)     | heading breadcrumb |
| `file_path`     | String (9)     | absolute path on the laptop (ignore on device) |
| `created_at`    | String (9)     | ISO-8601 ingestion time |
| `doc_date`      | Long (6)       | document's own date as `yyyymmdd` (0 = unknown) |
| `text`          | String (9)     | the chunk text used for context |
| `embedding`     | Float32Vector (28) | **HNSW index, dimensions = 768, distance = COSINE** |

The vector index parameters (`dimensions=768`, `distance_type=COSINE`) live in
`core/config.py` (`EMBED_DIM`, `VECTOR_DISTANCE`) and `core/vector_store.py`. They
**must** be identical on Android.

### Matching Kotlin entity (illustrative)
```kotlin
@Entity
data class Chunk(
    @Id var id: Long = 0,
    @Index var chunkId: String = "",      // maps to property name "chunk_id"
    @Index var docId: String = "",        // "doc_id"
    var filename: String = "",
    var docType: String = "",             // "doc_type"
    var sectionTitle: String = "",        // "section_title"
    var filePath: String = "",            // "file_path"
    var createdAt: String = "",           // "created_at"
    var docDate: Long = 0,                // "doc_date"
    var text: String = "",
    @HnswIndex(dimensions = 768, distanceType = VectorDistanceType.COSINE)
    var embedding: FloatArray = floatArrayOf(),
)
```
If you let Gradle generate fresh UIDs, the copied data file will **not** open.
Reproduce the UIDs from `core/objectbox-model.json` (in Kotlin via `@Uid(...)` on
the entity and each property, or by seeding the Android `objectbox-models/`
default model with the same JSON).

## Putting the file on the device
The store directory is `rag_db/objectbox/` (`data.mdb` + `lock.mdb`). Copy
`data.mdb` to the app before first DB access:
- Place it under the app's files dir at `objectbox/objectbox/data.mdb` (default
  `BoxStore` location), **or**
- Use `BoxStoreBuilder.initialDbFile(preparedFile)` so ObjectBox seeds the store
  from the prepared file on first open.
Open the store *before* the app touches the ObjectBox API for the first time.

## Open risks to validate on-device
- **HNSW index portability is not yet confirmed.** ObjectBox documents data-file
  portability, but neither the docs nor the FAQ explicitly guarantee the HNSW
  vector index transfers intact across a copied file. Test a real
  `nearest_neighbor` query on Android against the copied DB before relying on it.
  If the index doesn't survive the copy, re-inserting the rows on-device rebuilds
  it (the vectors themselves are stored regardless).
- **Query embeddings on-device.** Retrieval needs the *query* embedded with the
  same model that produced the stored vectors (`nomic-embed-text`, 768-d). That
  embedding step must exist on Android too; it is out of scope for this repo.
