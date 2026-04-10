# ChromaDB ingestion spec: bank feedback (AI agent reference)

**Purpose:** Single source of truth for implementing Python that loads **`vector-db-writer/data/bank_feedback/*.jsonl`** into **ChromaDB**, using **`verbatim`** as the only embedded text and all other JSON fields as **metadata**.

**Embedding model:** **Qwen3-Embedding-8B** (or the exact Hugging Face repo id you deploy). Ingestion and query **must** use the **same** model, pooling, normalization, and truncation policy. Record the model id and dimension in collection metadata or a small config file alongside the persisted DB.

**Vector store:** **ChromaDB** (persistent client recommended for local dev: e.g. `chromadb.PersistentClient(path=...)`).

---

## 1. Source data

- **Root directory (relative to repo):** `vector-db-writer/data/bank_feedback/`
- **Files:** one or more `*.jsonl` files (e.g. `BR-BALI-003.jsonl`, `BR-JAVA-006.jsonl`).
- **Encoding:** UTF-8.
- **Grain:** **one JSON object per line** = one logical complaint/feedback **case**.

### 1.1 Input JSON fields (per line)

Each line is a flat JSON object. Observed / expected keys:

| Field | Role |
|--------|------|
| `verbatim` | **Embedding text** — main customer narrative; required for vectorization. |
| `case_id` | **Primary business id** — use as Chroma `ids` when unique (see §3). |
| `ticket_number` | Often same as `case_id`; keep in metadata for exports. |
| `record_type` | Metadata (e.g. `bank_complaint_raw`). |
| `submission_type` | Metadata (e.g. `standard_feedback_form`, `detailed_complaint_ticket`). |
| `branch_code` | Metadata. |
| `region` | Metadata. |
| `channel` | Metadata. |
| `created_at` | Metadata (ISO 8601 string). |
| `updated_at` | Metadata (ISO 8601 string). |
| `status` | Metadata. |
| `product_bucket` | Metadata. |
| `subject` | Metadata (short line); do **not** embed unless using optional mode §6. |
| `language` | Metadata. |
| `has_attachments` | Metadata (boolean). |
| `customer_ref` | Metadata (string). |
| `assigned_queue` | Metadata; may be JSON `null`. |
| `synthetic` | Metadata (boolean). |
| `generator_version` | Metadata (string). |
| `model` | Metadata (string; e.g. LLM used to generate the record). |

**Do not** duplicate `case_id` inside `metadata` if the Chroma `id` is already `case_id` (optional omission to reduce redundancy).

---

## 2. Chroma collection mapping

| Chroma concept | Mapping |
|----------------|---------|
| **Collection name** | Suggested: `bank_feedback` (single collection for all JSONL under `bank_feedback/`). |
| **`ids`** | Prefer **`case_id`** per row. Must be **unique** within the collection. If a future batch can collide, use a stable composite: e.g. `{source_stem}:{case_id}` where `source_stem` is the JSONL filename without `.jsonl`. |
| **`documents`** | **`verbatim`** string only (default). Non-empty string expected; skip or log rows with missing/empty `verbatim`. |
| **`embeddings`** | Float vectors from **Qwen3-Embedding-8B** (or pass `None` only if using a Chroma embedding function that wraps the **same** model — preferably embed in application code for full control). |
| **`metadatas`** | Flat dict: **all fields except `verbatim`**, normalized per §4. Optional extra: `source_file` (JSONL basename) for lineage. |

**Ingestion rule:** **one Chroma point per JSONL line** (one `case_id`).

---

## 3. Identifier strategy

1. Read `case_id` from JSON.
2. If missing, skip row with error and increment failure counter **or** derive id from hash of full row (only if product requirements allow — prefer failing loud).
3. If ingesting multiple files and `case_id` might collide across batches, set  
   `id = f"{Path(path).stem}:{case_id}"`  
   and document that choice in README/CLI help.

---

## 4. Metadata normalization (Chroma compatibility)

Chroma metadata values must be **scalar** types Chroma accepts (`str`, `int`, `float`, `bool`). Nested objects/arrays are not suitable.

**Required normalizations:**

- **`assigned_queue` is `null`:** use empty string `""` or literal `"none"`; be consistent across the project.
- **Booleans:** keep Python `bool` for `has_attachments` and `synthetic`.
- **All other fields:** stringify if necessary; dates are already ISO strings in source — keep as `str`.

**Do not** put the full `verbatim` into metadata (redundant and huge).

---

## 5. Embedding pipeline (Qwen3-Embedding-8B)

Implementation guidance for the coding agent:

1. Load tokenizer + model per model card (device, dtype, trust_remote_code if required).
2. For each batch of `documents` (verbatim strings):
   - Apply the **same** truncation/max-length policy documented for Qwen3-Embedding-8B.
   - Produce one **fixed-size** embedding per row (length = model output dim).
   - If the model card specifies **L2 normalize** or special pooling, follow it for parity with later **query** embeddings.
3. Call `collection.add(ids=..., embeddings=..., documents=..., metadatas=...)` in batches sized to fit memory.

**Persistence:** store alongside the Chroma directory (or in collection metadata) at minimum:

- `embedding_model_id` (e.g. Hugging Face repo id)
- `embedding_dim`
- optional: `max_length`, `normalize`, framework version

so future agents do not mix incompatible vectors in the same collection.

---

## 6. Optional: richer document text (not default)

If retrieval quality is low on short subjects, support a **flag** (e.g. `--include-subject-in-document`):

- `document = subject.strip() + "\n\n" + verbatim`  
  when `subject` is non-empty; otherwise `document = verbatim`.

Default for automated pipelines should remain **`verbatim` only** unless the product spec opts in.

---

## 7. Suggested implementation checklist

- [ ] Recursively or explicitly glob `data/bank_feedback/*.jsonl`.
- [ ] Parse JSONL line-by-line; handle malformed JSON with line number in errors.
- [ ] Validate required keys: at minimum `case_id`, `verbatim`.
- [ ] Normalize metadata (null `assigned_queue` → `""` or `"none"`).
- [ ] Batch embed with Qwen3-Embedding-8B; batch add to Chroma.
- [ ] Idempotent runs: either **`collection.upsert`** or delete-by-id before add; document behavior.
- [ ] Log counts: lines read, added, skipped, failed.
- [ ] CLI or module entrypoint: inputs = data dir, Chroma persist path, collection name, model id/path.

---

## 8. Example logical row (after mapping)

```text
id:          "BK-2025-00000077"
document:    <verbatim only>
embeddings:  <list[float], dim = embedding_dim>
metadata:    {
               "record_type": "...",
               "submission_type": "...",
               "ticket_number": "...",
               "branch_code": "...",
               "region": "...",
               "channel": "...",
               "created_at": "...",
               "updated_at": "...",
               "status": "...",
               "product_bucket": "...",
               "subject": "...",
               "language": "...",
               "has_attachments": false,
               "customer_ref": "...",
               "assigned_queue": "",
               "synthetic": true,
               "generator_version": "...",
               "model": "...",
               "source_file": "BR-BALI-003.jsonl"
             }
```

---

## 9. Cross-reference

Synthetic raw-layer field meanings and operational context: `ai-data-generator/docs/bank_branch_raw_feedback_context.md`.

This file is **normative for vector-db-writer → Chroma** ingestion of **bank_feedback** JSONL only. Other corpora (e.g. telesales) need their own spec.
