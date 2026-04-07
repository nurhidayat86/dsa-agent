# Telesales ↔ customer conversations — file structure for synthetic data (VibeVoice + metadata)

**Purpose:** Context for implementing an **artificial data generator** that outputs **many conversations per generation run**, each suitable for **Microsoft VibeVoice-style TTS** (multi-speaker plain text) plus **machine-readable metadata**.

**Scope:** Mock phone conversations between a **bank customer** and a **tele-sales / outbound sales** agent (cross-sell / upsell). Not complaint intake; focus on scripted sales dialogue.

---

## Design principles

1. **One batch = one metadata JSON file listing every conversation** in that run (`conversations` array). Avoids N small JSON files when you generate hundreds of dialogues at once.
2. **One logical conversation = one entry inside that array + one transcript `.txt` on disk** (path stored on the entry). Keeps VibeVoice input clean in plain text.
3. **`transcript.txt` is only what TTS should see:** lines like `Speaker 0:` / `Speaker 1:` and spoken text—no JSON, no stage directions unless the TTS pipeline strips them.
4. **Per-conversation fields** (customer ref, branch, products, voice mapping, `transcript_path`) live on each object in `conversations`. **Batch-level fields** (batch id, generator version, model) live once at the root of the same JSON.

---

## Recommended directory layout (per generation batch)

Use a **batch folder** per run (timestamp or batch id). Example:

```text
output/
  batches/
    2026-04-07T143022Z_synth/
      metadata.json                        # single file: all conversations for this batch
      conversations/
        conv-00001/
          transcript.txt
        conv-00002/
          transcript.txt
        ...
```

**Optional:** `manifest.jsonl` (see below) only if you want a line-oriented duplicate index for streaming tools.

**Alternative (flat transcripts):** `metadata.json` at root + `transcripts/tel-2026-000001.txt`, paths stored in each conversation object.

---

## `metadata.json` (one file per batch — multiple conversations)

Single JSON document. Root holds **batch-wide** metadata; **`conversations`** is an ordered array of **N** items (one per generated dialogue).

### Root-level fields

| Field | Type | Required | Description |
|--------|------|----------|-------------|
| `batch_id` | string | yes | Unique id for this generation run |
| `record_type` | string | yes | e.g. `bank_telesales_batch` |
| `synthetic` | boolean | yes | Always `true` |
| `generator_version` | string | yes | Semver or build id |
| `model` | string | yes | LLM used to generate dialogue (e.g. `gemini-…`) |
| `created_at` | string | yes | ISO 8601 when the batch was written |
| `conversation_count` | integer | yes | Redundant check: must equal `conversations.length` |
| `conversations` | array[object] | yes | One object per conversation—see below |

### Each element of `conversations[]`

| Field | Type | Required | Description |
|--------|------|----------|-------------|
| `conversation_id` | string | yes | Stable id, unique within project (e.g. `tel-2026-00000001`) |
| `record_type` | string | yes | e.g. `bank_telesales_conversation` |
| `scenario` / `campaign` | string | yes | e.g. `credit_card_upsell`, `savings_cross_sell` |
| `language` | string | yes | BCP-47 or ISO (`en`, `id`) |
| **Customer / bank context** | | | |
| `customer_ref` | string | no | Synthetic token only; empty if anonymous |
| `branch_code` | string | no | e.g. `BR-JAVA-014` |
| `region` | string | no | |
| **Temporal** | | | |
| `conversation_datetime` | string | yes | **Simulated date and time of the call**, ISO 8601 with timezone (e.g. `2026-04-07T14:30:22+07:00`). Recommended for new generators. |
| **Offer** | | | |
| `primary_product` | string | yes | Normalized code, e.g. `credit_card_gold` |
| `secondary_products` | array[string] | no | Additional cross-sell targets |
| **Files / VibeVoice** | | | |
| `transcript_path` | string | yes | Path to **`transcript.txt` relative to batch folder** (or absolute if your tooling fixes roots) |
| `speaker_map` | object | yes | Maps `Speaker` index → role and display name |
| `vibevoice_speaker_names` | array[string] | yes | Ordered: index `0` = `Speaker 0`, etc. |

Do **not** repeat `generator_version` / `model` / `synthetic` on every conversation object unless you need self-contained rows for ETL; the batch root is the source of truth for provenance.

**Example `speaker_map` (2 speakers):**

```json
{
  "0": {"role": "tele_sales_agent", "display_name": "Agent"},
  "1": {"role": "customer", "display_name": "Customer"}
}
```

**Example `vibevoice_speaker_names`:**  
`["Alice", "Frank"]` — same order as `Speaker 0`, `Speaker 1`.

> **VibeVoice note:** Public examples use `Speaker 0:`, `Speaker 1:`, … up to four speakers, with a parallel list of voice names by index. Confirm CLI/API for your deployment.

### Minimal full-file example (2 conversations, abbreviated)

```json
{
  "batch_id": "2026-04-07T143022Z_synth",
  "record_type": "bank_telesales_batch",
  "synthetic": true,
  "generator_version": "0.1.0",
  "model": "gemini-2.0-flash",
  "created_at": "2026-04-07T14:30:22+07:00",
  "conversations": [
    {
      "conversation_id": "tel-2026-00000001",
      "record_type": "bank_telesales_conversation",
      "scenario": "credit_card_upsell",
      "language": "en",
      "conversation_datetime": "2026-04-07T10:15:00+07:00",
      "customer_ref": "SYN-CUST-ABC123",
      "branch_code": "BR-JAVA-014",
      "primary_product": "credit_card_gold",
      "transcript_path": "conversations/conv-00001/transcript.txt",
      "speaker_map": {
        "0": {"role": "tele_sales_agent", "display_name": "Agent"},
        "1": {"role": "customer", "display_name": "Customer"}
      },
      "vibevoice_speaker_names": ["Alice", "Frank"]
    },
    {
      "conversation_id": "tel-2026-00000002",
      "record_type": "bank_telesales_conversation",
      "scenario": "savings_cross_sell",
      "language": "id",
      "conversation_datetime": "2026-04-07T16:45:30+07:00",
      "transcript_path": "conversations/conv-00002/transcript.txt",
      "speaker_map": {
        "0": {"role": "tele_sales_agent", "display_name": "Agent"},
        "1": {"role": "customer", "display_name": "Customer"}
      },
      "vibevoice_speaker_names": ["Alice", "Frank"]
    }
  ]
}
```

---

## `manifest.jsonl` (optional)

Use only when you need **JSON Lines** (e.g. Spark, incremental ingestion) **without** loading the full `metadata.json`.

Each line can mirror one `conversations[]` element plus `batch_id` and minimal batch provenance (including **`conversation_datetime`** if present), **or** only `conversation_id` + `transcript_path` + pointers. If you maintain `manifest.jsonl`, keep it **consistent** with `metadata.json` (same ids, paths, and times).

---

## `transcript.txt` (per conversation — VibeVoice-oriented)

- **Encoding:** UTF-8.
- **Format:** One **turn** per line (recommended).

**Pattern:**

```text
Speaker 0: Good afternoon, am I speaking with the account holder?
Speaker 1: Yes, speaking.
Speaker 0: I'm calling from Example Bank regarding a preferred rate on our savings bundle…
Speaker 1: I'm not interested in new products right now.
```

- **Speaker indices** must match `vibevoice_speaker_names` on the **corresponding** object in `metadata.json` → `conversations[]`.
- **No** JSON, markdown, or bracketed stage directions unless a preprocessor strips them.
- **Punctuation:** For mixed Chinese/English, some VibeVoice docs suggest English-style commas/periods for stability.
- **Length:** Chunk long monologues into multiple `Speaker k:` lines if synthesis runs too fast.

---

## Many conversations per generation (batch contract)

1. Generator accepts parameters: `num_conversations`, language, scenario mix, branch count, etc.
2. Build the root object with `batch_id`, provenance, and empty `conversations: []`.
3. For **each** conversation `i` in `1..N`:
   - Create `conversations/conv-{i:05d}/transcript.txt` (or flat `transcripts/...`).
   - Append one object to `conversations` with `transcript_path` and all per-dialogue fields.
4. Write **`metadata.json`** once at the end (or stream-append only if your format allows; usually write atomically after all transcripts exist).
5. Optionally append **`manifest.jsonl`** per line, or emit **`batch_summary.json`** for QA counts.

---

## QA checklist for implementers

- [ ] `conversation_count` (if present) equals `conversations.length`.
- [ ] Every `conversation_id` is unique within the batch.
- [ ] Every `transcript_path` exists on disk and resolves from the batch folder.
- [ ] Every `transcript.txt` uses only `Speaker 0:` … `Speaker K:` with **K + 1** entries in that row’s `vibevoice_speaker_names`.
- [ ] No real PII: synthetic `customer_ref` only.
- [ ] Root `synthetic: true`; optional per-row duplication only if downstream requires it.
- [ ] If you populate time of call, use `conversation_datetime` (ISO 8601); avoid contradicting `conversation_date` / `conversation_start`.

---

## Relation to other docs

- [`data_structure_context.md`](data_structure_context.md) — general customer-voice envelope patterns.
- [`bank_branch_raw_feedback_context.md`](bank_branch_raw_feedback_context.md) — branch-attributed **complaint/feedback** raw records (different record shape; can share `branch_code` / `customer_ref` conventions).

---

*Document version: 0.3 — batch `metadata.json` with embedded `conversations[]`; per-row `conversation_datetime`; one `transcript.txt` per dialogue.*
