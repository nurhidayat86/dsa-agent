# Bank branch feedback & complaints — raw layer (coding agent context)

**Purpose:** Use this file when implementing the artificial data generator so synthetic **feedback and complaint** records resemble **real, pre-processed** operational exports from a **large retail/commercial bank** with **20–32 physical branches**.

**Related:** See `data_structure_context.md` for cross-cutting envelopes (JSONL, provenance) and other VOC types (NPS, reviews, etc.).

---

## Domain assumptions (generator invariants)

- **Institution:** One bank brand; multi-branch retail footprint.
- **Branch count:** **20–32** distinct branches—each record must carry a **branch identifier** (`branch_code` or equivalent), not isolated per-branch databases.
- **Channel reality:** Most complaints/feedback are captured in a **central** case/ticket or CRM system; **branch** is an **attribute** (org unit / cost center / servicing branch), not a separate storage silo.
- **“Raw” definition:** Data **before** NLP, topic modeling, sentiment scoring, deduplication, or curated taxonomies. Optional fields may be empty; classifications may be coarse or wrong.

---

## How raw data is structured in the real world (what to mimic)

### Cardinality

- **One primary row per case/ticket/submission** (plus separate tables/files for comment history or attachments in full systems—for generation, a **single flat record** or `comments[]` array is enough unless you need thread fidelity).

### Branch attribution

- **`branch_code`** (or `branch_id`): internal code (string), unique per branch within the bank.
- Optional: **`region`**, **`cost_center`**—align if you want regional rollups.

### Typical operational / export columns (pre-analytics)

Use these as the **baseline schema** for synthetic **raw** bank complaints/feedback:

| Field | Usage for generator | Notes |
|--------|----------------------|--------|
| `case_id` / `ticket_number` | Required; unique string | Stable ID across exports |
| `branch_code` | Required | From a closed set of 20–32 codes |
| `channel` | Required | e.g. `branch_counter`, `branch_tablet`, `phone`, `email`, `mobile_app`, `internet_banking`, `social`, `ombudsman_escalation` |
| `created_at` / `updated_at` | ISO 8601 strings | `updated_at` optional or nullable |
| `status` | e.g. `open`, `in_progress`, `resolved`, `closed`, `escalated` | Raw dumps often inconsistent casing—pick one convention for synthetic data |
| `product_bucket` | Optional; coarse enum | e.g. `account`, `lending`, `card`, `fees`, `digital_channel`, `mortgage`, `other`—often frontline dropdown |
| `subject` | Optional string | Short line; may duplicate start of body |
| `verbatim` / `body` | **Required** main free text | Complaint or feedback narrative; mixed tone/length |
| `language` | Optional | e.g. `en`, `id` if bilingual jurisdiction |
| `has_attachments` | boolean | True occasionally; no need to generate file binaries |
| `customer_ref` | Optional; **synthetic token only** | e.g. hashed customer key—**do not** generate realistic national IDs, full account numbers, or phone numbers unless explicitly required and flagged |
| `assigned_queue` / `owner` | Optional | Internal handling; can be null in raw intake |

**Usually *not* present at true raw source** (add only if simulating “lightly processed” export): `sentiment`, `topic`, `priority`, `severity`, `resolution_summary`. If you include them, label them as optional post-intake fields or a separate “silver” dataset to avoid conflating layers.

---

## Storage patterns to simulate (not implement)

- **Source of truth:** Enterprise CRM / case management—relational or SaaS with APIs.
- **“Bronze” / landing for data teams:** Immutable **CSV or JSONL** batches in a object store or lake, often partitioned by **`ingest_date`**; **`branch_code` lives in the row**, not as “one folder per branch” (both exist; row-level is more common for unified loads).
- **Attachments:** Represent as **`has_attachments`** + optional `attachment_uri` placeholder—no real files required for MVP.

**Generator implication:** Prefer **JSONL** (one JSON object per line) with a **fixed schema** or a documented **discriminated union** if you add ticket history later.

---

## Realism rules (edge cases coding agents should encode or sample)

- **Missing data:** Sometimes null `subject`, null `product_bucket`, or empty `assigned_queue`.
- **Wrong branch:** Occasional mismatch between narrative (“I was at the downtown branch…”) and `branch_code`—low probability if you want messy realism.
- **Duplicates / retries:** Same customer narrative with new `case_id` (simulate resubmission).
- **Long text:** Pasted emails, bullet lists, ALL CAPS fragments.
- **Channel mix:** Not all records are `branch_counter`; include app and call center with branch inferred from account.
- **Timestamps:** Business hours skew for branch-originated rows; wider spread for digital.

---

## Privacy and provenance (mandatory for synthetic output)

- Set **`synthetic: true`** (or equivalent) on every record or in batch metadata.
- Avoid realistic PII: no real names, addresses, phone numbers, or government IDs; use placeholders or obviously fake tokens.
- Include **`generator_version`** and **`model`** (e.g. Gemini model id) for audit trails.

---

## Suggested JSON shape (example for prompts / parsers)

Single **raw** complaint/feedback record (illustrative—adjust names to your codebase):

```json
{
  "record_type": "bank_complaint_raw",
  "case_id": "BK-2026-00018472",
  "branch_code": "BR-JKT-014",
  "region": "JAVA",
  "channel": "branch_counter",
  "created_at": "2026-03-15T10:22:00+07:00",
  "updated_at": "2026-03-15T14:05:00+07:00",
  "status": "in_progress",
  "product_bucket": "fees",
  "subject": "Question about monthly account fee",
  "verbatim": "The teller could not explain why the fee differed from last month. I asked for a printed statement of fees and was told to use the app.",
  "language": "en",
  "has_attachments": false,
  "customer_ref": "SYN-CUST-A8F29C",
  "synthetic": true,
  "generator_version": "0.1.0",
  "model": "gemini-..."
}
```

**Branch set:** Generate or load a static list of **20–32** `branch_code` values (and optional `region`) so distributions and joins stay consistent across runs.

---

## Checklist before merging generator code

- [ ] Every record has `case_id`, `branch_code`, `channel`, `created_at`, and `verbatim` (or your renamed equivalents).
- [ ] Branch cardinality respects **20–32** distinct codes across the dataset.
- [ ] Schema matches **raw** layer: no mandatory ML labels unless documented as a separate layer.
- [ ] Provenance and **`synthetic: true`** are always present.
- [ ] Output format documented (JSONL path, charset UTF-8).
