# Customer voice data — structure and real-world storage

This document captures how artificial customer voice data should be modeled for the `ai-data-generator` project (e.g. Gemini-generated records), and how analogous data is typically stored in production systems.

---

## 1. Customer feedback and complaints

**Bank, multi-branch raw layer:** For a large bank with **20–32 branches** and **pre-processing** operational shape, see [`bank_branch_raw_feedback_context.md`](bank_branch_raw_feedback_context.md).

### What to model

| Area | Typical fields |
|------|----------------|
| Identity & linkage | `feedback_id`, optional `customer_id` / `account_id` (often pseudonymized in analytics), `channel` (email, web form, app, social, store) |
| Content | `subject` (if any), `body` or `verbatim` (main text), `language`, optional `attachments` flag |
| Classification | `category` / `topic`, `subcategory`, `product_line`, `severity` or `priority`, `sentiment` (often human- or ML-labeled) |
| Operational | `created_at`, `status` (open/resolved), `resolution_summary`, SLA metadata, `agent_id` if support-driven |

### How it is stored in practice

- **Operational**: Case/ticket systems (e.g. Salesforce Service Cloud, Zendesk, Freshdesk)—one row (or entity) per ticket; comments or attachments may live in related tables.
- **Analytics / ML**: Replication or streaming into a **data warehouse** (BigQuery, Snowflake, Databricks) as fact + dimension tables; raw text in `STRING`/`TEXT` columns or object storage with a pointer; sometimes **JSON** for flexible attributes.
- **Privacy**: PII is often stripped or tokenized before leaving the CRM; synthetic datasets should mirror **only** fields that would be allowed in downstream use.

---

## 2. NPS survey

### What to model

| Area | Typical fields |
|------|----------------|
| Survey metadata | `survey_id`, `wave_id` or `campaign_name`, `channel` (email, in-app, SMS), `completed_at` |
| Core NPS | `score` (integer **0–10**), derived `segment`: **promoter** (9–10), **passive** (7–8), **detractor** (0–6) |
| Follow-ups | `reason_text` or “why” open-end; optional driver questions (e.g. satisfaction with support, product) |
| Context | `customer_id`, cohort / `segment`, `tenure`, `product`, etc., for analysis |

### How it is stored in practice

- **Source**: Survey platforms (Qualtrics, Medallia, SurveyMonkey, in-house)— commonly **one response row per respondent per wave**.
- **Warehouse**: Long tables of individual responses; reporting often uses **aggregates** (e.g. daily/weekly NPS, counts by segment). Open-ended answers live in text columns or a **verbatim** table keyed by `response_id`.
- **Consistency**: NPS is standardized—synthetic data should keep **0–10** scores and **segment rules** fixed so results align with typical dashboards.

---

## 3. Other customer voice (VOC)

Examples worth supporting in a generator (choose based on use case):

| Source | Typical fields |
|--------|----------------|
| App / store reviews | `rating` (1–5), `title`, `review_text`, `app_version`, `date` |
| Social / public mentions | `platform`, `post_url` (synthetic in generated data), `text`, engagement counts |
| Chat / messaging | `conversation_id`, ordered `messages[]` with `role` (user/agent/bot), `timestamp` |
| Call center (ASR) | `call_id`, `transcript` or `turns[]`, `duration_sec`, `disposition` |
| Community / forum | `thread_id`, `post_text`, `topic_tags` |
| CRM notes | Shorter internal “voice of customer” notes tied to accounts or opportunities |

### How it is stored

- Often **heterogeneous**: each source has its own pipeline; **unified VOC** layers add `source_system`, `source_type`, `unified_text`, `ingested_at` for search and analytics.
- Very text- or media-heavy sources may use **object storage** (e.g. recordings, long transcripts) with metadata in a database; full-text search may use Elastic, OpenSearch, or warehouse-native search.

**Synthetic telesales ↔ customer (multi-turn, TTS):** One batch **`metadata.json`** holds a **`conversations` array** (all dialogues from that run) plus per-row **`transcript_path`** to plain **`transcript.txt`** files (VibeVoice `Speaker 0:` / `Speaker 1:`). Optional `manifest.jsonl`. See [`telesales_vibevoice_data_structure.md`](telesales_vibevoice_data_structure.md).

---

## Cross-cutting structure (recommended for synthetic output)

1. **Common envelope** (all record types): `record_type`, `id`, `created_at`, `locale`, optional `customer_ref`, optional `brand` / `product`.
2. **Type-specific payload**: e.g. `complaint`, `nps_response`, `review`, `chat_transcript`, `bank_telesales_batch` / `bank_telesales_conversation` (batch JSON + transcript files per [`telesales_vibevoice_data_structure.md`](telesales_vibevoice_data_structure.md)).
3. **Serialization**: **JSON Lines (JSONL)** per type, or one stream with a discriminating `record_type`— convenient for LLM generation and loading into BigQuery or Parquet.

### Provenance (recommended for generated data)

Include fields such as `synthetic: true`, `generator_version`, and `model` so synthetic data does not mix ambiguously with real PII in production paths.

---

## Quick reference

| Kind | Core content | Typical real storage |
|------|----------------|----------------------|
| Feedback / complaints | Verbatim + category + channel + timestamps + status | CRM → warehouse text / JSON |
| NPS | 0–10 score, segment, wave, optional open-end | Survey DB → response tables + aggregates |
| Other VOC | Reviews, chats, calls, social—source-specific fields | Per-source store; often a unified VOC layer |

---

*Context for `ai-data-generator`; align schemas and prompts with downstream consumers (e.g. LLM training, BI, or CRM simulation).*
