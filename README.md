# DSA Agent

This repository hosts tools and agents for data-science and analytics workflows. Each major capability lives in its **own top-level folder**; this file summarizes what exists today. **Additional folders** will be described here (and may include their own `README.md`) as the project grows.

---

## Repository layout

| Folder | Purpose |
|--------|---------|
| [`ai-data-generator/`](ai-data-generator/) | Synthetic **customer voice** for banks via Google **Gemini**: branch **feedback / complaints** and **telesales call-center** dialogues (VibeVoice-oriented transcripts) |
| [`predictive-model-agent/`](predictive-model-agent/) | **Agent / MCP-style** Python tools for **tabular predictive modeling**: splits, drift-style metrics (PSI, rates over time), binning helpers, and a sample **HELOC** dataset for notebooks |
| [`vector-db-writer/`](vector-db-writer/) | **Ingest** synthetic (or real) bank-feedback **JSONL** into a **persistent ChromaDB** collection with **Qwen3-Embedding-8B** or **Gemini** embeddings |
| [`topic-modelling/`](topic-modelling/) | **Cluster and summarize** Chroma-loaded documents: **UMAP** + **HDBSCAN**, **KeyBERT** (local Qwen), **Gemini** per-cluster summaries and a **Markdown** helicopter-view report |

Root files such as [`.gitignore`](.gitignore) apply across the repo (for example ignoring local `config.yaml`, Chroma persist trees, and generated outputs).

---

## `ai-data-generator/`

Two generators share **`config.yaml`** (Gemini) and docs under **`docs/`**:

1. **Bank feedback / complaints** — raw-layer style records per branch (JSONL **per branch**).
2. **Call-center telesales** — batched **`metadata.json`** plus one **`.txt`** transcript per conversation (multi-speaker lines for TTS pipelines such as VibeVoice).

### What to use

| Piece | Role |
|-------|------|
| [`bank_feedback_generator/generator.py`](ai-data-generator/bank_feedback_generator/generator.py) | `generate_bank_feedback_data()` / `run_bank_feedback_generation()` for complaints & feedback |
| [`bank_feedback_generator/README.md`](ai-data-generator/bank_feedback_generator/README.md) | Bank generator: run, import, conda env, configuration |
| [`call_center_data/generator.py`](ai-data-generator/call_center_data/generator.py) | `generate_call_center_data()` / `run_call_center_generation()` for telesales dialogues |
| [`call_center_data/README.md`](ai-data-generator/call_center_data/README.md) | Call-center generator: outputs, duration roles, usage |
| [`config-example.yaml`](ai-data-generator/config-example.yaml) | Committed template for Gemini and generator settings |
| `config.yaml` | **Local only** (gitignored)—copy from `config-example.yaml`; prefer `GEMINI_API_KEY` in the environment |
| [`docs/`](ai-data-generator/docs/) | Data-shape context (`data_structure_context.md`, `bank_branch_raw_feedback_context.md`, `telesales_vibevoice_data_structure.md`) |
| [`requirements.txt`](ai-data-generator/requirements.txt) / [`pyproject.toml`](ai-data-generator/pyproject.toml) | Python dependencies / optional editable install |
| `out/` | **Generated** tree (gitignored): branch JSONL for the bank generator; `metadata.json` and `conversations/` for call-center batches |

### Quick start (from repo root)

**Bank feedback / complaints**

```bash
cd ai-data-generator/bank_feedback_generator
conda run -n google-adk python generator.py
```

Adjust parameters in the `run_bank_feedback_generation(...)` call at the bottom of `generator.py`, or import `generate_bank_feedback_data` from another project (see [`bank_feedback_generator/README.md`](ai-data-generator/bank_feedback_generator/README.md)).

**Call-center / telesales**

```bash
cd ai-data-generator/call_center_data
conda run -n google-adk python generator.py
```

Edit `run_call_center_generation(...)` at the bottom of `generator.py`, or import `generate_call_center_data` (see [`call_center_data/README.md`](ai-data-generator/call_center_data/README.md)).

---

## `vector-db-writer/`

Loads **bank-feedback JSONL** (under `vector-db-writer/data/bank_feedback/` by convention) into **ChromaDB** with embeddings from either:

- **Qwen** — local **Qwen3-Embedding-8B** via `transformers`, or  
- **Gemini** — **`gemini-embedding-001`** via `google.genai` and an API key.

Details, CLI / `__main__` defaults, and import examples: **[`vector-db-writer/README.md`](vector-db-writer/README.md)**.  
Normative ingestion mapping: **[`vector-db-writer/docs/chroma_bank_feedback_ingestion.md`](vector-db-writer/docs/chroma_bank_feedback_ingestion.md)**.

**Typical flow:** generate JSONL with `ai-data-generator` → copy or symlink into `vector-db-writer/data/bank_feedback/` → run `ingest_bank_feedback.py` → use the Chroma persist path as input to **topic-modelling**.

---

## `predictive-model-agent/`

Python module **[`agent_tools.py`](predictive-model-agent/agent_tools.py)** (import as `agent_tools` after adding the folder to `PYTHONPATH` or running from that directory) exposes functions intended for **AI agents** or **MCP** servers: validation, explicit return shapes, and docstrings for tool schemas. Topics include **train / valid / test / OOT** style splits, **PSI**, missingness over time, feature and score diagnostics, and optional **optbinning** integration.

| Piece | Role |
|-------|------|
| [`agent_tools.py`](predictive-model-agent/agent_tools.py) | Main library: `split_data`, `compute_psi`, timely metrics, `get_optimal_bin`, etc. |
| [`notebook/testing_tools.ipynb`](predictive-model-agent/notebook/testing_tools.ipynb) | Example usage against [`data/heloc_dataset_v1.parquet`](predictive-model-agent/data/heloc_dataset_v1.parquet) (and CSV) |

**Note:** If you still see an old `credit-risk-data-scientist/` directory after a rename, close anything locking it (e.g. Jupyter) and remove the leftover folder so only `predictive-model-agent/` remains.

---

## `topic-modelling/`

End-to-end **topic discovery** on the same Chroma collection: **L2** embeddings → **UMAP** → **HDBSCAN** → **KeyBERT** (Qwen, must match ingest geometry) → **Gemini** cluster summaries → **Gemini** Markdown report plus CSV sidecars.

See **[`topic-modelling/README.md`](topic-modelling/README.md)** for entry points (`generate_topic`, `generate_topic_model`), environment notes, and how to run [`topic-modelling-ai.py`](topic-modelling/topic-modelling-ai.py). The optional notebook [`topic-modelling/notebook/chrome_viewer.ipynb`](topic-modelling/notebook/chrome_viewer.ipynb) mirrors the pipeline for exploration.

---

## Future modules

When new top-level folders are added (for example training pipelines, evaluation, or other agents), this README will be updated to list them. Individual folders may also gain their own `README.md` for detailed usage.
