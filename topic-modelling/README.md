# Topic modelling

This folder contains a **batch pipeline** and an **optional Jupyter notebook** that turn **ChromaDB-stored embeddings** (for example bank feedback ingested by `vector-db-writer`) into **topic clusters**, **per-cluster keywords**, **Gemini-written summaries**, and a **Markdown management report**.

---

## What runs here

| Artifact | Role |
|----------|------|
| [`topic-modelling-ai.py`](topic-modelling-ai.py) | Main script: load Chroma → L2-normalize embeddings → **UMAP** (5D, cosine) → **HDBSCAN** → **KeyBERT** with local **Qwen3-Embedding-8B** → **Gemini** per-cluster text summaries → **Gemini** Markdown report with short cluster labels |
| [`notebook/chrome_viewer.ipynb`](notebook/chrome_viewer.ipynb) | Interactive exploration of the same flow (paths, plots, intermediate tables) |

Intermediate CSVs (under `cluster_summary_path`, default `./out`):

- `summary_llm.csv` — cluster, `text_summary`, `cluster_count`
- `keyword_summary.csv` — cluster, `keywords`, `cosine_similarities`

The Markdown report path is whatever you pass as `md_file_path` (for example `./out/report.md`).

---

## Prerequisites

- **Python** with the same stack as [`vector-db-writer`](../vector-db-writer): `chromadb`, `pandas`, `numpy`, `umap-learn`, `hdbscan`, `scikit-learn`, `torch`, `transformers`, `keybert`, `google-genai`, `pyyaml`, plus a local **Qwen3-Embedding-8B** snapshot that matches the embedding model used at ingest time. Installing [`vector-db-writer/requirements.txt`](../vector-db-writer/requirements.txt) in the **`google-adk`** conda env is the usual approach for this repo.
- **Gemini API key** for summaries and the final report: a parent `config.yaml` with a `gemini:` block (often this folder’s [`config.yaml`](config.yaml)), or `GEMINI_API_KEY` / `GOOGLE_API_KEY`. Do not commit real keys.
- **Working directory** at or under the **dsa-agent** repo when using default relative paths (for example `../vector-db-writer/...`) so Chroma and Qwen paths resolve.

---

## Public entry points

The module file is named `topic-modelling-ai.py` (hyphen), so it is not importable as a normal top-level package name. From another script in the same directory, use for example:

```python
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "topic_modelling_ai",
    Path(__file__).resolve().parent / "topic-modelling-ai.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

_mod.generate_topic(...)
```

- **`generate_topic(chroma_db_path, umap_n_neighbors, md_file_path, *, ...)`** — full pipeline with optional `collection_name`, `qwen_model_path`, `gemini_report_model`, `cluster_summary_path`.
- **`generate_topic_model(...)`** — convenience wrapper with defaults aligned to local layout; forwards to `generate_topic`.

---

## Running the script

Edit the variables under `if __name__ == "__main__":` in [`topic-modelling-ai.py`](topic-modelling-ai.py) (Chroma directory, `umap_n_neighbors`, report path, collection name, Qwen snapshot path, Gemini model id, CSV output directory), then:

```bash
cd topic-modelling
conda run -n google-adk python topic-modelling-ai.py
```

Ensure the Chroma path points at the **same directory** you use with `chromadb.PersistentClient` when ingesting (for example `vector-db-writer/data/vector-db/chroma-db` if that is your persist folder).

---

## Relation to `vector-db-writer`

1. **Ingest** JSONL bank feedback into Chroma with `vector-db-writer` (Qwen or Gemini embeddings).
2. **Run** `topic-modelling-ai.py` against that persist folder to cluster, summarize, and emit the report.

The notebook assumes the repo layout (`vector-db-writer` next to `topic-modelling`); the script uses explicit or relative paths you set in `__main__` or in your own caller.
