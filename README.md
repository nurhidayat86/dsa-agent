# Agentic VOC Insight

**Agentic VOC Insight** is a repository built around **AI-driven topic modelling** on customer voice (VOC) data. The centerpiece is [`topic-modelling/topic-modelling-ai.py`](topic-modelling/topic-modelling-ai.py): an agentic pipeline that loads embedded feedback from **ChromaDB**, discovers topics with **UMAP + HDBSCAN**, enriches them with **KeyBERT** (local embeddings) and **Google Gemini** (summaries and narrative reporting), and involves you in the loop when choosing UMAP parameters.

Everything else in this repo exists to **feed that pipeline**: synthetic or real-like bank feedback in JSONL, ingestion into a vector store with consistent embeddings, and an optional notebook for exploratory analysis.

---

## Install the Python environment

Use the **root** [`requirements.txt`](requirements.txt). It lists everything needed to run **`topic-modelling-ai.py`**, **`vector-db-writer`**, **`ai-data-generator`**, and the Jupyter notebook [`topic-modelling/notebook/chrome_viewer.ipynb`](topic-modelling/notebook/chrome_viewer.ipynb) (including `google-adk`, Chroma, UMAP/HDBSCAN, KeyBERT, PyTorch/transformers, and OpenTelemetry pins compatible with Chroma).

**Prerequisites:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda. Python **3.10+** (3.12 is a good default).

From the **repository root** (`agentic-voc-insight/`):

```bash
conda create -n google-adk python=3.12 -y
conda activate google-adk
pip install -r requirements.txt
```

If you prefer not to activate the env (for example in scripts or CI):

```bash
conda run -n google-adk pip install -r requirements.txt
```

After installation, use `conda activate google-adk` or prefix commands with `conda run -n google-adk` as shown elsewhere in this README.

**Optional — GPU PyTorch:** `requirements.txt` installs a generic `torch` from PyPI. For a specific CUDA build, install the matching wheel from [pytorch.org](https://pytorch.org/get-started/locally/) first, then run `pip install -r requirements.txt` again (pip will usually keep your existing `torch` if the version already satisfies the constraint).

**Note:** Older copies of [`vector-db-writer/requirements.txt`](vector-db-writer/requirements.txt) and [`ai-data-generator/requirements.txt`](ai-data-generator/requirements.txt) are subsets for partial installs; for one environment for the whole repo, prefer the **root** `requirements.txt`.

---

## North star: `topic-modelling-ai.py`

| What it does | How AI is used |
|----------------|----------------|
| **Load & normalise** | Reads documents and vectors from Chroma; L2-normalises embeddings for clustering. |
| **UMAP parameter choice** | A **Google ADK** `LlmAgent` (Gemini) sweeps `n_neighbors` on 2D UMAP projections (default 15–25), reads diagnostic stats (fragmentation vs. one big blob), picks a candidate, and can adjust from your **natural-language** feedback. |
| **Human in the loop** | Saves a **2D scatter** PNG, opens it in your OS viewer; you approve or steer the agent before the heavy clustering step. |
| **Cluster & label** | **UMAP** (5D, cosine) → **HDBSCAN** → **KeyBERT** with local **Qwen3-Embedding-8B** (must match how vectors were produced at ingest). |
| **Understand & report** | **Gemini** writes per-cluster text summaries and a **Markdown** helicopter-view report with short cluster titles; a deterministic appendix lists **KeyBERT keywords per cluster** (with relevance scores). |

**Outputs** (paths are configurable; defaults are under `topic-modelling/out/`):

- `report.md` — executive-style Markdown report plus verbatim KeyBERT appendix  
- `summary_llm.csv` — cluster id, Gemini summary, document count  
- `keyword_summary.csv` — cluster id, keywords, cosine similarity (and weight when present)  
- `umap_2d_n*.png` — 2D UMAP plots used during human review  

**Run** (from `topic-modelling/`, using the same **`google-adk`** conda env as the rest of the repo):

```bash
cd topic-modelling
conda run -n google-adk python topic-modelling-ai.py
```

**Configuration:** place a `config.yaml` with a `gemini:` block in the working directory or a parent (see `topic-modelling/config-example.yaml` patterns elsewhere), or set `GEMINI_API_KEY` / `GOOGLE_API_KEY`. The script propagates keys from YAML into the environment so the ADK agent and `google-genai` clients both see them.

**Imports:** the file is named `topic-modelling-ai.py` (hyphen). Import it with `importlib.util.spec_from_file_location` if you call it from another module (see [`topic-modelling/README.md`](topic-modelling/README.md)).

More detail: [`topic-modelling/README.md`](topic-modelling/README.md) · exploratory flow: [`topic-modelling/notebook/chrome_viewer.ipynb`](topic-modelling/notebook/chrome_viewer.ipynb)

---

## Supporting piece 1: `vector-db-writer/`

**Role:** Turn **bank-feedback JSONL** into a **persistent ChromaDB** collection whose **embeddings** `topic-modelling-ai.py` will cluster.

| Input | Output |
|-------|--------|
| `data/bank_feedback/*.jsonl` | Chroma persist directory (e.g. `data/vector-db/chroma-db`) |

**Embeddings** must align with topic modelling:

- **`qwen`** — local **Qwen3-Embedding-8B** (same family KeyBERT expects in the topic script), or  
- **`gemini`** — `gemini-embedding-001` via API key.

Entry point: [`vector-db-writer/ingest_bank_feedback.py`](vector-db-writer/ingest_bank_feedback.py). Spec: [`vector-db-writer/docs/chroma_bank_feedback_ingestion.md`](vector-db-writer/docs/chroma_bank_feedback_ingestion.md).

```bash
cd vector-db-writer
conda run -n google-adk python ingest_bank_feedback.py
```

(Install Python packages once from the repo root with `pip install -r requirements.txt` as described in [Install the Python environment](#install-the-python-environment).)

Use the **same Chroma path and collection name** you configure in `topic-modelling-ai.py`.

---

## Supporting piece 2: `ai-data-generator/`

**Role:** Produce **synthetic customer voice** (bank branch feedback / complaints and optional call-center style transcripts) as **JSONL** so you can stress-test ingestion and topic modelling without production data.

| Area | Entry |
|------|--------|
| Bank feedback JSONL | [`ai-data-generator/bank_feedback_generator/generator.py`](ai-data-generator/bank_feedback_generator/generator.py) |
| Call-center batches | [`ai-data-generator/call_center_data/generator.py`](ai-data-generator/call_center_data/generator.py) |

Typical path: generate JSONL → copy or symlink into `vector-db-writer/data/bank_feedback/` → run ingest → run `topic-modelling-ai.py`.

Uses Gemini via `config.yaml` (see [`ai-data-generator/config-example.yaml`](ai-data-generator/config-example.yaml)). Per-folder READMEs describe parameters and layout.

---

## End-to-end flow (VOC → topics → report)

1. **(Optional)** Generate synthetic feedback with **`ai-data-generator`**.  
2. **Ingest** JSONL into Chroma with **`vector-db-writer`** (pick Qwen or Gemini embeddings and remember the persist path + collection name).  
3. **Run** **`topic-modelling/topic-modelling-ai.py`**: agent-assisted UMAP choice + human review → HDBSCAN → KeyBERT → Gemini summaries → Markdown report and CSVs.

---

## Environment and repo hygiene

- **Conda env:** use the **`google-adk`** environment created in [Install the Python environment](#install-the-python-environment) for all Python in this repo (see also [`.cursor/rules/python-conda-google-adk.mdc`](.cursor/rules/python-conda-google-adk.mdc)).  
- **Secrets:** do not commit real API keys or full Chroma trees; [`.gitignore`](.gitignore) ignores common local paths (`config.yaml`, vector DB artifacts, generated `out/` trees, etc.).

---

## Repository layout (at a glance)

| Path | Supports topic modelling by… |
|------|-------------------------------|
| [`topic-modelling/`](topic-modelling/) | **Running** the agentic pipeline (`topic-modelling-ai.py`), config, outputs, notebook. |
| [`vector-db-writer/`](vector-db-writer/) | **Building** the Chroma index the pipeline reads. |
| [`ai-data-generator/`](ai-data-generator/) | **Seeding** JSONL VOC data for ingest and experiments. |

This README is the map; folder-level READMEs drill into each component.
