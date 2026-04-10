# vector-db-writer

## Environment

Use the **`google-adk`** Conda environment (same as `ai-data-generator`).

```bash
conda activate google-adk
pip install -r requirements.txt
```

On Windows without activating:

```bash
conda run -n google-adk pip install -r requirements.txt
```

`requirements.txt` pins **OpenTelemetry 1.38.x** so **`chromadb`** does not upgrade past **`google-adk`**’s supported range.

## Bank feedback → ChromaDB

Spec: [`docs/chroma_bank_feedback_ingestion.md`](docs/chroma_bank_feedback_ingestion.md).

- **Input:** `data/bank_feedback/*.jsonl`
- **Chroma persist (default):** `data/vector-db/` (gitignored)
- **Embeddings:** choose **`qwen`** (local **`Qwen/Qwen3-Embedding-8B`** via `transformers` + `model_path`) or **`gemini`** (**`gemini-embedding-001`** via `google.genai`, API key).

Import and call with explicit parameters (no argv):

```python
from ingest_bank_feedback import ingest_bank_feedback

ingest_bank_feedback(
    embedding_provider="gemini",
    gemini_api_key=None,  # optional: else vector-db-writer/config.yaml gemini.*, else GEMINI_API_KEY
    configure_logging=True,
)
```

Or run the script after editing the defaults in `if __name__ == "__main__"`:

```bash
cd vector-db-writer
conda run -n google-adk python ingest_bank_feedback.py
```

The `__main__` block defaults to **`embedding_provider="gemini"`** and **`GEMINI_API_KEY`**. For Qwen, set `_EMBEDDING_PROVIDER = "qwen"` and `_MODEL_PATH` to your local snapshot directory.
