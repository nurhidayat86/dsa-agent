# DSA Agent

This repository hosts tools and agents for data-science and analytics workflows. Each major capability lives in its **own top-level folder**; this file summarizes what exists today. **Additional folders** will be described here (and may include their own `README.md`) as the project grows.

---

## Repository layout

| Folder | Purpose |
|--------|---------|
| [`ai-data-generator/`](ai-data-generator/) | Synthetic **customer voice** for banks via Google **Gemini**: branch **feedback / complaints** and **telesales call-center** dialogues (VibeVoice-oriented transcripts) |

Root files such as [`.gitignore`](.gitignore) apply across the repo (e.g. ignoring local `config.yaml` and generated outputs).

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

## Future modules

When new top-level folders are added (e.g. training pipelines, evaluation, or other agents), this README will be updated to list them. Individual folders may also gain their own `README.md` for detailed usage.
