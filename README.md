# DSA Agent

This repository hosts tools and agents for data-science and analytics workflows. Each major capability lives in its **own top-level folder**; this file summarizes what exists today. **Additional folders** will be described here (and may include their own `README.md`) as the project grows.

---

## Repository layout

| Folder | Purpose |
|--------|---------|
| [`ai-data-generator/`](ai-data-generator/) | Synthetic **customer voice** data for banks (complaints / feedback) via Google **Gemini** |

Root files such as [`.gitignore`](.gitignore) apply across the repo (e.g. ignoring local `config.yaml` and generated outputs).

---

## `ai-data-generator/`

Generates **artificial, structured** raw-layer records that resemble operational exports from a multi-branch bank: feedback and complaints with branch codes, channels, verbatim text, optional anonymous customer IDs, and JSONL output **per branch**.

### What to use

| Piece | Role |
|-------|------|
| [`bank_feedback_generator/generator.py`](ai-data-generator/bank_feedback_generator/generator.py) | Core logic: `generate_bank_feedback_data()` (library API) and `run_bank_feedback_generation()` (logging + CLI-style entry) |
| [`bank_feedback_generator/README.md`](ai-data-generator/bank_feedback_generator/README.md) | How to run, import paths, conda env, configuration |
| [`config-example.yaml`](ai-data-generator/config-example.yaml) | Committed template for Gemini and generator settings |
| `config.yaml` | **Local only** (gitignored)—copy from `config-example.yaml`; prefer `GEMINI_API_KEY` in the environment |
| [`docs/`](ai-data-generator/docs/) | Data-shape context for generators (`data_structure_context.md`, `bank_branch_raw_feedback_context.md`) |
| [`requirements.txt`](ai-data-generator/requirements.txt) / [`pyproject.toml`](ai-data-generator/pyproject.toml) | Python dependencies / optional editable install |
| `out/*.jsonl` | **Generated** files (gitignored); one JSONL per branch when writing to disk |

### Quick start (from repo root)

```bash
cd ai-data-generator/bank_feedback_generator
conda run -n google-adk python generator.py
```

Adjust parameters in the `run_bank_feedback_generation(...)` call at the bottom of `generator.py`, or import `generate_bank_feedback_data` from another project (see the package README).

---

## Future modules

When new top-level folders are added (e.g. training pipelines, evaluation, or other agents), this README will be updated to list them. Individual folders may also gain their own `README.md` for detailed usage.
