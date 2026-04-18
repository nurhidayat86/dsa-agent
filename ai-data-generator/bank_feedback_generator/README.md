# Bank feedback generator

`generator.py` creates synthetic **bank complaint / feedback** records (raw-layer JSON) with the Google Gemini API. Use **`run_bank_feedback_generation`** from a script, **`generate_bank_feedback_data`** as a library, or run the file after editing `if __name__ == "__main__"`.

## Prerequisites

- **Python** 3.10+
- **Conda environment** `google-adk` (or any env with `google-genai` and `pyyaml`; see parent [`requirements.txt`](../requirements.txt)).
- **API key:** `GEMINI_API_KEY` in the environment (see `../config.yaml` or `../config-example.yaml` under `gemini.api_key_env`).

## Configuration

From the **`ai-data-generator`** directory (parent of this folder), ensure `config.yaml` exists. If you only have the example:

```bash
# Windows (cmd)
copy config-example.yaml config.yaml

# Unix
cp config-example.yaml config.yaml
```

Prefer the key in the environment rather than committing `gemini.api_key` in YAML.

## Run as a script

Edit `run_bank_feedback_generation(...)` at the bottom of `generator.py`, then:

```bash
cd ai-data-generator/bank_feedback_generator
conda run -n google-adk python generator.py
```

Defaults resolve `config_path` and `output_jsonl_dir` to the parent folder (`../config.yaml`, `../out/`). Each branch is written as `../out/{branch_code}.jsonl`.

## Import from another project

Add **`ai-data-generator`** (the folder that *contains* `bank_feedback_generator`) to `PYTHONPATH`, then import the package. Do **not** add only `bank_feedback_generator` as the sole path entry.

```python
import sys
from pathlib import Path

ADA = Path("/path/to/dsa-agent/ai-data-generator")  # adjust
sys.path.insert(0, str(ADA))

from bank_feedback_generator import generate_bank_feedback_data

records = generate_bank_feedback_data(
    num_branches=5,
    min_records_per_branch=3,
    max_records_per_branch=8,
    allow_anonymous_customer_id=True,
    min_customers_per_branch=2,
    min_chars_standard_feedback=40,
    max_chars_standard_feedback=800,
    min_chars_detailed_complaint=200,
    max_chars_detailed_complaint=3500,
    complaint_language="en",
    config_path=ADA / "config.yaml",
    output_jsonl_dir=ADA / "out",
    seed=42,
)
print(len(records), "records")
```

| You import | Defined in |
|------------|------------|
| `generate_bank_feedback_data`, `run_bank_feedback_generation` | `generator.py` (re-exported from `__init__.py`) |

## One-liner from a shell

With current directory **`ai-data-generator`**:

```bash
conda run -n google-adk python -c "
import sys
sys.path.insert(0, '.')
from bank_feedback_generator import generate_bank_feedback_data
from pathlib import Path
n = len(generate_bank_feedback_data(
    2, 2, 4, False, 1, 30, 400, 100, 2000,
    complaint_language='en',
    config_path=Path('config.yaml'),
    output_jsonl_dir=Path('out'),
    seed=1,
))
print('wrote', n, 'records; one jsonl per branch under out/')
"
```

## Outputs

- **Return value:** `list[dict]` — one dict per record (`record_type`, `submission_type`, `verbatim`, branch fields, provenance, …). Schema aligns with [`../docs/bank_branch_raw_feedback_context.md`](../docs/bank_branch_raw_feedback_context.md).
- **Language:** `complaint_language` (default `en`) drives prompts and the `language` field; verbatim length is clamped using the min/max parameters (short outputs may be padded with generic bank phrases).
- **Created date:** pass `created_date="2025-12-31"` (or a `datetime.date`) so every `created_at` / `updated_at` falls on that calendar day with random times; omit for fully random dates.
- **Files:** if `output_jsonl_dir` is set, writes **one UTF-8 JSONL per branch** (`{branch_code}.jsonl`). Use `output_jsonl_dir=None` to skip writing.

## Troubleshooting

- **`ValueError: Missing Gemini API key`:** set `GEMINI_API_KEY` or `gemini.api_key` in `config.yaml`.
- **`min_customers_per_branch` errors:** must be **≤ `min_records_per_branch`**.
- **Slow runs:** large character limits or `gemini.max_output_tokens` increase latency. Watch `INFO Gemini: branch=... chunk ...` logs. Reduce limits, lower `generator.chunk_size`, or tune `gemini.http_timeout_ms` in `config.yaml` (default long timeout per request if omitted).

Full parameter documentation: docstring of `generate_bank_feedback_data` in `generator.py`.
