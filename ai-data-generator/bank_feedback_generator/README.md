# Bank feedback generator

`generator.py` implements synthetic **bank complaint / feedback** records (raw-layer JSON) using the Google Gemini API. You can **import** `generate_bank_feedback_data` or `run_bank_feedback_generation` from another project, or run the module after editing the arguments in the `if __name__ == "__main__"` block:

```bash
cd bank_feedback_generator
conda run -n google-adk python generator.py
```

## Prerequisites

- **Python** 3.10+
- **Conda environment** `google-adk` (or another env with `google-genai` and `pyyaml`; see parent folder `requirements.txt`).
- **API key**: set `GEMINI_API_KEY` in your environment (see `../config.yaml` or `../config-example.yaml` under `gemini.api_key_env`).

## Configuration

1. From the **`ai-data-generator`** directory (parent of this folder), ensure `config.yaml` exists. If you only have the example file, copy it:

   ```bash
   copy config-example.yaml config.yaml
   ```

   (On Unix: `cp config-example.yaml config.yaml`.)

2. Put your Gemini key in the environment (recommended), or set `gemini.api_key` in `config.yaml` for local use only.

## How to run / use `generator.py`

### Option C: Run as a script (edit `__main__` call first)

Open `generator.py`, adjust the arguments passed to `run_bank_feedback_generation(...)` at the bottom, then:

```bash
cd bank_feedback_generator
conda run -n google-adk python generator.py
```

Paths for `config_path` and `output_jsonl_dir` default to the parent **`ai-data-generator`** folder (`../config.yaml`, `../out/`). Each branch is written as `../out/{branch_code}.jsonl`.

---

Add **`ai-data-generator`** (the folder that *contains* `bank_feedback_generator`) to `PYTHONPATH`, then import the public function from the package.

### Option A: From another project (recommended)

```python
import sys
from pathlib import Path

ADA = Path(r"f:/datascience/dsa-agent/ai-data-generator")  # adjust to your machine
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

### Option B: One-liner from a shell (current directory = `ai-data-generator`)

```bash
conda run -n google-adk python -c "
import sys
sys.path.insert(0, '.')
from bank_feedback_generator import generate_bank_feedback_data
from pathlib import Path
n = len(generate_bank_feedback_data(
    2, 2, 4, False, 1, 30, 400, 100, 2000,
    complaint_language="en",
    config_path=Path('config.yaml'),
    output_jsonl_dir=Path('out'),
    seed=1,
))
print('wrote', n, 'records; one jsonl per branch under out/')
"
```

### Imports map

| You import | Defined in |
|------------|------------|
| `generate_bank_feedback_data` | `generator.py` (re-exported from `__init__.py`) |

**Do not** add `bank_feedback_generator` itself as the only path element; the import root must be **`ai-data-generator`** so that `bank_feedback_generator` resolves as a package.

## Outputs

- **Return value**: `list[dict]` — one dict per synthetic record (`record_type`, `submission_type`, `verbatim`, branch fields, provenance, etc.). Schema aligns with `../docs/bank_branch_raw_feedback_context.md`.
- **Language**: `complaint_language` (default `en`) controls the prompt and the `language` field on each record; `verbatim` length is enforced to stay between per-type min/max (short outputs are padded with generic bank phrases if needed).
- **Created date**: pass `created_date="2025-12-31"` (or a `datetime.date`) so every `created_at` / `updated_at` stays on that day with random times; omit or `None` for the previous random calendar behaviour.
- **Optional files**: if `output_jsonl_dir` is set, the code writes **one UTF-8 JSONL file per branch** in that directory: `{branch_code}.jsonl`, each line one JSON object. Pass `output_jsonl_dir=None` to skip writing.

## Troubleshooting

- **`ValueError: Missing Gemini API key`**: set `GEMINI_API_KEY` or `gemini.api_key` in `config.yaml`.
- **`min_customers_per_branch` errors**: must be **≤ `min_records_per_branch`** so each branch has enough rows for distinct customers.
- **Looks “hung”**: each Gemini chunk can take **several minutes** if `max_chars_detailed_complaint` / `max_chars_standard_feedback` or `gemini.max_output_tokens` in `config.yaml` are very large. Watch for `INFO Gemini: branch=... chunk ...` lines. Lower character limits, reduce `max_output_tokens`, or set `generator.chunk_size` smaller. `gemini.http_timeout_ms` (default 15 minutes per request if the key is omitted from YAML) avoids waiting forever; set to `null` or `0` for the SDK default.

For full parameter documentation, see the docstring of `generate_bank_feedback_data` in `generator.py`.
