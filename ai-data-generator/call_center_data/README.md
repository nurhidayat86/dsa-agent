# Call center / telesales synthetic data (Gemini)

This package generates **mock phone conversations** between a bank **tele-sales agent** and a **customer** (cross-sell / upsell). Output matches the batch layout described in [`../docs/telesales_vibevoice_data_structure.md`](../docs/telesales_vibevoice_data_structure.md):

- One **`metadata.json`** per run (batch metadata + a `conversations` array).
- One **`conv-00001.txt`**, **`conv-00002.txt`**, … file per dialogue, in **VibeVoice-style** plain text: `Speaker 0:` / `Speaker 1:` lines (`Speaker 0` = agent, `Speaker 1` = customer).

Dialogue text is produced by **Google Gemini** using structured JSON from the model, then written as transcripts. **Speaker 0** is aligned with `tele_sales_speaker_name`; **Speaker 1** uses a name picked from `customer_speaker_names`.

---

## What the code does

| Piece | Role |
|--------|------|
| [`generator.py`](generator.py) | `generate_call_center_data()` (library), `run_call_center_generation()` (logging + script entry), and `if __name__ == "__main__"` defaults |
| [`__init__.py`](__init__.py) | Re-exports the public functions |

**Duration handling:** Calls are steered toward a **spoken-length window** in seconds (`min_duration`, `max_duration`). The model is given **word-count bounds** derived from those seconds (a TTS proxy). Each conversation has a **`duration_slot`**:

- **`minimum`**: first dialogue when there are two or more (targets ~`min_duration`).
- **`maximum`**: last dialogue when there are two or more, or the only dialogue when `number_of_conversation == 1` (targets ~`max_duration`).
- **`random`**: middle dialogues get a random target time in `[min_duration, max_duration]`.

If the global word band is too narrow to assign separate windows safely, every row may share the same word min/max while roles and target seconds stay as above (`duration_word_band_mode` in metadata).

**API batching:** For reliability with `gemini-flash-lite` and structured output, the generator uses **one conversation per Gemini request** (the `chunk_size` value in `config.yaml` does not apply to this module).

---

## Requirements

- **Conda env:** `google-adk` (same as the rest of this repo).
- **Dependencies:** install parent [`../requirements.txt`](../requirements.txt) (includes `pyyaml`, `google-genai`).
- **Config:** [`../config.yaml`](../config.yaml) (local; copy from [`../config-example.yaml`](../config-example.yaml)). Prefer **`GEMINI_API_KEY`** in the environment.

---

## Run as a script

From repo root (PowerShell-friendly):

```bash
cd ai-data-generator/call_center_data
conda run -n google-adk python generator.py
```

Edit the `run_call_center_generation(...)` call at the bottom of `generator.py` for counts, products, language, duration range, paths, and seed.

Default layout (adjust in code): `output_path_json` → parent of the batch, e.g. `../out/call_center_sample/metadata.json`; transcripts → `../out/call_center_sample/conversations/`.

---

## Use from another Python project

Add the folder that **contains** `call_center_data` (i.e. **`ai-data-generator`**) to `PYTHONPATH`, then import:

```python
import sys
from pathlib import Path

ADA = Path("/path/to/dsa-agent/ai-data-generator")  # adjust
sys.path.insert(0, str(ADA))

from call_center_data import generate_call_center_data

batch = generate_call_center_data(
    number_of_conversation=5,
    featured_products=["credit_card_gold", "savings_premium"],
    lang="en",
    tele_sales_speaker_name="Alice",
    customer_speaker_names=["Frank", "Grace"],
    conversation_dates=None,  # or [start, end] with optional None bounds
    output_path_json=ADA / "out" / "my_batch" / "metadata.json",
    output_path_txt=ADA / "out" / "my_batch" / "conversations",
    min_duration=45.0,
    max_duration=120.0,
    config_path=ADA / "config.yaml",
    seed=42,
)
# batch is the same dict written to metadata.json
```

---

## Outputs

- **`metadata.json`:** `batch_id`, `record_type`, `conversations[]` with `transcript_path`, `vibevoice_speaker_names`, `duration_slot`, `target_spoken_seconds`, `spoken_word_count_bounds`, etc.
- **`conversations/conv-NNNNN.txt`:** UTF-8, one turn per line with `Speaker 0:` / `Speaker 1:`.

Generated trees under `out/` are typically gitignored; see the repo [`.gitignore`](../../.gitignore).

---

## Related docs

- [`../docs/telesales_vibevoice_data_structure.md`](../docs/telesales_vibevoice_data_structure.md) — batch + transcript contract.
- [`../docs/data_structure_context.md`](../docs/data_structure_context.md) — how this fits with other customer-voice shapes.
