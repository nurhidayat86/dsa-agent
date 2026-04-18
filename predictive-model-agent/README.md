# Predictive model agent

Tabular **modeling and monitoring** for AI agents or MCP tools, plus a **multi-phase credit-style scorecard pipeline** with human-in-the-loop gates.

## Layout

| Path | Description |
|------|-------------|
| [`agent_tools.py`](agent_tools.py) | Core library: splits, PSI, WoE, logistic regression, optbinning/scorecard helpers, etc. Import as `agent_tools` with this directory on `PYTHONPATH` (or run from this folder). |
| [`requirements.txt`](requirements.txt) | Minimum third-party packages for `agent_tools` and the scorecard stack (`pip install -r requirements.txt`). |
| [`run_scorecard.py`](run_scorecard.py) | CLI launcher; equivalent to `python -m scorecard.cli` with the repo root on the path. |
| [`scorecard/`](scorecard/) | Pipeline package: schemas, tools (wrappers over `agent_tools`), orchestrator, ADK agents, HITL, CLI. |
| [`docs/multi-agent-scorecard-design.md`](docs/multi-agent-scorecard-design.md) | Design: phases, gates **H1–H6**, artifacts. |
| [`notebook/testing_tools.ipynb`](notebook/testing_tools.ipynb) | Worked examples using [`data/heloc_dataset_v1.parquet`](data/heloc_dataset_v1.parquet) (and CSV). |
| [`data/`](data/) | Sample HELOC-style dataset (Parquet and CSV). |
| `scorecard_runs/<run_id>/` | Created per pipeline run: contracts, binning, branches, scorecard tables, validation parquet, `model_documentation.md`, `run_manifest.json`, optional `production/scoring.py`. |

## Setup

Use the **`google-adk`** conda environment (same as other folders in this repo) or any Python 3.10+ env with the packages in [`requirements.txt`](requirements.txt):

```bash
cd predictive-model-agent
conda run -n google-adk pip install -r requirements.txt
```

**Gemini / ADK:** add `config.yaml` next to this README (or pass `--config` to the CLI). It must include a `gemini:` block; see [`scorecard/settings.py`](scorecard/settings.py). Set **`GEMINI_API_KEY`** or **`GOOGLE_API_KEY`** in the environment (or `gemini.api_key` in YAML for local use). The loader mirrors the key into `GOOGLE_API_KEY` for the GenAI SDK.

## What `agent_tools` covers (high level)

### Splits and cohort labels

- **`split_data`** — `train` / `valid` / `test` / `hoot` / `oot` from a time column and thresholds.

### Population stability (PSI)

- **`compute_psi`** — numeric feature vs reference quantile bins.
- **`get_timely_vars_psi`** — per-variable PSI over time (quantile bins on raw values).
- **`get_timely_feature_psi_woe`** — WoE-level PSI (wide: features × periods).
- **`get_timely_psi`** — single variable over `col_period` (numeric quantile bins or categorical levels).

### Target rates and score power

- **`get_timely_binary_target_rate`**, **`get_timely_target_rate_feature_segment`**
- **`get_score_predictive_power_timely`**, **`get_score_predictive_power_data_type`**, bootstrap and champion–challenger comparison helpers.

### Logistic regression

- **`train_logreg_l1_tune_cv`** / **`train_logreg_l2_tune_cv`** — CV tuning; **scikit-learn ≥ 1.8** uses `l1_ratio` instead of deprecated `penalty` for elastic-style control.
- **`logreg_predict`**

### Feature selection and binning

- Stepwise AIC / BIC / AUC (forward/backward), greedy AUC/IV with correlation caps.
- **Optbinning:** `get_optimal_bin`, `modify_optimal_bin`, binning tables, **`get_woe_from_bp`**, scorecard fitting and production code emission (see docstrings and **`__all__`** in `agent_tools.py`).

### Quick library check

```bash
cd predictive-model-agent
python -c "import agent_tools as at; print([x for x in dir(at) if not x.startswith('_')][:20])"
```

For Jupyter, open `notebook/testing_tools.ipynb` and use a kernel whose working directory includes this folder so `import agent_tools` resolves.

## Multi-agent scorecard pipeline

The `scorecard/` package runs: ingest → EDA → splits → binning → multi-branch feature search → training → ranking → scorecard build → validation → model documentation, with **HITL gates H1–H6** between phases (rewinds on `revise` where applicable).

| Module | Role |
|--------|------|
| [`scorecard/schemas.py`](scorecard/schemas.py) | Pydantic v2 contracts for gate payloads and artifacts. |
| [`scorecard/tools.py`](scorecard/tools.py) | Validated wrappers over `agent_tools` (no LLM imports). |
| [`scorecard/agents.py`](scorecard/agents.py) | Google ADK `LlmAgent` instances for narrative steps. |
| [`scorecard/orchestrator.py`](scorecard/orchestrator.py) | Phase machine, gates, persistence under `scorecard_runs/`. |
| [`scorecard/hitl.py`](scorecard/hitl.py) | `HitlInterface` and CLI / auto-approve / scripted implementations. |
| [`scorecard/cli.py`](scorecard/cli.py) | Argparse entry point (`python -m scorecard.cli`). |
| [`scorecard/settings.py`](scorecard/settings.py) | Loads `config.yaml`, Gemini block, API key handling. |

**Run** (from this directory, HELOC sample):

```bash
conda run -n google-adk python run_scorecard.py --data data/heloc_dataset_v1.parquet
```

Non-interactive smoke test:

```bash
conda run -n google-adk python run_scorecard.py --data data/heloc_dataset_v1.parquet --auto-approve
```

Same via module:

```bash
conda run -n google-adk python -m scorecard.cli --data data/heloc_dataset_v1.parquet --auto-approve
```

Useful flags (see [`scorecard/cli.py`](scorecard/cli.py)): `--config`, `--artifacts-root`, `--run-id`, `--col-target`, `--col-time`, `--max-iterations`, `--hitl-script` (JSONL decisions).
