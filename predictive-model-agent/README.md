# Predictive model agent

Tabular **modeling and monitoring** helpers for AI agents or MCP tools. The main module is [`agent_tools.py`](agent_tools.py): validated inputs, stable return shapes, and docstrings suitable for tool schemas.

## Layout

| Path | Description |
|------|-------------|
| `agent_tools.py` | Main module (`import agent_tools` with this directory on `PYTHONPATH`, or `cd` here) |
| `notebook/testing_tools.ipynb` | Examples using `data/heloc_dataset_v1.parquet` |
| `data/` | Sample HELOC-style dataset (CSV and Parquet) |

## What `agent_tools` covers

### Splits and cohort labels

- **`split_data`** — Labels rows as `train`, `valid`, `test`, `hoot`, and/or `oot` from a time column and thresholds (core rows can be stratified into train/valid/test).

### Population stability (PSI)

- **`compute_psi`** — PSI for one numeric feature using reference quantile bins.
- **`get_timely_vars_psi`** — PSI for each variable in `col_vars` vs. reference `df_ref`, by production period `col_time` (quantile bins via `compute_psi`); optional `prod_time_values`. **Return shape:** index = `time`, columns = variable names, values = PSI.
- **`get_timely_feature_psi_woe`** — Unlike **`get_timely_vars_psi`** (quantile bins on raw variables), WoE PSI uses **each distinct WoE value as its own bin** (no new bins). Returns a **wide** table: **index = feature names**, **columns = periods** (e.g. month), **values = PSI** (no count column).
- **`get_timely_psi`** — PSI for a **single** variable across `col_period`: **numeric** columns use quantile bins (`n_bin`); **categorical / string / bool** use native categories as bins. Columns: `time_period`, `psi`, **`count data`**.

### Target rates over time or segments

- **`get_timely_binary_target_rate`** — Mean and count of the target by `col_period`.
- **`get_timely_target_rate_feature_segment`** — For each period, each binned/categorical feature, and each **segment** (distinct feature value), returns **`time period`**, **`feature name`**, **`segment`**, **`count data`**, **`count positive`**, **`positive rate`** (positive class = numerically higher target value when the global target is binary).

### Score discrimination (ROC-AUC, Gini)

- **`get_score_predictive_power_timely`** — AUC (and Gini, counts, positive rate) of one score by **`col_period`**.
- **`get_score_predictive_power_data_type`** — Same metrics by **`col_type`** (e.g. train / valid / test / oot); output uses `time_period` as the cohort key for tooling parity.
- **`get_score_predictive_power_data_type_bootstrap`** — Bootstrap distributions of AUC/Gini (and mean positive rate) per `col_type`.
- **`compare_score_predictive_power_data_type_bootstrap`** — Paired bootstrap comparison of **champion** vs **challenger** scores per `col_type` (CIs and means for AUC and Gini).

### Logistic regression (tuning and scoring)

- **`train_logreg_l1_tune_cv`** / **`train_logreg_l2_tune_cv`** — Tune `C` with stratified CV on `train`, pick best on `valid`, report on train/valid/test; optional `min_delta` for step improvement; default `C` grids; **sklearn ≥ 1.8** uses `l1_ratio` instead of deprecated `penalty`.
- **`logreg_predict`** — Returns `proba_0`, `proba_1` (lists) and a dict of feature alignment issues.

### Feature selection and binning

- **Stepwise selection** — `select_features_*_{forward,backward}` for AIC, BIC, or training ROC-AUC, with optional **`min_delta`** and correlation caps.
- **Greedy filters** — `select_features_auc_max_corr`, `select_features_iv_max_corr`.
- **Optbinning** — `get_optimal_bin`, `modify_optimal_bin`, binning tables, **`get_woe_from_bp`**, etc.

### Other utilities

- Missingness over time, PSI-style timely views, dtype helpers, virtual date helpers, and more (see **`__all__`** in `agent_tools.py` for the exported names).

## Quick start

From this folder (or with `PYTHONPATH` including it):

```bash
cd predictive-model-agent
python -c "import agent_tools as at; print([x for x in dir(at) if not x.startswith('_')][:15])"
```

Open the notebook under `notebook/` with Jupyter; set the kernel working directory so imports resolve (see the first notebook cell).

## Multi-agent scorecard pipeline

The `scorecard/` package implements the workflow documented in
[`docs/multi-agent-scorecard-design.md`](docs/multi-agent-scorecard-design.md):
ingest → EDA → splits → binning → multi-branch feature search → training →
ranking → scorecard build → validation → model documentation, with HITL
gates **H1–H6** between phases.

Key modules:

| Module | Role |
|--------|------|
| `scorecard/schemas.py` | Pydantic v2 contracts (single source of truth for gate payloads & artifacts). |
| `scorecard/tools.py` | Pydantic-validated wrappers around `agent_tools.py` (no LLM imports). |
| `scorecard/agents.py` | `google-adk` `LlmAgent` instances for narrative work (EDA summary, ranking rationale, doc prose). |
| `scorecard/orchestrator.py` | Phase state machine, HITL gates, rewinds on `revise`. |
| `scorecard/hitl.py` | `HitlInterface` + CLI / auto-approve / scripted implementations (GUI will subclass this). |
| `scorecard/cli.py` | Argparse entry point. |

Run the full pipeline on the HELOC sample with interactive CLI gates:

```bash
conda run -n google-adk python run_scorecard.py --data data/heloc_dataset_v1.parquet
```

Or non-interactively for a smoke test:

```bash
conda run -n google-adk python run_scorecard.py \
    --data data/heloc_dataset_v1.parquet --auto-approve
```

Artifacts (data / problem contracts, binning process, per-branch results,
proposal, PDO points table, validation tables, `model_documentation.md` and
`run_manifest.json`) land under `scorecard_runs/<run_id>/`.

LLM agents read the `gemini:` block in [`config.yaml`](config.yaml); set
`GEMINI_API_KEY` (or the inline `api_key`) before running.

This package was previously named **`credit-risk-data-scientist`**; update any local scripts or bookmarks to **`predictive-model-agent`**.
