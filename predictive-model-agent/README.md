# Predictive model agent

Tabular **modeling and monitoring** helpers for AI agents or MCP tools: `split_data`, PSI, timely rates, feature/score diagnostics, optional **optbinning**, and related utilities in [`agent_tools.py`](agent_tools.py).

## Layout

| Path | Description |
|------|-------------|
| `agent_tools.py` | Main module (`import agent_tools` with this directory on `PYTHONPATH`, or `cd` here) |
| `notebook/testing_tools.ipynb` | Examples using `data/heloc_dataset_v1.parquet` |
| `data/` | Sample HELOC-style dataset (CSV and Parquet) |

## Quick start

From this folder (or with `PYTHONPATH` including it):

```bash
cd predictive-model-agent
python -c "import agent_tools as at; print([x for x in dir(at) if not x.startswith('_')][:10])"
```

Open the notebook under `notebook/` with Jupyter; set the kernel working directory so imports resolve (see the first notebook cell).

This package was previously named **`credit-risk-data-scientist`**; update any local scripts or bookmarks to **`predictive-model-agent`**.
