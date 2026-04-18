"""Run artifact persistence helpers.

Every orchestrator phase writes its outputs under
``<project_root>/scorecard_runs/<run_id>/``. Pydantic models are serialized as
JSON, pandas tables as parquet, and objects that cannot be pickled safely
are represented by metadata sidecars only.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel


class ArtifactStore:
    """Simple on-disk store scoped to a single ``run_id`` directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "branches").mkdir(exist_ok=True)
        (self.root / "hitl").mkdir(exist_ok=True)

    def path(self, relative: str) -> Path:
        p = self.root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_model(self, relative: str, model: BaseModel) -> Path:
        p = self.path(relative)
        p.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        return p

    def save_json(self, relative: str, obj: Any) -> Path:
        p = self.path(relative)
        p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return p

    def save_parquet(self, relative: str, df: pd.DataFrame) -> Path:
        p = self.path(relative)
        try:
            df.to_parquet(p, index=True)
        except Exception:
            p = p.with_suffix(".csv")
            df.to_csv(p, index=True)
        return p

    def save_csv(self, relative: str, df: pd.DataFrame, *, index: bool = False) -> Path:
        """Write ``df`` as UTF-8 CSV (no special format fallbacks).

        Used for artifacts that are explicitly intended for a human reviewer
        in a spreadsheet — e.g. the H3 detailed binning table. ``index`` is
        ``False`` by default because we expect the caller to have already
        materialized any meaningful index as a column.
        """

        p = self.path(relative)
        df.to_csv(p, index=index)
        return p

    def save_pickle(self, relative: str, obj: Any) -> Path:
        p = self.path(relative)
        with open(p, "wb") as fh:
            pickle.dump(obj, fh)
        return p

    def append_jsonl(self, relative: str, record: dict[str, Any]) -> Path:
        p = self.path(relative)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return p

    def save_text(self, relative: str, text: str) -> Path:
        p = self.path(relative)
        p.write_text(text, encoding="utf-8")
        return p
