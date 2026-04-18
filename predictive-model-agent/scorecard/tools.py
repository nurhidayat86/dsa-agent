"""Pydantic-validated wrappers over ``agent_tools.py``.

Each wrapper takes a ``RunContext`` plus a small typed input, validates it,
calls the underlying function from ``agent_tools`` and returns either a
Pydantic summary model or a pandas ``DataFrame``. Large tables remain in the
``RunContext`` or on disk — they are never round-tripped through LLM prompts.

This module intentionally has **no** LLM / agent / ADK imports so the wrappers
can be unit-tested in isolation and reused from the future GUI backend.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Make sure agent_tools.py (living one directory above this package) is importable
# when the scorecard CLI is invoked from an arbitrary working directory.
_PKG_PARENT = Path(__file__).resolve().parents[1]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

import agent_tools as at  # noqa: E402  -- intentional path manipulation above

from .schemas import (
    BinningConfig,
    BinningRevision,
    BranchMetrics,
    BranchResult,
    DataColumnContract,
    FeatureSearchConfig,
    LogisticHyperparams,
    PdoParams,
    ProblemContract,
    RecipeSpec,
    SplitConfig,
)
from .state import RunContext


# --------------------------------------------------------------------------- #
# Phase 1 — ingest + schema
# --------------------------------------------------------------------------- #


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read parquet or CSV into a DataFrame (no validation)."""

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"data file not found: {p}")
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"unsupported file type: {p.suffix}")


def auto_detect_contract(df: pd.DataFrame) -> DataColumnContract:
    """Best-effort column mapping used when the user accepts defaults.

    Looks for a datetime-like column for ``col_time`` and a 0/1 column for
    ``col_target`` (``label`` or ``target`` take priority). Everything else
    numeric becomes a candidate feature.
    """

    time_candidates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not time_candidates:
        for c in df.columns:
            try:
                pd.to_datetime(df[c])
                time_candidates.append(c)
                break
            except Exception:
                continue
    col_time = time_candidates[0] if time_candidates else df.columns[0]

    target_priority = ["label", "target", "y", "bad_flag"]
    col_target = next((c for c in target_priority if c in df.columns), None)
    if col_target is None:
        for c in df.columns:
            s = df[c].dropna().unique()
            if len(s) == 2 and set(map(int, np.nan_to_num(s, nan=-1).astype(int))) <= {0, 1}:
                col_target = c
                break
    if col_target is None:
        raise ValueError("Could not auto-detect a binary target column; set it manually.")

    exclude = {col_time, col_target}
    cols_feat: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            cols_feat.append(c)

    return DataColumnContract(
        col_time=col_time,
        col_target=col_target,
        cols_feat=cols_feat,
    )


def _to_month_start(series: pd.Series) -> pd.Series:
    """Collapse any datetime-like series to a month-start ``Timestamp``."""

    if pd.api.types.is_datetime64_any_dtype(series):
        ts = series
    else:
        ts = pd.to_datetime(series, errors="coerce")
    return ts.dt.to_period("M").dt.to_timestamp()


def ensure_month_column(ctx: RunContext) -> None:
    """Materialize ``ctx.df_work[col_month]`` as a month-start Timestamp.

    Monthly is the default time granularity across the pipeline: EDA timely
    stats, target-rate monitoring, PSI, score-over-time discrimination and
    the OOT boundary in ``split_data`` all group on this column. We derive
    it from ``col_time`` exactly once in phase 1 so downstream tools never
    have to recompute it.
    """

    assert ctx.df_work is not None and ctx.data_contract is not None
    dc = ctx.data_contract
    ctx.df_work[dc.col_month] = _to_month_start(ctx.df_work[dc.col_time])


def ensure_binary_target(df: pd.DataFrame, col_target: str) -> pd.DataFrame:
    """Coerce the target column to 0/1 ints if it is binary."""

    s = df[col_target]
    if pd.api.types.is_numeric_dtype(s):
        vals = set(s.dropna().unique().tolist())
        if vals <= {0, 1}:
            df[col_target] = s.astype(int)
            return df
    # Map a 2-class object column to 0/1 (alphabetical second value = 1).
    vals_sorted = sorted(s.dropna().unique().tolist())
    if len(vals_sorted) == 2:
        df[col_target] = (s == vals_sorted[1]).astype(int)
        return df
    raise ValueError(f"Target {col_target!r} is not binary: {s.dropna().unique()[:10]}")


# --------------------------------------------------------------------------- #
# Phase 2 — EDA / QC
# --------------------------------------------------------------------------- #


def eda_summaries(
    ctx: RunContext,
) -> dict[str, pd.DataFrame]:
    """Return nan rate + monthly target rate tables.

    All timely aggregates group on ``col_month`` (materialized in phase 1)
    so the default period is one calendar month.
    """

    assert ctx.df_work is not None and ctx.data_contract is not None
    dc = ctx.data_contract
    df = ctx.df_work
    if dc.col_month not in df.columns:
        ensure_month_column(ctx)
        df = ctx.df_work

    nan_overall = at.get_nan_rate(df, dc.cols_feat)
    nan_timely = at.get_nan_rate_timely(df, dc.cols_feat, dc.col_month)
    target_timely = at.get_timely_binary_target_rate(df, dc.col_month, dc.col_target)
    return {
        "nan_overall": nan_overall,
        "nan_timely": nan_timely,
        "target_timely": target_timely,
    }


def monthly_target_rate(ctx: RunContext) -> pd.DataFrame:
    """``df_stats_target_mean_per_month`` — target mean / count per month.

    Thin alias around ``agent_tools.get_timely_binary_target_rate`` using
    ``col_month``. Output columns: ``mean``, ``count``, ``count_positive``
    indexed by month-start Timestamp.
    """

    assert ctx.df_work is not None and ctx.data_contract is not None
    dc = ctx.data_contract
    if dc.col_month not in ctx.df_work.columns:
        ensure_month_column(ctx)
    return at.get_timely_binary_target_rate(
        ctx.df_work, col_period=dc.col_month, col_target=dc.col_target
    )


def split_sanity(
    ctx: RunContext,
    min_rows: int = 200,
    min_positive: int = 50,
) -> dict[str, Any]:
    """Count rows / positives per ``col_type`` and flag thin cohorts.

    Returns a dict with ``counts`` (rows per cohort), ``positives`` (sum of the
    binary target per cohort) and ``violations`` — human-readable strings
    describing cohorts that fall below ``min_rows`` or ``min_positive``.
    Surfaced to the user at gate H2 and embedded in the split proposer prompt.
    """

    assert ctx.df_work is not None and ctx.data_contract is not None
    dc = ctx.data_contract
    if dc.col_type not in ctx.df_work.columns:
        return {"counts": {}, "positives": {}, "violations": ["col_type not computed"]}
    grp = ctx.df_work.groupby(dc.col_type, dropna=False)
    rows = grp.size().astype(int)
    pos = grp[dc.col_target].sum().astype(int)
    violations: list[str] = []
    for cohort in rows.index:
        if int(rows[cohort]) < min_rows:
            violations.append(f"{cohort}: rows={int(rows[cohort])} < {min_rows}")
        if int(pos[cohort]) < min_positive:
            violations.append(f"{cohort}: positives={int(pos[cohort])} < {min_positive}")
    return {
        "counts": {str(k): int(v) for k, v in rows.to_dict().items()},
        "positives": {str(k): int(v) for k, v in pos.to_dict().items()},
        "violations": violations,
        "min_rows": int(min_rows),
        "min_positive": int(min_positive),
    }


# --------------------------------------------------------------------------- #
# Phase 3 — split
# --------------------------------------------------------------------------- #


def run_split(ctx: RunContext, cfg: SplitConfig) -> pd.Series:
    """Assign each row to train / valid / test / oot on monthly granularity.

    Uses ``col_month`` (month-start Timestamp) as the period column so the
    OOT threshold is interpreted on a month boundary rather than a specific
    day. ``cfg.oot_th`` / ``cfg.hoot_th`` are parsed as dates and snapped
    to month-start for consistency.
    """

    assert ctx.df_work is not None and ctx.data_contract is not None
    dc = ctx.data_contract
    if dc.col_month not in ctx.df_work.columns:
        ensure_month_column(ctx)

    def _snap(th: str | None) -> Any:
        if not th:
            return None
        return pd.to_datetime(th).to_period("M").to_timestamp()

    oot = _snap(cfg.oot_th)
    hoot = _snap(cfg.hoot_th)
    col_type_series = at.split_data(
        ctx.df_work,
        col_target=dc.col_target,
        col_period=dc.col_month,
        oot_th=oot,
        hoot_th=hoot,
        test_perc=cfg.test_perc,
        valid_perc=cfg.valid_perc,
    )
    ctx.df_work[dc.col_type] = col_type_series
    return col_type_series


# --------------------------------------------------------------------------- #
# Phase 4 — binning + WoE
# --------------------------------------------------------------------------- #


def run_binning(ctx: RunContext, cfg: BinningConfig) -> pd.DataFrame:
    """Fit optbinning on train, extract binning tables, materialize WoE cols."""

    assert ctx.df_work is not None and ctx.data_contract is not None
    dc = ctx.data_contract
    df_train = ctx.df_work[ctx.df_work[dc.col_type] == "train"]
    cat_cols = [c for c in cfg.categorical_features if c in dc.cols_feat]
    bin_dict, bp, bt, issues = at.get_optimal_bin(
        df_train,
        cols_feat=dc.cols_feat,
        col_target=dc.col_target,
        min_nbin=cfg.min_nbin,
        max_nbin=cfg.max_nbin,
        cols_feat_cat=cat_cols or None,
    )
    ctx.bin_dict = bin_dict
    ctx.binning_process = bp
    ctx.binning_tables = bt
    ctx.binning_issues = list(issues or [])
    _attach_woe(ctx, bp)
    return bt


def binning_detail_table(ctx: RunContext) -> pd.DataFrame:
    """Return the per-bin detail table for every feature in the binning process.

    Row-binds ``BinningProcess.get_binned_variable(f).binning_table.build()``
    across all features in ``ctx.data_contract.cols_feat`` via
    :func:`agent_tools.get_binning_tables_from_bp`. The result is the
    human-readable table the H3 reviewer actually needs (Feature Name / Bin /
    Count / Event / Event rate / WoE / IV / ...).
    """

    assert ctx.data_contract is not None
    if ctx.binning_process is None:
        return pd.DataFrame()
    df, _unrecognized = at.get_binning_tables_from_bp(
        ctx.binning_process, ctx.data_contract.cols_feat
    )
    return df


def _attach_woe(ctx: RunContext, bp: Any) -> None:
    """Attach ``*_woe`` columns to ``ctx.df_work`` without dropping other cols.

    ``get_woe_from_bp`` returns a *new* DataFrame containing only the WoE
    columns, so we concatenate it onto the existing working frame and drop
    any stale ``_woe`` columns from a prior run.
    """

    assert ctx.df_work is not None and ctx.data_contract is not None
    df_woe, cols_woe, woe_issues = at.get_woe_from_bp(ctx.df_work, ctx.data_contract.cols_feat, bp)
    base = ctx.df_work.drop(columns=[c for c in ctx.df_work.columns if c.endswith("_woe")])
    ctx.df_work = base.join(df_woe)
    ctx.cols_feat_woe = list(cols_woe)
    if woe_issues:
        ctx.binning_issues.extend(woe_issues)


def apply_binning_revision(ctx: RunContext, rev: BinningRevision) -> pd.DataFrame:
    """Apply user bin overrides via ``modify_optimal_bin`` and refresh WoE."""

    assert ctx.df_work is not None and ctx.data_contract is not None
    if not rev.overrides:
        return ctx.binning_tables  # type: ignore[return-value]
    dc = ctx.data_contract
    df_train = ctx.df_work[ctx.df_work[dc.col_type] == "train"]
    new_dict: dict[str, Any] = dict(ctx.bin_dict or {})
    new_dict.update(rev.overrides)
    cfg = ctx.binning_config
    cat_cols = [c for c in (cfg.categorical_features if cfg else []) if c in dc.cols_feat]
    bp, bt, issues = at.modify_optimal_bin(
        df_train,
        dict_bin=new_dict,
        cols_feat=dc.cols_feat,
        col_target=dc.col_target,
        cols_feat_cat=cat_cols or None,
    )
    ctx.bin_dict = new_dict
    ctx.binning_process = bp
    ctx.binning_tables = bt
    ctx.binning_issues = list(issues or [])
    _attach_woe(ctx, bp)
    return bt


# --------------------------------------------------------------------------- #
# Phase 5 — feature selection recipes
# --------------------------------------------------------------------------- #


_RECIPE_DISPATCH: dict[str, tuple[Any, str]] = {
    # tool_name -> (callable, positional kwarg name for feature list)
    "select_features_auc_max_corr": (at.select_features_auc_max_corr, "cols_feat"),
    "select_features_iv_max_corr": (at.select_features_iv_max_corr, "cols_feat_woe"),
    "select_features_aic_forward": (at.select_features_aic_forward, "cols_feat"),
    "select_features_aic_backward": (at.select_features_aic_backward, "cols_feat"),
    "select_features_bic_forward": (at.select_features_bic_forward, "cols_feat"),
    "select_features_bic_backward": (at.select_features_bic_backward, "cols_feat"),
    "select_features_auc_forward": (at.select_features_auc_forward, "cols_feat"),
    "select_features_auc_backward": (at.select_features_auc_backward, "cols_feat"),
}


def run_recipe(ctx: RunContext, recipe: RecipeSpec) -> tuple[list[str], pd.DataFrame]:
    """Dispatch one ``select_features_*`` call on the **train** slice."""

    assert ctx.df_work is not None and ctx.data_contract is not None
    dc = ctx.data_contract
    df_train = ctx.df_work[ctx.df_work[dc.col_type] == "train"]
    fn, feat_kw = _RECIPE_DISPATCH[recipe.tool]
    selected, info = fn(
        df_train,
        **{feat_kw: ctx.cols_feat_woe},
        col_target=dc.col_target,
        **(recipe.kwargs or {}),
    )
    return list(selected), info


def default_recipes() -> FeatureSearchConfig:
    """Reasonable default recipe set mirroring the notebook patterns."""

    recipes = [
        RecipeSpec(recipe_id="iv_corr05", tool="select_features_iv_max_corr", kwargs={"max_corr": 0.5}),
        RecipeSpec(recipe_id="auc_corr05", tool="select_features_auc_max_corr", kwargs={"max_corr": 0.5}),
        RecipeSpec(recipe_id="aic_forward", tool="select_features_aic_forward", kwargs={"max_corr": 0.6}),
        RecipeSpec(recipe_id="bic_backward", tool="select_features_bic_backward", kwargs={"max_corr": 0.6}),
    ]
    return FeatureSearchConfig(recipes=recipes)


# --------------------------------------------------------------------------- #
# Phase 6 — per-branch logistic training
# --------------------------------------------------------------------------- #


def train_branch(
    ctx: RunContext,
    branch_id: str,
    recipe_id: str,
    features_woe: list[str],
    penalty: str = "l2",
) -> BranchResult:
    assert ctx.df_work is not None and ctx.data_contract is not None
    dc = ctx.data_contract
    if not features_woe:
        return BranchResult(
            branch_id=branch_id,
            recipe_id=recipe_id,
            features_woe=[],
            metrics=BranchMetrics(),
            passed_filters=False,
            filter_notes=["empty feature set from recipe"],
        )
    fn = at.train_logreg_l1_tune_cv if penalty == "l1" else at.train_logreg_l2_tune_cv
    summary, model, _probs = fn(
        ctx.df_work,
        cols_feat_woe=features_woe,
        col_target=dc.col_target,
        col_type=dc.col_type,
    )
    tr = summary.get("train", {}) or {}
    va = summary.get("valid", {}) or {}
    te = summary.get("test", {}) or {}
    metrics = BranchMetrics(
        auc_train=_f(tr.get("best_auc")),
        auc_valid=_f(va.get("best_auc")),
        auc_test=_f(te.get("best_auc")),
        gini_train=_f(tr.get("best_gini")),
        gini_valid=_f(va.get("best_gini")),
        gini_test=_f(te.get("best_gini")),
    )
    ctx.branch_models[branch_id] = model
    return BranchResult(
        branch_id=branch_id,
        recipe_id=recipe_id,
        features_woe=list(features_woe),
        metrics=metrics,
    )


def _f(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Phase 9 — scorecard
# --------------------------------------------------------------------------- #


def build_scorecard(
    ctx: RunContext,
    champion: BranchResult,
    pdo: PdoParams,
    logistic_hp: LogisticHyperparams,
) -> pd.DataFrame:
    """Fit an optbinning ``Scorecard`` restricted to the champion features.

    ``optbinning.Scorecard.fit`` internally slices ``X`` by
    ``binning_process.variable_names`` so we cannot reuse the full binning
    process fitted in phase 4. We rebuild a smaller process over just the
    champion raw features (reusing their approved bin definitions) via
    ``modify_optimal_bin``.
    """

    assert ctx.df_work is not None and ctx.data_contract is not None and ctx.bin_dict is not None
    dc = ctx.data_contract
    df_train = ctx.df_work[ctx.df_work[dc.col_type] == "train"]
    raw_cols = [c[: -len("_woe")] for c in champion.features_woe if c.endswith("_woe")]
    if not raw_cols:
        raise ValueError("Champion has no _woe features to build a scorecard from.")
    cfg = ctx.binning_config or BinningConfig()
    cat_cols = [c for c in cfg.categorical_features if c in raw_cols]
    sub_dict = {k: v for k, v in ctx.bin_dict.items() if k in raw_cols}
    sub_bp, _bt, _issues = at.modify_optimal_bin(
        df_train,
        dict_bin=sub_dict,
        cols_feat=raw_cols,
        col_target=dc.col_target,
        cols_feat_cat=cat_cols or None,
    )
    points, sc_model = at.create_scorecard_model(
        df_train,
        cols_feat=raw_cols,
        logistic_regression_hyperparameters=logistic_hp.model_dump(),
        col_target=dc.col_target,
        binning_process=sub_bp,
        base_score=pdo.base_score,
        pdo=pdo.pdo,
        odds=pdo.odds,
    )
    ctx.scorecard_model = sc_model
    ctx.scorecard_points = points
    # Compute score column on the full frame via predict_proba of the trained logreg.
    model = ctx.branch_models.get(champion.branch_id)
    if model is not None:
        proba0, proba1, _align = at.logreg_predict(ctx.df_work, model, champion.features_woe)
        # Map proba to points using PDO (classic log-odds -> score transformation).
        p = np.clip(np.asarray(proba1), 1e-9, 1 - 1e-9)
        odds = p / (1.0 - p)
        factor = pdo.pdo / np.log(2.0)
        offset = pdo.base_score - factor * np.log(pdo.odds)
        scores = offset + factor * np.log(odds)
        ctx.df_work[dc.col_score] = scores
    return points


# --------------------------------------------------------------------------- #
# Production code export (runs after H5 approval, before validation)
# --------------------------------------------------------------------------- #


def export_production_code(ctx: RunContext, code_path: str | Path) -> Path:
    """Write a standalone ``.py`` scorer for the approved champion model.

    Thin wrapper around :func:`agent_tools.writre_production_code`. The
    generated module depends only on ``numpy``: it embeds the champion's bin
    definitions, WoE values, per-bin points and intercept, and exposes
    ``get_score`` / ``get_woe`` functions that accept ``dict[str, Any]``
    feature payloads — exactly the shape a downstream service would feed.
    """

    assert ctx.scorecard_model is not None, (
        "export_production_code called before the scorecard is fit; "
        "build_scorecard must run first."
    )
    bp = getattr(ctx.scorecard_model, "binning_process_", None)
    if bp is None:
        raise RuntimeError(
            "scorecard_model.binning_process_ is not available — did "
            "Scorecard.fit run successfully?"
        )
    p = Path(code_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # ``writre_production_code`` returns the absolute path as a ``str``.
    written = at.writre_production_code(bp, ctx.scorecard_model, str(p))
    return Path(written)


# --------------------------------------------------------------------------- #
# Phase 10 — validation
# --------------------------------------------------------------------------- #


def run_validation(ctx: RunContext) -> dict[str, pd.DataFrame]:
    assert ctx.df_work is not None and ctx.data_contract is not None
    dc = ctx.data_contract
    out: dict[str, pd.DataFrame] = {}
    if dc.col_month not in ctx.df_work.columns:
        ensure_month_column(ctx)
    if dc.col_score in ctx.df_work.columns:
        out["score_by_type"] = at.get_score_predictive_power_data_type(
            ctx.df_work, col_score=dc.col_score, col_type=dc.col_type, col_target=dc.col_target
        )
        # Discrimination-over-time is a monitoring signal, so it must stay
        # out-of-sample. Pooling ``train`` rows back in makes the in-window
        # months look optimistically good (the scorecard was fit on them)
        # and muddies the comparison against the OOT / HOOT months.
        df_scored_oos = ctx.df_work[ctx.df_work[dc.col_type] != "train"]
        out["score_timely"] = at.get_score_predictive_power_timely(
            df_scored_oos,
            col_score=dc.col_score,
            col_period=dc.col_month,
            col_target=dc.col_target,
        )
    # PSI of score (train as reference) across months.
    if dc.col_score in ctx.df_work.columns:
        df_ref = ctx.df_work[ctx.df_work[dc.col_type] == "train"]
        out["score_psi_timely"] = at.get_timely_psi(
            df_ref, ctx.df_work, col_var=dc.col_score, col_period=dc.col_month
        )
    # Feature WoE PSI (monthly). We sort by (feature_name, time) so readers of
    # model_documentation.md can scan one feature's PSI trajectory in a
    # contiguous block instead of hopping across all 20 features for each
    # month.
    if ctx.cols_feat_woe:
        df_ref = ctx.df_work[ctx.df_work[dc.col_type] == "train"]
        try:
            feat_psi = at.get_timely_feature_psi_woe(
                df_ref, ctx.df_work, cols_feats=ctx.cols_feat_woe, col_period=dc.col_month
            )
            out["feature_woe_psi_timely"] = feat_psi
        except Exception as e:  # noqa: BLE001
            out["feature_woe_psi_timely"] = pd.DataFrame({"error": [str(e)]})
    ctx.validation_tables = out
    return out
