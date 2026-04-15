"""
Agent / MCP-oriented utilities for predictive modeling workflows.

This module is intended to be exposed as tools for an AI agent. Functions
include input validation, explicit return shapes, and docstrings suitable
for tool schema generation.
"""

from __future__ import annotations
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

def _serialize_optbinning_splits(optb: Any) -> Any:
    """
    Convert optbinning ``OptimalBinning.splits`` to JSON-friendly Python types.

    Numeric variables: ``splits`` is a 1-D array of cut points (finite floats).
    Categorical variables: ``splits`` is a list of arrays, each listing
    category values merged into that bin.
    """
    splits = getattr(optb, "splits", None)
    if splits is None:
        return None
    if isinstance(splits, np.ndarray):
        out: list[float] = []
        for x in splits.tolist():
            if x is None or (isinstance(x, float) and np.isnan(x)):
                continue
            out.append(float(x))
        return out
    if isinstance(splits, list):
        serialized: list[Any] = []
        for item in splits:
            if isinstance(item, np.ndarray):
                serialized.append(item.tolist())
            else:
                serialized.append(item)
        return serialized
    return splits


def _validate_columns(
    df: pd.DataFrame,
    name: str,
    columns: Iterable[str],
) -> None:
    """Raise ValueError if any required column is missing from df."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _stabilize_distribution(probs: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Floor small probabilities and renormalize so they sum to 1.

    Avoids log(0) in PSI while keeping a proper discrete distribution.
    """
    p = np.asarray(probs, dtype=float)
    p = np.maximum(p, epsilon)
    total = p.sum()
    if total <= 0:
        raise ValueError("Non-positive total after stabilizing distribution.")
    return p / total


def compute_psi(
    ref: pd.Series,
    prod: pd.Series,
    *,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Population Stability Index (PSI) for one numeric feature.

    Bins are defined by quantiles of the reference sample (``ref``), then the
    same bin edges are applied to the production sample (``prod``). Expected
    proportions come from ``ref``; actual proportions from ``prod``.

    Parameters
    ----------
    ref : pd.Series
        Reference distribution (e.g. training or baseline window).
    prod : pd.Series
        Production or comparison distribution.
    n_bins : int, default 10
        Number of quantile bins (deciles when 10).
    epsilon : float, default 1e-6
        Minimum bucket probability after stabilization (avoids ``log(0)``).

    Returns
    -------
    float
        PSI value, or ``numpy.nan`` if computation is not defined (too few
        reference rows, constant reference, empty production, etc.).

    Notes
    -----
    Common interpretation (rule of thumb, not universal): PSI < 0.1 stable;
    0.1–0.25 review; > 0.25 strong shift. Domain policies may differ.
    """
    ref_clean = pd.to_numeric(ref, errors="coerce").dropna()
    prod_clean = pd.to_numeric(prod, errors="coerce").dropna()

    if len(ref_clean) < max(2, n_bins) or len(prod_clean) < 1:
        return float(np.nan)

    # Constant or near-constant reference: deciles are ill-defined.
    if ref_clean.nunique() <= 1:
        return float(np.nan)

    try:
        # Quantile bins on reference; same edges for production.
        ref_binned, bin_edges = pd.qcut(
            ref_clean,
            q=n_bins,
            retbins=True,
            duplicates="drop",
        )
    except (ValueError, TypeError):
        return float(np.nan)

    if bin_edges is None or len(np.unique(bin_edges)) < 3:
        return float(np.nan)

    expected_counts = ref_binned.value_counts().sort_index()
    prod_cut = pd.cut(
        prod_clean,
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    )
    actual_counts = prod_cut.value_counts().reindex(expected_counts.index).fillna(0)

    n_ref = float(expected_counts.sum())
    n_prod = float(actual_counts.sum())
    if n_ref <= 0 or n_prod <= 0:
        return float(np.nan)

    expected = (expected_counts / n_ref).to_numpy(dtype=float)
    actual = (actual_counts / n_prod).to_numpy(dtype=float)

    expected = _stabilize_distribution(expected, epsilon)
    actual = _stabilize_distribution(actual, epsilon)

    ratio = np.divide(actual, expected, out=np.full_like(actual, np.nan), where=expected > 0)
    if not np.all(np.isfinite(ratio)) or np.any(ratio <= 0):
        return float(np.nan)

    psi = float(np.sum((actual - expected) * np.log(ratio)))
    return psi


def _psi_discrete_unique_levels(
    ref: pd.Series,
    prod: pd.Series,
    *,
    epsilon: float,
) -> float:
    """
    PSI when each distinct numeric value is its own category (e.g. WoE levels).

    Expected proportions come from ``ref``; actual from ``prod``. Categories
    are the union of distinct values appearing in either sample.
    """
    ref_clean = pd.to_numeric(ref, errors="coerce").dropna()
    prod_clean = pd.to_numeric(prod, errors="coerce").dropna()
    if len(ref_clean) < 1 or len(prod_clean) < 1:
        return float(np.nan)
    ref_vc = ref_clean.value_counts()
    prod_vc = prod_clean.value_counts()
    n_ref = float(ref_vc.sum())
    n_prod = float(prod_vc.sum())
    if n_ref <= 0 or n_prod <= 0:
        return float(np.nan)
    idx = ref_vc.index.union(prod_vc.index, sort=False)
    expected = (ref_vc.reindex(idx, fill_value=0) / n_ref).to_numpy(dtype=float)
    actual = (prod_vc.reindex(idx, fill_value=0) / n_prod).to_numpy(dtype=float)
    expected = _stabilize_distribution(expected, epsilon)
    actual = _stabilize_distribution(actual, epsilon)
    ratio = np.divide(actual, expected, out=np.full_like(actual, np.nan), where=expected > 0)
    if not np.all(np.isfinite(ratio)) or np.any(ratio <= 0):
        return float(np.nan)
    return float(np.sum((actual - expected) * np.log(ratio)))


def _psi_discrete_category_union(
    ref: pd.Series,
    prod: pd.Series,
    *,
    epsilon: float,
) -> float:
    """
    PSI for categorical (or string/bool) labels: each observed category is a bin.
    """
    ref_clean = ref.dropna()
    prod_clean = prod.dropna()
    if len(ref_clean) < 1 or len(prod_clean) < 1:
        return float(np.nan)
    ref_vc = ref_clean.value_counts()
    prod_vc = prod_clean.value_counts()
    n_ref = float(ref_vc.sum())
    n_prod = float(prod_vc.sum())
    if n_ref <= 0 or n_prod <= 0:
        return float(np.nan)
    idx = ref_vc.index.union(prod_vc.index, sort=False)
    expected = (ref_vc.reindex(idx, fill_value=0) / n_ref).to_numpy(dtype=float)
    actual = (prod_vc.reindex(idx, fill_value=0) / n_prod).to_numpy(dtype=float)
    expected = _stabilize_distribution(expected, epsilon)
    actual = _stabilize_distribution(actual, epsilon)
    ratio = np.divide(actual, expected, out=np.full_like(actual, np.nan), where=expected > 0)
    if not np.all(np.isfinite(ratio)) or np.any(ratio <= 0):
        return float(np.nan)
    return float(np.sum((actual - expected) * np.log(ratio)))


def _col_var_use_category_bins(series: pd.Series) -> bool:
    """True if ``col_var`` should use native categories as PSI bins (no quantiles)."""
    dt = series.dtype
    if isinstance(dt, pd.CategoricalDtype):
        return True
    if dt == object or pd.api.types.is_string_dtype(series) or pd.api.types.is_bool_dtype(series):
        return True
    return False


def get_timely_feature_psi(
    df_ref: pd.DataFrame,
    df_prod: pd.DataFrame,
    col_feats: list[str],
    col_time: str,
    *,
    n_bins: int = 10,
    epsilon: float = 1e-6,
    prod_time_values: Optional[Iterable] = None,
) -> pd.DataFrame:
    """
    PSI per feature and per time period in production data.

    For each distinct value of ``col_time`` in production (or a subset you
    pass via ``prod_time_values``), and for each feature in ``col_feats``,
    this compares the distribution of that feature in
    ``df_prod[df_prod[col_time] == t]`` to the reference distribution in
    ``df_ref`` using :func:`compute_psi`.

    The time column is typically an integer period label (e.g. 202501, 202502
    for months; or YYYYMMDD; or year-quarter encoding). Any sortable discrete
    label works.

    Parameters
    ----------
    df_ref : pd.DataFrame
        Reference / baseline sample (e.g. model development window).
    df_prod : pd.DataFrame
        Production sample; compared to ``df_ref`` by time slice.
    col_feats : list of str
        Numeric feature column names present in both dataframes.
    col_time : str
        Name of the time period column in ``df_prod`` (integer or sortable).
    n_bins : int, default 10
        Number of quantile bins for PSI (see :func:`compute_psi`).
    epsilon : float, default 1e-6
        Stabilization for empty buckets (see :func:`compute_psi`).
    prod_time_values : iterable, optional
        If provided, only these ``col_time`` values are evaluated (must exist
        in ``df_prod``). If omitted, all unique ``col_time`` values in
        ``df_prod`` are used, sorted.

    Returns
    -------
    pd.DataFrame
        Columns: ``time``, ``feature_name``, ``psi``. One row per
        (time period, feature). ``psi`` may be ``NaN`` where PSI cannot be
        computed for that slice.

    Raises
    ------
    ValueError
        If required columns are missing or ``col_feats`` is empty.

    Examples
    --------
    >>> # illustrative only
    >>> ref = pd.DataFrame({"score": np.random.randn(5000)})
    >>> prod = pd.DataFrame({
    ...     "score": np.random.randn(2000),
    ...     "period": [202501] * 1000 + [202502] * 1000,
    ... })
    >>> out = get_timely_feature_psi(ref, prod, ["score"], "period")
    >>> set(out.columns) == {"time", "feature_name", "psi"}
    True
    """
    if not col_feats:
        raise ValueError("col_feats must be a non-empty list of column names.")

    feat_cols = list(col_feats)
    _validate_columns(df_ref, "df_ref", feat_cols)
    _validate_columns(df_prod, "df_prod", feat_cols + [col_time])

    if prod_time_values is not None:
        times = list(prod_time_values)
    else:
        times = sorted(df_prod[col_time].dropna().unique().tolist())

    rows: list[dict] = []
    for t in times:
        mask = df_prod[col_time] == t
        df_slice = df_prod.loc[mask]
        for feat in feat_cols:
            psi_val = compute_psi(
                df_ref[feat],
                df_slice[feat],
                n_bins=n_bins,
                epsilon=epsilon,
            )
            rows.append(
                {
                    "time": t,
                    "feature_name": feat,
                    "psi": psi_val,
                }
            )

    out = pd.DataFrame(rows, columns=["time", "feature_name", "psi"])
    return out


def get_timely_feature_psi_woe(
    df_ref: pd.DataFrame,
    df_prod: pd.DataFrame,
    cols_feats: list[str],
    col_period: str,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """
    PSI per WoE feature and per production period, using distinct WoE values as bins.

    Unlike :func:`get_timely_feature_psi` (quantile bins on a continuous score),
    each unique numeric WoE level in the union of reference and production
    counts as its own category; expected proportions follow ``df_ref`` and
    actual proportions follow ``df_prod`` restricted to
    ``df_prod[col_period] == t``.

    Parameters
    ----------
    df_ref : pd.DataFrame
        Reference sample (e.g. training); must contain ``cols_feats``.
    df_prod : pd.DataFrame
        Production sample; must contain ``cols_feats`` and ``col_period``.
    cols_feats : list of str
        WoE feature columns (numeric coercion applied).
    col_period : str
        Period column in ``df_prod``; all distinct non-null values are used,
        sorted.
    epsilon : float, default 1e-6
        Floor for category probabilities before the PSI log ratio (same role
        as in :func:`compute_psi`).

    Returns
    -------
    pd.DataFrame
        Columns ``time``, ``feature_name``, ``psi``, and ``count data`` (number
        of non-null numeric values of the feature in that production slice).
        ``time`` holds ``col_period`` values.

    Raises
    ------
    ValueError
        If ``cols_feats`` is empty or required columns are missing.
    """
    if not cols_feats:
        raise ValueError("cols_feats must be a non-empty list of column names.")

    feat_cols = list(cols_feats)
    _validate_columns(df_ref, "df_ref", feat_cols)
    _validate_columns(df_prod, "df_prod", feat_cols + [col_period])

    times = sorted(df_prod[col_period].dropna().unique().tolist())
    rows: list[dict[str, Any]] = []
    for t in times:
        mask = df_prod[col_period] == t
        df_slice = df_prod.loc[mask]
        for feat in feat_cols:
            psi_val = _psi_discrete_unique_levels(
                df_ref[feat],
                df_slice[feat],
                epsilon=epsilon,
            )
            n_data = int(
                pd.to_numeric(df_slice[feat], errors="coerce").notna().sum()
            )
            rows.append(
                {
                    "time": t,
                    "feature_name": feat,
                    "psi": psi_val,
                    "count data": n_data,
                }
            )

    return pd.DataFrame(rows, columns=["time", "feature_name", "psi", "count data"])


def get_timely_psi(
    df_ref: pd.DataFrame,
    df_prod: pd.DataFrame,
    col_var: str,
    col_period: str,
    n_bin: int = 10,
    *,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """
    PSI for one variable across production periods (or segment labels).

    For each distinct ``col_period`` value in ``df_prod`` (calendar periods,
    cohort segments such as ``train`` / ``valid``, etc.), compares the
    distribution of ``col_var`` in ``df_prod`` at that slice to the reference
    distribution in ``df_ref[col_var]``.

    - **Numeric** (integer/float): uses quantile bins on the reference sample
      (same as :func:`compute_psi`) with ``n_bin`` bins.
    - **Categorical** (``category``, ``object``, string, or boolean dtypes):
      each observed category is its own bin; no new bins are created.

    Parameters
    ----------
    df_ref : pd.DataFrame
        Reference sample; must contain ``col_var``.
    df_prod : pd.DataFrame
        Production sample; must contain ``col_var`` and ``col_period``.
    col_var : str
        Variable whose stability is measured.
    col_period : str
        Period or segment column in ``df_prod``; unique non-null values are
        used, sorted for deterministic row order.
    n_bin : int, default 10
        Number of quantile bins when ``col_var`` is numeric (ignored for
        categorical columns).
    epsilon : float, default 1e-6
        Probability floor for PSI stabilization (see :func:`compute_psi`).

    Returns
    -------
    pd.DataFrame
        Columns ``time_period``, ``psi``, and ``count data`` (number of
        non-missing ``col_var`` values in that production slice used for the
        actual distribution), one row per distinct ``col_period`` value in
        ``df_prod``.

    Raises
    ------
    ValueError
        If required columns are missing or ``n_bin`` is invalid for numeric PSI.
    """
    _validate_columns(df_ref, "df_ref", [col_var])
    _validate_columns(df_prod, "df_prod", [col_var, col_period])

    s_decide = df_ref[col_var]
    if s_decide.notna().any():
        use_categories = _col_var_use_category_bins(s_decide)
    else:
        use_categories = _col_var_use_category_bins(df_prod[col_var])

    if not use_categories and int(n_bin) < 2:
        raise ValueError("n_bin must be at least 2 for numeric PSI.")

    times = sorted(df_prod[col_period].dropna().unique().tolist())
    rows: list[dict[str, Any]] = []
    for t in times:
        mask = df_prod[col_period] == t
        df_slice = df_prod.loc[mask]
        if use_categories:
            psi_val = _psi_discrete_category_union(
                df_ref[col_var],
                df_slice[col_var],
                epsilon=epsilon,
            )
            n_data = int(df_slice[col_var].notna().sum())
        else:
            psi_val = compute_psi(
                df_ref[col_var],
                df_slice[col_var],
                n_bins=int(n_bin),
                epsilon=epsilon,
            )
            n_data = int(
                pd.to_numeric(df_slice[col_var], errors="coerce").notna().sum()
            )
        rows.append(
            {
                "time_period": t,
                "psi": float(psi_val),
                "count data": n_data,
            }
        )

    return pd.DataFrame(rows, columns=["time_period", "psi", "count data"])


def get_timely_binary_target_rate(
    df: pd.DataFrame,
    col_period: str,
    col_target: str,
) -> pd.DataFrame:
    """
    Positive rate (mean label) and count of ``col_target`` by time period.

    For each distinct value of ``col_period``, computes the sample mean of
    ``col_target`` (for a 0/1 binary label this is the event rate in that
    period) and the number of rows in that slice. Equivalent to:

    ``df[[col_period, col_target]].groupby(col_period).agg(['mean', 'count'])[col_target]``

    Parameters
    ----------
    df : pd.DataFrame
        Input data with a period column and a numeric/binary target column.
    col_period : str
        Column used as the grouping key (e.g. month ``YYYYMM``, day
        ``YYYYMMDD``).
    col_target : str
        Column whose mean and count are aggregated within each period.

    Returns
    -------
    pd.DataFrame
        Index: unique values of ``col_period`` (pandas groupby index; sorted
        by default when ``sort=True``). Columns: ``mean`` (label mean in the
        period) and ``count`` (row count in the period).

    Raises
    ------
    ValueError
        If ``col_period`` or ``col_target`` is not a column of ``df``.

    Notes
    -----
    **Agent / MCP:** stable two-column result plus period index; suitable for
    charts (rate over time) and volume checks (count per period).
    """
    _validate_columns(df, "get_timely_binary_target_rate", [col_period, col_target])
    return (
        df[[col_period, col_target]]
        .groupby(col_period)
        .agg(["mean", "count"])[col_target]
    )


def get_timely_target_rate_feature_segment(
    df: pd.DataFrame,
    cols_feat_bin: list[str],
    col_target: str,
    col_period: str,
) -> pd.DataFrame:
    """
    Target rate by pre-binned feature segment within each period or cohort.

    For each distinct ``col_period`` value (e.g. calendar month or a split label
    like ``train`` / ``valid``), each feature in ``cols_feat_bin``, and each
    **distinct non-null** value of that feature observed in that slice (the bin
    / segment), this reports how many rows have a non-null target, how many are
    the positive class, and the positive rate.

    The positive class is the **numerically larger** of the two distinct
    ``col_target`` values after coercion on rows where ``col_target`` is
    non-null globally; if the global target is not binary, rates are ``NaN``.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    cols_feat_bin : list of str
        Binned or categorical feature columns (deduplicated, first occurrence
        wins). Bin identity is the raw value in the column for that slice.
    col_target : str
        Binary target (coerced with ``pd.to_numeric``; two distinct values
        required globally among non-null targets).
    col_period : str
        Period or segment column; unique non-null values are processed, sorted
        for deterministic row order.

    Returns
    -------
    pd.DataFrame
        Columns ``time period``, ``feature name``, ``segment`` (bin/category
        value), ``count data`` (rows with non-null target in that bin and
        period), ``count positive`` (rows in that set with target equal to the
        positive class), ``positive rate`` (ratio, ``NaN`` when count data is
        0 or target is not globally binary).

    Raises
    ------
    ValueError
        If ``cols_feat_bin`` is empty or required columns are missing from
        ``df``.
    """
    if not cols_feat_bin:
        raise ValueError(
            "cols_feat_bin must be a non-empty list of column names."
        )
    feat_cols = list(dict.fromkeys(cols_feat_bin))
    _validate_columns(df, "df", feat_cols + [col_target, col_period])

    y_all = pd.to_numeric(df[col_target], errors="coerce").dropna()
    hi: float | None
    if len(y_all) < 2 or y_all.nunique() != 2:
        hi = None
    else:
        hi = float(sorted(y_all.unique().tolist())[-1])

    def _bin_sort_key(v: Any) -> tuple:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return (2, "")
        if isinstance(v, (bool, np.bool_)):
            return (0, int(bool(v)))
        if isinstance(v, (int, np.integer)):
            return (0, int(v))
        if isinstance(v, (float, np.floating)):
            return (0, float(v))
        return (1, str(v))

    periods = sorted(df[col_period].dropna().unique().tolist(), key=str)
    rows: list[dict[str, Any]] = []
    for t in periods:
        df_t = df.loc[df[col_period] == t]
        for feat in feat_cols:
            bins = df_t[feat].dropna().unique().tolist()
            bins_sorted = sorted(bins, key=_bin_sort_key)
            for b in bins_sorted:
                sub = df_t.loc[df_t[feat] == b]
                y_sub = pd.to_numeric(sub[col_target], errors="coerce")
                n_data = int(y_sub.notna().sum())
                if hi is None or n_data == 0:
                    rows.append(
                        {
                            "time period": t,
                            "feature name": feat,
                            "segment": b,
                            "count data": n_data,
                            "count positive": float(np.nan),
                            "positive rate": float(np.nan),
                        }
                    )
                else:
                    c_pos = int((y_sub == hi).sum())
                    rows.append(
                        {
                            "time period": t,
                            "feature name": feat,
                            "segment": b,
                            "count data": n_data,
                            "count positive": c_pos,
                            "positive rate": float(c_pos / n_data),
                        }
                    )

    return pd.DataFrame(
        rows,
        columns=[
            "time period",
            "feature name",
            "segment",
            "count data",
            "count positive",
            "positive rate",
        ],
    )


def get_nan_rate_timely(
    df: pd.DataFrame,
    col_features: list[str],
    col_time: str,
) -> pd.DataFrame:
    """
    Missing (NaN) rate per feature and per time period.

    For each distinct value of ``col_time`` in ``df``, computes the share of
    rows where that feature is
    missing, expressed as a percentage. Missing is evaluated with
    ``pandas.Series.isna()`` (covers ``NaN`` for floats, ``NaT`` for datetimes,
    and ``pandas.NA`` for nullable dtypes; it does **not** treat empty string
    as missing unless you coerce those values to NA beforehand).

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing features and a time-period column.
    col_features : list of str
        Feature column names to score; each must exist in ``df``.
    col_time : str
        Column name for the period label (e.g. monthly ``202501``, daily
        ``20250411``, quarter ``20251``, year ``2025``). Any hashable / sortable
        label is fine; periods are taken as all non-null unique values in
        ``df[col_time]``, sorted ascending.

    Returns
    -------
    pd.DataFrame
        Columns:

        - ``time_period``: value from ``col_time``
        - ``feature_name``: feature column name
        - ``missing_rate``: percent of missing values in that slice, rounded
          to two decimal places

        One row per (time period, feature). If ``df`` is empty or no periods
        apply, returns an empty frame with the same columns.

    Raises
    ------
    ValueError
        If ``col_features`` is empty or any listed column is missing from
        ``df``.

    Notes
    -----
    Intended for agent / MCP tools: deterministic column names, explicit
    validation, and stable rounding for reporting.
    """
    if not col_features:
        raise ValueError("col_features must be a non-empty list of column names.")

    feat_cols = list(col_features)
    _validate_columns(df, "df", feat_cols + [col_time])

    periods = sorted(df[col_time].dropna().unique().tolist())

    rows: list[dict] = []
    for t in periods:
        sub = df.loc[df[col_time] == t]
        n_rows = len(sub)
        if n_rows == 0:
            continue
        for feat in feat_cols:
            miss_share = sub[feat].isna().mean()
            pct = float(miss_share) * 100.0
            rows.append(
                {
                    "time_period": t,
                    "feature_name": feat,
                    "missing_rate": round(pct, 2),
                }
            )

    out = pd.DataFrame(
        rows,
        columns=["time_period", "feature_name", "missing_rate"],
    )

    out = out.set_index(['feature_name', 'time_period']).unstack()
    return out


def get_nan_rate(
    df: pd.DataFrame,
    col_features: list[str],
) -> pd.DataFrame:
    """
    Missing (NaN) rate per feature over the full dataframe.

    Simplified sibling of :func:`get_nan_rate_timely`: same definition of
    missing (``Series.isna()``), same percentage scale, but no time slicing.
    Each row is one feature across all rows in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing the listed feature columns.
    col_features : list of str
        Feature column names; each must exist in ``df``.

    Returns
    -------
    pd.DataFrame
        Columns:

        - ``feature_name``: feature column name
        - ``missing_rate``: share of missing values in ``df`` for that column,
          as a percent rounded to two decimal places

        One row per feature. If ``df`` has zero rows, ``missing_rate`` is
        ``NaN`` for each feature (columns still present).

    Raises
    ------
    ValueError
        If ``col_features`` is empty or any listed column is missing from
        ``df``.

    Notes
    -----
    Does **not** treat empty strings as missing unless values are NA already.
    Intended for agent / MCP use with a small, stable schema.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", None, "z"]})
    >>> out = get_nan_rate(df, ["a", "b"])
    >>> list(out.columns)
    ['feature_name', 'missing_rate']
    >>> float(out.loc[out["feature_name"] == "a", "missing_rate"].iloc[0])
    33.33
    """
    if not col_features:
        raise ValueError("col_features must be a non-empty list of column names.")

    feat_cols = list(col_features)
    _validate_columns(df, "df", feat_cols)

    rows: list[dict] = []
    for feat in feat_cols:
        miss_share = df[feat].isna().mean()
        pct = float(miss_share) * 100.0
        rows.append(
            {
                "feature_name": feat,
                "missing_rate": round(pct, 2),
            }
        )

    out = pd.DataFrame(rows, columns=["feature_name", "missing_rate"])
    return out


def get_target_rate_sample(
    df: pd.DataFrame,
    col_type: str,
    col_target: str,
) -> pd.DataFrame:
    """
    Row count and mean label per distinct ``col_type`` value (sample / cohort).

    Uses ``df[[col_type, col_target]].groupby(col_type).agg(['count', 'mean'])``
    on ``col_target`` so each group reports how many rows were used and the
    mean of the target in that slice (e.g. default rate by data split).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    col_type : str
        Column used as the group key (e.g. split / cohort label).
    col_target : str
        Label column; ``mean`` is the group-wise average of this column.

    Returns
    -------
    pd.DataFrame
        Index: ``col_type`` categories. Columns ``count`` and ``mean`` for
        ``col_target`` (pandas ``groupby`` / ``agg`` layout).

    Raises
    ------
    ValueError
        If ``col_type`` or ``col_target`` is not a column of ``df``.
    """
    _validate_columns(df, "get_target_rate_sample", [col_type, col_target])
    return (
        df[[col_type, col_target]]
        .groupby(col_type)
        .agg(["count", "mean"])[col_target]
    )


def _roc_auc_binary_feature(y_true: pd.Series, y_score: pd.Series) -> float:
    """
    ROC AUC treating ``y_score`` as a ranking score for binary ``y_true``.

    Returns NaN when AUC is undefined (e.g. fewer than two classes after
    cleaning, or too few rows).
    """
    y = pd.to_numeric(y_true, errors="coerce")
    x = pd.to_numeric(y_score, errors="coerce")
    valid = y.notna() & x.notna()
    y = y.loc[valid]
    x = x.loc[valid]
    if len(y) < 2:
        return float(np.nan)
    if y.nunique() != 2:
        return float(np.nan)
    try:
        return float(roc_auc_score(y.to_numpy(), x.to_numpy()))
    except ValueError:
        return float(np.nan)


def get_feature_predictive_power_timely(
    df: pd.DataFrame,
    col_feats: list[str],
    col_time: str,
    col_target: str,
) -> pd.DataFrame:
    """
    Univariate ROC AUC per feature and per time period (ranking vs. binary label).

    For each distinct ``col_time`` value and each feature in ``col_feats``,
    rows in that time slice are used to compute ``roc_auc_score`` from
    scikit-learn, with the feature as ``y_score`` and ``col_target`` as the
    binary ground truth. This measures how well the feature ranks positives
    vs. negatives in that slice (not a trained model).

    Parameters
    ----------
    df : pd.DataFrame
        Input rows with features, time period, and binary label.
    col_feats : list of str
        Feature columns to evaluate; each must exist in ``df``.
    col_time : str
        Period column (e.g. monthly ``202501``, daily integer, etc.). All
        non-null unique values are used, sorted ascending.
    col_target : str
        Binary label column (two distinct values after coercion to numeric;
        typical encoding is ``0``/``1``). Non-numeric labels are coerced with
        ``errors='coerce'`` and dropped with missing features.

    Returns
    -------
    pd.DataFrame
        Columns: ``time_period``, ``feature_name``, ``aucroc``. One row per
        (period, feature). ``aucroc`` is ``NaN`` when undefined for that slice
        (e.g. only one class present).

    Raises
    ------
    ValueError
        If ``col_feats`` is empty or required columns are missing from ``df``.

    Notes
    -----
    - **Direction:** AUC below ``0.5`` means higher feature values associate
      with the negative class more than the positive under scikit-learn's
      convention; you may interpret ``1 - aucroc`` as "inverted" strength if
      useful.
    - **Agent / MCP:** Deterministic output columns and validation for stable
      tool contracts.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({
    ...     "t": np.repeat([202501, 202502], n // 2),
    ...     "x": x,
    ...     "y": (rng.normal(size=n) + x * 0.5) > 0,
    ... })
    >>> out = get_feature_predictive_power_timely(df, ["x"], "t", "y")
    >>> set(out.columns) == {"time_period", "feature_name", "aucroc"}
    True
    """
    if not col_feats:
        raise ValueError("col_feats must be a non-empty list of column names.")

    feat_cols = list(col_feats)
    _validate_columns(df, "df", feat_cols + [col_time, col_target])

    periods = sorted(df[col_time].dropna().unique().tolist())
    rows: list[dict] = []
    for t in periods:
        sub = df.loc[df[col_time] == t]
        if len(sub) == 0:
            continue
        for feat in feat_cols:
            auc = _roc_auc_binary_feature(sub[col_target], sub[feat])
            rows.append(
                {
                    "time_period": t,
                    "feature_name": feat,
                    "aucroc": auc,
                }
            )

    out = pd.DataFrame(
        rows,
        columns=["time_period", "feature_name", "aucroc"],
    )
    return out


def get_feature_predictive_power(
    df: pd.DataFrame,
    col_feats: list[str],
    col_target: str,
) -> pd.DataFrame:
    """
    Univariate ROC AUC per feature over the full dataframe (vs. binary label).

    Simplified sibling of :func:`get_feature_predictive_power_timely`: same
    use of ``sklearn.metrics.roc_auc_score`` with each feature as ``y_score``
    and ``col_target`` as binary ground truth, but **no** time slicing—one AUC
    per feature across all rows in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Input rows with features and binary label.
    col_feats : list of str
        Feature columns to evaluate; each must exist in ``df``.
    col_target : str
        Binary label (two distinct values after numeric coercion; typically
        ``0``/``1``). Coerced with ``errors='coerce'``; rows missing label or
        feature after coercion are dropped per feature inside
        :func:`_roc_auc_binary_feature`.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature_name``, ``aucroc``. One row per feature. ``aucroc``
        is ``NaN`` when undefined (e.g. fewer than two label classes in ``df``
        after cleaning for that feature).

    Raises
    ------
    ValueError
        If ``col_feats`` is empty or required columns are missing from ``df``.

    Notes
    -----
    See :func:`get_feature_predictive_power_timely` Notes on AUC ``< 0.5``.
    **Agent / MCP:** fixed schema ``(feature_name, aucroc)``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(3)
    >>> n = 200
    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({"x": x, "y": (rng.normal(size=n) + x * 0.5) > 0})
    >>> out = get_feature_predictive_power(df, ["x"], "y")
    >>> list(out.columns)
    ['feature_name', 'aucroc']
    >>> bool(out.loc[out["feature_name"] == "x", "aucroc"].notna().iloc[0])
    True
    """
    if not col_feats:
        raise ValueError("col_feats must be a non-empty list of column names.")

    feat_cols = list(col_feats)
    _validate_columns(df, "df", feat_cols + [col_target])

    rows: list[dict] = []
    for feat in feat_cols:
        auc = _roc_auc_binary_feature(df[col_target], df[feat])
        rows.append({"feature_name": feat, "aucroc": auc})

    out = pd.DataFrame(rows, columns=["feature_name", "aucroc"])
    return out


def _standardized_binary_auc(auc: float) -> float:
    """Map univariate binary ROC AUC to [0.5, 1] strength (invert if below 0.5)."""
    if auc is None or not np.isfinite(auc):
        return float(np.nan)
    if auc < 0.5:
        return float(1.0 - auc)
    return float(auc)


def _abs_corr_pair(corr: pd.DataFrame, a: str, b: str) -> float:
    """Absolute Pearson correlation between rows ``a`` and ``b``; NaN treated as 0."""
    if a not in corr.index or b not in corr.index:
        return 0.0
    v = corr.loc[a, b]
    if pd.isna(v):
        return 0.0
    return abs(float(v))


def select_features_auc_max_corr(
    df: pd.DataFrame,
    cols_feat: list[str],
    col_target: str,
    max_corr: float = 0.5,
) -> tuple[list[str], pd.DataFrame]:
    """
    Greedy feature selection by standardized ROC AUC with a correlation cap.

    Univariate binary ROC AUC is computed per feature (same semantics as
    :func:`_roc_auc_binary_feature`). The **standardized** score is ``auc`` if
    ``auc >= 0.5``, else ``1 - auc``. Features are ranked by standardized AUC
    (descending); a greedy pass keeps a feature only if its absolute Pearson
    correlation to **every** already-selected feature is at most ``max_corr``
    (using pairwise-complete numeric coercion on ``df`` for the correlation
    matrix). The returned feature list is then re-sorted by standardized AUC
    (descending).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    cols_feat : list of str
        Candidate feature columns (deduplicated, first occurrence wins).
    col_target : str
        Binary target column (two classes after numeric coercion; see
        :func:`get_feature_predictive_power`).
    max_corr : float, default 0.5
        Maximum allowed absolute Pearson correlation between any pair of
        selected features. Must lie in ``[0, 1]``.

    Returns
    -------
    selected_features : list of str
        Selected names sorted by standardized AUC descending.
    summary : pandas.DataFrame
        Columns ``feature_name``, ``max_correlation``, ``auc_roc``,
        ``standardized_auc_roc``. One row per selected feature, in the same
        order as ``selected_features``. ``max_correlation`` is the largest
        absolute correlation between that feature and another selected feature
        (``0.0`` when only one feature is selected).

    Raises
    ------
    ValueError
        If ``cols_feat`` is empty, ``max_corr`` is outside ``[0, 1]``, or
        required columns are missing from ``df``.
    """
    if not cols_feat:
        raise ValueError("cols_feat must be a non-empty list of column names.")

    if max_corr < 0 or max_corr > 1:
        raise ValueError("max_corr must be between 0 and 1 inclusive.")

    order = list(dict.fromkeys(cols_feat))
    _validate_columns(df, "select_features_auc_max_corr", [col_target] + order)

    auc_by_feat: dict[str, float] = {}
    for name in order:
        auc_by_feat[name] = _roc_auc_binary_feature(df[col_target], df[name])

    usable = [n for n in order if np.isfinite(auc_by_feat[n])]
    if not usable:
        empty = pd.DataFrame(
            columns=[
                "feature_name",
                "max_correlation",
                "auc_roc",
                "standardized_auc_roc",
            ]
        )
        return [], empty

    std_by_feat = {n: _standardized_binary_auc(auc_by_feat[n]) for n in usable}
    X_num = df[usable].apply(pd.to_numeric, errors="coerce")
    corr = X_num.corr(method="pearson")

    ranked = sorted(usable, key=lambda f: (-std_by_feat[f], f))

    selected: list[str] = []
    for f in ranked:
        if not selected:
            selected.append(f)
            continue
        peak = max(_abs_corr_pair(corr, f, s) for s in selected)
        if peak <= max_corr:
            selected.append(f)

    selected_sorted = sorted(selected, key=lambda f: (-std_by_feat[f], f))

    rows: list[dict[str, Any]] = []
    for f in selected_sorted:
        others = [s for s in selected_sorted if s != f]
        if not others:
            mx = 0.0
        else:
            mx = max(_abs_corr_pair(corr, f, s) for s in others)
        rows.append(
            {
                "feature_name": f,
                "max_correlation": float(mx),
                "auc_roc": float(auc_by_feat[f]),
                "standardized_auc_roc": float(std_by_feat[f]),
            }
        )

    summary = pd.DataFrame(
        rows,
        columns=[
            "feature_name",
            "max_correlation",
            "auc_roc",
            "standardized_auc_roc",
        ],
    )
    return selected_sorted, summary


def _iv_binary_from_woe_column(
    y: pd.Series,
    woe: pd.Series,
    *,
    n_bins: int = 10,
) -> float:
    """
    Information value for a binary label vs. a WoE column via quantile bins.

    ``y`` is coerced to numeric; the larger of the two distinct classes is
    treated as the event (1). ``woe`` is coerced to numeric. Rows with missing
    ``y`` or ``woe`` are dropped. ``woe`` is split into at most ``n_bins``
    quantile bins (``pd.qcut``, ``duplicates='drop'``), then IV is

    ``sum_i (dist_non_event_i - dist_event_i) * ln((dist_non_event_i + eps) /
    (dist_event_i + eps))`` over bins, matching the usual weight-of-evidence
    total IV for a binned numeric characteristic.
    """
    yn = pd.to_numeric(y, errors="coerce")
    w = pd.to_numeric(woe, errors="coerce")
    valid = yn.notna() & w.notna()
    if int(valid.sum()) < 2:
        return float(np.nan)
    yn2 = yn.loc[valid]
    x = w.loc[valid]
    if yn2.nunique() != 2:
        return float(np.nan)
    lo, hi = sorted(yn2.unique().tolist())[:2]
    y_ev = (yn2 == hi).to_numpy(dtype=int)
    total_e = int(y_ev.sum())
    total_ne = int(len(y_ev) - total_e)
    if total_e == 0 or total_ne == 0:
        return float(np.nan)
    n_q = min(int(n_bins), int(x.nunique(dropna=True)))
    if n_q < 2:
        return float(np.nan)
    try:
        bins = pd.qcut(x, q=n_q, duplicates="drop")
    except (ValueError, TypeError):
        return float(np.nan)
    ep = 1e-10
    iv = 0.0
    for b in bins.cat.categories:
        m = (bins == b).to_numpy()
        e = int(y_ev[m].sum())
        ne = int(m.sum() - e)
        dist_e = e / total_e
        dist_ne = ne / total_ne
        woe_i = np.log((dist_ne + ep) / (dist_e + ep))
        iv += (dist_ne - dist_e) * woe_i
    return float(iv)


def select_features_iv_max_corr(
    df: pd.DataFrame,
    cols_feat_woe: list[str],
    col_target: str,
    max_corr: float = 0.5,
) -> tuple[list[str], pd.DataFrame]:
    """
    Greedy feature selection by information value (IV) with a correlation cap.

    IV is computed on each WoE-like column by quantile-binning the values and
    applying the standard binary IV formula (event = numerically higher of the
    two ``col_target`` classes). Univariate binary ROC AUC uses
    :func:`_roc_auc_binary_feature` on ``(col_target, column)``. The **standardized**
    AUC is ``auc`` if ``auc >= 0.5``, else ``1 - auc``. Features are ranked by
    IV (descending); a greedy pass keeps a feature only if its absolute Pearson
    correlation to every already-selected feature is at most ``max_corr``. The
    returned list is re-sorted by IV (descending).

    Parameters
    ----------
    df : pd.DataFrame
        Input data (typically includes WoE-encoded feature columns).
    cols_feat_woe : list of str
        Candidate WoE columns (deduplicated, first occurrence wins).
    col_target : str
        Binary target (two classes after numeric coercion; see
        :func:`get_feature_predictive_power`).
    max_corr : float, default 0.5
        Maximum allowed absolute Pearson correlation between any pair of
        selected features. Must lie in ``[0, 1]``.

    Returns
    -------
    selected_features : list of str
        Selected names sorted by IV descending.
    summary : pandas.DataFrame
        Columns ``feature_name``, ``max_correlation``, ``iv``, ``auc_roc``,
        ``standardized_auc_roc``. One row per selected feature, same order as
        ``selected_features``. ``max_correlation`` is the largest absolute
        correlation vs. other selected features (``0.0`` if only one selected).

    Raises
    ------
    ValueError
        If ``cols_feat_woe`` is empty, ``max_corr`` is outside ``[0, 1]``, or
        required columns are missing from ``df``.

    Notes
    -----
    IV depends on the chosen quantile bin count (fixed at 10); it approximates
    the IV of the underlying score treated as a continuous characteristic.
    """
    if not cols_feat_woe:
        raise ValueError(
            "cols_feat_woe must be a non-empty list of column names."
        )

    if max_corr < 0 or max_corr > 1:
        raise ValueError("max_corr must be between 0 and 1 inclusive.")

    order = list(dict.fromkeys(cols_feat_woe))
    _validate_columns(df, "select_features_iv_max_corr", [col_target] + order)

    iv_by_feat: dict[str, float] = {}
    auc_by_feat: dict[str, float] = {}
    for name in order:
        iv_by_feat[name] = _iv_binary_from_woe_column(df[col_target], df[name])
        auc_by_feat[name] = _roc_auc_binary_feature(df[col_target], df[name])

    usable = [n for n in order if np.isfinite(iv_by_feat[n])]
    if not usable:
        empty = pd.DataFrame(
            columns=[
                "feature_name",
                "max_correlation",
                "iv",
                "auc_roc",
                "standardized_auc_roc",
            ]
        )
        return [], empty

    std_by_feat = {n: _standardized_binary_auc(auc_by_feat[n]) for n in usable}
    X_num = df[usable].apply(pd.to_numeric, errors="coerce")
    corr = X_num.corr(method="pearson")

    ranked = sorted(usable, key=lambda f: (-iv_by_feat[f], f))

    selected: list[str] = []
    for f in ranked:
        if not selected:
            selected.append(f)
            continue
        peak = max(_abs_corr_pair(corr, f, s) for s in selected)
        if peak <= max_corr:
            selected.append(f)

    selected_sorted = sorted(selected, key=lambda f: (-iv_by_feat[f], f))

    rows: list[dict[str, Any]] = []
    for f in selected_sorted:
        others = [s for s in selected_sorted if s != f]
        if not others:
            mx = 0.0
        else:
            mx = max(_abs_corr_pair(corr, f, s) for s in others)
        rows.append(
            {
                "feature_name": f,
                "max_correlation": float(mx),
                "iv": float(iv_by_feat[f]),
                "auc_roc": float(auc_by_feat[f]),
                "standardized_auc_roc": float(std_by_feat[f]),
            }
        )

    summary = pd.DataFrame(
        rows,
        columns=[
            "feature_name",
            "max_correlation",
            "iv",
            "auc_roc",
            "standardized_auc_roc",
        ],
    )
    return selected_sorted, summary


def _logistic_fit_loglik_params_probs(
    X: np.ndarray,
    y01: np.ndarray,
    *,
    eps: float = 1e-15,
    random_state: int = 0,
) -> tuple[float, int, np.ndarray]:
    """
    Fit binary logistic regression (large ``C``) and return
    ``(log_likelihood, k_params_including_intercept, p_hat)``.
    """
    if X.shape[0] != y01.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if X.shape[1] == 0:
        p = np.clip(
            np.full_like(y01, float(y01.mean()), dtype=float), eps, 1.0 - eps
        )
        ll = float(
            (y01 * np.log(p + eps) + (1.0 - y01) * np.log(1.0 - p + eps)).sum()
        )
        return ll, 1, p
    clf = LogisticRegression(
        C=1e12,
        solver="lbfgs",
        max_iter=8000,
        tol=1e-7,
        random_state=random_state,
        fit_intercept=True,
    )
    clf.fit(X, y01)
    prob = clf.predict_proba(X)[:, 1].astype(float)
    prob = np.clip(prob, eps, 1.0 - eps)
    ll = float(
        (y01 * np.log(prob + eps) + (1.0 - y01) * np.log(1.0 - prob + eps)).sum()
    )
    k = int(X.shape[1]) + 1
    return ll, k, prob


def _aic_intercept_only_logistic(y01: np.ndarray, *, eps: float = 1e-15) -> float:
    """AIC for intercept-only Bernoulli model (MLE prevalence)."""
    n = int(y01.shape[0])
    if n == 0:
        return float("nan")
    p = float(np.clip(y01.mean(), eps, 1.0 - eps))
    ll = float(
        (y01 * np.log(p + eps) + (1.0 - y01) * np.log(1.0 - p + eps)).sum()
    )
    k = 1
    return float(2.0 * k - 2.0 * ll)


def _bic_intercept_only_logistic(y01: np.ndarray, *, eps: float = 1e-15) -> float:
    """BIC for intercept-only Bernoulli model (MLE prevalence). ``n = len(y01)``."""
    n = int(y01.shape[0])
    if n == 0:
        return float("nan")
    p = float(np.clip(y01.mean(), eps, 1.0 - eps))
    ll = float(
        (y01 * np.log(p + eps) + (1.0 - y01) * np.log(1.0 - p + eps)).sum()
    )
    k = 1
    return float(k * np.log(n) - 2.0 * ll)


def _aic_and_probs_logistic_mle(
    X: np.ndarray,
    y01: np.ndarray,
    *,
    eps: float = 1e-15,
    random_state: int = 0,
) -> tuple[float, np.ndarray]:
    """
    Fit unpenalized binary logistic regression (MLE) and return (AIC, p_hat).

    ``AIC = 2k - 2 log L`` with ``k = n_features + 1`` (intercept included).
    """
    ll, k, prob = _logistic_fit_loglik_params_probs(
        X, y01, eps=eps, random_state=random_state
    )
    return float(2.0 * k - 2.0 * ll), prob


def _bic_and_probs_logistic_mle(
    X: np.ndarray,
    y01: np.ndarray,
    *,
    eps: float = 1e-15,
    random_state: int = 0,
) -> tuple[float, np.ndarray]:
    """
    Fit binary logistic regression (MLE) and return (BIC, p_hat).

    ``BIC = k * ln(n) - 2 * log L`` (natural log) with ``k = n_features + 1`` and
    ``n = len(y)``.
    """
    n = int(y01.shape[0])
    ll, k, prob = _logistic_fit_loglik_params_probs(
        X, y01, eps=eps, random_state=random_state
    )
    return float(k * np.log(n) - 2.0 * ll), prob


def _max_pairwise_abs_corr_subset(
    feats: list[str],
    corr_df: pd.DataFrame,
) -> float:
    """Largest absolute Pearson correlation among distinct pairs in ``feats``."""
    if len(feats) < 2:
        return 0.0
    mx = 0.0
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            mx = max(mx, _abs_corr_pair(corr_df, feats[i], feats[j]))
    return float(mx)


def select_features_aic_backward(
    df: pd.DataFrame,
    cols_feat: list[str],
    col_target: str,
    max_corr: float = 0.5,
    min_delta: float = 1e-9,
) -> tuple[list[str], pd.DataFrame]:
    """
    Backward stepwise logistic regression minimizing AIC with a correlation cap.

    Rows with missing ``col_target`` or any candidate feature (after numeric
    coercion) are dropped. The target is coerced to numeric; the **larger**
    distinct value is class 1. An **initial** active set is built by scanning
    ``cols_feat`` in order and greedily keeping each feature with non-zero
    variance whose absolute correlation to every feature already kept is at
    most ``max_corr``. Starting from the logistic model on that set, each step
    removes the single feature whose removal yields the **lowest** AIC among
    all removals (tie-break by name); the step is taken only if AIC decreases
    by at least ``min_delta`` versus before. Stops when no removal meets that
    threshold. The same ``LogisticRegression`` settings as
    :func:`select_features_aic_forward` apply.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling data.
    cols_feat : list of str
        Candidate predictors (deduplicated, first occurrence wins).
    col_target : str
        Binary outcome (two distinct numeric values after coercion).
    max_corr : float, default 0.5
        Used when building the initial set (pairwise absolute correlation bound
        for inclusion). Must lie in ``[0, 1]``.
    min_delta : float, default 1e-9
        Minimum AIC decrease for a backward step (``aic_before - aic_after``).
        Must be finite and non-negative.

    Returns
    -------
    selected_features : list of str
        Remaining predictors after backward elimination, ordered as in
        ``cols_feat``.
    history : pandas.DataFrame
        Columns: ``step_number``, ``eliminated_feature_name``, ``max_correlation``,
        ``total_aic``, ``delta_aic``, ``roc_auc``, ``gini``. ``max_correlation`` is the
        maximum absolute pairwise correlation among features **still in the
        model** after the step (``0.0`` if fewer than two remain).
        ``delta_aic`` is ``total_aic`` after minus before the step (negative
        means AIC decreased). ``roc_auc`` is training ROC-AUC on the reduced model;
        ``gini`` is ``2 * roc_auc - 1`` (clipped to ``[-1, 1]``).

    Raises
    ------
    ValueError
        If ``cols_feat`` is empty, ``max_corr`` is outside ``[0, 1]``, required
        columns are missing, no feature passes the initial correlation screen,
        or after cleaning there are fewer than five rows or fewer than two
        target classes.
    """
    if not cols_feat:
        raise ValueError("cols_feat must be a non-empty list of column names.")

    if max_corr < 0 or max_corr > 1:
        raise ValueError("max_corr must be between 0 and 1 inclusive.")

    if not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be a finite non-negative float.")

    order = list(dict.fromkeys(cols_feat))
    _validate_columns(df, "select_features_aic_backward", [col_target] + order)

    sub = df[[col_target] + order].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 5:
        raise ValueError(
            "Need at least five complete-case rows for col_target and cols_feat."
        )

    y_raw = sub[col_target].to_numpy(dtype=float)
    if len(np.unique(y_raw)) != 2:
        raise ValueError(
            "col_target must have exactly two distinct numeric values on complete cases."
        )
    lo, hi = sorted(np.unique(y_raw).tolist())[:2]
    y01 = (y_raw == hi).astype(int)

    X_all = sub[order].to_numpy(dtype=float)
    corr_df = pd.DataFrame(X_all, columns=order).corr(method="pearson")

    var_ok = {
        order[j]
        for j in range(len(order))
        if float(np.nanstd(X_all[:, j])) > 1e-12
    }

    active: list[str] = []
    for f in order:
        if f not in var_ok:
            continue
        if not active:
            active.append(f)
            continue
        if max(_abs_corr_pair(corr_df, f, s) for s in active) <= max_corr:
            active.append(f)

    if not active:
        raise ValueError(
            "No feature could be placed in the initial model under max_corr; "
            "relax max_corr or adjust cols_feat."
        )

    try:
        aic_cur, prob_cur = _aic_and_probs_logistic_mle(
            sub[active].to_numpy(dtype=float), y01
        )
    except Exception as exc:
        raise ValueError(
            "Could not fit initial logistic model on the correlation-screened set."
        ) from exc

    rows: list[dict[str, Any]] = []
    step = 0

    while True:
        trials: list[tuple[float, str, list[str], np.ndarray]] = []
        for v in active:
            nxt = [x for x in active if x != v]
            if not nxt:
                aic_new = _aic_intercept_only_logistic(y01)
                p0 = float(np.clip(y01.mean(), 1e-15, 1.0 - 1e-15))
                prob = np.full_like(y01, p0, dtype=float)
            else:
                try:
                    aic_new, prob = _aic_and_probs_logistic_mle(
                        sub[nxt].to_numpy(dtype=float), y01
                    )
                except Exception:
                    continue
            if not np.isfinite(aic_new):
                continue
            if (aic_cur - aic_new) >= min_delta:
                trials.append((aic_new, v, nxt, prob))

        if not trials:
            break

        aic_new, v_out, active_next, prob = min(trials, key=lambda t: (t[0], t[1]))
        step += 1
        delta_aic = float(aic_new - aic_cur)
        mx_pair = _max_pairwise_abs_corr_subset(active_next, corr_df)
        try:
            auc = float(roc_auc_score(y01, prob))
        except ValueError:
            auc = float(np.nan)

        rows.append(
            {
                "step_number": int(step),
                "eliminated_feature_name": v_out,
                "max_correlation": float(mx_pair),
                "total_aic": float(aic_new),
                "delta_aic": float(delta_aic),
                "roc_auc": auc,
                "gini": _gini_binary_from_auc(auc),
            }
        )
        active = active_next
        aic_cur = float(aic_new)

    selected = [f for f in order if f in active]
    history = pd.DataFrame(
        rows,
        columns=[
            "step_number",
            "eliminated_feature_name",
            "max_correlation",
            "total_aic",
            "delta_aic",
            "roc_auc",
            "gini",
        ],
    )
    return selected, history


def select_features_aic_forward(
    df: pd.DataFrame,
    cols_feat: list[str],
    col_target: str,
    max_corr: float = 0.5,
    min_delta: float = 1e-9,
) -> tuple[list[str], pd.DataFrame]:
    """
    Forward stepwise logistic regression minimizing AIC with a correlation cap.

    Rows with missing ``col_target`` or any candidate feature (after numeric
    coercion) are dropped. The target is coerced to numeric; the **larger**
    distinct value is treated as class 1. At each step, among features not yet
    selected whose absolute Pearson correlation to every already-selected
    feature is at most ``max_corr`` (and with non-zero variance on the modeling
    frame), the model that lowers AIC by at least ``min_delta`` when the
    feature is added is chosen; ties break by feature name. Stops when no
    eligible feature meets that threshold. Fits ``LogisticRegression`` with a
    very large ``C`` (near-unpenalized L2) so AIC from ``2k - 2 log L`` matches
    usual MLE stepwise practice.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling data.
    cols_feat : list of str
        Candidate predictors (deduplicated, first occurrence wins).
    col_target : str
        Binary outcome column (two distinct numeric values after coercion).
    max_corr : float, default 0.5
        Maximum allowed absolute Pearson correlation between the newly added
        feature and **each** feature already in the model. Must lie in ``[0, 1]``.
    min_delta : float, default 1e-9
        Minimum AIC decrease for a forward step (``aic_before - aic_after``).
        Must be finite and non-negative.

    Returns
    -------
    selected_features : list of str
        Names in the order they entered the model.
    history : pandas.DataFrame
        Columns: ``step_number``, ``added_feature_name``, ``max_correlation``,
        ``total_aic``, ``delta_aic``, ``roc_auc``, ``gini``. ``delta_aic`` is
        ``total_aic`` after minus before the step (negative means AIC
        decreased). ``roc_auc`` is the training ROC-AUC of predicted
        probabilities vs. ``y``; ``gini`` is ``2 * roc_auc - 1`` (clipped to ``[-1, 1]``).

    Raises
    ------
    ValueError
        If ``cols_feat`` is empty, ``max_corr`` is outside ``[0, 1]``, required
        columns are missing, or after cleaning there are fewer than five rows
        or fewer than two target classes.
    """
    if not cols_feat:
        raise ValueError("cols_feat must be a non-empty list of column names.")

    if max_corr < 0 or max_corr > 1:
        raise ValueError("max_corr must be between 0 and 1 inclusive.")

    if not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be a finite non-negative float.")

    order = list(dict.fromkeys(cols_feat))
    _validate_columns(df, "select_features_aic_forward", [col_target] + order)

    sub = df[[col_target] + order].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 5:
        raise ValueError(
            "Need at least five complete-case rows for col_target and cols_feat."
        )

    y_raw = sub[col_target].to_numpy(dtype=float)
    if len(np.unique(y_raw)) != 2:
        raise ValueError(
            "col_target must have exactly two distinct numeric values on complete cases."
        )
    lo, hi = sorted(np.unique(y_raw).tolist())[:2]
    y01 = (y_raw == hi).astype(int)

    X_all = sub[order].to_numpy(dtype=float)
    corr_df = pd.DataFrame(X_all, columns=order).corr(method="pearson")

    var_ok = {
        order[j]
        for j in range(len(order))
        if float(np.nanstd(X_all[:, j])) > 1e-12
    }

    selected: list[str] = []
    aic_old = _aic_intercept_only_logistic(y01)
    rows: list[dict[str, Any]] = []
    step = 0

    while True:
        trials: list[tuple[float, str, float, np.ndarray]] = []
        for f in order:
            if f in selected or f not in var_ok:
                continue
            if selected:
                peak = max(_abs_corr_pair(corr_df, f, s) for s in selected)
                if peak > max_corr:
                    continue
            else:
                peak = 0.0
            cols = selected + [f]
            jidx = [order.index(c) for c in cols]
            X = X_all[:, jidx]
            try:
                aic_new, prob = _aic_and_probs_logistic_mle(X, y01)
            except Exception:
                continue
            if not np.isfinite(aic_new):
                continue
            if (aic_old - aic_new) >= min_delta:
                trials.append((aic_new, f, float(peak), prob))

        if not trials:
            break

        aic_new, f_add, peak, prob = min(trials, key=lambda t: (t[0], t[1]))
        step += 1
        delta_aic = float(aic_new - aic_old)
        try:
            auc = float(roc_auc_score(y01, prob))
        except ValueError:
            auc = float(np.nan)

        rows.append(
            {
                "step_number": int(step),
                "added_feature_name": f_add,
                "max_correlation": float(peak),
                "total_aic": float(aic_new),
                "delta_aic": float(delta_aic),
                "roc_auc": auc,
                "gini": _gini_binary_from_auc(auc),
            }
        )
        selected.append(f_add)
        aic_old = float(aic_new)

    history = pd.DataFrame(
        rows,
        columns=[
            "step_number",
            "added_feature_name",
            "max_correlation",
            "total_aic",
            "delta_aic",
            "roc_auc",
            "gini",
        ],
    )
    return selected, history


def select_features_bic_backward(
    df: pd.DataFrame,
    cols_feat: list[str],
    col_target: str,
    max_corr: float = 0.5,
    min_delta: float = 1e-9,
) -> tuple[list[str], pd.DataFrame]:
    """
    Backward stepwise logistic regression minimizing BIC with a correlation cap.

    Same procedure as :func:`select_features_aic_backward`, but each step
    minimizes **BIC** ``= k ln(n) - 2 log L`` on the modeling sample instead of
    AIC. A backward step is taken only if BIC decreases by at least
    ``min_delta``. History columns use ``total_bic``, ``delta_bic``,
    ``roc_auc``, and ``gini``.
    """
    if not cols_feat:
        raise ValueError("cols_feat must be a non-empty list of column names.")

    if max_corr < 0 or max_corr > 1:
        raise ValueError("max_corr must be between 0 and 1 inclusive.")

    if not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be a finite non-negative float.")

    order = list(dict.fromkeys(cols_feat))
    _validate_columns(df, "select_features_bic_backward", [col_target] + order)

    sub = df[[col_target] + order].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 5:
        raise ValueError(
            "Need at least five complete-case rows for col_target and cols_feat."
        )

    y_raw = sub[col_target].to_numpy(dtype=float)
    if len(np.unique(y_raw)) != 2:
        raise ValueError(
            "col_target must have exactly two distinct numeric values on complete cases."
        )
    lo, hi = sorted(np.unique(y_raw).tolist())[:2]
    y01 = (y_raw == hi).astype(int)

    X_all = sub[order].to_numpy(dtype=float)
    corr_df = pd.DataFrame(X_all, columns=order).corr(method="pearson")

    var_ok = {
        order[j]
        for j in range(len(order))
        if float(np.nanstd(X_all[:, j])) > 1e-12
    }

    active: list[str] = []
    for f in order:
        if f not in var_ok:
            continue
        if not active:
            active.append(f)
            continue
        if max(_abs_corr_pair(corr_df, f, s) for s in active) <= max_corr:
            active.append(f)

    if not active:
        raise ValueError(
            "No feature could be placed in the initial model under max_corr; "
            "relax max_corr or adjust cols_feat."
        )

    try:
        bic_cur, _ = _bic_and_probs_logistic_mle(
            sub[active].to_numpy(dtype=float), y01
        )
    except Exception as exc:
        raise ValueError(
            "Could not fit initial logistic model on the correlation-screened set."
        ) from exc

    rows: list[dict[str, Any]] = []
    step = 0

    while True:
        trials: list[tuple[float, str, list[str], np.ndarray]] = []
        for v in active:
            nxt = [x for x in active if x != v]
            if not nxt:
                bic_new = _bic_intercept_only_logistic(y01)
                p0 = float(np.clip(y01.mean(), 1e-15, 1.0 - 1e-15))
                prob = np.full_like(y01, p0, dtype=float)
            else:
                try:
                    bic_new, prob = _bic_and_probs_logistic_mle(
                        sub[nxt].to_numpy(dtype=float), y01
                    )
                except Exception:
                    continue
            if not np.isfinite(bic_new):
                continue
            if (bic_cur - bic_new) >= min_delta:
                trials.append((bic_new, v, nxt, prob))

        if not trials:
            break

        bic_new, v_out, active_next, prob = min(trials, key=lambda t: (t[0], t[1]))
        step += 1
        delta_bic = float(bic_new - bic_cur)
        mx_pair = _max_pairwise_abs_corr_subset(active_next, corr_df)
        try:
            auc = float(roc_auc_score(y01, prob))
        except ValueError:
            auc = float(np.nan)

        rows.append(
            {
                "step_number": int(step),
                "eliminated_feature_name": v_out,
                "max_correlation": float(mx_pair),
                "total_bic": float(bic_new),
                "delta_bic": float(delta_bic),
                "roc_auc": auc,
                "gini": _gini_binary_from_auc(auc),
            }
        )
        active = active_next
        bic_cur = float(bic_new)

    selected = [f for f in order if f in active]
    history = pd.DataFrame(
        rows,
        columns=[
            "step_number",
            "eliminated_feature_name",
            "max_correlation",
            "total_bic",
            "delta_bic",
            "roc_auc",
            "gini",
        ],
    )
    return selected, history


def select_features_bic_forward(
    df: pd.DataFrame,
    cols_feat: list[str],
    col_target: str,
    max_corr: float = 0.5,
    min_delta: float = 1e-9,
) -> tuple[list[str], pd.DataFrame]:
    """
    Forward stepwise logistic regression minimizing BIC with a correlation cap.

    Same procedure as :func:`select_features_aic_forward`, but each step
    minimizes **BIC** ``= k ln(n) - 2 log L`` on the modeling sample instead of
    AIC. A feature is added only if BIC decreases by at least ``min_delta``.
    History columns use ``total_bic``, ``delta_bic``, ``roc_auc``, and ``gini``.
    """
    if not cols_feat:
        raise ValueError("cols_feat must be a non-empty list of column names.")

    if max_corr < 0 or max_corr > 1:
        raise ValueError("max_corr must be between 0 and 1 inclusive.")

    if not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be a finite non-negative float.")

    order = list(dict.fromkeys(cols_feat))
    _validate_columns(df, "select_features_bic_forward", [col_target] + order)

    sub = df[[col_target] + order].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 5:
        raise ValueError(
            "Need at least five complete-case rows for col_target and cols_feat."
        )

    y_raw = sub[col_target].to_numpy(dtype=float)
    if len(np.unique(y_raw)) != 2:
        raise ValueError(
            "col_target must have exactly two distinct numeric values on complete cases."
        )
    lo, hi = sorted(np.unique(y_raw).tolist())[:2]
    y01 = (y_raw == hi).astype(int)

    X_all = sub[order].to_numpy(dtype=float)
    corr_df = pd.DataFrame(X_all, columns=order).corr(method="pearson")

    var_ok = {
        order[j]
        for j in range(len(order))
        if float(np.nanstd(X_all[:, j])) > 1e-12
    }

    selected: list[str] = []
    bic_old = _bic_intercept_only_logistic(y01)
    rows: list[dict[str, Any]] = []
    step = 0

    while True:
        trials: list[tuple[float, str, float, np.ndarray]] = []
        for f in order:
            if f in selected or f not in var_ok:
                continue
            if selected:
                peak = max(_abs_corr_pair(corr_df, f, s) for s in selected)
                if peak > max_corr:
                    continue
            else:
                peak = 0.0
            cols = selected + [f]
            jidx = [order.index(c) for c in cols]
            X = X_all[:, jidx]
            try:
                bic_new, prob = _bic_and_probs_logistic_mle(X, y01)
            except Exception:
                continue
            if not np.isfinite(bic_new):
                continue
            if (bic_old - bic_new) >= min_delta:
                trials.append((bic_new, f, float(peak), prob))

        if not trials:
            break

        bic_new, f_add, peak, prob = min(trials, key=lambda t: (t[0], t[1]))
        step += 1
        delta_bic = float(bic_new - bic_old)
        try:
            auc = float(roc_auc_score(y01, prob))
        except ValueError:
            auc = float(np.nan)

        rows.append(
            {
                "step_number": int(step),
                "added_feature_name": f_add,
                "max_correlation": float(peak),
                "total_bic": float(bic_new),
                "delta_bic": float(delta_bic),
                "roc_auc": auc,
                "gini": _gini_binary_from_auc(auc),
            }
        )
        selected.append(f_add)
        bic_old = float(bic_new)

    history = pd.DataFrame(
        rows,
        columns=[
            "step_number",
            "added_feature_name",
            "max_correlation",
            "total_bic",
            "delta_bic",
            "roc_auc",
            "gini",
        ],
    )
    return selected, history


def _roc_auc_and_probs_logistic_train(
    X: np.ndarray,
    y01: np.ndarray,
    *,
    random_state: int = 0,
) -> tuple[float, np.ndarray]:
    """Fit logistic (large ``C``) and return ``(train ROC-AUC, p_hat)``."""
    _ll, _k, prob = _logistic_fit_loglik_params_probs(
        X, y01, random_state=random_state
    )
    try:
        auc = float(roc_auc_score(y01, prob))
    except ValueError:
        auc = float(np.nan)
    return auc, prob


def _gini_binary_from_auc(auc: float) -> float:
    """Binary Gini from ROC-AUC: ``Gini = 2 * AUC - 1`` (clipped to ``[-1, 1]``)."""
    if auc is None or not np.isfinite(auc):
        return float(np.nan)
    return float(np.clip(2.0 * float(auc) - 1.0, -1.0, 1.0))


def select_features_auc_backward(
    df: pd.DataFrame,
    cols_feat: list[str],
    col_target: str,
    max_corr: float = 0.5,
    min_delta: float = 0.005,
) -> tuple[list[str], pd.DataFrame]:
    """
    Backward stepwise logistic regression maximizing training ROC-AUC.

    Same initial correlation screen as :func:`select_features_aic_backward`.
    Each step removes the feature whose removal yields the **highest** training
    ROC-AUC among all single removals (tie-break: lexicographically smallest
    eliminated name), provided training ROC-AUC increases by at least
    ``min_delta`` versus the current model. Stops when no removal meets that
    threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling data.
    cols_feat : list of str
        Candidate predictors (deduplicated, first occurrence wins).
    col_target : str
        Binary outcome (two distinct numeric values after coercion).
    max_corr : float, default 0.5
        Initial-set correlation bound; must lie in ``[0, 1]``.
    min_delta : float, default 1e-9
        Minimum training ROC-AUC increase for a backward step
        (``auc_after - auc_before``). Must be finite and non-negative.

    Returns
    -------
    selected_features : list of str
        Remaining predictors after backward steps, ordered as in ``cols_feat``.
    history : pandas.DataFrame
        Columns ``step_number``, ``feature_name`` (eliminated at that step),
        ``max_correlation``, ``delta_auc``, ``roc_auc``, ``gini``.
    """
    if not cols_feat:
        raise ValueError("cols_feat must be a non-empty list of column names.")

    if max_corr < 0 or max_corr > 1:
        raise ValueError("max_corr must be between 0 and 1 inclusive.")

    if not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be a finite non-negative float.")

    order = list(dict.fromkeys(cols_feat))
    _validate_columns(df, "select_features_auc_backward", [col_target] + order)

    sub = df[[col_target] + order].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 5:
        raise ValueError(
            "Need at least five complete-case rows for col_target and cols_feat."
        )

    y_raw = sub[col_target].to_numpy(dtype=float)
    if len(np.unique(y_raw)) != 2:
        raise ValueError(
            "col_target must have exactly two distinct numeric values on complete cases."
        )
    lo, hi = sorted(np.unique(y_raw).tolist())[:2]
    y01 = (y_raw == hi).astype(int)

    X_all = sub[order].to_numpy(dtype=float)
    corr_df = pd.DataFrame(X_all, columns=order).corr(method="pearson")

    var_ok = {
        order[j]
        for j in range(len(order))
        if float(np.nanstd(X_all[:, j])) > 1e-12
    }

    active: list[str] = []
    for f in order:
        if f not in var_ok:
            continue
        if not active:
            active.append(f)
            continue
        if max(_abs_corr_pair(corr_df, f, s) for s in active) <= max_corr:
            active.append(f)

    if not active:
        raise ValueError(
            "No feature could be placed in the initial model under max_corr; "
            "relax max_corr or adjust cols_feat."
        )

    try:
        auc_cur, _ = _roc_auc_and_probs_logistic_train(
            sub[active].to_numpy(dtype=float), y01
        )
    except Exception as exc:
        raise ValueError(
            "Could not fit initial logistic model on the correlation-screened set."
        ) from exc

    if not np.isfinite(auc_cur):
        raise ValueError("Initial model ROC-AUC is undefined on this sample.")

    rows: list[dict[str, Any]] = []
    step = 0

    while True:
        trials: list[tuple[float, str, list[str], np.ndarray]] = []
        for v in active:
            nxt = [x for x in active if x != v]
            if not nxt:
                _, prob = _roc_auc_and_probs_logistic_train(
                    np.empty((len(y01), 0), dtype=float), y01
                )
                try:
                    auc_new = float(roc_auc_score(y01, prob))
                except ValueError:
                    auc_new = float(np.nan)
            else:
                try:
                    auc_new, prob = _roc_auc_and_probs_logistic_train(
                        sub[nxt].to_numpy(dtype=float), y01
                    )
                except Exception:
                    continue
            if not np.isfinite(auc_new):
                continue
            if (auc_new - auc_cur) >= min_delta:
                trials.append((auc_new, v, nxt, prob))

        if not trials:
            break

        auc_new, v_out, active_next, prob = sorted(
            trials, key=lambda t: (-t[0], t[1])
        )[0]
        step += 1
        delta_auc = float(auc_new - auc_cur)
        mx_pair = _max_pairwise_abs_corr_subset(active_next, corr_df)
        gini = _gini_binary_from_auc(auc_new)

        rows.append(
            {
                "step_number": int(step),
                "feature_name": v_out,
                "max_correlation": float(mx_pair),
                "delta_auc": float(delta_auc),
                "roc_auc": float(auc_new),
                "gini": gini,
            }
        )
        active = active_next
        auc_cur = float(auc_new)

    selected = [f for f in order if f in active]
    history = pd.DataFrame(
        rows,
        columns=[
            "step_number",
            "feature_name",
            "max_correlation",
            "delta_auc",
            "roc_auc",
            "gini",
        ],
    )
    return selected, history


def select_features_auc_forward(
    df: pd.DataFrame,
    cols_feat: list[str],
    col_target: str,
    max_corr: float = 0.5,
    min_delta: float = 0.005,
) -> tuple[list[str], pd.DataFrame]:
    """
    Forward stepwise logistic regression maximizing training ROC-AUC.

    Same correlation rule as :func:`select_features_aic_forward`. Starts from
    an intercept-only probability; each step adds the eligible feature whose
    inclusion yields the **highest** training ROC-AUC, provided training AUC
    increases by at least ``min_delta`` (tie-break: lexicographically
    smallest added name). Stops when no candidate meets that threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling data.
    cols_feat : list of str
        Candidate predictors (deduplicated, first occurrence wins).
    col_target : str
        Binary outcome column (two distinct numeric values after coercion).
    max_corr : float, default 0.5
        Maximum absolute Pearson correlation to each already-selected feature.
        Must lie in ``[0, 1]``.
    min_delta : float, default 1e-9
        Minimum training ROC-AUC increase to add a feature
        (``auc_after - auc_before``). Must be finite and non-negative.

    Returns
    -------
    selected_features : list of str
        Names in the order they entered the model.
    history : pandas.DataFrame
        Columns ``step_number``, ``feature_name`` (added at that step),
        ``max_correlation``, ``delta_auc``, ``roc_auc``, ``gini``.
    """
    if not cols_feat:
        raise ValueError("cols_feat must be a non-empty list of column names.")

    if max_corr < 0 or max_corr > 1:
        raise ValueError("max_corr must be between 0 and 1 inclusive.")

    if not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be a finite non-negative float.")

    order = list(dict.fromkeys(cols_feat))
    _validate_columns(df, "select_features_auc_forward", [col_target] + order)

    sub = df[[col_target] + order].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 5:
        raise ValueError(
            "Need at least five complete-case rows for col_target and cols_feat."
        )

    y_raw = sub[col_target].to_numpy(dtype=float)
    if len(np.unique(y_raw)) != 2:
        raise ValueError(
            "col_target must have exactly two distinct numeric values on complete cases."
        )
    lo, hi = sorted(np.unique(y_raw).tolist())[:2]
    y01 = (y_raw == hi).astype(int)

    X_all = sub[order].to_numpy(dtype=float)
    corr_df = pd.DataFrame(X_all, columns=order).corr(method="pearson")

    var_ok = {
        order[j]
        for j in range(len(order))
        if float(np.nanstd(X_all[:, j])) > 1e-12
    }

    auc_old, _ = _roc_auc_and_probs_logistic_train(
        np.empty((len(y01), 0), dtype=float), y01
    )
    if not np.isfinite(auc_old):
        auc_old = 0.5

    selected: list[str] = []
    rows: list[dict[str, Any]] = []
    step = 0

    while True:
        trials: list[tuple[float, str, float, np.ndarray]] = []
        for f in order:
            if f in selected or f not in var_ok:
                continue
            if selected:
                peak = max(_abs_corr_pair(corr_df, f, s) for s in selected)
                if peak > max_corr:
                    continue
            else:
                peak = 0.0
            cols = selected + [f]
            jidx = [order.index(c) for c in cols]
            X = X_all[:, jidx]
            try:
                auc_new, prob = _roc_auc_and_probs_logistic_train(X, y01)
            except Exception:
                continue
            if not np.isfinite(auc_new):
                continue
            if (auc_new - auc_old) >= min_delta:
                trials.append((auc_new, f, float(peak), prob))

        if not trials:
            break

        auc_new, f_add, peak, prob = sorted(
            trials, key=lambda t: (-t[0], t[1])
        )[0]
        step += 1
        delta_auc = float(auc_new - auc_old)
        gini = _gini_binary_from_auc(auc_new)

        rows.append(
            {
                "step_number": int(step),
                "feature_name": f_add,
                "max_correlation": float(peak),
                "delta_auc": float(delta_auc),
                "roc_auc": float(auc_new),
                "gini": gini,
            }
        )
        selected.append(f_add)
        auc_old = float(auc_new)

    history = pd.DataFrame(
        rows,
        columns=[
            "step_number",
            "feature_name",
            "max_correlation",
            "delta_auc",
            "roc_auc",
            "gini",
        ],
    )
    return selected, history


def get_score_predictive_power_timely(
    df: pd.DataFrame,
    col_score: str,
    col_period: str,
    col_target: str,
) -> pd.DataFrame:
    """
    ROC AUC of a model score vs. binary label, by time period.

    For each distinct ``col_period`` value, rows in that slice are used to
    compute ``sklearn.metrics.roc_auc_score`` with ``col_score`` as the
    ranking score and ``col_target`` as the binary outcome. This summarizes
    how well the **same score column** separates the classes within each
    period (e.g. for monitoring discrimination over time).

    Parameters
    ----------
    df : pd.DataFrame
        Rows containing model score, time period, and label.
    col_score : str
        Name of the numeric score column (e.g. probability, logit, or risk
        score). Coerced with ``pd.to_numeric(..., errors="coerce")``; rows
        with missing score or label are dropped for that period's AUC.
    col_period : str
        Period column (daily, monthly, etc.). All non-null unique values are
        used, sorted ascending.
    col_target : str
        Binary label (two distinct values after numeric coercion; typically
        ``0``/``1``).

    Returns
    -------
    pd.DataFrame
        One row per period. Columns: ``time_period`` (value from ``col_period``),
        ``aucroc``, ``gini`` (``2 * aucroc - 1`` clipped to ``[-1, 1]``, or
        ``NaN`` when AUC is undefined), ``count`` (complete-case row count for
        that slice's AUC), ``count_positive`` (count of the numerically higher
        label when exactly two classes exist among complete cases; else
        ``NaN``), ``positive_rate`` (``count_positive / count``).

    Raises
    ------
    ValueError
        If ``col_score``, ``col_period``, or ``col_target`` is not in ``df``.

    Notes
    -----
    - Same semantics as :func:`get_feature_predictive_power_timely`, but for
      one model output column; see that function's Notes on AUC ``< 0.5``.
    - **Agent / MCP:** Stable column set aligned with
      :func:`get_score_predictive_power_data_type`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> n = 300
    >>> s = rng.normal(size=n)
    >>> df = pd.DataFrame({
    ...     "t": np.repeat([202501, 202502], n // 2),
    ...     "score": s,
    ...     "y": (rng.normal(size=n) + s * 0.6) > 0,
    ... })
    >>> out = get_score_predictive_power_timely(df, "score", "t", "y")
    >>> list(out.columns)
    ['time_period', 'aucroc', 'gini', 'count', 'count_positive', 'positive_rate']
    """
    _validate_columns(df, "df", [col_score, col_period, col_target])

    periods = sorted(df[col_period].dropna().unique().tolist())
    rows: list[dict[str, Any]] = []
    for t in periods:
        sub = df.loc[df[col_period] == t]
        if len(sub) == 0:
            continue
        y = pd.to_numeric(sub[col_target], errors="coerce")
        x = pd.to_numeric(sub[col_score], errors="coerce")
        valid = y.notna() & x.notna()
        yv = y.loc[valid]
        n = int(len(yv))
        auc = _roc_auc_binary_feature(sub[col_target], sub[col_score])
        gini = _gini_binary_from_auc(auc)
        if n > 0 and yv.nunique() == 2:
            hi = float(sorted(yv.unique().tolist())[-1])
            count_positive = int((yv == hi).sum())
            positive_rate = float(count_positive / n)
        else:
            count_positive = float(np.nan)
            positive_rate = float(np.nan)
        rows.append(
            {
                "time_period": t,
                "aucroc": auc,
                "gini": gini,
                "count": n,
                "count_positive": count_positive,
                "positive_rate": positive_rate,
            }
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "time_period",
            "aucroc",
            "gini",
            "count",
            "count_positive",
            "positive_rate",
        ],
    )
    return out


def get_score_predictive_power_data_type(
    df: pd.DataFrame,
    col_score: str,
    col_type: str,
    col_target: str,
) -> pd.DataFrame:
    """
    ROC AUC of a model score vs. binary label, by data split / cohort label.

    For each distinct non-null value in ``col_type`` (e.g. ``"train"``,
    ``"valid"``, ``"test"``, ``"oot"`` for out-of-time, ``"hold"`` / holdout),
    rows in that slice are used to compute ``sklearn.metrics.roc_auc_score``
    with ``col_score`` as the ranking score and ``col_target`` as the binary
    outcome. Use this to compare score discrimination across modeling splits
    rather than calendar periods.

    Parameters
    ----------
    df : pd.DataFrame
        Rows containing model score, split label, and label.
    col_score : str
        Numeric model score column. Coerced with ``pd.to_numeric(...,
        errors="coerce")``; rows with missing score or label are dropped for
        that slice's AUC.
    col_type : str
        String column naming the data subset (e.g. train / valid / test).
        All non-null unique values are used, sorted ascending by string
        representation for deterministic row order.
    col_target : str
        Binary label (two distinct values after numeric coercion; typically
        ``0``/``1``).

    Returns
    -------
    pd.DataFrame
        One row per distinct ``col_type`` value (sorted). Columns:

        - ``time_period``: cohort key (split string from ``col_type``), same
          name as :func:`get_score_predictive_power_timely` for tooling.
        - ``aucroc``: ROC AUC on rows with non-null score and label after
          numeric coercion (``NaN`` when undefined).
        - ``gini``: ``2 * aucroc - 1`` clipped to ``[-1, 1]``, or ``NaN`` when
          AUC is undefined.
        - ``count``: number of complete-case rows used for that slice's AUC.
        - ``count_positive``: count of the numerically **higher** label class
          among those rows (only when exactly two distinct labels exist after
          coercion; otherwise ``NaN``).
        - ``positive_rate``: ``count_positive / count`` (``NaN`` when
          ``count_positive`` is undefined or ``count`` is 0).

    Raises
    ------
    ValueError
        If ``col_score``, ``col_type``, or ``col_target`` is not in ``df``.

    Notes
    -----
    - Same AUC semantics as :func:`get_score_predictive_power_timely`; see
      that function's Notes on AUC ``< 0.5``.
    - **Agent / MCP:** Stable column set including ``time_period``, ``aucroc``,
      ``gini``, ``count``, ``count_positive``, ``positive_rate``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(2)
    >>> n = 240
    >>> s = rng.normal(size=n)
    >>> kinds = np.array(["train"] * 80 + ["valid"] * 80 + ["test"] * 80)
    >>> df = pd.DataFrame({
    ...     "split": kinds,
    ...     "score": s,
    ...     "y": (rng.normal(size=n) + s * 0.5) > 0,
    ... })
    >>> out = get_score_predictive_power_data_type(df, "score", "split", "y")
    >>> list(out.columns)
    ['time_period', 'aucroc', 'gini', 'count', 'count_positive', 'positive_rate']
    >>> len(out)
    3
    """
    _validate_columns(df, "df", [col_score, col_type, col_target])

    splits = sorted(df[col_type].dropna().astype(str).unique().tolist())
    rows: list[dict[str, Any]] = []
    for label in splits:
        sub = df.loc[df[col_type].astype(str) == label]
        if len(sub) == 0:
            continue
        y = pd.to_numeric(sub[col_target], errors="coerce")
        x = pd.to_numeric(sub[col_score], errors="coerce")
        valid = y.notna() & x.notna()
        yv = y.loc[valid]
        n = int(len(yv))
        auc = _roc_auc_binary_feature(sub[col_target], sub[col_score])
        gini = _gini_binary_from_auc(auc)
        if n > 0 and yv.nunique() == 2:
            hi = float(sorted(yv.unique().tolist())[-1])
            count_positive = int((yv == hi).sum())
            positive_rate = float(count_positive / n)
        else:
            count_positive = float(np.nan)
            positive_rate = float(np.nan)
        rows.append(
            {
                "time_period": label,
                "aucroc": auc,
                "gini": gini,
                "count": n,
                "count_positive": count_positive,
                "positive_rate": positive_rate,
            }
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "time_period",
            "aucroc",
            "gini",
            "count",
            "count_positive",
            "positive_rate",
        ],
    )
    return out


def get_score_predictive_power_data_type_bootstrap(
    df: pd.DataFrame,
    col_score: str,
    col_type: str,
    col_target: str,
    *,
    n_data: int = 100,
    n_iteration: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap distribution of score AUC / Gini by ``col_type`` stratum.

    For each distinct non-null ``col_type`` value (same strata as
    :func:`get_score_predictive_power_data_type`), complete cases (non-null
    numeric target and score) form the resampling pool. Each of ``n_iteration``
    iterations draws ``n_data`` rows **with replacement** from that pool,
    computes ROC AUC and Gini (``2 * AUC - 1``, clipped), and the     positive rate
    of the numerically higher label (defined on the **full** pool when it has
    exactly two classes). ``mean_positive_rate`` is the mean of that rate over
    **all** iterations. AUC/Gini percentiles and means use only iterations where
    the bootstrap draw contains both classes so AUC is defined.

    Parameters
    ----------
    df : pd.DataFrame
        Rows with score, split label, and binary label.
    col_score : str
        Score column; coerced with ``pd.to_numeric(..., errors="coerce")``.
    col_type : str
        Split / cohort label column; unique non-null string values, sorted.
    col_target : str
        Binary label (two distinct values after coercion; typically ``0``/``1``).
    n_data : int, default 100
        Bootstrap sample size (drawn with replacement per iteration).
    n_iteration : int, default 1000
        Number of bootstrap draws per stratum.
    random_seed : int, default 42
        Seed for :class:`numpy.random.Generator`.

    Returns
    -------
    pd.DataFrame
        One row per ``col_type`` stratum. Columns: ``time_period`` (stratum
        key), ``CI_5_gini``, ``mean_gini``, ``CI_95_gini``, ``CI_5_aucroc``,
        ``mean_auc``, ``CI_95_aucroc``, ``mean_positive_rate``. CI columns use
        the 5th and 95th percentiles of the bootstrap AUC/Gini distributions.
        ``mean_auc`` / ``mean_gini`` use only finite AUC/Gini bootstrap values;
        if none, they are ``NaN``. ``mean_positive_rate`` averages the positive
        rate (fraction with the numerically higher full-pool label) over every
        iteration.

    Raises
    ------
    ValueError
        If required columns are missing, ``n_data < 2``, or ``n_iteration < 1``.
    """
    _validate_columns(df, "df", [col_score, col_type, col_target])
    if int(n_data) < 2:
        raise ValueError("n_data must be at least 2.")
    if int(n_iteration) < 1:
        raise ValueError("n_iteration must be at least 1.")

    def _pct5_95_mean(vals: list[float]) -> tuple[float, float, float]:
        if not vals:
            return float(np.nan), float(np.nan), float(np.nan)
        a = np.asarray(vals, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float(np.nan), float(np.nan), float(np.nan)
        return (
            float(np.percentile(a, 5.0)),
            float(np.mean(a)),
            float(np.percentile(a, 95.0)),
        )

    rng = np.random.default_rng(int(random_seed))
    splits = sorted(df[col_type].dropna().astype(str).unique().tolist())
    rows: list[dict[str, Any]] = []

    for label in splits:
        sub = df.loc[df[col_type].astype(str) == label]
        if len(sub) == 0:
            continue
        y = pd.to_numeric(sub[col_target], errors="coerce")
        x = pd.to_numeric(sub[col_score], errors="coerce")
        valid = y.notna() & x.notna()
        yv = y.loc[valid]
        y_arr = yv.to_numpy(dtype=float)
        x_arr = x.loc[valid].to_numpy(dtype=float)
        n_pool = int(len(y_arr))
        if n_pool < 2:
            rows.append(
                {
                    "time_period": label,
                    "CI_5_gini": float(np.nan),
                    "mean_gini": float(np.nan),
                    "CI_95_gini": float(np.nan),
                    "CI_5_aucroc": float(np.nan),
                    "mean_auc": float(np.nan),
                    "CI_95_aucroc": float(np.nan),
                    "mean_positive_rate": float(np.nan),
                }
            )
            continue

        uniq = np.unique(y_arr)
        if len(uniq) != 2:
            rows.append(
                {
                    "time_period": label,
                    "CI_5_gini": float(np.nan),
                    "mean_gini": float(np.nan),
                    "CI_95_gini": float(np.nan),
                    "CI_5_aucroc": float(np.nan),
                    "mean_auc": float(np.nan),
                    "CI_95_aucroc": float(np.nan),
                    "mean_positive_rate": float(np.nan),
                }
            )
            continue

        hi = float(sorted(uniq.tolist())[-1])
        nd = int(n_data)
        ni = int(n_iteration)

        aucs: list[float] = []
        ginis: list[float] = []
        pos_rates: list[float] = []

        for _ in range(ni):
            idx = rng.integers(0, n_pool, size=nd, endpoint=False)
            y_b = y_arr[idx]
            x_b = x_arr[idx]
            pos_rates.append(float(np.mean(y_b == hi)))
            if len(np.unique(y_b)) < 2:
                continue
            try:
                auc_b = float(roc_auc_score(y_b, x_b))
            except ValueError:
                continue
            ginis.append(float(_gini_binary_from_auc(auc_b)))
            aucs.append(auc_b)

        ci5_g, mean_g, ci95_g = _pct5_95_mean(ginis)
        ci5_a, mean_a, ci95_a = _pct5_95_mean(aucs)
        mean_pr = float(np.mean(pos_rates))

        rows.append(
            {
                "time_period": label,
                "CI_5_gini": ci5_g,
                "mean_gini": mean_g,
                "CI_95_gini": ci95_g,
                "CI_5_aucroc": ci5_a,
                "mean_auc": mean_a,
                "CI_95_aucroc": ci95_a,
                "mean_positive_rate": mean_pr,
            }
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "time_period",
            "CI_5_gini",
            "mean_gini",
            "CI_95_gini",
            "CI_5_aucroc",
            "mean_auc",
            "CI_95_aucroc",
            "mean_positive_rate",
        ],
    )
    return out


def compare_score_predictive_power_data_type_bootstrap(
    df: pd.DataFrame,
    col_score_champion: str,
    col_score_challenger: str,
    col_type: str,
    col_target: str,
    n_data: int = 100,
    n_iteration: int = 1000,
    *,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap AUC and Gini for champion vs. challenger score by ``col_type``.

    For each distinct non-null ``col_type`` stratum (same keys as
    :func:`get_score_predictive_power_data_type`), complete cases with non-null
    numeric ``col_target``, ``col_score_champion``, and ``col_score_challenger``
    form the resampling pool for that stratum. Each iteration draws ``n_data``
    rows with replacement and evaluates both scores on the same ``y`` draw.
    ``mean_positive_rate`` averages the positive-class fraction over all
    iterations; AUC/Gini summaries use only iterations where the bootstrap ``y``
    has two classes.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    col_score_champion : str
        Champion score column (numeric coercion).
    col_score_challenger : str
        Challenger score column (numeric coercion).
    col_type : str
        Split / cohort label; unique non-null string values, sorted ascending.
    col_target : str
        Binary label (exactly two distinct values after coercion on each pool).
    n_data : int, default 100
        Bootstrap sample size per iteration.
    n_iteration : int, default 1000
        Number of bootstrap iterations per stratum.
    random_seed : int, default 42
        Base seed; each stratum uses a deterministic offset for its RNG.

    Returns
    -------
    pd.DataFrame
        One row per ``col_type`` value. Columns: ``time_period`` (stratum key),
        then champion/challenger CI and mean columns for Gini and AUC, and
        ``mean_positive_rate``.

    Raises
    ------
    ValueError
        If columns are missing, ``n_data < 2``, or ``n_iteration < 1``.
    """
    _validate_columns(
        df,
        "df",
        [col_score_champion, col_score_challenger, col_type, col_target],
    )
    if int(n_data) < 2:
        raise ValueError("n_data must be at least 2.")
    if int(n_iteration) < 1:
        raise ValueError("n_iteration must be at least 1.")

    def _pct5_95_mean(vals: list[float]) -> tuple[float, float, float]:
        if not vals:
            return float(np.nan), float(np.nan), float(np.nan)
        a = np.asarray(vals, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float(np.nan), float(np.nan), float(np.nan)
        return (
            float(np.percentile(a, 5.0)),
            float(np.mean(a)),
            float(np.percentile(a, 95.0)),
        )

    base_seed = int(random_seed)
    splits = sorted(df[col_type].dropna().astype(str).unique().tolist())
    rows: list[dict[str, Any]] = []

    for si, label in enumerate(splits):
        sub = df.loc[df[col_type].astype(str) == label]
        if len(sub) == 0:
            continue
        y = pd.to_numeric(sub[col_target], errors="coerce")
        xc = pd.to_numeric(sub[col_score_champion], errors="coerce")
        xh = pd.to_numeric(sub[col_score_challenger], errors="coerce")
        valid = y.notna() & xc.notna() & xh.notna()
        y_arr = y.loc[valid].to_numpy(dtype=float)
        champ_arr = xc.loc[valid].to_numpy(dtype=float)
        chall_arr = xh.loc[valid].to_numpy(dtype=float)
        n_pool = int(len(y_arr))

        nan_tail = {
            "champion_CI_5_gini": float(np.nan),
            "champion_mean_gini": float(np.nan),
            "champion_CI_95_gini": float(np.nan),
            "challenger_CI_5_gini": float(np.nan),
            "challenger_mean_gini": float(np.nan),
            "challenger_CI_95_gini": float(np.nan),
            "champion_CI_5_aucroc": float(np.nan),
            "champion_mean_auc": float(np.nan),
            "champion_CI_95_aucroc": float(np.nan),
            "challenger_CI_5_aucroc": float(np.nan),
            "challenger_mean_auc": float(np.nan),
            "challenger_CI_95_aucroc": float(np.nan),
            "mean_positive_rate": float(np.nan),
        }

        if n_pool < 2:
            rows.append({"time_period": label, **nan_tail})
            continue

        uniq = np.unique(y_arr)
        if len(uniq) != 2:
            rows.append({"time_period": label, **nan_tail})
            continue

        hi = float(sorted(uniq.tolist())[-1])
        nd = int(n_data)
        ni = int(n_iteration)
        rng = np.random.default_rng(base_seed + si * 1_000_003)

        auc_champ: list[float] = []
        gini_champ: list[float] = []
        auc_chall: list[float] = []
        gini_chall: list[float] = []
        pos_rates: list[float] = []

        for _ in range(ni):
            idx = rng.integers(0, n_pool, size=nd, endpoint=False)
            y_b = y_arr[idx]
            c_b = champ_arr[idx]
            h_b = chall_arr[idx]
            pos_rates.append(float(np.mean(y_b == hi)))
            if len(np.unique(y_b)) < 2:
                continue
            try:
                auc_c = float(roc_auc_score(y_b, c_b))
                auc_h = float(roc_auc_score(y_b, h_b))
            except ValueError:
                continue
            auc_champ.append(auc_c)
            gini_champ.append(float(_gini_binary_from_auc(auc_c)))
            auc_chall.append(auc_h)
            gini_chall.append(float(_gini_binary_from_auc(auc_h)))

        c5g, cmg, c95g = _pct5_95_mean(gini_champ)
        h5g, hmg, h95g = _pct5_95_mean(gini_chall)
        c5a, cma, c95a = _pct5_95_mean(auc_champ)
        h5a, hma, h95a = _pct5_95_mean(auc_chall)
        mean_pr = float(np.mean(pos_rates)) if pos_rates else float(np.nan)

        rows.append(
            {
                "time_period": label,
                "champion_CI_5_gini": c5g,
                "champion_mean_gini": cmg,
                "champion_CI_95_gini": c95g,
                "challenger_CI_5_gini": h5g,
                "challenger_mean_gini": hmg,
                "challenger_CI_95_gini": h95g,
                "champion_CI_5_aucroc": c5a,
                "champion_mean_auc": cma,
                "champion_CI_95_aucroc": c95a,
                "challenger_CI_5_aucroc": h5a,
                "challenger_mean_auc": hma,
                "challenger_CI_95_aucroc": h95a,
                "mean_positive_rate": mean_pr,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "time_period",
            "champion_CI_5_gini",
            "champion_mean_gini",
            "champion_CI_95_gini",
            "challenger_CI_5_gini",
            "challenger_mean_gini",
            "challenger_CI_95_gini",
            "champion_CI_5_aucroc",
            "champion_mean_auc",
            "champion_CI_95_aucroc",
            "challenger_CI_5_aucroc",
            "challenger_mean_auc",
            "challenger_CI_95_aucroc",
            "mean_positive_rate",
        ],
    )


def get_feature_dtype(
    df: pd.DataFrame,
    cols_feat: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Classify feature columns into numeric, categorical, and temporal dtypes.

    Uses pandas dtype checks on ``df[cols_feat]``. Intended as a lightweight
    profiling step before modeling or encoding (e.g. for an AI agent choosing
    imputers, scalers, or one-hot targets).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the listed columns.
    cols_feat : list of str
        Feature column names to classify. Each name must exist in ``df`` when
        the list is non-empty.

    Returns
    -------
    cols_feat_num : list of str
        Columns whose dtype is numeric (integer, float, nullable ``Int64``,
        etc.). Boolean columns are treated as numeric per
        ``pandas.api.types.is_numeric_dtype``.
    cols_feat_cat : list of str
        Columns that are neither classified as temporal nor numeric (e.g.
        ``object``, pandas ``string``, ``category``).
    cols_feat_time : list of str
        Columns with ``datetime64`` (timezone-aware or naive) or
        ``timedelta64`` dtypes. **Note:** dates stored as ``object`` (e.g.
        Python ``datetime.date``) are **not** detected here unless converted to
        ``datetime64`` first.

    Returns are three lists in the order above. A column appears in exactly
    one list. Order within each list follows the order of ``cols_feat``.

    Raises
    ------
    ValueError
        If any name in ``cols_feat`` is not a column of ``df`` (only checked
        when ``cols_feat`` is non-empty).

    Notes
    -----
    - **Agent / MCP:** stable return shape ``(list, list, list)``; callers
      can map to JSON as three array fields.
    - For ambiguous ``object`` columns, the result is categorical unless you
      cast to datetime or numeric beforehand.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "n": [1, 2],
    ...     "s": ["a", "b"],
    ...     "d": pd.to_datetime(["2020-01-01", "2020-01-02"]),
    ... })
    >>> num, cat, time = get_feature_dtype(df, ["n", "s", "d"])
    >>> num, cat, time
    (['n'], ['s'], ['d'])
    >>> get_feature_dtype(df, [])
    ([], [], [])
    """
    if not cols_feat:
        return [], [], []

    feat_cols = list(cols_feat)
    _validate_columns(df, "df", feat_cols)

    cols_feat_num: list[str] = []
    cols_feat_cat: list[str] = []
    cols_feat_time: list[str] = []

    for col in feat_cols:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(
            series
        ):
            cols_feat_time.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            cols_feat_num.append(col)
        else:
            cols_feat_cat.append(col)

    return cols_feat_num, cols_feat_cat, cols_feat_time


def get_optimal_bin(
    df: pd.DataFrame,
    cols_feat: list[str],
    col_target: str,
    min_nbin: int = 2,
    max_nbin: int = 6,
    cols_feat_cat: list[str] | None = None,
) -> tuple[dict[str, Any], Any, pd.DataFrame, list[str]]:
    """
    Fit supervised optimal binning (optbinning) for multiple features.

    Wraps ``optbinning.BinningProcess`` to produce bin cut points / category
    groups, the fitted process object (for transforms or persistence), and a
    summary table (IV, Gini, etc.). Requires the optional dependency
    ``optbinning`` (``pip install optbinning``).

    Only feature names that exist in ``df`` are fit; any problem with requested
    feature configuration is listed in the fourth return value (see Returns).

    ``col_target`` is required (supervised binning); it is placed after
    ``cols_feat`` so call order stays ``(df, features, target, …)``.

    Parameters
    ----------
    df : pd.DataFrame
        Training-like frame containing feature columns and the target. Missing
        ``cols_feat`` / ``cols_feat_cat`` entries are reported in
        ``feature_issues`` and omitted from the fit.
    cols_feat : list of str
        Feature columns to bin jointly in one ``BinningProcess``.
    col_target : str
        Target column in ``df`` (binary, continuous, or multiclass per
        optbinning). Rows with missing target are dropped before fitting.
    min_nbin : int, default 2
        Minimum bins per variable (passed as ``min_n_bins``).
    max_nbin : int, default 6
        Maximum bins per variable (passed as ``max_n_bins``).
    cols_feat_cat : list of str, optional
        Names to treat as **categorical** (nominal) in optbinning (including
        numeric codes stored as numbers that should be treated as categorical).
        Entries not listed in ``cols_feat`` or missing from ``df`` are recorded
        in ``feature_issues`` and ignored for the categorical override. Pass
        ``None`` or ``[]`` if none.

    Returns
    -------
    dict_bin : dict[str, Any]
        Maps each variable name in the binning summary to serialized split
        information: list of float cut points for typical numeric features, or
        a list of category lists for nominal variables (see
        :func:`_serialize_optbinning_splits`). Empty list means no internal
        splits (e.g. near-constant feature).
    model : BinningProcess
        Fitted ``optbinning.BinningProcess`` instance (not typed at import time
        to keep ``optbinning`` optional until this function runs).
    binning_stats : pandas.DataFrame
        Copy of ``BinningProcess.summary()`` (columns typically include
        ``name``, ``dtype``, ``status``, ``selected``, ``n_bins``, ``iv``,
        ``js``, ``gini``, ``quality_score``). Empty when no usable feature
        columns remain after validation.
    feature_issues : list of str
        Deduplicated names, in order of discovery: entries from ``cols_feat``
        missing from ``df``; then entries from ``cols_feat_cat`` not listed in
        ``cols_feat``; then entries from ``cols_feat_cat`` that are in
        ``cols_feat`` but missing from ``df``. Empty when there are no such
        problems.

    Raises
    ------
    ImportError
        If ``optbinning`` is not installed.
    ValueError
        For invalid arguments, empty ``cols_feat``, invalid bin counts,
        insufficient non-null target rows, or ``col_target`` missing from
        ``df``.

    Notes
    -----
    - **Agent / MCP:** return is a 4-tuple; JSON serializers may need to handle
      ``dict_bin`` and ``feature_issues`` only and pickle ``model`` separately.
    - Uses ``fit(..., check_input=False)`` like common sklearn-style examples;
      ensure ``df`` is already cleaned if strict validation is required.

    Examples
    --------
    Illustrative (requires ``pip install optbinning``)::

        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(0)
        n = 400
        d = pd.DataFrame(
            {
                "x1": rng.normal(size=n),
                "x2": rng.choice(["p", "q"], size=n),
                "y": (rng.random(n) > 0.65).astype(int),
            }
        )
        d_bin, proc, stats, issues = get_optimal_bin(d, ["x1", "x2"], "y", 2, 5, ["x2"])
    """
    try:
        from optbinning import BinningProcess
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "get_optimal_bin requires the 'optbinning' package. "
            "Install with: pip install optbinning"
        ) from exc

    if not cols_feat:
        raise ValueError("cols_feat must be a non-empty list of column names.")

    if min_nbin < 1 or max_nbin < min_nbin:
        raise ValueError("Require 1 <= min_nbin <= max_nbin.")

    feat_order = list(dict.fromkeys(cols_feat))
    cat_order = list(dict.fromkeys(cols_feat_cat)) if cols_feat_cat else []
    feat_set = set(feat_order)

    feature_issues: list[str] = []
    _issue_seen: set[str] = set()

    def _push_issue(n: str) -> None:
        if n not in _issue_seen:
            _issue_seen.add(n)
            feature_issues.append(n)

    for c in feat_order:
        if c not in df.columns:
            _push_issue(c)
    for c in cat_order:
        if c not in feat_set:
            _push_issue(c)
        elif c not in df.columns:
            _push_issue(c)

    valid_feats = [c for c in feat_order if c in df.columns]
    cat_for_bp = [c for c in cat_order if c in feat_set and c in df.columns]

    _validate_columns(df, "df", [col_target])

    if not valid_feats:
        return {}, None, pd.DataFrame(), feature_issues

    mask = df[col_target].notna()
    if int(mask.sum()) < 2:
        raise ValueError("Need at least two rows with non-null col_target to fit binning.")

    X = df.loc[mask, valid_feats]
    y = df.loc[mask, col_target]

    bp = BinningProcess(
        variable_names=valid_feats,
        min_n_bins=min_nbin,
        max_n_bins=max_nbin,
        categorical_variables=cat_for_bp if cat_for_bp else None,
    )
    bp.fit(X, y, check_input=False)

    binning_stats = bp.summary().copy()
    binning_stats['gini_power'] = binning_stats['gini']
    binning_stats.loc[binning_stats['gini'] < 0.5, 'gini_power'] = 1-binning_stats['gini']
    dict_bin: dict[str, Any] = {}
    for name in binning_stats["name"].astype(str).tolist():
        optb = bp.get_binned_variable(name)
        dict_bin[str(name)] = _serialize_optbinning_splits(optb)

    return dict_bin, bp, binning_stats, feature_issues


def _dict_bin_splits_well_formed(splits: Any, is_categorical: bool) -> bool:
    """Return True if ``dict_bin`` entry can be passed to optbinning as user splits."""
    if splits is None:
        return False
    if not isinstance(splits, (list, tuple)):
        return False
    if is_categorical:
        if len(splits) == 0:
            return False

        def _is_bin_piece(b: Any) -> bool:
            if isinstance(b, (str, bytes)):
                return False
            if isinstance(b, (list, tuple, np.ndarray)):
                return True
            return hasattr(b, "__iter__") and hasattr(b, "__len__")

        return all(_is_bin_piece(b) for b in splits)
    for z in splits:
        if isinstance(z, (bool, np.bool_)):
            return False
        if not isinstance(z, (int, float, np.integer, np.floating)):
            return False
        if isinstance(z, float) and np.isnan(float(z)):
            return False
    return True


def _optimal_binning_class_for_target_type(target_kind: str) -> Any:
    try:
        from optbinning import (
            ContinuousOptimalBinning,
            MulticlassOptimalBinning,
            OptimalBinning,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "modify_optimal_bin requires the 'optbinning' package. "
            "Install with: pip install optbinning"
        ) from exc
    if target_kind == "binary":
        return OptimalBinning
    if target_kind == "continuous":
        return ContinuousOptimalBinning
    if target_kind == "multiclass":
        return MulticlassOptimalBinning
    raise ValueError(
        f"Target type {target_kind!r} is not supported for modify_optimal_bin "
        '(expected "binary", "continuous", or "multiclass").'
    )


def _coerce_categorical_user_splits(
    splits: list[Any] | tuple[Any, ...],
) -> list[list[Any]]:
    """Normalize ``dict_bin`` categorical splits to list-of-lists for optbinning."""
    out: list[list[Any]] = []
    for b in splits:
        if isinstance(b, np.ndarray):
            out.append(b.tolist())
        elif isinstance(b, (list, tuple)):
            out.append(list(b))
        else:
            out.append(np.asarray(b).tolist())
    return out


def _optb_from_dict_bin_splits(
    name: str,
    dtype: str,
    splits: list[Any] | tuple[Any, ...],
    OptB: Any,
) -> Any:
    """Construct an unfitted optbinning object using serialized ``dict_bin`` splits."""
    if dtype == "categorical":
        user_splits = _coerce_categorical_user_splits(splits)
        return OptB(name=name, dtype="categorical", user_splits=user_splits)
    arr = np.asarray(splits, dtype=float)
    if arr.size == 0:
        return OptB(name=name, dtype="numerical")
    fixed = [True] * int(arr.shape[0])
    return OptB(
        name=name,
        dtype="numerical",
        user_splits=arr,
        user_splits_fixed=fixed,
    )


def modify_optimal_bin(
    df: pd.DataFrame,
    dict_bin: dict[str, Any],
    cols_feat: list[str],
    col_target: str,
    cols_feat_cat: list[str] | None = None,
) -> tuple[Any, pd.DataFrame, list[str]]:
    """
    Rebuild a ``BinningProcess`` from ``df`` and serialized splits in ``dict_bin``.

    For each usable feature, fits an ``OptimalBinning``-family object with
    ``user_splits`` taken from ``dict_bin`` (same format as produced by
    :func:`get_optimal_bin`), then assembles a process via
    ``BinningProcess.fit_from_dict``. Requires a target column for supervised
    statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Data used to refit binning (typically train or another sample).
    dict_bin : dict[str, Any]
        Maps feature names to serialized splits (numeric: list of cut points;
        categorical: list of category lists per bin), as from
        :func:`get_optimal_bin`.
    cols_feat : list of str
        Feature names to include (deduplicated, order preserved).
    col_target : str
        Target column in ``df`` (supervised refit).
    cols_feat_cat : list of str, optional
        Subset of ``cols_feat`` to treat as categorical. Entries not in
        ``cols_feat`` or missing from ``df`` are recorded in the issues list.

    Returns
    -------
    bp : BinningProcess or None
        Fitted process, or ``None`` if no variable could be fit.
    binning_stats : pandas.DataFrame
        ``bp.summary()`` copy with ``gini_power`` added (same convention as
        :func:`get_optimal_bin`). Empty if ``bp`` is ``None``.
    feature_issues : list of str
        Deduplicated problem feature names: missing from ``df``, absent from
        ``dict_bin``, malformed ``dict_bin`` entry, listed as categorical but
        not in ``cols_feat``, categorical-only problems in ``cols_feat_cat``,
        incompatible with multiclass target, or fit-time failure.

    Raises
    ------
    ImportError
        If ``optbinning`` is not installed.
    TypeError
        If ``dict_bin`` is not a ``dict``.
    ValueError
        If ``cols_feat`` is empty, ``col_target`` is empty, ``df`` lacks
        ``col_target``, or the target column has fewer than two non-null rows.
    """
    try:
        from optbinning import BinningProcess
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "modify_optimal_bin requires the 'optbinning' package. "
            "Install with: pip install optbinning"
        ) from exc

    if not isinstance(dict_bin, dict):
        raise TypeError("dict_bin must be a dict.")

    if not cols_feat:
        raise ValueError("cols_feat must be a non-empty list of column names.")

    if not str(col_target).strip():
        raise ValueError("col_target must be a non-empty column name.")

    _validate_columns(df, "modify_optimal_bin", [col_target])

    feat_order = list(dict.fromkeys(cols_feat))
    cat_order = list(dict.fromkeys(cols_feat_cat)) if cols_feat_cat else []
    feat_set = set(feat_order)
    db = {str(k): v for k, v in dict_bin.items()}

    feature_issues: list[str] = []
    _issue_seen: set[str] = set()

    def _push_issue(n: str) -> None:
        if n not in _issue_seen:
            _issue_seen.add(n)
            feature_issues.append(n)

    for c in feat_order:
        if c not in df.columns:
            _push_issue(c)
    for c in cat_order:
        if c not in feat_set:
            _push_issue(c)
        elif c not in df.columns:
            _push_issue(c)

    cat_names = {c for c in cat_order if c in feat_set and c in df.columns}

    for name in feat_order:
        if name not in df.columns:
            continue
        if name not in db:
            _push_issue(name)
            continue
        is_cat = name in cat_names
        if not _dict_bin_splits_well_formed(db[name], is_cat):
            _push_issue(name)

    mask = df[col_target].notna()
    if int(mask.sum()) < 2:
        raise ValueError(
            "Need at least two rows with non-null col_target to refit binning."
        )

    y_valid = df.loc[mask, col_target]
    from sklearn.utils.multiclass import type_of_target

    target_kind = type_of_target(y_valid)
    if target_kind not in ("binary", "continuous", "multiclass"):
        raise ValueError(
            f"col_target dtype is not supported for binning: {target_kind!r}."
        )

    OptB = _optimal_binning_class_for_target_type(target_kind)

    fit_names: list[str] = []
    for name in feat_order:
        if name not in df.columns or name not in db:
            continue
        is_cat = name in cat_names
        if not _dict_bin_splits_well_formed(db[name], is_cat):
            continue
        if target_kind == "multiclass" and is_cat:
            _push_issue(name)
            continue
        fit_names.append(name)

    if not fit_names:
        return None, pd.DataFrame(), feature_issues

    dict_optb: dict[str, Any] = {}
    for name in fit_names:
        is_cat = name in cat_names
        dtype = "categorical" if is_cat else "numerical"
        try:
            optb = _optb_from_dict_bin_splits(name, dtype, db[name], OptB)
            optb.fit(
                df.loc[mask, name].values,
                y_valid.values,
                check_input=False,
            )
            dict_optb[name] = optb
        except Exception:
            _push_issue(name)

    if not dict_optb:
        return None, pd.DataFrame(), feature_issues

    cat_for_bp = [c for c in cat_order if c in dict_optb]
    bp = BinningProcess(
        variable_names=list(dict_optb.keys()),
        categorical_variables=cat_for_bp if cat_for_bp else None,
    )
    bp.fit_from_dict(dict_optb)

    binning_stats = bp.summary().copy()
    binning_stats["gini_power"] = binning_stats["gini"]
    binning_stats.loc[binning_stats["gini"] < 0.5, "gini_power"] = (
        1 - binning_stats["gini"]
    )

    return bp, binning_stats, feature_issues


def get_binning_tables_from_bp(
    optb: Any, cols_feat: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Concatenate optbinning binning tables for selected features from a process.

    For each name in ``cols_feat`` (deduplicated, order preserved), loads
    ``optb.get_binned_variable(name).binning_table.build()`` and row-binds the
    tables after inserting a ``Feature Name`` column as the first column.

    Parameters
    ----------
    optb : BinningProcess
        Fitted ``optbinning.BinningProcess`` (e.g. from :func:`get_optimal_bin`).
    cols_feat : list of str
        Feature names to include (deduplicated, first occurrence wins). Names
        not in ``optb.variable_names`` are skipped; those names appear in the
        second return value.

    Returns
    -------
    binning_tables : pandas.DataFrame
        Columns are ``Feature Name`` followed by the union of all binning table
        columns (e.g. ``Bin``, ``Count``, ``WoE`` for binary targets; schemas can
        differ for multiclass, in which case :func:`pandas.concat` aligns with
        missing values where a column does not apply). Empty when ``cols_feat``
        is empty or when no requested name is recognized.
    unrecognized : list of str
        Names from ``cols_feat`` (after deduplication) that are not in
        ``optb.variable_names``. Empty when ``cols_feat`` is ``[]``.

    Raises
    ------
    TypeError
        If ``optb`` lacks ``get_binned_variable`` or a variable has no
        ``binning_table``.
    """
    if not hasattr(optb, "get_binned_variable"):
        raise TypeError(
            "optb must expose 'get_binned_variable' (e.g. a fitted "
            "optbinning.BinningProcess)."
        )

    if not cols_feat:
        return pd.DataFrame(), []

    order = list(dict.fromkeys(cols_feat))
    bp_names = {str(v) for v in getattr(optb, "variable_names", ())}
    unrecognized = [c for c in order if c not in bp_names]
    recognized = [c for c in order if c in bp_names]
    if not recognized:
        return pd.DataFrame(), unrecognized

    parts: list[pd.DataFrame] = []
    for name in recognized:
        binned = optb.get_binned_variable(name)
        bt = getattr(binned, "binning_table", None)
        if bt is None:
            raise TypeError(
                f"get_binned_variable({name!r}) returned an object without "
                "'binning_table'."
            )
        raw = bt.build()
        if not isinstance(raw, pd.DataFrame):
            raw = pd.DataFrame(raw)
        df_part = raw.copy()
        df_part.insert(0, "Feature Name", str(name))
        parts.append(df_part)

    out = pd.concat(parts, axis=0, ignore_index=True, sort=False)
    col_order = ["Feature Name"] + [c for c in out.columns if c != "Feature Name"]
    return out.loc[:, col_order], unrecognized


def get_woe_from_bp(
    df: pd.DataFrame,
    cols_feat: list[str],
    bp: Any,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Apply WoE encoding from a fitted ``optbinning.BinningProcess`` to a frame.

    Transforms only names that appear in ``cols_feat`` (deduplicated, order
    preserved), are listed in ``bp.variable_names``, **and** are columns of
    ``df``. Any other ``cols_feat`` entry is skipped and listed in the third
    return value (not recognized by ``bp``, or recognized but absent from
    ``df``).

    Parameters
    ----------
    df : pd.DataFrame
        Rows to transform.
    cols_feat : list of str
        Candidate feature names to try in order.
    bp : BinningProcess
        Fitted ``optbinning.BinningProcess`` (e.g. from :func:`get_optimal_bin`).

    Returns
    -------
    df_woe : pandas.DataFrame
        WoE values from ``bp.transform(..., metric="woe")`` on the usable
        columns only, renamed to ``<name>_woe``. Index matches ``df``. Empty
        columns when nothing is both recognized and present in ``df``.
    col_names : list of str
        Column names of ``df_woe`` (same as ``list(df_woe.columns)``).
    skipped : list of str
        Names from ``cols_feat`` (after deduplication) that were not transformed
        because they are not in ``bp.variable_names`` or not in ``df.columns``.

    Raises
    ------
    TypeError
        If ``bp`` lacks ``variable_names`` or ``get_binned_variable``.

    Notes
    -----
    Requires ``optbinning`` to be installed. WoE is computed with
    ``bp.get_binned_variable(name).transform(...)`` for each usable column so the
    frame need not include every variable that was fit in the process (unlike
    ``BinningProcess.transform``, which requires all selected variables to be
    present). ``metric_special`` and ``metric_missing`` use optbinning defaults
    (numeric ``0``).
    """
    if not hasattr(bp, "variable_names") or not hasattr(bp, "get_binned_variable"):
        raise TypeError(
            "bp must expose 'variable_names' and 'get_binned_variable' (e.g. a "
            "fitted optbinning.BinningProcess)."
        )

    bp_names = {str(v) for v in getattr(bp, "variable_names", ())}
    order = list(dict.fromkeys(cols_feat))
    df_cols = set(df.columns)
    to_transform = [c for c in order if c in bp_names and c in df_cols]
    skipped = [c for c in order if c not in to_transform]

    if not to_transform:
        df_woe = pd.DataFrame(index=df.index)
        return df_woe, [], skipped

    woe_data: dict[str, np.ndarray] = {}
    for name in to_transform:
        binned = bp.get_binned_variable(name)
        w = binned.transform(df[name], metric="woe", check_input=False)
        woe_data[f"{name}_woe"] = np.asarray(w, dtype=float).reshape(-1)

    df_woe = pd.DataFrame(woe_data, index=df.index)
    col_names = list(df_woe.columns)
    return df_woe, col_names, skipped


def _stratify_labels(y: pd.Series) -> pd.Series | None:
    """Return ``y`` for stratify= if sklearn constraints hold; else ``None``."""
    y_clean = y.dropna()
    if len(y_clean) < 4:
        return None
    counts = y_clean.value_counts()
    if len(counts) < 2 or int(counts.min()) < 2:
        return None
    return y


def split_data(
    df: pd.DataFrame,
    col_target: str,
    col_period: str,
    oot_th=None,
    hoot_th=None,
    test_perc: float = 0.25,
    valid_perc: float = 0.25,
) -> pd.Series:
    """
    Assign each row a data split label: ``hoot``, ``train``, ``test``,
    ``valid``, and/or ``oot``.

    Period-style cuts (when thresholds are set): ``df[col_period]`` and
    ``hoot_th`` / ``oot_th`` are compared as plain numbers (``<=`` / ``>=``).
    There is no parsing of dates or special calendar rules; choose one integer
    encoding for the column and use thresholds in that **same** encoding.

    - **hoot** (historical out-of-time): ``df[col_period] <= hoot_th``
    - **oot** (out-of-time holdout): ``df[col_period] >= oot_th``
    - **Core** (everything else): stratified into ``train`` / ``test`` /
      ``valid`` using ``test_perc`` and ``valid_perc`` as fractions of the
      **core** subset only.

    ``hoot`` and ``oot`` masks must not overlap; otherwise a ``ValueError``
    is raised (often means ``hoot_th`` and ``oot_th`` are not on the same
    numeric scale as ``col_period``).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    col_target : str
        Binary or discrete target used for stratified sampling in the core
        slice (when stratification is feasible per class counts).
    col_period : str
        Column whose values are **comparable to** ``hoot_th`` and ``oot_th``
        (typically integers). The caller must keep period values and
        thresholds on the same scale; this function only applies order
        comparisons.
    oot_th : optional
        If not ``None``, rows with ``df[col_period] >= oot_th`` are labeled ``oot``.
    hoot_th : optional
        If not ``None``, rows with ``df[col_period] <= hoot_th`` are labeled ``hoot``.
    test_perc : float, default 0.25
        Fraction of **core** rows assigned to ``test``. If ``0``, no ``test``
        label is used (core is only ``train`` and/or ``valid``).
    valid_perc : float, default 0.25
        Fraction of **core** rows assigned to ``valid``. If ``0``, no ``valid``
        label is used.

    Returns
    -------
    pandas.Series
        Same index as ``df``, dtype object, values among the labels that apply
        (e.g. if both thresholds are ``None``, only ``train``/``test``/``valid``
        appear as configured).

    Raises
    ------
    ValueError
        For missing columns, invalid proportions, overlapping hoot/oot, or
        impossible stratified splits.

    Notes
    -----
    - Rows with missing ``col_target`` in the core slice cannot be stratified
      consistently; they are labeled ``train`` and excluded from the stratified
      index set (see implementation).
    - **Agent / MCP:** deterministic label strings; align with
      :func:`get_score_predictive_power_data_type` ``col_type`` values.
    - Uses a fixed ``random_state`` (42 / 43) in ``train_test_split`` for
      reproducible splits.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "period": list(range(24)),
    ...     "y": ([0, 1] * 12),
    ... })
    >>> s = split_data(df, "y", "period", oot_th=22, hoot_th=2, test_perc=0.25, valid_perc=0.25)
    >>> set(s.unique()) <= {"hoot", "train", "test", "valid", "oot"}
    True
    """
    _validate_columns(df, "df", [col_target, col_period])

    if not (0.0 <= float(test_perc) <= 1.0 and 0.0 <= float(valid_perc) <= 1.0):
        raise ValueError("test_perc and valid_perc must be in [0, 1].")
    tp0 = float(test_perc)
    vp0 = float(valid_perc)
    if tp0 + vp0 > 1.0 + 1e-9:
        raise ValueError("test_perc + valid_perc must not exceed 1.0 (fractions of core).")
    if tp0 + vp0 >= 1.0 - 1e-12 and (tp0 > 0 and vp0 > 0):
        raise ValueError(
            "When both test_perc and valid_perc are positive, their sum must be < 1 "
            "so the core retains a train share."
        )
    if tp0 >= 1.0 or vp0 >= 1.0:
        raise ValueError("test_perc and valid_perc must be < 1 when used (use 0 to disable a split).")

    period = df[col_period]
    mask_hoot = pd.Series(False, index=df.index) if hoot_th is None else (period <= hoot_th)
    mask_oot = pd.Series(False, index=df.index) if oot_th is None else (period >= oot_th)

    overlap = mask_hoot & mask_oot
    if bool(overlap.any()):
        raise ValueError(
            "hoot (col_period <= hoot_th) and oot (col_period >= oot_th) overlap for some rows. "
            "Use thresholds so the intervals are disjoint (typically hoot_th < oot_th)."
        )

    mask_core = ~(mask_hoot | mask_oot)
    out = pd.Series(index=df.index, dtype=object)
    out.loc[mask_hoot] = "hoot"
    out.loc[mask_oot] = "oot"

    core_idx = df.index[mask_core]
    if len(core_idx) == 0:
        return out

    out.loc[core_idx] = "train"

    y_core = df.loc[core_idx, col_target]

    usable = core_idx[y_core.notna()]
    if len(usable) == 0:
        return out

    y_use = y_core.loc[usable]
    strat = _stratify_labels(y_use)
    strat_arr = strat.to_numpy() if strat is not None else None

    tp = float(test_perc)
    vp = float(valid_perc)
    rs1, rs2 = 42, 43

    if tp <= 0 and vp <= 0:
        out.loc[usable] = "train"
        return out

    if len(usable) < 2:
        out.loc[usable] = "train"
        return out

    if tp > 0 and vp > 0:
        train_val, test_idx = train_test_split(
            usable.to_numpy(),
            test_size=tp,
            stratify=strat_arr,
            random_state=rs1,
            shuffle=True,
        )
        train_val = pd.Index(train_val)
        test_idx = pd.Index(test_idx)
        out.loc[test_idx] = "test"
        rel_valid = vp / (1.0 - tp) if tp < 1.0 else 0.0
        y_tv = df.loc[train_val, col_target]
        strat_tv = _stratify_labels(y_tv)
        strat_tv_arr = strat_tv.to_numpy() if strat_tv is not None else None
        train_idx, valid_idx = train_test_split(
            train_val.to_numpy(),
            test_size=rel_valid,
            stratify=strat_tv_arr,
            random_state=rs2,
            shuffle=True,
        )
        out.loc[pd.Index(train_idx)] = "train"
        out.loc[pd.Index(valid_idx)] = "valid"
        return out

    if tp > 0:
        train_idx, test_idx = train_test_split(
            usable.to_numpy(),
            test_size=tp,
            stratify=strat_arr,
            random_state=rs1,
            shuffle=True,
        )
        out.loc[pd.Index(train_idx)] = "train"
        out.loc[pd.Index(test_idx)] = "test"
        return out

    # vp > 0, tp == 0
    train_idx, valid_idx = train_test_split(
        usable.to_numpy(),
        test_size=vp,
        stratify=strat_arr,
        random_state=rs1,
        shuffle=True,
    )
    out.loc[pd.Index(train_idx)] = "train"
    out.loc[pd.Index(valid_idx)] = "valid"
    return out


def _calendar_step_index(min_date: pd.Timestamp, max_date: pd.Timestamp, frequency: str) -> pd.DatetimeIndex:
    """Inclusive calendar steps between bounds for ``d`` / ``m`` / ``y``."""
    t0 = pd.Timestamp(min_date)
    t1 = pd.Timestamp(max_date)
    if t1 < t0:
        raise ValueError("max_date must be greater than or equal to min_date.")

    f = frequency.lower()
    if f == "d":
        return pd.date_range(t0.normalize(), t1.normalize(), freq="D")
    if f == "m":
        p0 = t0.to_period("M")
        p1 = t1.to_period("M")
        pr = pd.period_range(p0, p1, freq="M")
        return pr.to_timestamp(how="start")
    if f == "y":
        p0 = t0.to_period("Y")
        p1 = t1.to_period("Y")
        pr = pd.period_range(p0, p1, freq="Y")
        return pr.to_timestamp(how="start")
    raise ValueError("frequency must be one of 'd', 'm', 'y' (day, month, year steps).")


def get_virtual_date(
    df: pd.DataFrame,
    min_date,
    max_date,
    frequency: str = "d",
) -> pd.Series:
    """
    Build a virtual calendar column with uniform spacing across rows.

    Constructs an ordered sequence of calendar dates from ``min_date`` to
    ``max_date`` (inclusive) at the given step (daily / monthly / yearly),
    then assigns one date per row of ``df`` so that rows are spread **evenly**
    across that sequence: early rows map to early dates, later rows to later
    dates, with each calendar step receiving nearly the same number of rows
    (remainder rows go to the first steps). This matches the common pattern
    “~every *k* rows advance one day” when ``len(df)`` is a multiple of the
    number of steps.

    Parameters
    ----------
    df : pd.DataFrame
        Only ``len(df)`` and ``df.index`` are used.
    min_date, max_date
        Parsed with ``pandas.to_datetime``. Inclusive range at the resolution
        implied by ``frequency`` (e.g. ``'d'``: each calendar day between the
        two dates; ``'m'``: month starts from the month of ``min_date`` through
        the month of ``max_date``).
    frequency : str, default ``'d'``
        ``'d'`` day, ``'m'`` month start, ``'y'`` year start (case-insensitive).

    Returns
    -------
    pandas.Series
        ``datetime64`` values, index aligned to ``df.index``, length ``len(df)``.

    Raises
    ------
    ValueError
        If bounds are invalid, ``frequency`` is unsupported, or the implied
        calendar sequence is empty.

    Notes
    -----
    - **Agent / MCP:** output is always the same length and index as ``df``.
    - For ``n`` rows and ``m`` calendar steps, step ``j`` receives either
      ``n // m`` or ``n // m + 1`` rows (first ``n % m`` steps get the extra).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": range(310)})
    >>> s = get_virtual_date(df, "2025-12-01", "2025-12-31", "d")
    >>> len(s) == 310
    True
    >>> s.iloc[0] == pd.Timestamp("2025-12-01")
    True
    >>> s.iloc[9] == pd.Timestamp("2025-12-01")
    True
    >>> s.iloc[10] == pd.Timestamp("2025-12-02")
    True
    >>> s.iloc[-1] == pd.Timestamp("2025-12-31")
    True
    """
    n = len(df)
    if n == 0:
        return pd.Series(dtype="datetime64[ns]", index=df.index)

    t0 = pd.to_datetime(min_date)
    t1 = pd.to_datetime(max_date)
    steps = _calendar_step_index(t0, t1, frequency)
    m = len(steps)
    if m == 0:
        raise ValueError("Empty calendar sequence for the given min_date, max_date, and frequency.")

    # Uniform in row space: ~equal rows per calendar step (see Notes).
    counts = np.full(m, n // m, dtype=np.int64)
    counts[: n % m] += 1
    vals = np.repeat(steps.to_numpy(dtype="datetime64[ns]"), counts)
    return pd.Series(vals, index=df.index, dtype="datetime64[ns]")


def get_day_from_date(df: pd.DataFrame, col_time: str) -> np.ndarray:
    """
    Encode datetimes in ``df[col_time]`` as integer calendar days ``YYYYMMDD``.

    Equivalent to
    ``pd.to_datetime(df[col_time]).dt.strftime('%Y%m%d').astype(int)``, returned
    as a one-dimensional NumPy integer array (same length as ``df``).

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.
    col_time : str
        Column name with values parseable by ``pandas.to_datetime``.

    Returns
    -------
    numpy.ndarray
        ``dtype`` typically ``int64``; each value is e.g. ``20250412``.

    Raises
    ------
    ValueError
        If ``col_time`` is not a column of ``df``.
    """
    _validate_columns(df, "get_day_from_date", [col_time])
    return (
        pd.to_datetime(df[col_time])
        .dt.strftime("%Y%m%d")
        .astype(int)
        .values
    )


def get_month_from_date(df: pd.DataFrame, col_time: str) -> list[int]:
    """
    Encode datetimes in ``df[col_time]`` as integer year-month ``YYYYMM``.

    Uses ``pd.to_datetime(df[col_time]).dt.strftime('%Y%m').astype(int)`` and
    returns a Python list of integers (one per row), e.g. ``202504`` for
    April 2025.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.
    col_time : str
        Column name with values parseable by ``pandas.to_datetime``.

    Returns
    -------
    list of int
        Length ``len(df)``; each element is ``YYYYMM``.

    Raises
    ------
    ValueError
        If ``col_time`` is not a column of ``df``.
    """
    _validate_columns(df, "get_month_from_date", [col_time])
    return (
        pd.to_datetime(df[col_time])
        .dt.strftime("%Y%m")
        .astype(int)
        .values
    )


def _prepare_xy_binary_logreg(
    df: pd.DataFrame,
    feat_cols: list[str],
    col_target: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Complete-case numeric matrix and binary ``y`` (higher distinct target = 1)."""
    order = list(dict.fromkeys(feat_cols))
    _validate_columns(df, "df", [col_target] + order)
    sub = df[[col_target] + order].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 2:
        raise ValueError("Insufficient complete-case rows for target and features.")
    y_raw = sub[col_target].to_numpy(dtype=float)
    uniq = np.unique(y_raw)
    if len(uniq) != 2:
        raise ValueError(
            "col_target must have exactly two distinct numeric values in this subset."
        )
    _lo, hi = sorted(uniq.tolist())[:2]
    y01 = (y_raw == hi).astype(int)
    X = sub[order].to_numpy(dtype=float)
    return X, y01, order


def _auc_gini_safe(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    try:
        auc = float(roc_auc_score(y_true, proba))
    except ValueError:
        auc = float(np.nan)
    return auc, _gini_binary_from_auc(auc)


def _sklearn_uses_l1_ratio_for_penalty() -> bool:
    """
    True for scikit-learn >= 1.8, where ``penalty=`` on ``LogisticRegression`` is
    deprecated in favor of ``l1_ratio`` (0 = L2, 1 = L1).
    """
    import sklearn

    s = str(getattr(sklearn, "__version__", "0")).split(" ")[0].strip()
    chunks = s.split(".")
    if not chunks:
        return False
    mj_s = "".join(ch for ch in chunks[0] if ch.isdigit())
    if not mj_s:
        return False
    major = int(mj_s)
    minor = 0
    if len(chunks) > 1:
        mn_s = "".join(ch for ch in chunks[1] if ch.isdigit())
        if mn_s:
            minor = int(mn_s)
    return (major, minor) >= (1, 8)


def _logreg_tune_kwds(
    *,
    regularization: str,
    C: float,
    random_state: int,
    max_iter: int,
) -> dict[str, Any]:
    """Keyword args for ``LogisticRegression`` (sklearn 1.8+ vs older)."""
    C = float(C)
    rs = int(random_state)
    mi = int(max_iter)
    if regularization not in ("l1", "l2"):
        raise ValueError("regularization must be 'l1' or 'l2'.")
    if _sklearn_uses_l1_ratio_for_penalty():
        if regularization == "l1":
            return {
                "C": C,
                "l1_ratio": 1.0,
                "solver": "saga",
                "max_iter": mi,
                "random_state": rs,
                "fit_intercept": True,
            }
        return {
            "C": C,
            "l1_ratio": 0.0,
            "solver": "lbfgs",
            "max_iter": mi,
            "random_state": rs,
            "fit_intercept": True,
        }
    if regularization == "l1":
        return {
            "penalty": "l1",
            "solver": "saga",
            "C": C,
            "max_iter": mi,
            "random_state": rs,
            "fit_intercept": True,
        }
    return {
        "penalty": "l2",
        "solver": "lbfgs",
        "C": C,
        "max_iter": mi,
        "random_state": rs,
        "fit_intercept": True,
    }


def _cv_mean_auc_gini_logreg(
    X: np.ndarray,
    y01: np.ndarray,
    C: float,
    *,
    penalty: str,
    random_state: int,
    kfold: int,
) -> tuple[list[float], list[float], float, float]:
    """Return per-fold AUCs/Ginis on validation folds and their means."""
    skf = StratifiedKFold(
        n_splits=int(kfold), shuffle=True, random_state=int(random_state)
    )
    fold_aucs: list[float] = []
    fold_ginis: list[float] = []
    for tr_idx, va_idx in skf.split(X, y01):
        X_tr, y_tr = X[tr_idx], y01[tr_idx]
        X_va, y_va = X[va_idx], y01[va_idx]
        if X_tr.shape[0] < 2 or len(np.unique(y_tr)) < 2:
            fold_aucs.append(float(np.nan))
            fold_ginis.append(float(np.nan))
            continue
        clf = LogisticRegression(
            **_logreg_tune_kwds(
                regularization=penalty,
                C=float(C),
                random_state=int(random_state),
                max_iter=10000,
            )
        )
        try:
            clf.fit(X_tr, y_tr)
            p_va = clf.predict_proba(X_va)[:, 1]
        except Exception:
            fold_aucs.append(float(np.nan))
            fold_ginis.append(float(np.nan))
            continue
        auc, gini = _auc_gini_safe(y_va, p_va)
        fold_aucs.append(auc)
        fold_ginis.append(gini)
    mean_auc = float(np.nanmean(fold_aucs)) if fold_aucs else float(np.nan)
    mean_gini = float(np.nanmean(fold_ginis)) if fold_ginis else float(np.nan)
    return fold_aucs, fold_ginis, mean_auc, mean_gini


def _train_logreg_tune_cv_core(
    df: pd.DataFrame,
    cols_feat_woe: list[str],
    col_target: str,
    col_type: str,
    hyperparam_space: list[float],
    *,
    penalty: str,
    best_param_key: str,
    random_seed: int,
    kfold: int,
) -> tuple[dict[str, Any], LogisticRegression, pd.Series]:
    """
    Shared CV tuning on ``train``, model choice by valid AUC, report on train/valid/test.

    Rows with ``col_type`` not in ``{'train','valid','test'}`` (e.g. ``oot``, ``hoot``)
    are not used.
    """
    if not cols_feat_woe:
        raise ValueError("cols_feat_woe must be a non-empty list of column names.")
    C_list = [float(c) for c in hyperparam_space]
    if not C_list:
        raise ValueError("hyperparam_space must be a non-empty iterable of C values.")
    if int(kfold) < 2:
        raise ValueError("kfold must be at least 2.")
    if penalty not in ("l1", "l2"):
        raise ValueError("penalty must be 'l1' or 'l2'.")

    _validate_columns(df, "df", [col_type, col_target])
    ct = df[col_type].astype(str)
    df_train = df.loc[ct == "train"]
    df_valid = df.loc[ct == "valid"]
    df_test = df.loc[ct == "test"]
    if len(df_train) == 0:
        raise ValueError("No rows with col_type == 'train'.")
    if len(df_valid) == 0:
        raise ValueError("No rows with col_type == 'valid' (required for tuning).")

    X_tr, y_tr, feat_order = _prepare_xy_binary_logreg(df_train, cols_feat_woe, col_target)
    X_va, y_va, _ = _prepare_xy_binary_logreg(df_valid, cols_feat_woe, col_target)
    X_te: np.ndarray | None = None
    y_te: np.ndarray | None = None
    if len(df_test) > 0:
        try:
            X_te, y_te, _ = _prepare_xy_binary_logreg(df_test, cols_feat_woe, col_target)
        except ValueError:
            X_te, y_te = None, None

    if len(np.unique(y_tr)) < 2:
        raise ValueError("Training subset must contain two classes in col_target.")
    if int(kfold) > len(X_tr):
        raise ValueError(
            f"kfold={kfold} exceeds number of training rows ({len(X_tr)})."
        )

    by_hp: list[dict[str, Any]] = []
    best_C: float | None = None
    best_idx = 0
    best_valid_auc = float("-inf")

    for i, C in enumerate(C_list):
        fold_aucs, fold_ginis, cv_mean_auc, cv_mean_gini = _cv_mean_auc_gini_logreg(
            X_tr,
            y_tr,
            C,
            penalty=penalty,
            random_state=int(random_seed),
            kfold=int(kfold),
        )
        clf_full = LogisticRegression(
            **_logreg_tune_kwds(
                regularization=penalty,
                C=float(C),
                random_state=int(random_seed),
                max_iter=10000,
            )
        )
        try:
            clf_full.fit(X_tr, y_tr)
            p_tr = clf_full.predict_proba(X_tr)[:, 1]
            p_va = clf_full.predict_proba(X_va)[:, 1]
        except Exception:
            train_auc = train_gini = float(np.nan)
            valid_auc = valid_gini = float(np.nan)
        else:
            train_auc, train_gini = _auc_gini_safe(y_tr, p_tr)
            valid_auc, valid_gini = _auc_gini_safe(y_va, p_va)

        row = {
            "C": float(C),
            "train_cv_fold_aucs": [float(x) for x in fold_aucs],
            "train_cv_fold_ginis": [float(x) for x in fold_ginis],
            "train_cv_mean_auc": float(cv_mean_auc),
            "train_cv_mean_gini": float(cv_mean_gini),
            "train_fit_auc": float(train_auc),
            "train_fit_gini": float(train_gini),
            "valid_auc": float(valid_auc),
            "valid_gini": float(valid_gini),
        }
        by_hp.append(row)

        if np.isfinite(valid_auc) and valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_C = float(C)
            best_idx = i

    if best_C is None:
        best_C = float(C_list[0])
        best_idx = 0

    best_model = LogisticRegression(
        **_logreg_tune_kwds(
            regularization=penalty,
            C=float(best_C),
            random_state=int(random_seed),
            max_iter=10000,
        )
    )
    best_model.fit(X_tr, y_tr)

    p_tr_best = best_model.predict_proba(X_tr)[:, 1]
    p_va_best = best_model.predict_proba(X_va)[:, 1]
    train_best_auc, train_best_gini = _auc_gini_safe(y_tr, p_tr_best)
    valid_best_auc, valid_best_gini = _auc_gini_safe(y_va, p_va_best)

    best_row = by_hp[best_idx]
    train_avg_auc = float(best_row["train_cv_mean_auc"])
    train_avg_gini = float(best_row["train_cv_mean_gini"])

    valid_aucs = [r["valid_auc"] for r in by_hp if np.isfinite(r["valid_auc"])]
    valid_avg_auc = float(np.mean(valid_aucs)) if valid_aucs else float(np.nan)
    valid_ginis = [r["valid_gini"] for r in by_hp if np.isfinite(r["valid_gini"])]
    valid_avg_gini = float(np.mean(valid_ginis)) if valid_ginis else float(np.nan)

    if (
        X_te is not None
        and y_te is not None
        and len(X_te) > 0
        and len(np.unique(y_te)) >= 2
    ):
        p_te_best = best_model.predict_proba(X_te)[:, 1]
        test_best_auc, test_best_gini = _auc_gini_safe(y_te, p_te_best)
        test_avg_auc = test_best_auc
        test_avg_gini = test_best_gini
    else:
        test_best_auc = test_best_gini = float(np.nan)
        test_avg_auc = test_avg_gini = float(np.nan)

    coef = np.asarray(best_model.coef_).ravel()
    importance = pd.Series(np.abs(coef), index=feat_order, name="importance")

    split_block = {
        "best_auc": float(train_best_auc),
        "best_gini": float(train_best_gini),
        best_param_key: float(best_C),
        "average_auc": train_avg_auc,
        "average_gini": train_avg_gini,
    }
    dict_statistics: dict[str, Any] = {
        best_param_key: float(best_C),
        "train": split_block,
        "valid": {
            "best_auc": float(valid_best_auc),
            "best_gini": float(valid_best_gini),
            best_param_key: float(best_C),
            "average_auc": valid_avg_auc,
            "average_gini": valid_avg_gini,
        },
        "test": {
            "best_auc": float(test_best_auc),
            "best_gini": float(test_best_gini),
            best_param_key: float(best_C),
            "average_auc": float(test_avg_auc),
            "average_gini": float(test_avg_gini),
        },
        "by_hyperparameter": by_hp,
    }

    return dict_statistics, best_model, importance


def train_logreg_l1_tune_cv(
    df: pd.DataFrame,
    cols_feat_woe: list[str],
    col_target: str,
    col_type: str,
    l1_space: Iterable[float] | None = None,
    random_seed: int = 42,
    kfold: int = 5,
) -> tuple[dict[str, Any], LogisticRegression, pd.Series]:
    """
    Tune ``LogisticRegression`` with L1 penalty using stratified CV on the train split.

    Only rows with ``col_type`` in ``{'train','valid','test'}`` are used; ``oot``,
    ``hoot``, and any other labels are ignored. For each inverse regularization
    strength ``C`` in ``l1_space``, stratified ``kfold`` CV is run on **train**
    rows only; the ``C`` with highest ROC-AUC on the **valid** split (model fit
    on all train rows) is selected. The returned model is refit on full train
    with that ``C``. Test metrics use the **test** split only (paper result).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``col_target``, ``col_type``, and WoE feature columns.
    cols_feat_woe : list of str
        Predictor columns (deduplicated, first occurrence wins).
    col_target : str
        Binary target (two distinct numeric values after coercion; larger = 1).
    col_type : str
        Column with values ``train`` / ``valid`` / ``test`` (and possibly others
        that are ignored).
    l1_space : iterable of float, optional
        ``C`` values for ``LogisticRegression`` (smaller ``C`` = stronger L1).
        If omitted, uses ``(1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)``.
    random_seed : int, default 42
        ``random_state`` for the estimator and CV shuffling.
    kfold : int, default 5
        Number of stratified folds on the train subset.

    Returns
    -------
    dict_statistics : dict
        Top-level ``best_l1_param`` (chosen ``C``), nested ``train`` / ``valid`` /
        ``test`` each with ``best_auc``, ``best_gini``, ``best_l1_param``,
        ``average_auc``, ``average_gini`` (train averages are CV means at the best
        ``C``; valid averages are means over the grid on the valid slice; test
        ``average_*`` match ``best_*`` when test is non-empty), and
        ``by_hyperparameter``: list of dicts with ``C``, CV fold AUCs/Ginis on
        train, CV means, in-sample train AUC/Gini after full-train fit, and
        valid AUC/Gini for that ``C``.
    best_model : LogisticRegression
        Fitted on all train rows at ``best_l1_param``.
    feature_importance : pandas.Series
        Absolute fitted coefficients (index = feature names).
    """
    # Default inverse-regularization ``C`` grid (sklearn: smaller ``C`` = stronger penalty).
    _DEFAULT_L1_TUNING_C_VALUES: tuple[float, ...] = (
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        0.1,
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
    )
    C_list = list(l1_space if l1_space is not None else _DEFAULT_L1_TUNING_C_VALUES)
    return _train_logreg_tune_cv_core(
        df,
        cols_feat_woe,
        col_target,
        col_type,
        C_list,
        penalty="l1",
        best_param_key="best_l1_param",
        random_seed=random_seed,
        kfold=kfold,
    )


def train_logreg_l2_tune_cv(
    df: pd.DataFrame,
    cols_feat_woe: list[str],
    col_target: str,
    col_type: str,
    l2_space: Iterable[float] | None = None,
    random_seed: int = 42,
    kfold: int = 5,
) -> tuple[dict[str, Any], LogisticRegression, pd.Series]:
    """
    Same as :func:`train_logreg_l1_tune_cv` but with L2 penalty and ``l2_space``
    giving ``C`` values (default ``(1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)`` when
    omitted). The statistics dict uses the key ``best_l2_param``.
    """
    # Default inverse-regularization ``C`` grid (sklearn: smaller ``C`` = stronger penalty).
    _DEFAULT_L2_TUNING_C_VALUES: tuple[float, ...] = (
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        0.1,
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
    )
    C_list = list(l2_space if l2_space is not None else _DEFAULT_L2_TUNING_C_VALUES)
    return _train_logreg_tune_cv_core(
        df,
        cols_feat_woe,
        col_target,
        col_type,
        C_list,
        penalty="l2",
        best_param_key="best_l2_param",
        random_seed=random_seed,
        kfold=kfold,
    )


def logreg_predict(
    df: pd.DataFrame,
    logreg_model: Any,
    cols_feat_woe: list[str],
) -> tuple[list[float], list[float], dict[str, list[str]]]:
    """
    Binary ``predict_proba`` from a fitted logistic regression on WoE columns.

    Builds a design matrix in the same column order the model expects, then
    calls ``predict_proba``. Column alignment uses ``feature_names_in_`` when
    present (e.g. model fit on a ``DataFrame``); otherwise the first
    ``coef_.shape[1]`` entries of ``cols_feat_woe`` are taken as the training
    column order (extras are listed under ``cols_feat_woe_not_in_model``), as
    with models from :func:`train_logreg_l1_tune_cv` /
    :func:`train_logreg_l2_tune_cv` fit on numpy arrays without names.

    Parameters
    ----------
    df : pd.DataFrame
        Scoring frame; may omit rows or columns not used.
    logreg_model
        Fitted classifier with ``coef_``, ``classes_``, and ``predict_proba``.
    cols_feat_woe : list of str
        Candidate feature names (deduplicated, first occurrence wins). Used
        for alignment checks and, when the model has no stored names, as the
        ordered feature list.

    Returns
    -------
    proba_0 : list of float
        Probability of class ``0`` per row (same order as ``df``), aligned to
        ``classes_``.
    proba_1 : list of float
        Probability of class ``1`` per row.
    feature_issue_prediction : dict of list of str
        ``cols_feat_woe_not_in_model``: names in ``cols_feat_woe`` not used by
        the model; ``model_feature_not_in_cols_feat_woe``: model features absent
        from ``cols_feat_woe``; ``model_feature_missing_from_dataframe``: model
        features with no column in ``df``.

    Raises
    ------
    ValueError
        If the model is not binary, ``classes_`` are not the integer labels
        ``0`` and ``1``, ``cols_feat_woe`` lists fewer names than ``coef_`` has columns
        (when there are no ``feature_names_in_``), alignment is inconsistent, or
        ``predict_proba`` is unavailable.
    """
    if not hasattr(logreg_model, "predict_proba"):
        raise ValueError("logreg_model must implement predict_proba.")
    coef = np.asarray(getattr(logreg_model, "coef_", None))
    if coef.ndim != 2 or coef.shape[0] < 1:
        raise ValueError("logreg_model must have a 2-D coef_ with one or more rows.")
    n_model_feat = int(coef.shape[1])

    order = list(dict.fromkeys(cols_feat_woe))
    if not order:
        raise ValueError("cols_feat_woe must be a non-empty list of column names.")

    fni = getattr(logreg_model, "feature_names_in_", None)
    if fni is not None:
        model_feats = [str(x) for x in np.asarray(fni).tolist()]
        if len(model_feats) != n_model_feat:
            raise ValueError(
                "logreg_model.feature_names_in_ length does not match coef_."
            )
    else:
        if len(order) < n_model_feat:
            raise ValueError(
                "logreg_model has no feature_names_in_; cols_feat_woe must list at "
                f"least {n_model_feat} names in training order (got {len(order)})."
            )
        if len(order) == n_model_feat:
            model_feats = list(order)
        else:
            model_feats = list(order[:n_model_feat])

    cols_set = set(order)
    model_set = set(model_feats)
    cols_feat_woe_not_in_model = [c for c in order if c not in model_set]
    model_feature_not_in_cols_feat_woe = [c for c in model_feats if c not in cols_set]

    n = len(df)
    X_cols: list[np.ndarray] = []
    model_feature_missing_from_dataframe: list[str] = []
    for name in model_feats:
        if name not in df.columns:
            model_feature_missing_from_dataframe.append(name)
            X_cols.append(np.zeros(n, dtype=float))
        else:
            v = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
            if v.shape[0] != n:
                raise ValueError(f"Column {name!r} length does not match len(df).")
            X_cols.append(v)
    X = np.column_stack(X_cols) if X_cols else np.empty((n, 0), dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    classes = np.asarray(logreg_model.classes_)
    if classes.size != 2:
        raise ValueError("logreg_predict only supports binary classifiers.")
    labs = [int(np.asarray(x).item()) for x in classes.tolist()]
    if set(labs) != {0, 1}:
        raise ValueError(
            "Binary model classes_ must be labels 0 and 1 (after coercion) for "
            "proba_0/proba_1."
        )
    i0 = int(labs.index(0))
    i1 = int(labs.index(1))

    proba_mat = logreg_model.predict_proba(X)
    p0 = proba_mat[:, i0].astype(float, copy=False)
    p1 = proba_mat[:, i1].astype(float, copy=False)

    proba_0 = p0.tolist()
    proba_1 = p1.tolist()

    feature_issue_prediction: dict[str, list[str]] = {
        "cols_feat_woe_not_in_model": cols_feat_woe_not_in_model,
        "model_feature_not_in_cols_feat_woe": model_feature_not_in_cols_feat_woe,
        "model_feature_missing_from_dataframe": model_feature_missing_from_dataframe,
    }
    return proba_0, proba_1, feature_issue_prediction


__all__ = [
    "compute_psi",
    "get_timely_feature_psi",
    "get_timely_feature_psi_woe",
    "get_timely_psi",
    "get_timely_binary_target_rate",
    "get_timely_target_rate_feature_segment",
    "get_nan_rate",
    "get_nan_rate_timely",
    "get_target_rate_sample",
    "get_feature_predictive_power",
    "select_features_auc_max_corr",
    "select_features_iv_max_corr",
    "select_features_aic_forward",
    "select_features_aic_backward",
    "select_features_bic_forward",
    "select_features_bic_backward",
    "select_features_auc_backward",
    "select_features_auc_forward",
    "get_feature_predictive_power_timely",
    "get_score_predictive_power_timely",
    "get_score_predictive_power_data_type",
    "get_score_predictive_power_data_type_bootstrap",
    "compare_score_predictive_power_data_type_bootstrap",
    "get_feature_dtype",
    "get_optimal_bin",
    "modify_optimal_bin",
    "get_binning_tables_from_bp",
    "get_woe_from_bp",
    "split_data",
    "get_virtual_date",
    "get_day_from_date",
    "get_month_from_date",
    "train_logreg_l1_tune_cv",
    "train_logreg_l2_tune_cv",
    "logreg_predict",
]
