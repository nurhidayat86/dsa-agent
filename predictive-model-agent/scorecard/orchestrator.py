"""Phase state machine with HITL gates H1–H6.

Phases are executed sequentially. After each gate the ``HitlInterface``
returns a decision (``approve | revise | reject``). ``revise`` payloads are
parsed into the appropriate Pydantic model and applied; the orchestrator
rewinds exactly the subgraph that the revision invalidates:

* H1 revise → reset problem contract, rerun from phase 2.
* H2 revise → reset splits, rerun from phase 3.
* H3 revise → apply binning overrides, rerun from phase 4 WoE refresh on
  through phase 7.
* H4 revise → merge feedback into ``ConstraintSpec``, rerun phases 5-7.
* H5 revise → rebuild scorecard with new PDO, rerun phase 10.
* H6 reject → regenerate docs after upstream changes; rerun gate.

Each phase writes artifacts via ``ArtifactStore``; a final ``run_manifest.json``
indexes everything. The ``run_pipeline`` entry point is the only public API.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import tools
from .agents import (
    compliance_checklist,
    docs_interpretations,
    eda_narrative,
    propose_split_config,
    ranking_rationale,
    revise_binning,
    revise_constraint_spec,
    revise_pdo_params,
    revise_problem_contract,
    revise_split_config,
)
from .artifacts import ArtifactStore
from .hitl import GatePayload, HitlInterface
from .schemas import (
    BinningConfig,
    BinningRevision,
    BranchResult,
    ConstraintSpec,
    DataColumnContract,
    FeatureSearchConfig,
    GateId,
    HitlAction,
    HitlDecision,
    LogisticHyperparams,
    ModelDocumentationMeta,
    PdoParams,
    ProblemContract,
    ProposalBundle,
    RankerWeights,
    RecipeSpec,
    RunManifest,
    SplitConfig,
)
from .settings import Settings, load_settings
from .state import RunContext


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


@dataclass
class PipelineOptions:
    data_path: str | Path
    artifacts_root: str | Path | None = None
    run_id: str | None = None
    col_target: str | None = None
    col_time: str | None = None
    cols_feat: list[str] | None = None
    max_iterations: int = 3


def run_pipeline(
    options: PipelineOptions,
    hitl: HitlInterface,
    settings: Settings | None = None,
) -> RunManifest:
    """Execute phases 1 → 11 with HITL gates and return the final manifest."""

    settings = settings or load_settings()
    run_id = options.run_id or _fresh_run_id()
    art_root = Path(options.artifacts_root or (settings.project_root / "scorecard_runs")).resolve()
    store = ArtifactStore(art_root / run_id)
    ctx = RunContext(run_id=run_id, data_source=str(options.data_path), store=store)

    print(f"[run_id] {run_id}")
    print(f"[artifacts] {store.root}")

    # ------------------------------ phase 1 ------------------------------ #
    _phase_1_ingest(ctx, options)

    # ------------------------------- gate H1 ----------------------------- #
    _gate_h1(ctx, hitl, settings)

    # ------------------------------ phase 2 ------------------------------ #
    _phase_2_eda(ctx, settings)

    # ------------------------------ phase 3 ------------------------------ #
    _phase_3_split(ctx, settings)

    # ------------------------------- gate H2 ----------------------------- #
    while True:
        decision = _gate_h2(ctx, hitl)
        if decision.action is HitlAction.APPROVE:
            break
        if decision.action is HitlAction.REJECT:
            raise RuntimeError("User rejected split design (H2). Aborting run.")
        new_cfg = _resolve_h2_revision(ctx, settings, decision.payload)
        ctx.split_config = new_cfg
        tools.run_split(ctx, new_cfg)
        ctx.store.save_model("split_config.json", new_cfg)

    # ------------------------------ phase 4 ------------------------------ #
    _phase_4_binning(ctx)

    # ------------------------------- gate H3 ----------------------------- #
    while True:
        decision = _gate_h3(ctx, hitl)
        if decision.action is HitlAction.APPROVE:
            break
        if decision.action is HitlAction.REJECT:
            raise RuntimeError("User rejected binning tables (H3). Aborting run.")
        rev = _resolve_h3_revision(ctx, settings, decision.payload)
        if not rev.overrides:
            print("[H3] no actionable overrides extracted from the request — keeping bins.")
            continue
        tools.apply_binning_revision(ctx, rev)

    # ------------------------------ phases 5-7 --------------------------- #
    iteration = 0
    while True:
        iteration += 1
        ctx.iteration = iteration
        _phase_5_and_6_feature_search_and_train(ctx)
        proposal = _phase_7_rank(ctx, settings)
        decision = _gate_h4(ctx, hitl, proposal)
        if decision.action is HitlAction.APPROVE:
            break
        if decision.action is HitlAction.REJECT:
            raise RuntimeError("User rejected model proposal (H4). Aborting run.")
        if iteration >= options.max_iterations:
            raise RuntimeError(f"Max H4 iterations ({options.max_iterations}) exceeded.")
        # revise — translate the NL request into a ConstraintSpec update
        revised = _resolve_h4_revision(ctx, settings, decision.payload)
        ctx.constraint_spec = revised
        store.save_model("constraint_spec.json", ctx.constraint_spec)
        print(f"[H4] iteration {iteration} feedback applied — rerunning phases 5-7")

    # ------------------------------ phase 9 ------------------------------ #
    _phase_9_scorecard(ctx)

    # ------------------------------- gate H5 ----------------------------- #
    while True:
        decision = _gate_h5(ctx, hitl)
        if decision.action is HitlAction.APPROVE:
            break
        if decision.action is HitlAction.REJECT:
            raise RuntimeError("User rejected PDO / scorecard (H5). Aborting run.")
        new_pdo = _resolve_h5_revision(ctx, settings, decision.payload)
        if new_pdo == ctx.pdo_params:
            print("[H5] reviser produced an identical PdoParams — keeping scorecard.")
            continue
        ctx.pdo_params = new_pdo
        _phase_9_scorecard(ctx)

    # ------------------- production code export (post-H5) --------------- #
    _phase_9b_production_code(ctx)

    # ------------------------------ phase 10 ----------------------------- #
    _phase_10_validation(ctx)

    # ------------------------------ phase 12 ----------------------------- #
    _phase_12_model_docs(ctx, settings)

    # ------------------------------- gate H6 ----------------------------- #
    while True:
        decision = _gate_h6(ctx, hitl)
        if decision.action is HitlAction.APPROVE:
            break
        if decision.action is HitlAction.REJECT:
            raise RuntimeError("User rejected final model (H6). Aborting run.")
        # revise at H6: regenerate docs only (numbers unchanged).
        print("[H6] regenerating model_documentation.md per revise request")
        _phase_12_model_docs(ctx, settings)

    # ------------------------------ phase 11 ----------------------------- #
    manifest = _phase_11_package(ctx)
    print(f"[done] manifest: {store.root / 'run_manifest.json'}")
    return manifest


# --------------------------------------------------------------------------- #
# Phase 1 — ingest + H1
# --------------------------------------------------------------------------- #


def _fresh_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _phase_1_ingest(ctx: RunContext, options: PipelineOptions) -> None:
    df = tools.load_dataset(options.data_path)
    contract = tools.auto_detect_contract(df)
    if options.col_target:
        contract.col_target = options.col_target
    if options.col_time:
        contract.col_time = options.col_time
    if options.cols_feat:
        contract.cols_feat = list(options.cols_feat)
    df = tools.ensure_binary_target(df, contract.col_target)
    ctx.df_raw = df
    ctx.df_work = df.copy()
    ctx.data_contract = contract
    # Materialize the canonical monthly period column up-front so every
    # downstream "timely" operation (EDA, PSI, target-rate, split OOT) groups
    # on the same month-start Timestamps.
    tools.ensure_month_column(ctx)
    ctx.store.save_model("data_contract.json", contract)
    print(
        f"[phase 1] loaded {df.shape[0]} rows × {df.shape[1]} cols — "
        f"target={contract.col_target} time={contract.col_time} "
        f"month={contract.col_month} features={len(contract.cols_feat)}"
    )


def _gate_h1(ctx: RunContext, hitl: HitlInterface, settings: Settings) -> None:
    assert ctx.data_contract is not None and ctx.df_work is not None
    dc = ctx.data_contract
    target_rate = float(ctx.df_work[dc.col_target].mean())
    proposed = ProblemContract(
        target_definition=f"binary 0/1 = {dc.col_target}",
        forbidden_features=[],
        forced_features=[],
    )

    # Terminal output truncates long lists, so write the full feature catalog
    # and column contract to a sidecar text file and point the user at it in
    # the gate summary. The reviewer can skim it in their editor while still
    # seeing a short preview in the console.
    metadata_path = _write_h1_metadata_file(ctx, target_rate)

    n_feat = len(dc.cols_feat)
    preview_n = min(8, n_feat)
    preview = ", ".join(dc.cols_feat[:preview_n])
    if n_feat > preview_n:
        preview += f" ... (+{n_feat - preview_n} more)"

    payload = GatePayload(
        gate_id=GateId.H1,
        title="problem contract",
        summary=(
            f"Confirm the target definition, any exclusions and feature eligibility "
            f"rules, and performance thresholds. Overall target rate = {target_rate:.3%}. "
            f"The full column contract ({n_feat} features) has been written to "
            f"`{metadata_path}` — open that file to review every feature before "
            "deciding. To revise, describe your changes in plain English — for "
            "example: \"forbid ExternalRiskEstimate and require "
            "MSinceOldestTradeOpen\", \"raise the minimum valid Gini to 0.3\"."
        ),
        proposal=proposed.model_dump(),
        tables={
            "column_contract (preview — full list in the file above)": pd.DataFrame(
                {
                    "role": ["col_time", "col_target", f"cols_feat ({n_feat} total)"],
                    "value": [dc.col_time, dc.col_target, preview],
                }
            ),
        },
    )
    decision = hitl.ask_gate(payload)
    hitl.log(ctx.store.root / "hitl", decision)
    if decision.action is HitlAction.REJECT:
        raise RuntimeError("User rejected the problem contract (H1). Aborting.")
    if decision.action is HitlAction.REVISE:
        ctx.problem_contract = _resolve_h1_revision(ctx, settings, proposed, decision.payload)
    else:
        ctx.problem_contract = proposed
    # Apply forbidden features to cols_feat immediately.
    if ctx.problem_contract.forbidden_features:
        ctx.data_contract.cols_feat = [
            c for c in ctx.data_contract.cols_feat
            if c not in set(ctx.problem_contract.forbidden_features)
        ]
    ctx.store.save_model("problem_contract.json", ctx.problem_contract)
    ctx.store.save_model("data_contract.json", ctx.data_contract)


def _write_h1_metadata_file(ctx: RunContext, target_rate: float) -> Path:
    """Write a human-readable column / row contract to ``hitl/h1_metadata.txt``.

    The terminal view of gate H1 truncates the feature list, so we mirror the
    full data contract plus per-feature dtype / missing-rate / target-rate
    hints into a sidecar file that the reviewer can open in an editor. The
    file path is returned so the gate summary can reference it.
    """

    assert ctx.data_contract is not None and ctx.df_work is not None
    dc = ctx.data_contract
    df = ctx.df_work

    lines: list[str] = []
    lines.append("# H1 — problem contract metadata")
    lines.append("")
    lines.append(f"run_id          : {ctx.run_id}")
    lines.append(f"data_source     : {ctx.data_source}")
    lines.append(f"rows            : {len(df):,}")
    lines.append(f"columns         : {df.shape[1]}")
    lines.append(f"target rate     : {target_rate:.4%}")
    lines.append("")
    lines.append("Column contract")
    lines.append("---------------")
    lines.append(f"col_time        : {dc.col_time}")
    lines.append(f"col_target      : {dc.col_target}")
    lines.append(f"col_month       : {dc.col_month}")
    lines.append(f"col_type        : {dc.col_type}")
    lines.append(f"col_score       : {dc.col_score}")
    lines.append(f"cols_feat total : {len(dc.cols_feat)}")
    lines.append("")
    lines.append(
        f"Candidate features ({len(dc.cols_feat)}) — dtype, missing%, target rate if non-null:"
    )
    lines.append("")
    target_series = df[dc.col_target] if dc.col_target in df.columns else None
    for i, feat in enumerate(dc.cols_feat, start=1):
        if feat not in df.columns:
            lines.append(f"{i:>3}. {feat}  — [MISSING FROM FRAME]")
            continue
        col = df[feat]
        dtype = str(col.dtype)
        missing_pct = float(col.isna().mean()) * 100.0
        tgt_note = ""
        if target_series is not None:
            mask = col.notna()
            if mask.any():
                tgt_rate = float(target_series[mask].mean())
                tgt_note = f", target_rate_when_present={tgt_rate:.3%}"
        lines.append(
            f"{i:>3}. {feat}  (dtype={dtype}, missing={missing_pct:.2f}%{tgt_note})"
        )
    lines.append("")
    lines.append(
        "Tip: to revise this contract, come back to the terminal and choose 'r' "
        "(revise). You can describe changes in plain English — for example: "
        "\"forbid <feature_name>\", \"force-include <feature_name>\", or "
        "\"raise the minimum valid Gini to 0.3\"."
    )
    lines.append("")

    return ctx.store.save_text("hitl/h1_metadata.txt", "\n".join(lines))


def _resolve_h1_revision(
    ctx: RunContext,
    settings: Settings,
    current: ProblemContract,
    payload: dict[str, Any],
) -> ProblemContract:
    """Translate an H1 revise payload into a validated ``ProblemContract``.

    The only accepted shape is ``{"_nl_request": "<free text>"}``; an empty
    payload keeps the proposed contract. Anything else is treated as free
    text by serializing to JSON and handing to the reviser agent.
    """

    if not payload:
        return current
    assert ctx.data_contract is not None and ctx.df_work is not None
    dc = ctx.data_contract
    nl = payload.get("_nl_request") if isinstance(payload, dict) else None
    if not isinstance(nl, str) or not nl.strip():
        nl = json.dumps(payload, default=str)
    reviser_payload = {
        "current": current.model_dump(),
        "feature_catalog": list(dc.cols_feat),
        "target_rate": float(ctx.df_work[dc.col_target].mean()),
        "user_request": nl.strip(),
    }
    revised = revise_problem_contract(settings, reviser_payload, fallback=current)
    print(f"[H1] problem_contract_reviser -> {revised.model_dump()}")
    return revised


# --------------------------------------------------------------------------- #
# Phase 2 — EDA / QC
# --------------------------------------------------------------------------- #


def _phase_2_eda(ctx: RunContext, settings: Settings) -> None:
    assert ctx.df_work is not None and ctx.data_contract is not None
    eda = tools.eda_summaries(ctx)
    ctx.store.save_parquet("eda/nan_overall.parquet", eda["nan_overall"])
    ctx.store.save_parquet("eda/nan_timely.parquet", eda["nan_timely"])
    ctx.store.save_parquet("eda/target_timely.parquet", eda["target_timely"])
    payload = _eda_llm_payload(ctx, eda)
    narrative = eda_narrative(settings, payload)
    ctx.store.save_text("eda/narrative.md", narrative or "(no narrative)")
    print("\n[phase 2 — eda narrative]\n" + (narrative or "(skipped)"))


def _eda_llm_payload(ctx: RunContext, eda: dict[str, pd.DataFrame]) -> dict[str, Any]:
    assert ctx.df_work is not None and ctx.data_contract is not None
    dc = ctx.data_contract
    nan_overall = eda["nan_overall"]
    top_nan = (
        nan_overall.sort_values(by=nan_overall.columns[-1], ascending=False)
        .head(8)
        .reset_index()
        .to_dict(orient="records")
    )
    target_timely = eda["target_timely"].to_dict(orient="records")
    return {
        "rows": int(len(ctx.df_work)),
        "features": len(dc.cols_feat),
        "target_prevalence_overall": float(ctx.df_work[dc.col_target].mean()),
        "top_missing_features": top_nan,
        "target_rate_over_time": target_timely,
    }


# --------------------------------------------------------------------------- #
# Phase 3 — split + H2
# --------------------------------------------------------------------------- #


# Minimum cohort thresholds used both by the proposer prompt and the
# post-split sanity check. Tuned for demo datasets; override via env later.
_MIN_ROWS_PER_COHORT = 200
_MIN_POSITIVE_PER_COHORT = 50


def _phase_3_split(ctx: RunContext, settings: Settings) -> None:
    """Propose an initial ``SplitConfig`` with an AI agent, then run the split.

    The proposer sees a monthly target-rate table (``get_timely_binary_target_rate``
    collapsed to month) plus the dataset size, overall target rate and the
    minimum per-cohort thresholds, and returns a structured ``SplitConfig``.
    If the agent is unreachable or returns an invalid payload, we fall back
    to ``SplitConfig()`` defaults so the pipeline never stalls.
    """

    if ctx.split_config is None:
        monthly = tools.monthly_target_rate(ctx)
        ctx.store.save_parquet("eda/target_monthly.parquet", monthly)
        payload = _split_proposer_payload(ctx, monthly)
        proposed = propose_split_config(settings, payload, fallback=SplitConfig())
        ctx.split_config = proposed
        print(f"[phase 3] split proposer -> {proposed.model_dump()}")
    else:
        print(f"[phase 3] using provided split config: {ctx.split_config.model_dump()}")
    tools.run_split(ctx, ctx.split_config)
    ctx.store.save_model("split_config.json", ctx.split_config)


def _split_proposer_payload(ctx: RunContext, monthly: pd.DataFrame) -> dict[str, Any]:
    assert ctx.df_work is not None and ctx.data_contract is not None
    dc = ctx.data_contract
    monthly_records = (
        monthly.reset_index()
        .rename(columns={monthly.index.name or "index": "month"})
        .to_dict(orient="records")
    )
    return {
        "rows": int(len(ctx.df_work)),
        "target_prevalence_overall": float(ctx.df_work[dc.col_target].mean()),
        "monthly_table": monthly_records,
        "min_rows_per_cohort": _MIN_ROWS_PER_COHORT,
        "min_positive_per_cohort": _MIN_POSITIVE_PER_COHORT,
        "column_contract": {
            "col_time": dc.col_time,
            "col_month": dc.col_month,
            "col_target": dc.col_target,
        },
        "time_granularity": "month",
    }


def _gate_h2(ctx: RunContext, hitl: HitlInterface) -> HitlDecision:
    assert ctx.df_work is not None and ctx.data_contract is not None and ctx.split_config is not None
    dc = ctx.data_contract
    sanity = tools.split_sanity(ctx, _MIN_ROWS_PER_COHORT, _MIN_POSITIVE_PER_COHORT)
    counts = ctx.df_work[dc.col_type].value_counts().sort_index()
    rates = ctx.df_work.groupby(dc.col_type)[dc.col_target].mean().reindex(counts.index)
    cohort_table = pd.DataFrame(
        {
            "count": counts.astype(int),
            "positives": [sanity["positives"].get(str(k), 0) for k in counts.index],
            "target_rate": rates.round(4),
        }
    ).reset_index()
    monthly = tools.monthly_target_rate(ctx).reset_index()
    monthly.columns = [str(c) for c in monthly.columns]
    violations = sanity.get("violations", [])
    v_suffix = (
        f" ⚠ sanity violations: {'; '.join(violations)}"
        if violations
        else " ✓ all cohorts meet minimum row/positive thresholds"
    )
    payload = GatePayload(
        gate_id=GateId.H2,
        title="split design",
        summary=(
            "AI-proposed split below. Approve, or REVISE either with a plain-text "
            "request (e.g. 'use last 2 months as OOT', 'shrink valid to 15%') or "
            "with a JSON SplitConfig override. A reviser agent will translate your "
            "text into a new SplitConfig, enforcing minimum cohort sizes."
            + v_suffix
        ),
        proposal=ctx.split_config.model_dump(),
        tables={
            "cohort_counts": cohort_table,
            "target_rate_per_month": monthly,
        },
    )
    decision = hitl.ask_gate(payload)
    hitl.log(ctx.store.root / "hitl", decision)
    return decision


def _resolve_h2_revision(
    ctx: RunContext,
    settings: Settings,
    payload: dict[str, Any],
) -> SplitConfig:
    """Turn an H2 revise payload into a validated ``SplitConfig``.

    Three input shapes are accepted:

    1. Empty payload — keep the current split.
    2. A dict of explicit ``SplitConfig`` field overrides (e.g. produced by
       a GUI form) — merged on top of the current split.
    3. ``{"_nl_request": "<free text>"}`` — dispatched to the ``split_reviser``
       LLM agent together with the monthly stats and current cohort sanity
       report; the resulting structured ``SplitConfig`` is returned.
    """

    current = ctx.split_config or SplitConfig()
    if not payload:
        return current

    nl = payload.get("_nl_request")
    if isinstance(nl, str) and nl.strip():
        monthly = tools.monthly_target_rate(ctx)
        sanity = tools.split_sanity(ctx, _MIN_ROWS_PER_COHORT, _MIN_POSITIVE_PER_COHORT)
        reviser_payload = {
            "current_split": current.model_dump(),
            "monthly_table": monthly.reset_index().to_dict(orient="records"),
            "cohort_stats": {
                "counts": sanity["counts"],
                "positives": sanity["positives"],
            },
            "thresholds": {
                "min_rows_per_cohort": _MIN_ROWS_PER_COHORT,
                "min_positive_per_cohort": _MIN_POSITIVE_PER_COHORT,
            },
            "user_request": nl.strip(),
        }
        revised = revise_split_config(settings, reviser_payload, fallback=current)
        print(f"[H2] split_reviser -> {revised.model_dump()}")
        return revised

    # Structured overrides (e.g. from a GUI) — merge onto the current config.
    merged = {**current.model_dump(), **payload}
    try:
        return SplitConfig.model_validate(merged)
    except Exception as e:  # noqa: BLE001
        print(f"[H2] invalid structured override ({e}); keeping current split.")
        return current


# --------------------------------------------------------------------------- #
# Phase 4 — binning + H3
# --------------------------------------------------------------------------- #


def _phase_4_binning(ctx: RunContext) -> None:
    cfg = ctx.binning_config or BinningConfig()
    ctx.binning_config = cfg
    bt = tools.run_binning(ctx, cfg)
    ctx.store.save_parquet("binning/binning_tables.parquet", bt)
    if ctx.bin_dict is not None:
        ctx.store.save_pickle("binning/binning_process.pkl", ctx.binning_process)
    ctx.store.save_json("binning/binning_feature_issues.json", ctx.binning_issues)
    print(f"[phase 4] binned {len(ctx.cols_feat_woe)} features (issues: {len(ctx.binning_issues)})")


def _gate_h3(ctx: RunContext, hitl: HitlInterface) -> HitlDecision:
    assert ctx.binning_tables is not None and ctx.data_contract is not None
    bt = ctx.binning_tables
    # Compact summary: one row per feature with bins and IV sum.
    try:
        iv_col = next((c for c in bt.columns if c.lower() in {"iv", "information_value"}), None)
        summary = (
            bt.assign(_iv=bt[iv_col] if iv_col else 0.0)
            .groupby(level=0)["_iv"].sum()
            .sort_values(ascending=False)
            .head(15)
            .to_frame(name="iv_total")
        )
    except Exception:  # noqa: BLE001
        summary = bt.head(15)

    # Write the per-bin detail table out to CSV so the reviewer can open it in
    # a spreadsheet alongside the terminal and judge whether any feature needs
    # a splits override. ``get_binning_tables_from_bp`` row-binds the per-
    # feature optbinning tables (Bin / Count / WoE / Event rate / IV / ...)
    # that a human actually wants to inspect — much richer than the per-
    # feature summary held in ``ctx.binning_tables``.
    detail_path = _write_h3_binning_detail_csv(ctx)

    payload = GatePayload(
        gate_id=GateId.H3,
        title="binning tables",
        summary=(
            f"The detailed per-bin binning table (Bin / Count / WoE / Event "
            f"rate / IV for every feature) has been written to "
            f"`{detail_path}`. Open that CSV in a spreadsheet to decide "
            "whether any feature's bins need to be adjusted. Then come back "
            "to the terminal and either approve the current splits, or "
            "describe your change in plain English — for example: \"merge "
            "the last two bins of AverageMInFile\" or \"use 5, 10 and 20 as "
            "the new cut points for NumTotalTrades\". An AI agent will "
            "translate your request into the required override structure."
        ),
        proposal={
            "n_features_woe": len(ctx.cols_feat_woe),
            "issues": ctx.binning_issues[:10],
            "binning_detail_csv": str(detail_path),
        },
        tables={"top_features_by_iv": summary},
    )
    decision = hitl.ask_gate(payload)
    hitl.log(ctx.store.root / "hitl", decision)
    return decision


def _write_h3_binning_detail_csv(ctx: RunContext) -> Path:
    """Materialize the per-bin detail table as a CSV for the H3 reviewer.

    Uses :func:`scorecard.tools.binning_detail_table` (a thin wrapper over
    ``agent_tools.get_binning_tables_from_bp``) so every feature's optbinning
    output is row-bound into a single, spreadsheet-friendly file under
    ``binning/binning_tables_detailed.csv``. Falls back to an empty CSV with
    a descriptive message if the binning process is not available.
    """

    try:
        df = tools.binning_detail_table(ctx)
    except Exception as e:  # noqa: BLE001
        df = pd.DataFrame({"error": [f"get_binning_tables_from_bp failed: {e}"]})
    if df is None or len(df) == 0:
        df = pd.DataFrame({"note": ["no binning tables available for this run"]})
    return ctx.store.save_csv("binning/binning_tables_detailed.csv", df, index=False)


def _resolve_h3_revision(
    ctx: RunContext,
    settings: Settings,
    payload: dict[str, Any],
) -> BinningRevision:
    """Translate an H3 revise payload into a ``BinningRevision``.

    Accepts ``{"_nl_request": "..."}`` and hands the user's request to the
    binning reviser agent, seeded with the current per-feature bin summary.
    """

    fallback = BinningRevision()
    if not payload:
        return fallback
    nl = payload.get("_nl_request") if isinstance(payload, dict) else None
    if not isinstance(nl, str) or not nl.strip():
        nl = json.dumps(payload, default=str)
    current_bins = _summarise_current_bins(ctx)
    reviser_payload = {
        "current_bins": current_bins,
        "user_request": nl.strip(),
    }
    revised = revise_binning(settings, reviser_payload, fallback=fallback)
    print(f"[H3] binning_reviser -> overrides for {list(revised.overrides.keys())}")
    return revised


def _summarise_current_bins(ctx: RunContext) -> list[dict[str, Any]]:
    """Build a compact ``current_bins`` list for the H3 reviser.

    Each entry is ``{"feature": str, "dtype": "numerical"|"categorical",
    "current_splits": <list>}``. We prefer the serialized ``ctx.bin_dict``
    splits since they are already in the exact shape ``modify_optimal_bin``
    expects; we fall back to the binning table summary when needed.
    """

    rows: list[dict[str, Any]] = []
    bin_dict = ctx.bin_dict or {}
    bt = ctx.binning_tables
    cat_set: set[str] = set()
    if ctx.binning_config is not None:
        cat_set = set(ctx.binning_config.categorical_features)
    if bt is not None and "dtype" in bt.columns:
        try:
            for name, dtype in bt["dtype"].items():  # type: ignore[assignment]
                if isinstance(dtype, str) and dtype.startswith("cat"):
                    cat_set.add(str(name))
        except Exception:  # noqa: BLE001
            pass
    for feat, splits in bin_dict.items():
        dtype = "categorical" if feat in cat_set else "numerical"
        rows.append({
            "feature": str(feat),
            "dtype": dtype,
            "current_splits": splits,
        })
    return rows


# --------------------------------------------------------------------------- #
# Phases 5-6 — multi-branch feature search + training
# --------------------------------------------------------------------------- #


def _phase_5_and_6_feature_search_and_train(ctx: RunContext) -> None:
    cfg = ctx.feature_search_config or tools.default_recipes()
    ctx.feature_search_config = cfg
    ctx.store.save_model("feature_search_config.json", cfg)
    ctx.branches = []
    ctx.branch_models.clear()

    disabled = set(ctx.constraint_spec.disable_recipes)
    recipes = [r for r in cfg.recipes if r.recipe_id not in disabled]
    if ctx.constraint_spec.max_corr is not None:
        for r in recipes:
            r.kwargs["max_corr"] = float(ctx.constraint_spec.max_corr)

    forbidden = set(ctx.problem_contract.forbidden_features if ctx.problem_contract else [])
    forbidden |= set(ctx.constraint_spec.add_forbidden_features)
    forced = set(ctx.constraint_spec.add_forced_features) | set(
        ctx.problem_contract.forced_features if ctx.problem_contract else []
    )

    for recipe in recipes:
        branch_id = f"br_{recipe.recipe_id}"
        try:
            selected, _info = tools.run_recipe(ctx, recipe)
        except Exception as e:  # noqa: BLE001
            print(f"[branch {branch_id}] recipe failed: {e}")
            ctx.branches.append(
                BranchResult(
                    branch_id=branch_id,
                    recipe_id=recipe.recipe_id,
                    features_woe=[],
                    metrics=tools.BranchMetrics(),
                    passed_filters=False,
                    filter_notes=[f"recipe error: {e}"],
                )
            )
            continue
        # apply forbidden / forced overrides (translated to *_woe names).
        forbidden_woe = {f"{c}_woe" for c in forbidden} | forbidden
        forced_woe = {f"{c}_woe" for c in forced} | forced
        selected = [c for c in selected if c not in forbidden_woe]
        for f in forced_woe:
            if f in ctx.cols_feat_woe and f not in selected:
                selected.append(f)
        if ctx.constraint_spec.max_features is not None:
            selected = selected[: int(ctx.constraint_spec.max_features)]
        branch = tools.train_branch(
            ctx, branch_id=branch_id, recipe_id=recipe.recipe_id, features_woe=selected
        )
        ctx.branches.append(branch)
        ctx.store.save_model(f"branches/{branch_id}.json", branch)
        print(
            f"[branch {branch_id}] n_features={len(branch.features_woe)} "
            f"gini_valid={branch.metrics.gini_valid} gini_test={branch.metrics.gini_test}"
        )


def _phase_7_rank(ctx: RunContext, settings: Settings) -> ProposalBundle:
    weights = RankerWeights()
    problem = ctx.problem_contract or ProblemContract()
    assert ctx.df_work is not None and ctx.data_contract is not None

    # Hard filters
    feasible: list[BranchResult] = []
    for b in ctx.branches:
        notes: list[str] = list(b.filter_notes)
        ok = b.passed_filters
        if not b.features_woe:
            ok = False
            notes.append("no features")
        if b.metrics.gini_valid is None:
            ok = False
            notes.append("no valid gini")
        elif b.metrics.gini_valid < problem.min_gini_valid:
            ok = False
            notes.append(f"gini_valid {b.metrics.gini_valid:.3f} < {problem.min_gini_valid}")
        b.passed_filters = ok
        b.filter_notes = notes
        if ok:
            feasible.append(b)

    def score(b: BranchResult) -> float:
        g_valid = b.metrics.gini_valid or 0.0
        g_test = b.metrics.gini_test if b.metrics.gini_test is not None else g_valid
        emphasis = ctx.constraint_spec.emphasize_cohort
        w_valid = weights.w_valid_gini + (0.2 if emphasis == "valid" else 0.0)
        w_test = weights.w_test_gini + (0.2 if emphasis in {"test", "oot"} else 0.0)
        parsimony = weights.w_parsimony / max(1, len(b.features_woe))
        return w_valid * g_valid + w_test * g_test + parsimony

    feasible.sort(key=score, reverse=True)
    if not feasible:
        raise RuntimeError("No branch passed the hard filters. Relax thresholds or revise H4.")
    champion = feasible[0]
    alternates = [b.branch_id for b in feasible[1:4]]

    rationale_payload = {
        "champion": champion.model_dump(),
        "shortlist": [b.model_dump() for b in feasible[:4]],
        "constraint_spec": ctx.constraint_spec.model_dump(),
    }
    rationale = ranking_rationale(settings, rationale_payload)

    proposal = ProposalBundle(
        champion_branch_id=champion.branch_id,
        alternate_branch_ids=alternates,
        branches=feasible,
        ranker_weights=weights,
        rationale=rationale or "(no rationale)",
    )
    ctx.proposal = proposal
    ctx.store.save_model("proposal.json", proposal)
    return proposal


def _gate_h4(ctx: RunContext, hitl: HitlInterface, proposal: ProposalBundle) -> HitlDecision:
    rows = [
        {
            "branch_id": b.branch_id,
            "recipe": b.recipe_id,
            "n_features": len(b.features_woe),
            "gini_valid": b.metrics.gini_valid,
            "gini_test": b.metrics.gini_test,
            "passed": b.passed_filters,
            "notes": "; ".join(b.filter_notes)[:80],
        }
        for b in proposal.branches[:8]
    ]
    summary = (
        f"Champion: {proposal.champion_branch_id}.\n\n"
        f"Rationale:\n{proposal.rationale}\n\n"
        "Approve the champion or describe what you want changed in plain "
        "English — for example: \"keep at most 8 features with correlation "
        "below 0.4 and weight OOT more heavily\", \"forbid "
        "NumTrades60Ever2DerogPubRec\", or \"don't use the IV recipe\". An "
        "AI agent will translate your request into a ConstraintSpec."
    )
    payload = GatePayload(
        gate_id=GateId.H4,
        title="model proposal",
        summary=summary,
        proposal={"champion": proposal.champion_branch_id, "alternates": proposal.alternate_branch_ids},
        tables={"branches": pd.DataFrame(rows)},
    )
    decision = hitl.ask_gate(payload)
    hitl.log(ctx.store.root / "hitl", decision)
    return decision


def _resolve_h4_revision(
    ctx: RunContext,
    settings: Settings,
    payload: dict[str, Any],
) -> ConstraintSpec:
    """Translate an H4 revise payload into a merged ``ConstraintSpec``.

    The reviser is seeded with the current constraints, the list of recent
    branches (for feature / recipe context), and the full feature + recipe
    catalogs so it can resolve plain-English references. Lists returned by
    the agent extend the existing ``ConstraintSpec`` lists rather than
    replace them, preserving prior-iteration decisions.
    """

    current = ctx.constraint_spec
    if not payload:
        return current
    nl = payload.get("_nl_request") if isinstance(payload, dict) else None
    if not isinstance(nl, str) or not nl.strip():
        nl = json.dumps(payload, default=str)
    assert ctx.data_contract is not None
    dc = ctx.data_contract
    branches_summary = [
        {
            "branch_id": b.branch_id,
            "recipe": b.recipe_id,
            "n_features": len(b.features_woe),
            "gini_valid": b.metrics.gini_valid,
            "gini_test": b.metrics.gini_test,
            "passed": b.passed_filters,
        }
        for b in ctx.branches
    ]
    recipe_catalog = sorted({b.recipe_id for b in ctx.branches})
    if ctx.feature_search_config is not None:
        recipe_catalog = sorted(
            {*recipe_catalog, *(r.recipe_id for r in ctx.feature_search_config.recipes)}
        )
    reviser_payload = {
        "current": current.model_dump(),
        "branches": branches_summary,
        "feature_catalog": list(dc.cols_feat),
        "recipe_catalog": recipe_catalog,
        "user_request": nl.strip(),
    }
    revised = revise_constraint_spec(settings, reviser_payload, fallback=current)
    # Preserve accumulated list-fields across iterations: UNION with current.
    cur_dump = current.model_dump()
    new_dump = revised.model_dump()
    for list_field in (
        "prefer_recipes",
        "disable_recipes",
        "add_forbidden_features",
        "add_forced_features",
    ):
        merged = list({*cur_dump[list_field], *new_dump[list_field]})
        new_dump[list_field] = merged
    try:
        return ConstraintSpec.model_validate(new_dump)
    except Exception:  # noqa: BLE001
        return revised


# --------------------------------------------------------------------------- #
# Phase 9 — scorecard + H5
# --------------------------------------------------------------------------- #


def _phase_9_scorecard(ctx: RunContext) -> None:
    assert ctx.proposal is not None
    champion = next(b for b in ctx.proposal.branches if b.branch_id == ctx.proposal.champion_branch_id)
    if ctx.pdo_params is None:
        ctx.pdo_params = PdoParams()
    logistic_hp = LogisticHyperparams()
    points = tools.build_scorecard(ctx, champion, ctx.pdo_params, logistic_hp)
    ctx.store.save_parquet("scorecard/points_table.parquet", points)
    ctx.store.save_model("scorecard/pdo_params.json", ctx.pdo_params)
    ctx.store.save_model("scorecard/logistic_hyperparams.json", logistic_hp)
    print(f"[phase 9] scorecard built — points table rows: {len(points)}")


def _gate_h5(ctx: RunContext, hitl: HitlInterface) -> HitlDecision:
    assert ctx.scorecard_points is not None and ctx.pdo_params is not None
    pts = ctx.scorecard_points
    score_col = next(
        (c for c in pts.columns if c.lower() in {"points", "score", "point"}), pts.columns[-1]
    )
    summary_table = pts[[score_col]].describe().T if score_col in pts.columns else pts.head()
    payload = GatePayload(
        gate_id=GateId.H5,
        title="PDO + points table",
        summary=(
            "Approve the PDO scaling and the business readability of the points table. "
            "To revise, describe the scale in plain English — for example: "
            "\"center around 660 with 30 points doubling the odds\", \"use "
            "reference odds of 72:1\"."
        ),
        proposal=ctx.pdo_params.model_dump(),
        tables={"points_summary": summary_table},
    )
    decision = hitl.ask_gate(payload)
    hitl.log(ctx.store.root / "hitl", decision)
    return decision


def _resolve_h5_revision(
    ctx: RunContext,
    settings: Settings,
    payload: dict[str, Any],
) -> PdoParams:
    """Translate an H5 revise payload into a ``PdoParams``."""

    current = ctx.pdo_params or PdoParams()
    if not payload:
        return current
    nl = payload.get("_nl_request") if isinstance(payload, dict) else None
    if not isinstance(nl, str) or not nl.strip():
        nl = json.dumps(payload, default=str)
    reviser_payload = {
        "current": current.model_dump(),
        "user_request": nl.strip(),
    }
    revised = revise_pdo_params(settings, reviser_payload, fallback=current)
    print(f"[H5] pdo_reviser -> {revised.model_dump()}")
    return revised


# --------------------------------------------------------------------------- #
# Phase 9b — production code export
# --------------------------------------------------------------------------- #


def _phase_9b_production_code(ctx: RunContext) -> None:
    """Emit a standalone ``.py`` scorer for the approved champion.

    Written once H5 is approved so the ``PdoParams`` and ``Scorecard`` are
    final. The file is self-contained (only depends on ``numpy``) and is what
    downstream services should deploy.
    """

    out_path = ctx.store.root / "production" / "scoring.py"
    written = tools.export_production_code(ctx, out_path)
    ctx.production_code_path = written
    try:
        rel = written.relative_to(ctx.store.root)
    except ValueError:
        rel = written
    print(f"[phase 9b] production scorer written: {rel}")


# --------------------------------------------------------------------------- #
# Phase 10 — validation
# --------------------------------------------------------------------------- #


def _phase_10_validation(ctx: RunContext) -> None:
    out = tools.run_validation(ctx)
    for name, df in out.items():
        ctx.store.save_parquet(f"validation/{name}.parquet", df)
    print(f"[phase 10] validation tables: {list(out.keys())}")


# --------------------------------------------------------------------------- #
# Phase 12 — model documentation
# --------------------------------------------------------------------------- #


REQUIRED_HEADINGS = [
    "## 1. EDA",
    "## 2. Data split statistics",
    "## 3. Feature transformation statistics",
    "## 4. Feature selection statistics",
    "## 5. Model performance",
    "## 6. Model stability",
    "## 7. Feature stability",
    "## 8. Final scorecard",
    "## 9. Problematic features",
    "## 10. Other important content",
]


def _df_to_md_table(df: pd.DataFrame) -> str:
    """Render ``df`` as a GitHub-flavored markdown pipe table.

    Unlike ``DataFrame.to_markdown`` (which silently requires the optional
    ``tabulate`` dependency and raises ``ImportError`` when it is missing) we
    always produce a real markdown table:

    * A non-default (i.e. labeled) index is promoted to a leading column so
      the month / feature labels survive in the rendered output.
    * A plain ``RangeIndex`` is dropped — the row numbers it contains add no
      information and made the old output noisy.
    * ``tabulate`` is used when available for nicer alignment; otherwise a
      deterministic manual renderer emits the same structure. Either way the
      caller gets markdown, never plain ``to_string`` whitespace output.
    """

    if df is None or len(df) == 0:
        return "_(empty)_"

    view = df.copy()
    if isinstance(view.index, pd.RangeIndex):
        view = view.reset_index(drop=True)
    else:
        view = view.reset_index()

    try:
        return view.to_markdown(index=False)
    except Exception:  # noqa: BLE001 -- tabulate missing: fall back to manual
        pass

    def _cell(value: Any) -> str:
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):  # unhashable / array-like scalar
            pass
        if isinstance(value, float):
            return f"{value:g}"
        text = str(value)
        return text.replace("|", r"\|").replace("\n", " ").strip()

    headers = [_cell(c) for c in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)


def _phase_12_model_docs(ctx: RunContext, settings: Settings) -> None:
    assert ctx.proposal is not None and ctx.data_contract is not None
    dc = ctx.data_contract
    champion = next(b for b in ctx.proposal.branches if b.branch_id == ctx.proposal.champion_branch_id)

    # --- evidence blocks -------------------------------------------------- #
    # ``rows=None`` (default) renders the whole DataFrame. We deliberately do
    # **not** truncate the timely tables (target rate, score / feature PSI,
    # discrimination over time) nor the binning table: otherwise the 20-row
    # head silently drops later months / features and the generated
    # documentation looks like it only covers a single period.
    #
    # Rendering is delegated to ``_df_to_md_table`` which always emits a
    # GitHub-flavored markdown pipe table (even when the optional
    # ``tabulate`` package is missing) so the documentation never degrades
    # to space-padded ``to_string`` output.
    def _mdtable(df: pd.DataFrame, rows: int | None = None) -> str:
        if df is None or len(df) == 0:
            return "_(empty)_"
        view = df if rows is None else df.head(rows)
        return _df_to_md_table(view)

    eda_nan = _mdtable(_safe_parquet(ctx.store.path("eda/nan_overall.parquet")))
    target_timely = _mdtable(_safe_parquet(ctx.store.path("eda/target_timely.parquet")))
    counts = ctx.df_work[dc.col_type].value_counts().to_frame("count")
    target_by_type = (
        ctx.df_work.groupby(dc.col_type)[dc.col_target].mean().to_frame("target_rate")
    )
    split_stats = counts.join(target_by_type)
    binning_tbl = _mdtable(ctx.binning_tables if ctx.binning_tables is not None else pd.DataFrame())
    branches_df = pd.DataFrame([
        {
            "branch_id": b.branch_id,
            "recipe": b.recipe_id,
            "n_features": len(b.features_woe),
            "gini_valid": b.metrics.gini_valid,
            "gini_test": b.metrics.gini_test,
            "passed": b.passed_filters,
        }
        for b in ctx.proposal.branches
    ])
    score_by_type = _mdtable(ctx.validation_tables.get("score_by_type", pd.DataFrame()))
    score_timely = _mdtable(ctx.validation_tables.get("score_timely", pd.DataFrame()))
    score_psi = _mdtable(ctx.validation_tables.get("score_psi_timely", pd.DataFrame()))
    feat_psi = _mdtable(ctx.validation_tables.get("feature_woe_psi_timely", pd.DataFrame()))
    scorecard_pts = _scorecard_points_for_doc(ctx)
    scorecard_table = _mdtable(scorecard_pts)

    headline_gini = champion.metrics.gini_valid

    # --- interpretations from the LLM docs agent ------------------------- #
    interp_prompts = {
        "eda": {"headline": f"rows={len(ctx.df_work)}, target_rate={ctx.df_work[dc.col_target].mean():.3%}", "table_md": eda_nan[:2000]},
        "split": {"table_md": _mdtable(split_stats.reset_index())},
        "selection": {"champion": champion.recipe_id, "table_md": _mdtable(branches_df)},
        "performance": {"headline_gini_valid": headline_gini, "table_md": score_by_type[:2000]},
        "stability": {"table_md": score_psi[:2000]},
    }
    interpretations = docs_interpretations(settings, interp_prompts) or {}

    # --- render ---------------------------------------------------------- #
    exec_summary = (
        f"**Run id:** `{ctx.run_id}`\\\n"
        f"**Data source:** `{ctx.data_source}`\\\n"
        f"**Champion branch:** `{champion.branch_id}` (recipe `{champion.recipe_id}`)\\\n"
        f"**Features:** {len(champion.features_woe)}\\\n"
        f"**Headline Gini (valid):** {headline_gini}\\\n"
        f"**PDO:** base={ctx.pdo_params.base_score if ctx.pdo_params else 'n/a'}, "
        f"pdo={ctx.pdo_params.pdo if ctx.pdo_params else 'n/a'}, "
        f"odds={ctx.pdo_params.odds if ctx.pdo_params else 'n/a'}"
    )
    feat_list = "\n".join(f"- `{f}`" for f in champion.features_woe) or "_(empty)_"

    md = _render_model_doc(
        exec_summary=exec_summary,
        eda_table=eda_nan,
        eda_interp=interpretations.get("eda", ""),
        target_timely=target_timely,
        split_table=_mdtable(split_stats.reset_index()),
        split_interp=interpretations.get("split", ""),
        binning_table=binning_tbl,
        branches_table=_mdtable(branches_df),
        selection_interp=interpretations.get("selection", ""),
        feature_list=feat_list,
        score_by_type=score_by_type,
        score_timely=score_timely,
        performance_interp=interpretations.get("performance", ""),
        score_psi=score_psi,
        stability_interp=interpretations.get("stability", ""),
        feat_psi=feat_psi,
        scorecard_table=scorecard_table,
        binning_issues=ctx.binning_issues,
        champion=champion,
        ctx=ctx,
    )
    path = ctx.store.save_text("model_documentation.md", md)
    ctx.model_documentation_path = path

    # CI heading check
    required_ok = all(h in md for h in REQUIRED_HEADINGS)
    ctx.store.save_model(
        "model_documentation_meta.json",
        ModelDocumentationMeta(
            path=str(path),
            doc_schema_version=1,
            champion_branch_id=champion.branch_id,
            required_headings_ok=required_ok,
        ),
    )
    print(f"[phase 12] model_documentation.md -> {path} (required_headings_ok={required_ok})")


def _scorecard_points_for_doc(ctx: RunContext) -> pd.DataFrame:
    """Return the champion scorecard points table for section 8 of the docs.

    Prefers the in-memory ``ctx.scorecard_points`` (produced by
    ``tools.build_scorecard`` -> ``agent_tools.create_scorecard_model``), and
    falls back to the persisted artifact at
    ``scorecard/points_table.parquet`` so the renderer still works when
    documentation is regenerated in a follow-up H6 pass.
    """

    pts = ctx.scorecard_points
    if pts is None or len(pts) == 0:
        pts = _safe_parquet(ctx.store.path("scorecard/points_table.parquet"))
    if pts is None or len(pts) == 0:
        return pd.DataFrame()
    if "Variable" in pts.columns and "Bin id" in pts.columns:
        pts = pts.sort_values(["Variable", "Bin id"]).reset_index(drop=True)
    return pts


def _safe_parquet(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        try:
            csv = path.with_suffix(".csv")
            if csv.exists():
                return pd.read_csv(csv)
        except Exception:  # noqa: BLE001
            pass
    return pd.DataFrame()


def _render_model_doc(
    *,
    exec_summary: str,
    eda_table: str,
    eda_interp: str,
    target_timely: str,
    split_table: str,
    split_interp: str,
    binning_table: str,
    branches_table: str,
    selection_interp: str,
    feature_list: str,
    score_by_type: str,
    score_timely: str,
    performance_interp: str,
    score_psi: str,
    stability_interp: str,
    feat_psi: str,
    scorecard_table: str,
    binning_issues: Iterable[str],
    champion: BranchResult,
    ctx: RunContext,
) -> str:
    issues_md = "\n".join(f"- {x}" for x in binning_issues) or "- None identified"
    pdo_line = (
        f"base_score={ctx.pdo_params.base_score if ctx.pdo_params else 'n/a'}, "
        f"pdo={ctx.pdo_params.pdo if ctx.pdo_params else 'n/a'}, "
        f"odds={ctx.pdo_params.odds if ctx.pdo_params else 'n/a'}"
    )
    if ctx.production_code_path is not None:
        try:
            rel_code = ctx.production_code_path.relative_to(ctx.store.root)
        except ValueError:
            rel_code = ctx.production_code_path
        production_code_line = (
            f"**Production code:** standalone scorer written to "
            f"`{rel_code}` (numpy-only; exposes ``get_score`` / ``get_woe`` over "
            f"``dict[str, Any]`` payloads — drop this file into the serving "
            f"service without the optbinning runtime)."
        )
    else:
        production_code_line = (
            "**Production code:** not yet generated for this run."
        )
    return f"""# Model documentation

{exec_summary}

## 1. EDA

{eda_interp}

### Missing rates (top)

{eda_table}

### Target rate over time

{target_timely}

## 2. Data split statistics

{split_interp}

{split_table}

## 3. Feature transformation statistics

Optbinning ``BinningProcess`` fit on the ``train`` cohort; WoE columns
derived via ``get_woe_from_bp``. Top entries of the binning table:

{binning_table}

## 4. Feature selection statistics

{selection_interp}

### All branches

{branches_table}

### Champion features

{feature_list}

## 5. Model performance

{performance_interp}

### Discrimination by cohort

{score_by_type}

### Discrimination over time

Monthly AUC / Gini are computed on **out-of-sample rows only** (``valid``,
``test``, ``oot`` and ``hoot`` — ``train`` is excluded) so that in-window
months are not inflated by fit-time performance. ``count`` /
``count_positive`` in the table below are therefore the OOS row counts for
each month, not the full population counts shown in §1's target-rate table.

{score_timely}

## 6. Model stability

{stability_interp}

### Score PSI over time

{score_psi}

## 7. Feature stability

### WoE feature PSI over time

{feat_psi}

## 8. Final scorecard

Per-feature, per-bin point allocations produced by ``create_scorecard_model``
in ``agent_tools.py``. Each row shows one bin of one feature with its
``Count`` / ``Event rate`` / ``WoE`` / ``IV`` plus the fitted logistic
``Coefficient`` and the final ``Points`` under the PDO scaling ({pdo_line}).
To score an application, look up the row matching each of the champion
features for the applicant's value and sum the ``Points`` column.

{scorecard_table}

## 9. Problematic features

{issues_md}

## 10. Other important content

**Model form:** binary logistic regression on WoE features with PDO scaling
({pdo_line}).

**Reproducibility:** run_id=`{ctx.run_id}`; artifacts at `{ctx.store.root}`;
data=`{ctx.data_source}`.

**Champion vs alternates:** champion=`{champion.branch_id}`; alternates=`{', '.join(ctx.proposal.alternate_branch_ids) if ctx.proposal else ''}`.

**Deployment notes:** input features map through the frozen ``BinningProcess``;
score column name=`{ctx.data_contract.col_score if ctx.data_contract else 'score'}`.

{production_code_line}

**Limitations:** sample size is limited to the data window; OOT horizon is
bounded by the ``SplitConfig``; known data defects are listed under
problematic features.
"""


def _gate_h6(ctx: RunContext, hitl: HitlInterface) -> HitlDecision:
    assert ctx.model_documentation_path is not None and ctx.proposal is not None
    doc_preview = ctx.model_documentation_path.read_text(encoding="utf-8")[:1500]
    payload = GatePayload(
        gate_id=GateId.H6,
        title="final model acceptance",
        summary=(
            "Review model_documentation.md. Approve to package the bundle. "
            "Revise re-runs the documentation writer (numbers unchanged)."
        ),
        proposal={"doc_path": str(ctx.model_documentation_path)},
        tables={"doc_preview (first 1500 chars)": pd.DataFrame({"excerpt": [doc_preview]})},
    )
    decision = hitl.ask_gate(payload)
    hitl.log(ctx.store.root / "hitl", decision)
    return decision


# --------------------------------------------------------------------------- #
# Phase 11 — package
# --------------------------------------------------------------------------- #


def _phase_11_package(ctx: RunContext) -> RunManifest:
    manifest = ctx.manifest()
    manifest.approved = True
    if ctx.model_documentation_path is not None:
        manifest.model_documentation = ModelDocumentationMeta(
            path=str(ctx.model_documentation_path),
            champion_branch_id=manifest.champion_branch_id or "",
        )
    if ctx.production_code_path is not None:
        manifest.production_code_path = str(ctx.production_code_path)
    ctx.store.save_model("run_manifest.json", manifest)
    return manifest
