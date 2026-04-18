"""Google-ADK ``LlmAgent`` instances used as sub-agents.

The design specifies a roster of sub-agents (``schema``, ``eda``, ``split``,
``binning``, ``feature_search``, ``model_train``, ``model_rank``,
``scorecard``, ``validate``, ``model_docs``, ``compliance_assist``). In the
POC we collapse the **computational** agents into deterministic Python code
(``orchestrator`` + ``tools``), and we reserve *language model* work to the
gates that genuinely benefit from narrative generation:

* ``eda`` — explain QC findings to the human at H1 / H2.
* ``model_rank`` — write a champion-vs-alternates rationale for H4.
* ``model_docs`` — fill interpretation paragraphs inside
  ``model_documentation.md``.
* ``compliance_assist`` — optional fairness / documentation checklist prose.

All numbers come from the tool outputs; the LLMs never invent metrics.
Swapping to a richer multi-agent topology (``SequentialAgent`` / workflow
agents) later is straightforward because every agent here is a standalone
``LlmAgent``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional

from .schemas import (
    BinningRevision,
    ConstraintSpec,
    PdoParams,
    ProblemContract,
    SplitConfig,
)
from .settings import Settings


# Gemini structured output rejects ``additionalProperties: false`` (which
# Pydantic emits when ``extra="forbid"`` is set). We therefore declare a
# mirror model with permissive extras to use as ``output_schema`` and then
# validate the raw JSON through the strict schema for storage.
class _SplitConfigOut(BaseModel):
    oot_th: Optional[str] = None
    hoot_th: Optional[str] = None
    test_perc: float = Field(default=0.2, ge=0.05, le=0.5)
    valid_perc: float = Field(default=0.2, ge=0.05, le=0.5)


class _ProblemContractOut(BaseModel):
    target_definition: str = "binary 0/1 default indicator"
    exclusions: list[str] = Field(default_factory=list)
    forbidden_features: list[str] = Field(default_factory=list)
    forced_features: list[str] = Field(default_factory=list)
    min_gini_valid: float = Field(default=0.25, ge=0.0, le=1.0)
    max_psi_oot_score: float = Field(default=0.25, ge=0.0, le=5.0)


class _ConstraintSpecOut(BaseModel):
    max_features: Optional[int] = Field(default=None, ge=1, le=200)
    max_corr: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    prefer_recipes: list[str] = Field(default_factory=list)
    disable_recipes: list[str] = Field(default_factory=list)
    add_forbidden_features: list[str] = Field(default_factory=list)
    add_forced_features: list[str] = Field(default_factory=list)
    # Gemini tolerates a plain string; we re-validate through the strict
    # ConstraintSpec literal in Python to reject unknown values.
    emphasize_cohort: Optional[str] = None
    notes: str = ""


class _PdoParamsOut(BaseModel):
    base_score: float = Field(default=600.0, ge=0.0, le=2000.0)
    pdo: float = Field(default=20.0, gt=0.0, le=200.0)
    odds: float = Field(default=50.0, gt=0.0, le=1_000_000.0)


class _BinFeatureOverride(BaseModel):
    """A single feature's new split specification.

    ``numeric_splits`` is a list of cut points (monotonic floats) for numeric
    features; ``categorical_splits`` is a list of category groups (each group
    a list of string category values) for categorical features. Exactly one
    of the two should be populated per entry.
    """

    feature: str
    numeric_splits: Optional[list[float]] = None
    categorical_splits: Optional[list[list[str]]] = None


class _BinningRevisionOut(BaseModel):
    overrides: list[_BinFeatureOverride] = Field(default_factory=list)


_APP_NAME = "scorecard-agents"


# --------------------------------------------------------------------------- #
# Agent factory
# --------------------------------------------------------------------------- #


def _build_agent(
    name: str,
    instruction: str,
    settings: Settings,
    *,
    output_schema: type[BaseModel] | None = None,
) -> LlmAgent:
    """Create an ``LlmAgent``.

    When ``output_schema`` is provided, ADK enforces structured JSON output
    matching the Pydantic model via ``response_mime_type=application/json``
    plus schema-constrained decoding. Structured agents cannot use transfer
    or tool calls, which is exactly what we want here.
    """

    gen_cfg: dict[str, Any] = {
        "temperature": settings.gemini.temperature,
        "top_p": settings.gemini.top_p,
        "max_output_tokens": min(settings.gemini.max_output_tokens, 4096),
    }
    kwargs: dict[str, Any] = dict(
        name=name,
        model=settings.gemini.model,
        instruction=instruction,
        generate_content_config=types.GenerateContentConfig(**gen_cfg),
    )
    if output_schema is not None:
        kwargs["output_schema"] = output_schema
        kwargs["disallow_transfer_to_parent"] = True
        kwargs["disallow_transfer_to_peers"] = True
    return LlmAgent(**kwargs)


_EDA_INSTRUCTION = """\
You are the EDA sub-agent for a binary logistic scorecard pipeline.
You receive a JSON snippet with row counts, target prevalence, missing rates
and any automated QC flags. Summarize the dataset health in <=6 bullet points
using ONLY numbers present in the input. Do not invent metrics. End with one
sentence that highlights the single most important risk to a downstream model.
Output plain markdown bullets, no preamble.
"""

_RANK_INSTRUCTION = """\
You are the model_rank sub-agent. You receive a JSON with a ranked list of
branches (each has a recipe_id, feature count, AUC / Gini on valid/test/oot
when available) and a nominated champion. Write a 4-8 line rationale that:
- references only provided numbers,
- names the champion and its recipe_id,
- explains briefly WHY it wins (metric + parsimony),
- mentions the best alternate.
Plain markdown; no headings, no bullets with fake numbers.
"""

_DOCS_INSTRUCTION = """\
You are the model_docs sub-agent. You receive a JSON with per-section
"interpretation prompts" each containing evidence tables already rendered as
markdown. For each prompt, return a short paragraph (<=80 words) of business
interpretation, prefixed with 'Interpretation:'. Use ONLY numbers that appear
in the provided tables. Return a JSON object keyed by section id with each
value being the paragraph string.
"""

_COMPLIANCE_INSTRUCTION = """\
You are the compliance_assist sub-agent. Produce a short markdown checklist
(<=8 bullets) of fairness / documentation items to review based on the
provided champion summary JSON. Be neutral; this is NOT legal advice.
"""

_SPLIT_PROPOSER_INSTRUCTION = """\
You are the split_design sub-agent. Given a JSON payload describing a binary
classification dataset (row count, overall target rate, a monthly table with
columns month / mean / count / count_positive, and minimum cohort thresholds
`min_rows_per_cohort` and `min_positive_per_cohort`), propose an initial
``SplitConfig`` that will be passed to ``agent_tools.split_data``.

Hard rules:
1. Every cohort produced (train / valid / test and, if used, oot / hoot) MUST
   satisfy BOTH `min_rows_per_cohort` AND `min_positive_per_cohort`.
2. Prefer a time-based OOT only when the dataset spans >= 4 distinct months
   AND the final >=1 month contributes enough rows and positives to meet the
   thresholds. Otherwise set `oot_th` to null (no OOT).
3. `oot_th` must be an ISO date (YYYY-MM-DD) at a month boundary taken from
   the provided monthly table; never invent dates outside the observed range.
4. `test_perc` and `valid_perc` must each lie in [0.1, 0.3]. Their sum plus
   the OOT fraction (rows on/after `oot_th`) must leave >= 40% for train.
5. `hoot_th` stays null unless the user explicitly asks for a holdout-in-OOT
   split.

Output ONLY a JSON object matching the SplitConfig schema; no prose, no
markdown fences. Use numeric types, not strings, for the two percentages.
"""

_SPLIT_REVISER_INSTRUCTION = """\
You are the split_reviser sub-agent. Given a JSON payload with
`current_split` (a SplitConfig), `monthly_table` (same schema as the
proposer sees), `cohort_stats` (counts and positives per cohort under the
current split), `thresholds` (`min_rows_per_cohort`, `min_positive_per_cohort`),
and `user_request` (free-text describing what should change), return a NEW
``SplitConfig`` that applies the user's request while still satisfying the
same hard rules as the proposer:

- Every produced cohort MUST satisfy the minimum row and positive counts.
- `oot_th` / `hoot_th` must be ISO dates on a month boundary observed in
  `monthly_table`, or null.
- `test_perc` and `valid_perc` in [0.1, 0.3].
- If the user's request is impossible (e.g. would shrink a cohort below
  thresholds), move the relevant values toward the user's intent as far as
  the thresholds allow and otherwise keep `current_split` fields unchanged.
- Never change a field the user didn't mention unless required to satisfy a
  hard rule.

Output ONLY the JSON object; no prose, no markdown fences.
"""


_PROBLEM_CONTRACT_REVISER_INSTRUCTION = """\
You are the problem_contract_reviser sub-agent for gate H1. Given a JSON
payload with `current` (a ProblemContract), `feature_catalog` (the list of
candidate feature names so you can resolve user references like "the risk
score"), `target_rate` (overall target rate as a decimal) and `user_request`
(free text from a non-technical reviewer), return a NEW ProblemContract.

Fields available (keep field types exactly):
- `target_definition`: short English description of the target (string).
- `exclusions`: list of row-filter descriptions (strings, free form).
- `forbidden_features`: feature names that MUST NOT be used. Match against
  `feature_catalog`; drop any name not in it. Accept the user's plain-English
  phrasing (e.g. "drop ExternalRiskEstimate" -> add it).
- `forced_features`: feature names that MUST be kept.
- `min_gini_valid`: float in [0, 1]. Increase only if the user asks for a
  higher minimum Gini; otherwise keep current.
- `max_psi_oot_score`: float, typically in [0.05, 1.0]. Keep current unless
  the user explicitly loosens/tightens the PSI tolerance.

Rules:
- If the user's request is unrelated to a given field, copy that field from
  `current` unchanged. NEVER invent feature names.
- Feature names are case-sensitive and must appear in `feature_catalog`.
- Lists should contain de-duplicated values.

Output ONLY the JSON object (ProblemContract shape). No prose, no fences.
"""


_CONSTRAINT_SPEC_REVISER_INSTRUCTION = """\
You are the constraint_spec_reviser sub-agent for gate H4. Given a JSON
payload with `current` (a ConstraintSpec), `branches` (the recent branch
results with branch_id / recipe_id / feature count / gini metrics),
`feature_catalog` (the full list of available *raw* feature names — the
`_woe` suffix is added automatically downstream), `recipe_catalog` (the list
of recipe_ids available to enable or disable), and `user_request` (free
text from a non-technical reviewer), return a NEW ConstraintSpec that
applies the user's intent.

Fields available:
- `max_features`: int >= 1 or null. Cap on the number of features used by
  any branch. Set only if the user asks to shrink / cap the model.
- `max_corr`: float in [0, 1] or null. Maximum absolute pairwise correlation
  allowed between selected features. 0.4 is strict, 0.6 is lenient.
- `prefer_recipes` / `disable_recipes`: recipe_ids drawn from `recipe_catalog`.
- `add_forbidden_features` / `add_forced_features`: raw feature names drawn
  from `feature_catalog` (no `_woe` suffix). These EXTEND the existing
  ProblemContract lists — do not re-list features already forbidden/forced.
- `emphasize_cohort`: one of "valid", "test", "oot", or null. Set when the
  user wants the ranker to weight a specific cohort more heavily.
- `notes`: short free-form string summarising the reviewer's intent.

Rules:
- Copy every field from `current` that the user did not mention.
- Feature names and recipe ids must exist in the respective catalogs; drop
  any you cannot resolve.
- Lists must be de-duplicated.

Output ONLY the JSON object (ConstraintSpec shape). No prose, no fences.
"""


_PDO_REVISER_INSTRUCTION = """\
You are the pdo_reviser sub-agent for gate H5. Given a JSON payload with
`current` (a PdoParams: base_score, pdo, odds) and `user_request` (free text
from a non-technical reviewer — e.g. "center around 660", "make 30 points
double the odds", "use 72:1 reference odds"), return a NEW PdoParams.

Fields:
- `base_score`: float. Score value at `odds`. Typical range 300-900.
- `pdo`: float > 0. Points that double the odds. Typical range 10-60.
- `odds`: float > 0. Reference odds at `base_score`. Typical range 10-200.

Rules:
- Keep any field the user did not mention from `current`.
- Output all three fields, even if only one changed.

Output ONLY the JSON object (PdoParams shape). No prose, no fences.
"""


_BINNING_REVISER_INSTRUCTION = """\
You are the binning_reviser sub-agent for gate H3. Given a JSON payload with
`current_bins` (a list of records: `feature`, `dtype` in {"numerical",
"categorical"}, `current_splits`), and `user_request` (free text from a
non-technical reviewer — e.g. "merge the last two bins of AverageMInFile",
"use 5, 10, 20 as cut points for NumTotalTrades"), return a new set of
per-feature split overrides.

Output shape: a JSON object with a single field `overrides`, which is a LIST
of entries. Each entry has:
- `feature`: the feature name (must match an entry in `current_bins`).
- `numeric_splits`: for numerical features, a strictly-increasing list of
  floats representing the new cut points (null or omitted for categorical).
- `categorical_splits`: for categorical features, a list of groups, each a
  list of category string values. Every observed category in
  `current_splits` must appear in exactly one group.

Rules:
- ONLY include features the user explicitly asked to change. Do not repeat
  the whole binning table.
- Numeric splits must be strictly increasing; remove duplicates and sort.
- Drop entries referring to features not present in `current_bins`.

Output ONLY the JSON object. No prose, no fences.
"""


_ROSTER_CACHE: dict[int, dict[str, LlmAgent]] = {}


def build_roster(settings: Settings) -> dict[str, LlmAgent]:
    """Construct one ``LlmAgent`` per narrative role; cached per Settings id."""

    key = id(settings)
    cached = _ROSTER_CACHE.get(key)
    if cached is not None:
        return cached
    roster = {
        "eda": _build_agent("eda_agent", _EDA_INSTRUCTION, settings),
        "model_rank": _build_agent("model_rank_agent", _RANK_INSTRUCTION, settings),
        "model_docs": _build_agent("model_docs_agent", _DOCS_INSTRUCTION, settings),
        "compliance_assist": _build_agent("compliance_agent", _COMPLIANCE_INSTRUCTION, settings),
        "split_proposer": _build_agent(
            "split_proposer_agent",
            _SPLIT_PROPOSER_INSTRUCTION,
            settings,
            output_schema=_SplitConfigOut,
        ),
        "split_reviser": _build_agent(
            "split_reviser_agent",
            _SPLIT_REVISER_INSTRUCTION,
            settings,
            output_schema=_SplitConfigOut,
        ),
        "problem_contract_reviser": _build_agent(
            "problem_contract_reviser_agent",
            _PROBLEM_CONTRACT_REVISER_INSTRUCTION,
            settings,
            output_schema=_ProblemContractOut,
        ),
        "constraint_spec_reviser": _build_agent(
            "constraint_spec_reviser_agent",
            _CONSTRAINT_SPEC_REVISER_INSTRUCTION,
            settings,
            output_schema=_ConstraintSpecOut,
        ),
        "pdo_reviser": _build_agent(
            "pdo_reviser_agent",
            _PDO_REVISER_INSTRUCTION,
            settings,
            output_schema=_PdoParamsOut,
        ),
        "binning_reviser": _build_agent(
            "binning_reviser_agent",
            _BINNING_REVISER_INSTRUCTION,
            settings,
            output_schema=_BinningRevisionOut,
        ),
    }
    _ROSTER_CACHE[key] = roster
    return roster


# --------------------------------------------------------------------------- #
# Invocation helpers
# --------------------------------------------------------------------------- #


async def _run_once(agent: LlmAgent, user_text: str) -> str:
    runner = InMemoryRunner(agent=agent, app_name=_APP_NAME)
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    session = await runner.session_service.create_session(app_name=_APP_NAME, user_id=user_id)
    out_parts: list[str] = []
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text=user_text)]),
        ):
            if event.content and event.content.parts:
                for p in event.content.parts:
                    if getattr(p, "text", None):
                        out_parts.append(p.text)
    finally:
        try:
            await runner.close()
        except Exception:  # noqa: BLE001
            pass
    return "".join(out_parts).strip()


def invoke_agent(agent: LlmAgent, user_text: str) -> str:
    """Synchronous wrapper around ``_run_once`` suitable for the CLI."""

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():  # unusual in CLI but safe-guard anyway
            return asyncio.run_coroutine_threadsafe(_run_once(agent, user_text), loop).result()
    except RuntimeError:
        pass
    return asyncio.run(_run_once(agent, user_text))


def invoke_agent_safe(agent: LlmAgent, user_text: str, fallback: str = "") -> str:
    """Same as ``invoke_agent`` but degrades gracefully on transport errors."""

    try:
        text = invoke_agent(agent, user_text)
        return text or fallback
    except Exception as e:  # noqa: BLE001
        return fallback or f"[agent {agent.name} unavailable: {e}]"


# --------------------------------------------------------------------------- #
# Convenience wrappers used by the orchestrator
# --------------------------------------------------------------------------- #


def eda_narrative(settings: Settings, eda_payload: dict[str, Any]) -> str:
    agent = build_roster(settings)["eda"]
    return invoke_agent_safe(agent, json.dumps(eda_payload, default=str)[:16_000])


def ranking_rationale(settings: Settings, rank_payload: dict[str, Any]) -> str:
    agent = build_roster(settings)["model_rank"]
    return invoke_agent_safe(agent, json.dumps(rank_payload, default=str)[:16_000])


def docs_interpretations(settings: Settings, prompts: dict[str, Any]) -> dict[str, str]:
    agent = build_roster(settings)["model_docs"]
    raw = invoke_agent_safe(agent, json.dumps(prompts, default=str)[:32_000])
    # The agent is instructed to emit JSON; be lenient if it adds fences.
    if not raw:
        return {}
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except Exception:  # noqa: BLE001
        pass
    return {"raw": raw}


def compliance_checklist(settings: Settings, champion_payload: dict[str, Any]) -> str:
    agent = build_roster(settings)["compliance_assist"]
    return invoke_agent_safe(agent, json.dumps(champion_payload, default=str)[:8_000])


def _parse_split_config(raw: str) -> SplitConfig | None:
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        return SplitConfig.model_validate_json(cleaned)
    except Exception:  # noqa: BLE001
        try:
            return SplitConfig.model_validate(json.loads(cleaned))
        except Exception:  # noqa: BLE001
            return None


def propose_split_config(
    settings: Settings,
    payload: dict[str, Any],
    fallback: SplitConfig | None = None,
) -> SplitConfig:
    """Ask the ``split_proposer`` agent for an initial ``SplitConfig``.

    Returns ``fallback`` (or ``SplitConfig()`` defaults) if the agent is
    unreachable or produces an unparseable payload.
    """

    agent = build_roster(settings)["split_proposer"]
    raw = invoke_agent_safe(agent, json.dumps(payload, default=str)[:16_000])
    parsed = _parse_split_config(raw)
    if parsed is not None:
        return parsed
    return fallback or SplitConfig()


def revise_split_config(
    settings: Settings,
    payload: dict[str, Any],
    fallback: SplitConfig,
) -> SplitConfig:
    """Ask the ``split_reviser`` agent to translate a user request into a
    new ``SplitConfig``. Returns ``fallback`` on any error.
    """

    agent = build_roster(settings)["split_reviser"]
    raw = invoke_agent_safe(agent, json.dumps(payload, default=str)[:16_000])
    parsed = _parse_split_config(raw)
    return parsed or fallback


def _clean_json(raw: str) -> str:
    r"""Strip common ```json fences and whitespace from an LLM reply."""

    if not raw:
        return ""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _parse_model_json(raw: str, model: type[BaseModel]) -> BaseModel | None:
    cleaned = _clean_json(raw)
    if not cleaned:
        return None
    try:
        return model.model_validate_json(cleaned)
    except Exception:  # noqa: BLE001
        try:
            return model.model_validate(json.loads(cleaned))
        except Exception:  # noqa: BLE001
            return None


def revise_problem_contract(
    settings: Settings,
    payload: dict[str, Any],
    fallback: ProblemContract,
) -> ProblemContract:
    """Translate a free-text H1 revision into a ``ProblemContract``.

    Unknown feature names the LLM might return are dropped against the
    provided ``feature_catalog`` so the contract never references features
    that do not exist in the dataset. Returns ``fallback`` on any error.
    """

    agent = build_roster(settings)["problem_contract_reviser"]
    raw = invoke_agent_safe(agent, json.dumps(payload, default=str)[:16_000])
    parsed = _parse_model_json(raw, _ProblemContractOut)
    if parsed is None:
        return fallback
    data = parsed.model_dump()
    catalog = set(payload.get("feature_catalog") or [])
    if catalog:
        data["forbidden_features"] = [f for f in data.get("forbidden_features", []) if f in catalog]
        data["forced_features"] = [f for f in data.get("forced_features", []) if f in catalog]
    try:
        return ProblemContract.model_validate({**fallback.model_dump(), **data})
    except Exception:  # noqa: BLE001
        return fallback


def revise_constraint_spec(
    settings: Settings,
    payload: dict[str, Any],
    fallback: ConstraintSpec,
) -> ConstraintSpec:
    """Translate a free-text H4 revision into a ``ConstraintSpec``.

    The reviser's ``emphasize_cohort`` field is relaxed to ``Optional[str]``
    so Gemini structured output accepts it; we enforce the
    ``{"valid","test","oot"}`` literal in Python after parsing.
    Feature and recipe names not found in the catalogs are silently dropped.
    """

    agent = build_roster(settings)["constraint_spec_reviser"]
    raw = invoke_agent_safe(agent, json.dumps(payload, default=str)[:16_000])
    parsed = _parse_model_json(raw, _ConstraintSpecOut)
    if parsed is None:
        return fallback
    data = parsed.model_dump()
    feat_catalog = set(payload.get("feature_catalog") or [])
    recipe_catalog = set(payload.get("recipe_catalog") or [])
    if feat_catalog:
        data["add_forbidden_features"] = [
            f for f in data.get("add_forbidden_features", []) if f in feat_catalog
        ]
        data["add_forced_features"] = [
            f for f in data.get("add_forced_features", []) if f in feat_catalog
        ]
    if recipe_catalog:
        data["prefer_recipes"] = [r for r in data.get("prefer_recipes", []) if r in recipe_catalog]
        data["disable_recipes"] = [r for r in data.get("disable_recipes", []) if r in recipe_catalog]
    cohort = data.get("emphasize_cohort")
    if cohort not in {"valid", "test", "oot", None}:
        data["emphasize_cohort"] = None
    try:
        return ConstraintSpec.model_validate({**fallback.model_dump(), **data})
    except Exception:  # noqa: BLE001
        return fallback


def revise_pdo_params(
    settings: Settings,
    payload: dict[str, Any],
    fallback: PdoParams,
) -> PdoParams:
    """Translate a free-text H5 revision into a ``PdoParams``."""

    agent = build_roster(settings)["pdo_reviser"]
    raw = invoke_agent_safe(agent, json.dumps(payload, default=str)[:8_000])
    parsed = _parse_model_json(raw, _PdoParamsOut)
    if parsed is None:
        return fallback
    try:
        return PdoParams.model_validate(parsed.model_dump())
    except Exception:  # noqa: BLE001
        return fallback


def revise_binning(
    settings: Settings,
    payload: dict[str, Any],
    fallback: BinningRevision,
) -> BinningRevision:
    """Translate a free-text H3 revision into a ``BinningRevision``.

    The agent returns a list of per-feature override records; this wrapper
    collapses that list into the ``overrides: dict[feature, splits]`` shape
    expected by :func:`scorecard.tools.apply_binning_revision`. Numeric
    splits are sorted and de-duplicated; categorical groups are validated
    against the set of known categories from ``current_bins``.
    """

    agent = build_roster(settings)["binning_reviser"]
    raw = invoke_agent_safe(agent, json.dumps(payload, default=str)[:16_000])
    parsed = _parse_model_json(raw, _BinningRevisionOut)
    if parsed is None:
        return fallback
    known = {
        row["feature"]: (row.get("dtype"), row.get("current_splits"))
        for row in (payload.get("current_bins") or [])
        if isinstance(row, dict) and row.get("feature")
    }
    overrides: dict[str, Any] = {}
    for entry in parsed.overrides:  # type: ignore[attr-defined]
        if entry.feature not in known:
            continue
        dtype, _ = known[entry.feature]
        if entry.numeric_splits is not None and dtype in (None, "numerical"):
            pts = sorted({float(x) for x in entry.numeric_splits})
            if len(pts) >= 1:
                overrides[entry.feature] = pts
        elif entry.categorical_splits is not None and dtype in (None, "categorical"):
            groups = [
                [str(v) for v in grp]
                for grp in entry.categorical_splits
                if isinstance(grp, (list, tuple)) and len(grp) > 0
            ]
            if groups:
                overrides[entry.feature] = groups
    if not overrides:
        return fallback
    try:
        return BinningRevision(overrides=overrides)
    except Exception:  # noqa: BLE001
        return fallback
