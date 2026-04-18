"""Pydantic v2 contracts for the scorecard multi-agent system.

Names track §4.1 of the design document. All orchestration state, HITL gate
payloads and persisted artifacts round-trip through these models so that the
CLI, the future GUI backend and any MCP/ADK schemas share one definition.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# --------------------------------------------------------------------------- #
# Column / problem contracts
# --------------------------------------------------------------------------- #


class DataColumnContract(BaseModel):
    """Semantic column map that flows through every phase (§4)."""

    model_config = ConfigDict(extra="forbid")

    col_time: str
    col_target: str
    cols_feat: list[str] = Field(min_length=1)
    col_type: str = "data_type"
    # ``col_month`` is a derived month-start Timestamp column materialized on
    # ``ctx.df_work`` during phase 1. All "timely" operations — target-rate
    # monitoring, PSI (score + feature WoE), score discrimination over time,
    # and the OOT boundary in ``split_data`` — use this column so monthly is
    # the canonical granularity. ``col_day`` is reserved for future sub-monthly
    # analyses (none used today).
    col_month: str = "data_month"
    col_day: Optional[str] = None
    col_score: str = "score"


class ProblemContract(BaseModel):
    """H1 output: business rules + success thresholds."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    target_definition: str = "binary 0/1 default indicator"
    exclusions: list[str] = Field(default_factory=list)
    forbidden_features: list[str] = Field(default_factory=list)
    forced_features: list[str] = Field(default_factory=list)
    min_gini_valid: float = 0.25
    max_psi_oot_score: float = 0.25


# --------------------------------------------------------------------------- #
# Split / binning
# --------------------------------------------------------------------------- #


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    oot_th: Optional[str] = None
    hoot_th: Optional[str] = None
    test_perc: float = 0.2
    valid_perc: float = 0.2


class BinningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_nbin: int = 2
    max_nbin: int = 6
    categorical_features: list[str] = Field(default_factory=list)


class BinningRevision(BaseModel):
    """Per-feature bin override applied via ``modify_optimal_bin``."""

    model_config = ConfigDict(extra="forbid")

    overrides: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Feature search / branches
# --------------------------------------------------------------------------- #


class RecipeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipe_id: str
    tool: Literal[
        "select_features_auc_max_corr",
        "select_features_iv_max_corr",
        "select_features_aic_forward",
        "select_features_aic_backward",
        "select_features_bic_forward",
        "select_features_bic_backward",
        "select_features_auc_forward",
        "select_features_auc_backward",
    ]
    kwargs: dict[str, Any] = Field(default_factory=dict)


class FeatureSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipes: list[RecipeSpec]
    max_features_global: Optional[int] = None


class BranchMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auc_train: float | None = None
    auc_valid: float | None = None
    auc_test: float | None = None
    gini_train: float | None = None
    gini_valid: float | None = None
    gini_test: float | None = None


class BranchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch_id: str
    recipe_id: str
    features_woe: list[str]
    metrics: BranchMetrics
    model_path: Optional[str] = None
    metrics_path: Optional[str] = None
    passed_filters: bool = True
    filter_notes: list[str] = Field(default_factory=list)


class RankerWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    w_valid_gini: float = 0.6
    w_test_gini: float = 0.3
    w_parsimony: float = 0.1


class ProposalBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    champion_branch_id: str
    alternate_branch_ids: list[str] = Field(default_factory=list)
    branches: list[BranchResult]
    ranker_weights: RankerWeights
    rationale: str = ""


class ConstraintSpec(BaseModel):
    """Post-H4 feedback persisted between iterations."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    max_features: Optional[int] = None
    max_corr: Optional[float] = None
    prefer_recipes: list[str] = Field(default_factory=list)
    disable_recipes: list[str] = Field(default_factory=list)
    add_forbidden_features: list[str] = Field(default_factory=list)
    add_forced_features: list[str] = Field(default_factory=list)
    emphasize_cohort: Optional[Literal["valid", "test", "oot"]] = None
    notes: str = ""


# --------------------------------------------------------------------------- #
# Scorecard
# --------------------------------------------------------------------------- #


class PdoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_score: float = 600
    pdo: float = 20
    odds: float = 50.0


class LogisticHyperparams(BaseModel):
    """A minimal, validated subset of ``LogisticRegression`` kwargs."""

    model_config = ConfigDict(extra="forbid")

    C: float = 1.0
    penalty: Literal["l1", "l2"] = "l2"
    solver: Literal["liblinear", "lbfgs", "saga"] = "liblinear"
    max_iter: int = 1000
    random_state: int = 42


# --------------------------------------------------------------------------- #
# HITL
# --------------------------------------------------------------------------- #


class GateId(str, Enum):
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"
    H4 = "H4"
    H5 = "H5"
    H6 = "H6"


class HitlAction(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


class HitlDecision(BaseModel):
    """Structured outcome of any HITL gate.

    ``extra="ignore"`` so decisions produced by older runs (which carried an
    unused ``comment`` field) still round-trip through
    :class:`scorecard.hitl.ScriptedHitl`.
    """

    model_config = ConfigDict(extra="ignore")

    gate_id: GateId
    action: HitlAction
    user: str = "cli"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _revise_requires_payload(self) -> "HitlDecision":
        if self.action is HitlAction.REVISE and not self.payload:
            raise ValueError("Revise actions require a non-empty payload.")
        return self


# --------------------------------------------------------------------------- #
# Run manifest
# --------------------------------------------------------------------------- #


class ModelDocumentationMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    doc_schema_version: int = 1
    champion_branch_id: str
    required_headings_ok: bool = True


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    run_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    data_source: str
    artifacts_dir: str
    data_contract: DataColumnContract
    problem_contract: ProblemContract
    split_config: SplitConfig
    binning_config: BinningConfig
    feature_search_config: FeatureSearchConfig
    pdo_params: PdoParams
    constraint_spec: ConstraintSpec = Field(default_factory=ConstraintSpec)
    model_documentation: Optional[ModelDocumentationMeta] = None
    production_code_path: Optional[str] = None
    champion_branch_id: Optional[str] = None
    approved: bool = False
