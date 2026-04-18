"""In-memory run context.

The orchestrator keeps the working dataset, fitted objects (``BinningProcess``,
branch ``LogisticRegression`` models, the optbinning ``Scorecard``), and
pointers to on-disk artifacts in a single ``RunContext`` instance. Individual
phase handlers and tool wrappers receive this context instead of passing
DataFrames through the LLM prompts, which would be both expensive and
unsafe for private data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .artifacts import ArtifactStore
from .schemas import (
    BinningConfig,
    BranchResult,
    ConstraintSpec,
    DataColumnContract,
    FeatureSearchConfig,
    PdoParams,
    ProblemContract,
    ProposalBundle,
    RunManifest,
    SplitConfig,
)


@dataclass
class RunContext:
    """Stateful bag of objects shared between phases."""

    run_id: str
    data_source: str
    store: ArtifactStore

    df_raw: Optional[pd.DataFrame] = None
    df_work: Optional[pd.DataFrame] = None  # with col_type, col_month, WoE columns

    data_contract: Optional[DataColumnContract] = None
    problem_contract: Optional[ProblemContract] = None
    split_config: Optional[SplitConfig] = None
    binning_config: Optional[BinningConfig] = None
    feature_search_config: Optional[FeatureSearchConfig] = None
    pdo_params: Optional[PdoParams] = None
    constraint_spec: ConstraintSpec = field(default_factory=ConstraintSpec)

    bin_dict: Optional[dict[str, Any]] = None
    binning_process: Optional[Any] = None
    binning_tables: Optional[pd.DataFrame] = None
    cols_feat_woe: list[str] = field(default_factory=list)
    binning_issues: list[str] = field(default_factory=list)

    branches: list[BranchResult] = field(default_factory=list)
    branch_models: dict[str, Any] = field(default_factory=dict)
    proposal: Optional[ProposalBundle] = None

    scorecard_model: Optional[Any] = None
    scorecard_points: Optional[pd.DataFrame] = None

    validation_tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    iteration: int = 0
    model_documentation_path: Optional[Path] = None
    production_code_path: Optional[Path] = None

    def manifest(self) -> RunManifest:
        assert self.data_contract is not None
        assert self.problem_contract is not None
        assert self.split_config is not None
        assert self.binning_config is not None
        assert self.feature_search_config is not None
        assert self.pdo_params is not None
        return RunManifest(
            run_id=self.run_id,
            data_source=self.data_source,
            artifacts_dir=str(self.store.root),
            data_contract=self.data_contract,
            problem_contract=self.problem_contract,
            split_config=self.split_config,
            binning_config=self.binning_config,
            feature_search_config=self.feature_search_config,
            pdo_params=self.pdo_params,
            constraint_spec=self.constraint_spec,
            champion_branch_id=self.proposal.champion_branch_id if self.proposal else None,
        )
