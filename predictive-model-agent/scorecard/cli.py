"""Command-line entry point for the scorecard multi-agent system.

Examples
--------

Interactive CLI HITL (default)::

    python -m scorecard.cli --data data/heloc_dataset_v1.parquet

Unattended smoke test (auto-approve every gate)::

    python -m scorecard.cli --data data/heloc_dataset_v1.parquet --auto-approve

Scripted HITL from a JSONL file::

    python -m scorecard.cli --data data/heloc_dataset_v1.parquet \
        --hitl-script decisions.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .hitl import AutoApproveHitl, CLIHitl, ScriptedHitl
from .orchestrator import PipelineOptions, run_pipeline
from .settings import load_settings


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scorecard", description="Run the scorecard multi-agent pipeline.")
    p.add_argument("--data", required=True, help="Path to parquet or CSV dataset.")
    p.add_argument("--config", default=None, help="Path to config.yaml (defaults to project-local).")
    p.add_argument("--artifacts-root", default=None, help="Root directory for run artifacts.")
    p.add_argument("--run-id", default=None, help="Explicit run id (auto-generated otherwise).")
    p.add_argument("--col-target", default=None)
    p.add_argument("--col-time", default=None)
    p.add_argument("--max-iterations", type=int, default=3, help="Max H4 feedback iterations.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--auto-approve", action="store_true", help="Approve every gate non-interactively.")
    group.add_argument("--hitl-script", default=None, help="JSONL file with pre-recorded HitlDecisions.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = load_settings(args.config)

    if args.auto_approve:
        hitl = AutoApproveHitl()
    elif args.hitl_script:
        hitl = ScriptedHitl.from_jsonl(args.hitl_script)
    else:
        hitl = CLIHitl()

    options = PipelineOptions(
        data_path=Path(args.data).resolve(),
        artifacts_root=args.artifacts_root,
        run_id=args.run_id,
        col_target=args.col_target,
        col_time=args.col_time,
        max_iterations=args.max_iterations,
    )
    manifest = run_pipeline(options, hitl=hitl, settings=settings)
    print(f"[ok] champion={manifest.champion_branch_id} approved={manifest.approved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
