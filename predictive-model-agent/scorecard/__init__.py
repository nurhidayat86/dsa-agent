"""Multi-agent scorecard pipeline.

Implementation of the design in ``docs/multi-agent-scorecard-design.md``.

Package layout:

``settings``      load ``config.yaml`` (Gemini model, keys).
``schemas``       Pydantic v2 contracts (one module = single source of truth).
``tools``         Pydantic-validated wrappers over ``agent_tools.py``.
``state``         In-memory run context (datasets, fitted objects).
``artifacts``     Persist / load JSON + parquet run artifacts.
``hitl``          ``HitlInterface`` abstraction + CLI implementation.
``agents``        Google-ADK ``LlmAgent`` instances for narrative work.
``orchestrator``  Phase state machine with HITL gates H1-H6 and rewinds.
``cli``           End-user entry point (argparse + interactive HITL).

The public API for running a pipeline is ``scorecard.orchestrator.run_pipeline``.
"""

from .settings import load_settings, Settings
from .orchestrator import run_pipeline

__all__ = ["load_settings", "Settings", "run_pipeline"]
