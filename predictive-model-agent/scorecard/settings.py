"""Settings loader for the scorecard multi-agent system.

Reads ``predictive-model-agent/config.yaml`` and exposes the Gemini model
parameters used by every LLM sub-agent. Also pushes the resolved API key
into the ``GOOGLE_API_KEY`` environment variable because that is what the
``google-genai`` SDK (used under google-adk) reads by default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


_REPO_CONFIG_NAME = "config.yaml"


class GeminiSettings(BaseModel):
    """Subset of the ``gemini:`` block in config.yaml used by agents."""

    model_config = ConfigDict(extra="ignore")

    model: str = "gemini-flash-lite-latest"
    temperature: float = 0.2
    top_p: float = 0.95
    max_output_tokens: int = 8192
    api_key_env: str = "GEMINI_API_KEY"
    api_key: str | None = None
    http_timeout_ms: int | None = 900000


class Settings(BaseModel):
    """Top-level settings container."""

    model_config = ConfigDict(extra="ignore")

    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    project_root: Path


def _find_config(explicit_path: str | os.PathLike[str] | None) -> Path:
    if explicit_path is not None:
        p = Path(explicit_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {p}")
        return p
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        cand = parent / _REPO_CONFIG_NAME
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "Could not locate config.yaml near the scorecard package; "
        "pass an explicit path to load_settings()."
    )


def load_settings(config_path: str | os.PathLike[str] | None = None) -> Settings:
    """Load ``config.yaml``, populate GOOGLE_API_KEY, return a ``Settings``."""

    cfg_path = _find_config(config_path)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    gemini = GeminiSettings.model_validate(raw.get("gemini", {}) or {})
    settings = Settings(gemini=gemini, project_root=cfg_path.parent)

    key = os.environ.get(gemini.api_key_env) or (gemini.api_key or "").strip() or None
    if key:
        os.environ.setdefault("GOOGLE_API_KEY", key)
        os.environ.setdefault(gemini.api_key_env, key)
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "0")

    return settings
