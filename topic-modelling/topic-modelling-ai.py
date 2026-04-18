"""
Agent-driven, end-to-end topic-modelling pipeline.

This module is a refactor of the original ``topic-modelling-ai.py`` that:

* Uses **Pydantic** models as data contracts (config, UMAP candidate metrics,
  cluster records, pipeline result).
* Wraps the whole flow as a **google-adk** ``LlmAgent`` whose tools execute the
  individual stages (load + normalise, sweep UMAP candidates, visualise the
  selected 2D projection, run the remaining HDBSCAN -> KeyBERT -> Gemini
  summary -> Markdown report stages).
* Lets **Gemini** (the agent itself) pick ``umap_n_neighbors`` from the result
  of ``UMAP(n_components=2).fit_transform`` swept across n_neighbors=15..25.
  The diagnostic stats sent to the agent capture the "shatters into hundreds
  of tiny islands" (n_neighbors too low) vs. "everything merges into one giant
  blob" (n_neighbors too high) heuristic from the reference notebook.
* Puts a **human in the loop**: the agent saves the 2D scatter plot to disk,
  opens it in the OS image viewer, and waits for natural-language feedback
  from the user. The user can approve the projection or describe what they
  want changed; the agent re-renders with a new ``n_neighbors`` until the
  user is satisfied. Then the rest of the pipeline runs to completion.

Run it interactively:

    conda run -n google-adk python topic-modelling-ai.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
import hdbscan
import matplotlib

matplotlib.use("Agg")  # render to file; visualisation is opened by OS viewer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import umap
import yaml
from google import genai
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types
from keybert import KeyBERT
from keybert.backend._base import BaseEmbedder
from pydantic import BaseModel, ConfigDict, Field
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


# ===========================================================================
# Pydantic data contracts
# ===========================================================================


class GeminiSettings(BaseModel):
    """Gemini client configuration loaded from ``config.yaml`` (or env)."""

    model: str = "gemini-flash-lite-latest"
    api_key: str | None = None
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    http_timeout_ms: int | None = None


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chroma_db_path: Path
    md_file_path: Path
    cluster_summary_path: Path = Path("./out")
    collection_name: str | None = None
    qwen_model_path: Path | None = None
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    umap_min_n_neighbors: int = 15
    umap_max_n_neighbors: int = 25
    umap_n_components: int = 5
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int = 1
    agent_model: str = "gemini-flash-lite-latest"


class UmapCandidateMetrics(BaseModel):
    """Diagnostic statistics for a single ``n_neighbors`` candidate.

    The metrics are designed for Gemini to reason about whether the projection
    "shatters" (n_neighbors too low) or "merges into one blob" (too high).
    """

    n_neighbors: int
    n_clusters: int
    n_noise_points: int
    n_tiny_islands: int = Field(
        description="Clusters with <=5 points; large counts suggest n_neighbors too low."
    )
    largest_cluster_share: float = Field(
        description="Fraction of points in the biggest cluster; ~1.0 suggests one giant blob."
    )
    spread_x: float
    spread_y: float
    bbox_area: float


class UmapEvaluation(BaseModel):
    """Result of sweeping UMAP n_neighbors across a range."""

    n_samples: int
    candidates: list[UmapCandidateMetrics]
    diagnostic_summary: str


class UmapVisualization(BaseModel):
    """Output of rendering a single 2D UMAP projection for the user."""

    n_neighbors: int
    plot_path: Path
    metrics: UmapCandidateMetrics
    opened_in_viewer: bool


class ClusterKeywordRecord(BaseModel):
    cluster: int
    keywords: str
    cosine_similarities: float
    weight: float = 0.0


class ClusterSummaryRecord(BaseModel):
    cluster: int
    text_summary: str
    cluster_count: int


class PipelineResult(BaseModel):
    """Final artefacts produced by the pipeline."""

    chosen_n_neighbors: int
    n_documents: int
    n_clusters: int
    keyword_csv: Path
    summary_csv: Path
    report_path: Path


# ===========================================================================
# Repo / Chroma helpers (unchanged behaviour from notebook)
# ===========================================================================


def _dsa_agent_repo_root() -> Path:
    """First ancestor of cwd that contains ``vector-db-writer`` (Qwen snapshot)."""
    p = Path.cwd().resolve()
    for d in (p, *p.parents):
        if (d / "vector-db-writer").is_dir():
            return d
    raise FileNotFoundError(
        "Could not find a parent directory containing vector-db-writer/. "
        "Set cwd to the dsa-agent repo root, or install Qwen3 under "
        "vector-db-writer/model/Qwen3-Embedding-8B."
    )


def load_chroma(chroma_path: str | Path, collection_name: str | None = None) -> pd.DataFrame:
    """Load a persistent Chroma collection into a DataFrame (id, document, embedding, meta_*)."""
    path = Path(chroma_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Chroma path is not a directory: {path}")

    client = chromadb.PersistentClient(path=str(path))
    existing = sorted({c.name for c in client.list_collections()}, key=str.lower)
    if not existing:
        raise ValueError(f"No collections in {path!s}.")
    if collection_name is None:
        if len(existing) == 1:
            collection_name = existing[0]
        else:
            raise ValueError("Pass collection_name=... One of: " + ", ".join(existing))
    elif collection_name not in existing:
        raise ValueError(f"Collection {collection_name!r} not found. Available: {existing}")

    col = client.get_collection(collection_name)

    def _docs_as_list(raw: Any, n_rows: int) -> list[Any]:
        if raw is None:
            return [None] * n_rows
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        if len(raw) != n_rows:
            raise ValueError(f"documents length {len(raw)} != ids length {n_rows}")
        return list(raw)

    def _embeddings_as_list(raw: Any, n_rows: int) -> list[Any]:
        if raw is None:
            return [None] * n_rows
        if isinstance(raw, np.ndarray):
            if raw.ndim == 2 and raw.shape[0] == n_rows:
                return [raw[i].astype(float, copy=False).tolist() for i in range(n_rows)]
            raise ValueError(f"Unexpected embeddings ndarray shape {raw.shape}")
        out: list[list[float]] = []
        for row in raw:
            if row is None:
                out.append(None)
            elif isinstance(row, np.ndarray):
                out.append(row.astype(float, copy=False).tolist())
            else:
                out.append(list(row))
        if len(out) != n_rows:
            raise ValueError(f"embeddings length {len(out)} != ids length {n_rows}")
        return out

    include = ["documents", "embeddings", "metadatas"]
    batch_size = 10_000
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        batch = col.get(include=include, limit=batch_size, offset=offset)
        ids = batch.get("ids") or []
        if not ids:
            break
        n = len(ids)
        docs = _docs_as_list(batch.get("documents"), n)
        embs = _embeddings_as_list(batch.get("embeddings"), n)
        frame = pd.DataFrame({"id": ids, "document": docs, "embedding": embs})
        metas = batch.get("metadatas") or [None] * n
        meta_df = pd.json_normalize(metas)
        meta_df.columns = [f"meta_{c}" for c in meta_df.columns]
        frame = pd.concat([frame.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)
        frames.append(frame)
        offset += n
        if n < batch_size:
            break

    if not frames:
        return pd.DataFrame(columns=["id", "document", "embedding"])
    return pd.concat(frames, ignore_index=True)


def get_array(df: pd.DataFrame, col_vector: str) -> np.ndarray:
    """Stack a column of per-row float vectors into a 2D ``float64`` array."""
    if col_vector not in df.columns:
        raise KeyError(f"Column {col_vector!r} not in DataFrame columns: {list(df.columns)}")
    s = df[col_vector]
    if s.isna().any():
        raise ValueError(f"Column {col_vector!r} contains NaN/None.")
    rows_out: list[np.ndarray] = []
    dim: int | None = None
    for i, v in enumerate(s):
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Row {i}: expected 1D vector, got shape {arr.shape}")
        if dim is None:
            dim = int(arr.shape[0])
        elif int(arr.shape[0]) != dim:
            raise ValueError(f"Row {i}: ragged vector length {arr.shape[0]} != {dim}")
        rows_out.append(arr)
    if not rows_out:
        return np.empty((0, 0), dtype=np.float64)
    return np.stack(rows_out, axis=0)


def normalize_embeddings_for_clustering(
    X: np.ndarray,
    method: str = "l2",
    **kwargs: Any,
) -> tuple[np.ndarray, Normalizer | StandardScaler | MinMaxScaler]:
    """Normalise embedding matrix for clustering."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    method = str(method).lower().strip()
    if method == "standard":
        scaler: Normalizer | StandardScaler | MinMaxScaler = StandardScaler(**kwargs)
    elif method == "l2":
        scaler = Normalizer(norm="l2", **kwargs)
    elif method == "minmax":
        scaler = MinMaxScaler(**kwargs)
    else:
        raise ValueError("method must be 'standard', 'l2', or 'minmax'")
    return scaler.fit_transform(X), scaler


# ===========================================================================
# Qwen3 KeyBERT backend (unchanged)
# ===========================================================================


def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


class Qwen3KeyBERTBackend(BaseEmbedder):
    """Qwen3-Embedding via transformers; subclasses BaseEmbedder for KeyBERT."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        max_length: int = 8192,
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        self._path = Path(model_path).expanduser().resolve()
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._max_length = max_length
        self._batch_size = max(1, int(batch_size))
        dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        tp = str(self._path)
        self._tokenizer = AutoTokenizer.from_pretrained(tp, padding_side="left", local_files_only=True)
        self._model = AutoModel.from_pretrained(tp, torch_dtype=dtype, local_files_only=True)
        self._model.to(self._device)
        self._model.eval()

    @torch.inference_mode()
    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        batch = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self._device) for k, v in batch.items()}
        out = self._model(**batch)
        emb = _last_token_pool(out.last_hidden_state, batch["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        return emb.float().cpu().numpy()

    def embed(self, documents: Any, verbose: bool = False) -> np.ndarray:
        if documents is None:
            return np.empty((0, 0), dtype=np.float32)
        if isinstance(documents, np.ndarray):
            documents = documents.ravel().tolist()
        texts = [("" if x is None else str(x)) for x in documents]
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        parts: list[np.ndarray] = []
        for i in range(0, len(texts), self._batch_size):
            chunk = texts[i : i + self._batch_size]
            parts.append(self._encode_batch(chunk))
        return np.vstack(parts)


def cluster_keywords_keybert(
    df: pd.DataFrame,
    *,
    doc_col: str = "document",
    cluster_col: str = "cluster",
    model_path: str | Path | None = None,
    top_n: int = 20,
    keyphrase_ngram_range: tuple[int, int] = (1, 3),
    stop_words: str | None = "english",
    max_cluster_chars: int = 12_000,
    join_sep: str = "\n\n",
    embed_batch_size: int = 16,
) -> pd.DataFrame:
    """Per-cluster KeyBERT keywords (Qwen3 backend)."""
    if model_path is None:
        model_path = (_dsa_agent_repo_root() / "vector-db-writer" / "model" / "Qwen3-Embedding-8B").resolve()
    model_path = Path(model_path).expanduser().resolve()

    backend = Qwen3KeyBERTBackend(model_path, batch_size=embed_batch_size)
    kw_model = KeyBERT(model=backend)

    rows: list[dict[str, Any]] = []
    for cid in sorted(df[cluster_col].dropna().unique(), key=lambda x: (str(type(x)), str(x))):
        texts = df.loc[df[cluster_col] == cid, doc_col].dropna().astype(str).tolist()
        blob = join_sep.join(t for t in texts if t.strip())
        if len(blob) > max_cluster_chars:
            blob = blob[:max_cluster_chars]
        if not blob.strip():
            continue
        kws = kw_model.extract_keywords(
            blob,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words=stop_words,
            top_n=top_n,
        )
        for phrase, sim in kws:
            rows.append(
                {
                    "cluster": cid,
                    "keywords": str(phrase),
                    "cosine_similarities": float(sim),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["cluster", "keywords", "cosine_similarities", "weight"])

    def _weights(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        tot = s.sum()
        if tot > 0:
            return s / tot
        return pd.Series(np.zeros(len(s)), index=s.index)

    out["weight"] = out.groupby("cluster", group_keys=False)["cosine_similarities"].apply(_weights)
    return out.reset_index(drop=True)


# ===========================================================================
# Gemini config + Gemini-driven cluster summary / report (unchanged behaviour)
# ===========================================================================


def _find_gemini_config_path() -> Path | None:
    root = Path.cwd().resolve()
    for d in (root, *root.parents):
        cand = d / "config.yaml"
        if not cand.is_file():
            continue
        try:
            raw = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if isinstance(raw, dict) and raw.get("gemini"):
            return cand
    return None


def load_gemini_settings() -> GeminiSettings:
    """Load Gemini settings from a parent ``config.yaml`` (with env fallback)."""
    cfg = _find_gemini_config_path()
    data: dict[str, Any] = {}
    if cfg is not None:
        raw = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        g = raw.get("gemini") or {}
        for key in ("model", "api_key", "api_key_env", "temperature", "top_p",
                    "max_output_tokens", "http_timeout_ms"):
            if g.get(key) is not None:
                data[key] = g[key]
    settings = GeminiSettings(**data)
    if not settings.api_key:
        env_key = os.environ.get(settings.api_key_env) or os.environ.get("GEMINI_API_KEY")
        if env_key:
            settings.api_key = env_key.strip()
    if not settings.api_key:
        raise ValueError(
            "Gemini API key not found: set gemini.api_key in config.yaml or "
            f"the {settings.api_key_env} env var."
        )
    # Propagate the key to env vars that the google-genai SDK (used by ADK's
    # LlmAgent under the hood) auto-discovers. Without this, an inline
    # api_key from config.yaml would be invisible to the agent's model client.
    for env_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", settings.api_key_env):
        if env_name and not os.environ.get(env_name):
            os.environ[env_name] = settings.api_key
    return settings


def _build_genai_client(settings: GeminiSettings) -> genai.Client:
    http_options = None
    if settings.http_timeout_ms and int(settings.http_timeout_ms) > 0:
        http_options = genai_types.HttpOptions(timeout=int(settings.http_timeout_ms))
    return genai.Client(api_key=settings.api_key, http_options=http_options)


def write_cluster_summary(
    df: pd.DataFrame, col_doc: str, cluster: str, settings: GeminiSettings
) -> pd.DataFrame:
    """Per-cluster Gemini text summaries."""
    cfg_fields: dict[str, Any] = {}
    if settings.temperature is not None:
        cfg_fields["temperature"] = float(settings.temperature)
    if settings.top_p is not None:
        cfg_fields["top_p"] = float(settings.top_p)
    if settings.max_output_tokens is not None:
        cfg_fields["max_output_tokens"] = int(settings.max_output_tokens)
    gen_config = genai_types.GenerateContentConfig(**cfg_fields) if cfg_fields else None

    client = _build_genai_client(settings)
    max_blob_chars = 600_000
    join_sep = "\n\n---\n\n"

    work = df[[cluster, col_doc]].copy()
    rows: list[dict[str, Any]] = []
    for cid in sorted(work[cluster].dropna().unique(), key=lambda x: (str(type(x)), str(x))):
        grp = work.loc[work[cluster] == cid]
        texts = [str(t).strip() for t in grp[col_doc].tolist() if str(t).strip()]
        blob = join_sep.join(texts)
        if len(blob) > max_blob_chars:
            blob = blob[:max_blob_chars] + "\n\n[... truncated for context limit ...]"
        if not blob:
            rows.append({"cluster": cid, "text_summary": ""})
            continue
        prompt = (
            "You summarize customer feedback that belongs to one cluster.\n"
            "Read all passages (they may repeat themes). Produce one clear summary: main themes, "
            "issues, sentiment if obvious, and concrete details (products, channels, branches) when "
            "they matter. Plain text only; about 200-400 words unless the cluster is very diverse.\n\n"
            "--- PASSAGES ---\n"
            f"{blob}"
        )
        resp = client.models.generate_content(
            model=settings.model,
            contents=prompt,
            config=gen_config,
        )
        rows.append({"cluster": cid, "text_summary": (getattr(resp, "text", None) or "").strip()})

    return pd.DataFrame(rows, columns=["cluster", "text_summary"]).reset_index(drop=True)


def _parse_json_dict(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if not s:
        return {}
    if "```" in s:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.I)
        if m:
            s = m.group(1).strip()
    s = s.strip()
    if not s.startswith("{"):
        i, j = s.find("{"), s.rfind("}")
        if i == -1 or j <= i:
            raise ValueError("No JSON object in model response.")
        s = s[i : j + 1]
    return json.loads(s)


def _label_max_three_words(s: str, max_words: int = 3) -> str:
    s = re.sub(r"\s+", " ", str(s).strip())
    parts = re.split(r"[^\w\-]+", s, flags=re.UNICODE)
    words = [p for p in parts if p]
    return " ".join(words[:max_words]) if words else "Unnamed cluster"


def generate_md_report(
    df_keywords: pd.DataFrame,
    df_summary: pd.DataFrame,
    md_path: str | Path,
    settings: GeminiSettings,
) -> Path:
    """Gemini: short cluster labels + Markdown helicopter-view report."""
    need_kw = {"keywords", "cluster", "cosine_similarities"}
    need_sm = {"cluster", "text_summary", "cluster_count"}
    if miss := need_kw - set(df_keywords.columns):
        raise ValueError(f"df_keywords missing columns: {sorted(miss)}")
    if miss := need_sm - set(df_summary.columns):
        raise ValueError(f"df_summary missing columns: {sorted(miss)}")

    out = Path(md_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    client = _build_genai_client(settings)
    cfg_labels = genai_types.GenerateContentConfig(temperature=0.25, max_output_tokens=4096)
    cfg_report = genai_types.GenerateContentConfig(temperature=0.4, max_output_tokens=12_288)

    sm = df_summary.copy()
    sm["_ck"] = sm["cluster"].map(lambda x: str(x))
    sm = sm.sort_values("_ck")

    label_rows: list[dict[str, Any]] = []
    for _, r in sm.iterrows():
        cc = pd.to_numeric(r["cluster_count"], errors="coerce")
        cnt_val = int(cc) if pd.notna(cc) else None
        label_rows.append(
            {
                "cluster_key": str(r["cluster"]),
                "cluster_count": cnt_val,
                "text_summary": str(r["text_summary"] or "")[:12_000],
            }
        )

    prompt_labels = (
        "You label customer-feedback clusters for executives.\n"
        "Input is a JSON array of objects with fields cluster_key, cluster_count, text_summary.\n"
        "Return ONE JSON object only (no markdown fences): keys must be every cluster_key as strings, "
        "values must be a short display name of AT MOST THREE WORDS reflecting that cluster's summary "
        "(Title Case, e.g. Satisfying Service, Fee Transparency Issues).\n\n"
        + json.dumps(label_rows, ensure_ascii=False)
    )

    raw_labels = client.models.generate_content(
        model=settings.model, contents=prompt_labels, config=cfg_labels
    ).text

    try:
        parsed = _parse_json_dict(raw_labels)
        labels_map = {str(k): _label_max_three_words(str(v)) for k, v in parsed.items()}
    except (json.JSONDecodeError, ValueError, TypeError):
        labels_map = {}
    for _, r in sm.iterrows():
        ck = str(r["cluster"])
        if ck not in labels_map or not labels_map[ck].strip():
            labels_map[ck] = _label_max_three_words(f"Cluster {ck}")

    kw = df_keywords.copy()
    kw["_ck"] = kw["cluster"].map(lambda x: str(x))

    summary_blocks: list[str] = []
    for _, r in sm.iterrows():
        ck = str(r["cluster"])
        title = labels_map.get(ck, ck)
        n = r["cluster_count"]
        summary_blocks.append(
            f"## {title}\n- **Cluster id:** `{ck}`\n- **Documents in cluster:** {n}\n\n"
            f"**Summary:**\n{r['text_summary']}\n"
        )

    keyword_blocks: list[str] = []
    for ck in sm["_ck"].unique():
        grp = kw.loc[kw["_ck"] == ck].copy()
        title = labels_map.get(str(ck), str(ck))
        if grp.empty:
            keyword_blocks.append(f"## {title} (`{ck}`)\n_No keyword rows._\n")
            continue
        grp = grp.nlargest(min(25, len(grp)), "cosine_similarities")
        lines = [
            f"- `{row['keywords']}` - relevance {float(row['cosine_similarities']):.3f}"
            for _, row in grp.iterrows()
        ]
        keyword_blocks.append(f"## {title} (`{ck}`)\n" + "\n".join(lines) + "\n")

    prompt_report = (
        "Write a professional Markdown report containing a helicopter view of "
        "customer feedback. Audience is non-technical; avoid terms like embeddings, cosine, or vectors.\n\n"
        "Use the cluster **titles** already chosen (the ## headings in the material below). Do not rename clusters.\n\n"
        "Include these sections with `##` headings (adjust wording but keep intent):\n"
        "1. Executive overview - 4-6 tight bullets on overall tone, main risks, and bright spots.\n"
        "2. Cluster-by-cluster insight - for each cluster, 2-4 bullets linking keywords to the summary; "
        "mention document counts where useful.\n"
        "3. Cross-cutting themes - tensions, recurring patterns across clusters.\n\n"
        "Do NOT include an appendix or any list of keywords per cluster: a separate, "
        "verbatim KeyBERT keyword section will be appended to the report automatically.\n\n"
        "Target length about 700-3000 words. Use Markdown lists and optional **bold** for emphasis.\n\n"
        "--- SOURCE: CLUSTER SUMMARIES ---\n\n"
        + "\n".join(summary_blocks)
        + "\n--- SOURCE: KEYPHRASES ---\n\n"
        + "\n".join(keyword_blocks)
    )

    report_body = client.models.generate_content(
        model=settings.model, contents=prompt_report, config=cfg_report
    ).text
    if not (report_body or "").strip():
        report_body = "_The model returned no text._\n"

    table = ["| Cluster id | Count | Display name (<=3 words) |", "| --- | ---: | --- |"]
    for _, r in sm.iterrows():
        ck = str(r["cluster"])
        disp = str(labels_map.get(ck, "")).replace("|", "/")
        cnt = r["cluster_count"]
        table.append(f"| `{ck}` | {cnt} | {disp} |")

    keybert_appendix = _build_keybert_appendix(kw, sm, labels_map)

    doc = (
        "<!-- generate_md_report: Gemini labels + narrative. -->\n\n"
        "# Cluster display names\n\n"
        + "\n".join(table)
        + "\n\n---\n\n"
        + report_body.strip()
        + "\n\n---\n\n"
        + keybert_appendix
        + "\n"
    )
    out.write_text(doc, encoding="utf-8")
    return out


def _build_keybert_appendix(
    kw: pd.DataFrame,
    sm: pd.DataFrame,
    labels_map: dict[str, str],
    *,
    top_n: int | None = None,
) -> str:
    """Render the deterministic per-cluster KeyBERT keyword listing.

    Args:
        kw: KeyBERT keyword DataFrame (columns: ``cluster``, ``keywords``,
            ``cosine_similarities``, optional ``_ck``).
        sm: Cluster summary DataFrame (used only for ordering / counts;
            must contain ``cluster`` and ``cluster_count``, and the
            same ``_ck`` ordering used for the rest of the report).
        labels_map: Mapping from cluster_key string to display name.
        top_n: If given, keep only the top-N keywords per cluster
            (by ``cosine_similarities``). ``None`` keeps all rows.
    """
    lines: list[str] = ["# Keywords per cluster (KeyBERT)\n"]
    if "_ck" not in kw.columns:
        kw = kw.copy()
        kw["_ck"] = kw["cluster"].map(lambda x: str(x))

    counts_by_ck: dict[str, Any] = {}
    if "cluster_count" in sm.columns:
        for _, r in sm.iterrows():
            counts_by_ck[str(r["cluster"])] = r["cluster_count"]

    for ck in sm["_ck"].unique():
        title = labels_map.get(str(ck), str(ck))
        cnt = counts_by_ck.get(str(ck))
        header = f"## {title} (`{ck}`)"
        if cnt is not None:
            header += f" - {cnt} documents"
        lines.append(header)

        grp = kw.loc[kw["_ck"] == ck].copy()
        if grp.empty:
            lines.append("_No KeyBERT keywords extracted for this cluster._\n")
            continue
        grp = grp.sort_values("cosine_similarities", ascending=False)
        if top_n is not None:
            grp = grp.head(int(top_n))
        for _, row in grp.iterrows():
            phrase = str(row["keywords"]).replace("`", "'")
            sim = float(row["cosine_similarities"])
            weight = row.get("weight")
            try:
                weight_str = f", weight {float(weight):.3f}" if weight is not None else ""
            except (TypeError, ValueError):
                weight_str = ""
            lines.append(f"- `{phrase}` - relevance {sim:.3f}{weight_str}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ===========================================================================
# UMAP 2D evaluation + visualisation (the part Gemini reasons about)
# ===========================================================================


def _summarise_2d_projection(
    n_neighbors: int,
    embeddings_2d: np.ndarray,
    *,
    micro_island_size: int = 5,
    min_cluster_size: int = 5,
) -> UmapCandidateMetrics:
    """Compute "shatter vs. blob" diagnostics for a 2D UMAP projection."""
    n = embeddings_2d.shape[0]
    if n == 0:
        return UmapCandidateMetrics(
            n_neighbors=n_neighbors,
            n_clusters=0,
            n_noise_points=0,
            n_tiny_islands=0,
            largest_cluster_share=0.0,
            spread_x=0.0,
            spread_y=0.0,
            bbox_area=0.0,
        )

    mcs = max(2, min(min_cluster_size, n))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=1, metric="euclidean")
    labels = clusterer.fit_predict(embeddings_2d.astype(np.float64, copy=False))
    counter = Counter(int(l) for l in labels)
    n_noise = int(counter.pop(-1, 0))
    cluster_sizes = sorted(counter.values(), reverse=True)
    n_clusters = len(cluster_sizes)
    largest = cluster_sizes[0] if cluster_sizes else 0
    n_tiny = sum(1 for s in cluster_sizes if s <= micro_island_size)

    spread_x = float(np.std(embeddings_2d[:, 0]))
    spread_y = float(np.std(embeddings_2d[:, 1]))
    bbox_w = float(np.ptp(embeddings_2d[:, 0]))
    bbox_h = float(np.ptp(embeddings_2d[:, 1]))

    return UmapCandidateMetrics(
        n_neighbors=n_neighbors,
        n_clusters=n_clusters,
        n_noise_points=n_noise,
        n_tiny_islands=n_tiny,
        largest_cluster_share=float(largest) / float(n) if n else 0.0,
        spread_x=spread_x,
        spread_y=spread_y,
        bbox_area=bbox_w * bbox_h,
    )


def _render_scatter_png(
    embeddings_2d: np.ndarray, n_neighbors: int, out_path: Path
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        s=5,
        alpha=0.5,
        edgecolor=None,
        ax=ax,
    )
    ax.set_title(f"UMAP 2D projection (n_neighbors={n_neighbors})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _open_with_system_viewer(path: Path) -> bool:
    """Best-effort open of an image in the OS image viewer."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return True
    except Exception as exc:
        print(f"[viewer] could not auto-open {path}: {exc}", file=sys.stderr)
        return False


# ===========================================================================
# Pipeline runtime singleton (heavy state lives here, not in ADK session)
# ===========================================================================


class PipelineRuntime:
    """Holds heavy artefacts (df, embeddings) shared across tool calls."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.df: pd.DataFrame | None = None
        self.X_all: np.ndarray | None = None
        self.evaluations: dict[int, UmapCandidateMetrics] = {}
        self.cached_2d: dict[int, np.ndarray] = {}
        self.last_evaluation: UmapEvaluation | None = None
        self.last_visualization: UmapVisualization | None = None
        self.chosen_n_neighbors: int | None = None
        self.result: PipelineResult | None = None

    def ensure_loaded(self) -> None:
        if self.df is not None and self.X_all is not None:
            return
        df = load_chroma(self.config.chroma_db_path, collection_name=self.config.collection_name)
        if df.empty:
            raise ValueError("Chroma collection returned no rows.")
        np_emb = get_array(df, "embedding")
        if np_emb.size == 0:
            raise ValueError("No embedding vectors loaded from Chroma.")
        X_all, _ = normalize_embeddings_for_clustering(np_emb, method="l2")
        self.df = df
        self.X_all = X_all

    def _fit_2d(self, n_neighbors: int) -> np.ndarray:
        if n_neighbors in self.cached_2d:
            return self.cached_2d[n_neighbors]
        assert self.X_all is not None
        n_samples = self.X_all.shape[0]
        nn = max(2, min(int(n_neighbors), max(2, n_samples - 1)))
        reducer = umap.UMAP(
            n_neighbors=nn,
            n_components=2,
            metric="cosine",
            random_state=42,
        )
        emb_2d = reducer.fit_transform(self.X_all)
        self.cached_2d[n_neighbors] = emb_2d
        return emb_2d


_RUNTIME: PipelineRuntime | None = None


def _runtime() -> PipelineRuntime:
    if _RUNTIME is None:
        raise RuntimeError("PipelineRuntime not initialised. Call bootstrap_runtime(config) first.")
    return _RUNTIME


def bootstrap_runtime(config: PipelineConfig) -> PipelineRuntime:
    """Initialise the global pipeline runtime and load Chroma data eagerly."""
    global _RUNTIME
    _RUNTIME = PipelineRuntime(config)
    _RUNTIME.ensure_loaded()
    return _RUNTIME


# ===========================================================================
# ADK tools
# ===========================================================================


def evaluate_umap_n_neighbors_range(
    min_n_neighbors: int,
    max_n_neighbors: int,
    tool_context: ToolContext,
) -> dict:
    """Sweep ``UMAP(n_components=2)`` across ``n_neighbors`` and return diagnostics.

    For each candidate, this fits a 2D UMAP and reports:

    - ``n_clusters`` / ``n_noise_points`` / ``n_tiny_islands`` from a quick
      HDBSCAN on the 2D projection. Very high ``n_clusters`` together with a
      large ``n_tiny_islands`` indicates the data **shatters into hundreds of
      tiny islands** (n_neighbors too low).
    - ``largest_cluster_share`` close to 1.0 indicates everything **merges into
      one giant blob** (n_neighbors too high).
    - ``spread_x`` / ``spread_y`` / ``bbox_area`` describe the 2D layout.

    Use these signals to pick the best ``n_neighbors`` before visualising.

    Args:
        min_n_neighbors: Lowest ``n_neighbors`` to try (e.g. 15).
        max_n_neighbors: Highest ``n_neighbors`` to try (e.g. 25); inclusive.

    Returns:
        Dict with ``n_samples``, ``candidates`` (list of metrics), and a
        ``diagnostic_summary`` string.
    """
    rt = _runtime()
    rt.ensure_loaded()
    if min_n_neighbors > max_n_neighbors:
        min_n_neighbors, max_n_neighbors = max_n_neighbors, min_n_neighbors
    min_n_neighbors = max(2, int(min_n_neighbors))
    max_n_neighbors = max(min_n_neighbors, int(max_n_neighbors))

    candidates: list[UmapCandidateMetrics] = []
    print(
        f"[evaluate] sweeping n_neighbors={min_n_neighbors}..{max_n_neighbors} "
        f"on {rt.X_all.shape[0]} points",
        file=sys.stderr,
    )
    for nn in range(min_n_neighbors, max_n_neighbors + 1):
        emb_2d = rt._fit_2d(nn)
        metrics = _summarise_2d_projection(nn, emb_2d)
        rt.evaluations[nn] = metrics
        candidates.append(metrics)
        print(
            f"[evaluate] n_neighbors={nn} -> clusters={metrics.n_clusters} "
            f"noise={metrics.n_noise_points} tiny={metrics.n_tiny_islands} "
            f"largest_share={metrics.largest_cluster_share:.2f}",
            file=sys.stderr,
        )

    summary_lines = [
        f"n_neighbors={c.n_neighbors}: clusters={c.n_clusters}, "
        f"tiny_islands={c.n_tiny_islands}, noise={c.n_noise_points}, "
        f"largest_share={c.largest_cluster_share:.2f}"
        for c in candidates
    ]
    evaluation = UmapEvaluation(
        n_samples=int(rt.X_all.shape[0]),
        candidates=candidates,
        diagnostic_summary="\n".join(summary_lines),
    )
    rt.last_evaluation = evaluation
    tool_context.state["last_evaluation"] = evaluation.model_dump(mode="json")
    return evaluation.model_dump(mode="json")


def visualize_umap_2d(n_neighbors: int, tool_context: ToolContext) -> dict:
    """Render the 2D UMAP scatter for ``n_neighbors`` and open it for the user.

    The PNG is saved under ``<cluster_summary_path>/umap_2d_n<NN>.png`` and
    opened in the system image viewer so the human can review it.

    Args:
        n_neighbors: The candidate ``n_neighbors`` to visualise.

    Returns:
        Dict with ``n_neighbors``, ``plot_path``, ``opened_in_viewer`` and the
        diagnostic ``metrics`` for the same projection.
    """
    rt = _runtime()
    rt.ensure_loaded()
    n_neighbors = int(n_neighbors)
    emb_2d = rt._fit_2d(n_neighbors)
    metrics = rt.evaluations.get(n_neighbors) or _summarise_2d_projection(n_neighbors, emb_2d)
    rt.evaluations[n_neighbors] = metrics

    out_dir = Path(rt.config.cluster_summary_path).expanduser().resolve()
    plot_path = out_dir / f"umap_2d_n{n_neighbors}.png"
    _render_scatter_png(emb_2d, n_neighbors, plot_path)
    opened = _open_with_system_viewer(plot_path)

    viz = UmapVisualization(
        n_neighbors=n_neighbors,
        plot_path=plot_path,
        metrics=metrics,
        opened_in_viewer=opened,
    )
    rt.last_visualization = viz
    tool_context.state["last_visualization"] = viz.model_dump(mode="json")
    print(
        f"[visualize] saved {plot_path} (opened={opened}) "
        f"n_neighbors={n_neighbors} clusters={metrics.n_clusters}",
        file=sys.stderr,
    )
    return viz.model_dump(mode="json")


def finalize_n_neighbors_and_run(n_neighbors: int, tool_context: ToolContext) -> dict:
    """Lock in ``n_neighbors`` and run the rest of the pipeline to completion.

    Reduces with ``UMAP(n_components=<config>)``, clusters with HDBSCAN, runs
    KeyBERT (Qwen3 backend) per cluster, generates Gemini per-cluster summaries,
    then writes the Markdown report.

    Args:
        n_neighbors: Approved ``n_neighbors`` value.

    Returns:
        Dict describing the final ``PipelineResult``.
    """
    rt = _runtime()
    rt.ensure_loaded()
    cfg = rt.config
    assert rt.df is not None and rt.X_all is not None

    n_neighbors = int(n_neighbors)
    n_samples = rt.X_all.shape[0]
    nn = max(2, min(n_neighbors, max(2, n_samples - 1)))

    print(f"[finalize] running full pipeline with n_neighbors={nn}", file=sys.stderr)
    reducer = umap.UMAP(
        n_neighbors=nn,
        n_components=cfg.umap_n_components,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(rt.X_all)

    min_cluster_size = min(cfg.hdbscan_min_cluster_size, max(2, n_samples))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        metric="euclidean",
        prediction_data=True,
    )
    cluster_labels = clusterer.fit_predict(reduced)
    df = rt.df.copy()
    df["cluster"] = cluster_labels

    print("[finalize] extracting KeyBERT keywords per cluster...", file=sys.stderr)
    summary_df = cluster_keywords_keybert(df, model_path=cfg.qwen_model_path)
    keyword_cols = [c for c in ["cluster", "keywords", "cosine_similarities", "weight"]
                    if c in summary_df.columns]
    keyword_summary = summary_df[keyword_cols].copy()

    print("[finalize] generating Gemini per-cluster text summaries...", file=sys.stderr)
    summary_llm = write_cluster_summary(df, "document", "cluster", cfg.gemini)
    counts = df.groupby("cluster", dropna=False)["document"].count()
    summary_llm["cluster_count"] = summary_llm["cluster"].map(lambda c: int(counts.get(c, 0)))

    out_dir = Path(cfg.cluster_summary_path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary_llm.csv"
    keyword_csv = out_dir / "keyword_summary.csv"
    summary_llm.to_csv(summary_csv, index=False)
    keyword_summary.to_csv(keyword_csv, index=False)

    print("[finalize] writing Markdown report...", file=sys.stderr)
    report_path = generate_md_report(
        keyword_summary,
        summary_llm,
        md_path=cfg.md_file_path,
        settings=cfg.gemini,
    )

    n_unique_clusters = int(df["cluster"].nunique(dropna=False))
    result = PipelineResult(
        chosen_n_neighbors=nn,
        n_documents=int(len(df)),
        n_clusters=n_unique_clusters,
        keyword_csv=keyword_csv,
        summary_csv=summary_csv,
        report_path=report_path,
    )
    rt.chosen_n_neighbors = nn
    rt.result = result
    tool_context.state["pipeline_result"] = result.model_dump(mode="json")
    print(f"[finalize] done. report -> {report_path}", file=sys.stderr)
    return result.model_dump(mode="json")


# ===========================================================================
# Agent definition
# ===========================================================================


_AGENT_INSTRUCTION = """\
You are an interactive topic-modelling assistant. The full pipeline is:

  load embeddings from Chroma -> L2-normalise
  -> UMAP -> HDBSCAN -> KeyBERT -> Gemini summaries -> Markdown report

Your job is to interactively pick the **best umap_n_neighbors** with the human's
approval, then run the rest of the pipeline.

Decision heuristic (from the reference notebook):
  * If the data shatters into hundreds of tiny islands -> n_neighbors too low.
  * If everything merges into one giant blob -> n_neighbors too high.
  * Pick a value that yields a moderate number of well-sized clusters.

Workflow (do these in order):

1. On the FIRST user turn, call `evaluate_umap_n_neighbors_range` with
   min_n_neighbors=15 and max_n_neighbors=25 (or whatever range the user
   requests).
2. Read the diagnostic metrics. Pick the n_neighbors that best balances the
   "shatter vs. blob" trade-off. Briefly explain your choice (1-3 sentences),
   then call `visualize_umap_2d(n_neighbors=<your_choice>)`. The tool saves a
   PNG and opens it in the user's image viewer.
3. After the visualisation tool call, tell the user which n_neighbors you
   picked, summarise the metrics for it, and ask: "Does this look good, or
   would you like to try a different n_neighbors?"
4. Read the user's natural-language reply:
     * If they approve (e.g. "looks good", "ok", "accept", "go ahead"), call
       `finalize_n_neighbors_and_run(n_neighbors=<approved value>)`. This runs
       the full pipeline (HDBSCAN + KeyBERT + Gemini + report).
     * If they want changes (e.g. "too many tiny clusters", "merge them more",
       "try fewer/larger groups"), translate their wish into a new
       n_neighbors value (lower n_neighbors -> finer / more / smaller clusters;
       higher n_neighbors -> coarser / fewer / bigger clusters) and call
       `visualize_umap_2d` again. You can also call
       `evaluate_umap_n_neighbors_range` with a different range if the user
       asks for it. Repeat step 3.
5. Once `finalize_n_neighbors_and_run` succeeds, present the resulting
   `report_path`, `keyword_csv` and `summary_csv` paths to the user and stop.

Rules:
* Always call exactly one tool per turn unless you are giving the final
  summary.
* Never skip the visualisation step; the human must always see the 2D scatter
  before you finalise.
* Be concise.
"""


def build_agent(model: str = "gemini-flash-lite-latest") -> LlmAgent:
    """Build the orchestration LlmAgent with the three pipeline tools."""
    return LlmAgent(
        name="topic_modelling_orchestrator",
        model=model,
        description=(
            "Picks UMAP n_neighbors with a human in the loop and runs the "
            "Chroma -> UMAP -> HDBSCAN -> KeyBERT -> Gemini topic pipeline."
        ),
        instruction=_AGENT_INSTRUCTION,
        tools=[
            evaluate_umap_n_neighbors_range,
            visualize_umap_2d,
            finalize_n_neighbors_and_run,
        ],
    )


# ===========================================================================
# Interactive driver
# ===========================================================================


async def _run_agent_turn(
    runner: InMemoryRunner,
    user_id: str,
    session_id: str,
    text: str,
) -> str:
    """Send one user message; print assistant text as it streams; return final text."""
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=text)])
    final_text_chunks: list[str] = []
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    if event.is_final_response():
                        final_text_chunks.append(part.text)
                    print(part.text, end="", flush=True)
        if event.is_final_response():
            print()
    return "".join(final_text_chunks).strip()


async def run_interactive(config: PipelineConfig) -> PipelineResult:
    """Run the agent loop until ``finalize_n_neighbors_and_run`` has been called."""
    bootstrap_runtime(config)
    rt = _runtime()
    print(
        f"[bootstrap] loaded {len(rt.df)} documents from "
        f"{config.chroma_db_path} (collection={config.collection_name!r})",
        file=sys.stderr,
    )

    agent = build_agent(model=config.agent_model)
    runner = InMemoryRunner(agent=agent, app_name="topic_modelling_orchestrator")
    user_id = "human-in-the-loop"
    session = await runner.session_service.create_session(
        app_name="topic_modelling_orchestrator", user_id=user_id
    )
    session_id = session.id

    initial = (
        f"Start the topic-modelling pipeline. Sweep n_neighbors from "
        f"{config.umap_min_n_neighbors} to {config.umap_max_n_neighbors}, "
        "pick a candidate, and show me the 2D scatter."
    )
    print("\n=== Topic Modelling Orchestrator ===")
    print("Loaded data, starting agent. The agent will pick an initial n_neighbors,")
    print("show you the 2D scatter, then ask for your feedback.\n")
    print(f"You: {initial}\n")
    print("Agent: ", end="", flush=True)
    await _run_agent_turn(runner, user_id, session_id, initial)

    while rt.result is None:
        try:
            user_msg = input("\nYou: ").strip()
        except EOFError:
            print("\n[input] EOF received; aborting.", file=sys.stderr)
            break
        if not user_msg:
            continue
        if user_msg.lower() in {"quit", "exit", "abort"}:
            print("[input] user aborted.", file=sys.stderr)
            break
        print("Agent: ", end="", flush=True)
        await _run_agent_turn(runner, user_id, session_id, user_msg)

    if rt.result is None:
        raise RuntimeError("Pipeline did not complete (no PipelineResult produced).")
    return rt.result


def generate_topic(
    chroma_db_path: str | Path,
    md_file_path: str | Path,
    *,
    collection_name: str | None = None,
    qwen_model_path: str | Path | None = None,
    gemini_report_model: str = "gemini-flash-lite-latest",
    cluster_summary_path: str | Path = "./out",
    umap_min_n_neighbors: int = 15,
    umap_max_n_neighbors: int = 25,
    agent_model: str | None = None,
) -> PipelineResult:
    """Public entry point: run the agent-driven pipeline interactively.

    Note: Unlike the previous version, ``umap_n_neighbors`` is NOT a parameter;
    it is selected by Gemini with human-in-the-loop approval.
    """
    gemini_settings = load_gemini_settings()
    gemini_settings.model = gemini_report_model
    config = PipelineConfig(
        chroma_db_path=Path(chroma_db_path),
        md_file_path=Path(md_file_path),
        cluster_summary_path=Path(cluster_summary_path),
        collection_name=collection_name,
        qwen_model_path=Path(qwen_model_path) if qwen_model_path else None,
        gemini=gemini_settings,
        umap_min_n_neighbors=umap_min_n_neighbors,
        umap_max_n_neighbors=umap_max_n_neighbors,
        agent_model=agent_model or gemini_report_model,
    )
    return asyncio.run(run_interactive(config))


if __name__ == "__main__":
    chroma_db_path = "../vector-db-writer/data/vector-db/chroma-db"
    md_file_path = "./out/report.md"
    collection_name = "bank_feedback"
    qwen_model_path = "../vector-db-writer/model/Qwen3-Embedding-8B"
    gemini_report_model = "gemini-flash-lite-latest"
    cluster_summary_path = "./out"

    result = generate_topic(
        chroma_db_path=chroma_db_path,
        md_file_path=md_file_path,
        collection_name=collection_name,
        qwen_model_path=qwen_model_path,
        gemini_report_model=gemini_report_model,
        cluster_summary_path=cluster_summary_path,
        umap_min_n_neighbors=15,
        umap_max_n_neighbors=25,
    )
    print("\n=== Pipeline complete ===")
    print(result.model_dump_json(indent=2))
