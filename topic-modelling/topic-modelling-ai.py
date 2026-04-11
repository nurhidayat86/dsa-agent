"""
End-to-end topic pipeline: Chroma load -> L2 embeddings -> UMAP -> HDBSCAN ->
KeyBERT (local Qwen3) -> per-cluster Gemini summaries -> Markdown report (Gemini).

Intended to mirror ``notebook/chrome_viewer.ipynb``. Import ``generate_topic`` or
``generate_topic_model`` from another script, or run ``python topic-modelling-ai.py``
after editing the variables in ``if __name__ == "__main__"``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import chromadb
import hdbscan
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import umap
import yaml
from google import genai
from google.genai import types
from keybert import KeyBERT
from keybert.backend._base import BaseEmbedder
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Repo / Chroma (from notebook)
# ---------------------------------------------------------------------------


def _dsa_agent_repo_root() -> Path:
    """First ancestor of cwd that contains ``vector-db-writer`` (Qwen snapshot path)."""
    p = Path.cwd().resolve()
    for d in (p, *p.parents):
        if (d / "vector-db-writer").is_dir():
            return d
    raise FileNotFoundError(
        "Could not find a parent directory containing vector-db-writer/. "
        "Set the process cwd to the dsa-agent repo root (or a subdirectory) before calling generate_topic, "
        "or install the Qwen3 embedding snapshot under vector-db-writer/model/."
    )


def load_chroma(chroma_path: str | Path, collection_name: str | None = None) -> pd.DataFrame:
    """Load Chroma persistent DB from ``chroma_path`` into a DataFrame (id, document, embedding, meta_*)."""
    path = Path(chroma_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Chroma path is not a directory: {path}")

    client = chromadb.PersistentClient(path=str(path))
    existing = sorted({c.name for c in client.list_collections()}, key=str.lower)

    if not existing:
        raise ValueError(
            f"No collections in {path!s}. Check chroma_db_path or run the ingest pipeline first."
        )

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
            raise ValueError(f"Unexpected embeddings ndarray shape {raw.shape} for n={n_rows}")
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


# ---------------------------------------------------------------------------
# Embeddings / UMAP / clustering (from notebook)
# ---------------------------------------------------------------------------


def get_array(df: pd.DataFrame, col_vector: str) -> np.ndarray:
    """Stack a column of per-row float vectors into a 2D ``float64`` array (n_rows, dim)."""
    if col_vector not in df.columns:
        raise KeyError(f"Column {col_vector!r} not in DataFrame columns: {list(df.columns)}")

    s = df[col_vector]
    if s.isna().any():
        raise ValueError(f"Column {col_vector!r} contains NaN/None; drop or fill those rows before stacking.")

    rows_out: list[np.ndarray] = []
    dim: int | None = None

    for i, v in enumerate(s):
        if v is None:
            raise ValueError(f"Row index {i}: null vector in column {col_vector!r}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Row index {i}: expected a 1D vector, got shape {arr.shape} in {col_vector!r}")
        if dim is None:
            dim = int(arr.shape[0])
        elif int(arr.shape[0]) != dim:
            raise ValueError(
                f"Row index {i}: vector length {arr.shape[0]} != {dim} (ragged vectors in {col_vector!r})"
            )
        rows_out.append(arr)

    if not rows_out:
        return np.empty((0, 0), dtype=np.float64)

    return np.stack(rows_out, axis=0)


def normalize_embeddings_for_clustering(
    X: np.ndarray,
    method: str = "l2",
    **kwargs: Any,
) -> tuple[np.ndarray, Normalizer | StandardScaler | MinMaxScaler]:
    """Normalize embedding matrix for clustering (fit on ``X``)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {X.shape}")

    method = str(method).lower().strip()
    if method == "standard":
        scaler: Normalizer | StandardScaler | MinMaxScaler = StandardScaler(**kwargs)
    elif method == "l2":
        scaler = Normalizer(norm="l2", **kwargs)
    elif method == "minmax":
        scaler = MinMaxScaler(**kwargs)
    else:
        raise ValueError("method must be 'standard', 'l2', or 'minmax'")

    X_out = scaler.fit_transform(X)
    return X_out, scaler


# ---------------------------------------------------------------------------
# KeyBERT + Qwen3 (from notebook)
# ---------------------------------------------------------------------------


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
    """Per cluster KeyBERT keywords (Qwen3 embedding backend)."""
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


# ---------------------------------------------------------------------------
# Gemini helpers (from notebook cells)
# ---------------------------------------------------------------------------


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


def _gemini_api_key() -> str:
    cfg = _find_gemini_config_path()
    if cfg is not None:
        raw = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        g = raw.get("gemini") or {}
        env_name = (g.get("api_key_env") or "GEMINI_API_KEY").strip() or "GEMINI_API_KEY"
        key = (g.get("api_key") or "").strip()
        if key:
            return key
        ev = (os.environ.get(env_name) or "").strip()
        if ev:
            return ev
    fallback = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if fallback:
        return fallback
    raise ValueError(
        "Gemini API key not found: set gemini.api_key (or api_key_env) in a parent config.yaml, "
        "or set GEMINI_API_KEY."
    )


def write_cluster_summary(df: pd.DataFrame, col_doc: str, cluster: str) -> pd.DataFrame:
    """Per-cluster Gemini text summaries. Columns: cluster, text_summary."""
    cfg_path = _find_gemini_config_path()
    if cfg_path is None:
        raise FileNotFoundError(
            "config.yaml with a gemini: block not found above cwd. Place topic-modelling/config.yaml "
            "or set GEMINI_API_KEY."
        )
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    g = raw.get("gemini") or {}

    model = (g.get("model") or "gemini-flash-lite-latest").strip()

    cfg_fields: dict[str, Any] = {}
    if g.get("temperature") is not None:
        cfg_fields["temperature"] = float(g["temperature"])
    if g.get("top_p") is not None:
        cfg_fields["top_p"] = float(g["top_p"])
    if g.get("max_output_tokens") is not None:
        cfg_fields["max_output_tokens"] = int(g["max_output_tokens"])
    gen_config = types.GenerateContentConfig(**cfg_fields) if cfg_fields else None

    timeout_ms = g.get("http_timeout_ms")
    http_options = None
    if timeout_ms is not None and int(timeout_ms) > 0:
        http_options = types.HttpOptions(timeout=int(timeout_ms))

    client = genai.Client(api_key=_gemini_api_key(), http_options=http_options)

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
            "they matter. Plain text only; about 200–400 words unless the cluster is very diverse.\n\n"
            "--- PASSAGES ---\n"
            f"{blob}"
        )

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=gen_config,
        )
        summary = (getattr(resp, "text", None) or "").strip()
        rows.append({"cluster": cid, "text_summary": summary})

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
    md_path: str | Path | None = None,
    model: str = "gemini-flash-lite-latest",
) -> Path:
    """Gemini: short cluster labels + Markdown helicopter report (matches notebook cell)."""
    need_kw = {"keywords", "cluster", "cosine_similarities"}
    need_sm = {"cluster", "text_summary", "cluster_count"}
    if miss := need_kw - set(df_keywords.columns):
        raise ValueError(f"df_keywords missing columns: {sorted(miss)}")
    if miss := need_sm - set(df_summary.columns):
        raise ValueError(f"df_summary missing columns: {sorted(miss)}")

    out = Path("./report.md") if md_path is None else Path(md_path).expanduser()
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=_gemini_api_key())
    cfg_labels = types.GenerateContentConfig(temperature=0.25, max_output_tokens=4096)
    cfg_report = types.GenerateContentConfig(temperature=0.4, max_output_tokens=12_288)

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
        model=model, contents=prompt_labels, config=cfg_labels
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
            f"## {title}\n- **Cluster id:** `{ck}`\n- **Documents in cluster:** {n}\n\n**Summary:**\n{r['text_summary']}\n"
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
            f"- `{row['keywords']}` — relevance {float(row['cosine_similarities']):.3f}"
            for _, row in grp.iterrows()
        ]
        keyword_blocks.append(f"## {title} (`{ck}`)\n" + "\n".join(lines) + "\n")

    prompt_report = (
        "Write a professional Markdown report containing a helicopter view of "
        "customer feedback. Audience is non-technical; avoid terms like embeddings, cosine, or vectors.\n\n"
        "Use the cluster **titles** already chosen (the ## headings in the material below). Do not rename clusters.\n\n"
        "Include these sections with `##` headings (adjust wording but keep intent):\n"
        "1. Executive overview — 4-6 tight bullets on overall tone, main risks, and bright spots.\n"
        "2. Cluster-by-cluster insight — for each cluster, 2-4 bullets linking keywords to the summary; "
        "mention document counts where useful.\n"
        "3. Cross-cutting themes — tensions, recurring patterns across clusters.\n"
        "4. Appendix — the data used to generate this report: (1) keywords per cluster, "
        "(2) text summary and document count per cluster.\n\n"
        "Target length about 700–3000 words. Use Markdown lists and optional **bold** for emphasis.\n\n"
        "--- SOURCE: CLUSTER SUMMARIES ---\n\n"
        + "\n".join(summary_blocks)
        + "\n--- SOURCE: KEYPHRASES ---\n\n"
        + "\n".join(keyword_blocks)
    )

    report_body = client.models.generate_content(
        model=model, contents=prompt_report, config=cfg_report
    ).text
    if not (report_body or "").strip():
        report_body = "_The model returned no text._\n"

    table = ["| Cluster id | Count | Display name (<=3 words) |", "| --- | ---: | --- |"]
    for _, r in sm.iterrows():
        ck = str(r["cluster"])
        disp = str(labels_map.get(ck, "")).replace("|", "/")
        cnt = r["cluster_count"]
        table.append(f"| `{ck}` | {cnt} | {disp} |")

    doc = (
        "<!-- generate_md_report: Gemini labels + narrative. -->\n\n"
        "# Cluster display names\n\n"
        + "\n".join(table)
        + "\n\n---\n\n"
        + report_body.strip()
        + "\n"
    )
    out.write_text(doc, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_topic(
    chroma_db_path: str | Path,
    umap_n_neighbors: int,
    md_file_path: str | Path,
    *,
    collection_name: str | None = None,
    qwen_model_path: str | Path | None = None,
    gemini_report_model: str = "gemini-flash-lite-latest",
    cluster_summary_path: str | Path = "./out",
) -> Path:
    """
    Run the full notebook pipeline and write the Markdown report to ``md_file_path``.

    Parameters
    ----------
    chroma_db_path
        Directory passed to ``chromadb.PersistentClient`` (the ``chroma-db`` folder).
    umap_n_neighbors
        ``n_neighbors`` for ``umap.UMAP`` (clamped to a valid range for the sample size).
    md_file_path
        Output path for the Gemini-generated report.
    collection_name
        Optional Chroma collection name when the DB has more than one collection.
    qwen_model_path
        Optional path to Qwen3-Embedding-8B; default uses ``vector-db-writer/model/...`` under repo root.
    gemini_report_model
        Model id for ``generate_md_report`` (labels + narrative).
    cluster_summary_path
        Directory for ``summary_llm.csv`` and ``keyword_summary.csv`` (created if missing).
    """
    df = load_chroma(chroma_db_path, collection_name=collection_name)
    np_embedding = get_array(df, "embedding")
    if np_embedding.size == 0:
        raise ValueError("No embedding vectors loaded from Chroma.")

    X_all, _ = normalize_embeddings_for_clustering(np_embedding, method="l2")

    n_samples = X_all.shape[0]
    nn = int(umap_n_neighbors)
    nn = max(2, min(nn, max(2, n_samples - 1)))

    reducer = umap.UMAP(
        n_neighbors=nn,
        n_components=5,
        metric="cosine",
        random_state=42,
    )
    reduced_embeddings = reducer.fit_transform(X_all)

    # Notebook used min_cluster_size=10; cap by dataset size for small Chroma exports.
    min_cluster_size = min(10, max(2, n_samples))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
        prediction_data=True,
    )
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    df = df.copy()
    df["cluster"] = cluster_labels

    summary_df = cluster_keywords_keybert(df, model_path=qwen_model_path)
    keyword_summary = summary_df[["cluster", "keywords", "cosine_similarities"]].copy()

    summary_llm = write_cluster_summary(df, "document", "cluster")
    counts = df.groupby("cluster", dropna=False)["document"].count()
    summary_llm = summary_llm.copy()
    summary_llm["cluster_count"] = summary_llm["cluster"].map(lambda c: int(counts.get(c, 0)))
    _out_dir = Path(cluster_summary_path).expanduser().resolve()
    _out_dir.mkdir(parents=True, exist_ok=True)
    summary_llm.to_csv(_out_dir / "summary_llm.csv", index=False)
    keyword_summary.to_csv(_out_dir / "keyword_summary.csv", index=False)
    print("Wrote cluster summary to", _out_dir, file=sys.stderr)

    return generate_md_report(
        keyword_summary,
        summary_llm,
        md_path=md_file_path,
        model=gemini_report_model,
    )
    
# def generate_topic_model(
#     chroma_db_path: str | Path,
#     umap_n_neighbors: int,
#     md_file_path: str | Path,
#     collection_name: str | None = None,
#     qwen_model_path: str | Path = "../vector-db-writer/model",
#     gemini_report_model: str = "gemini-flash-lite-latest",
#     cluster_summary_path: str | Path = "./out",
# ) -> Path:
#     """Run :func:`generate_topic` with the given paths and options; print output path to stderr."""
#     out = generate_topic(
#         chroma_db_path,
#         umap_n_neighbors,
#         md_file_path,
#         collection_name=collection_name,
#         qwen_model_path=qwen_model_path,
#         gemini_report_model=gemini_report_model,
#         cluster_summary_path=cluster_summary_path,
#     )
#     print("Wrote", out, file=sys.stderr)
#     return out


if __name__ == "__main__":
    chroma_db_path = "../vector-db-writer/data/vector-db/chroma-db"
    umap_n_neighbors = 15
    md_file_path = "./out/report.md"
    collection_name="bank_feedback"
    qwen_model_path="../vector-db-writer/model/Qwen3-Embedding-8B"
    gemini_report_model="gemini-flash-lite-latest"
    cluster_summary_path="./out"

    generate_topic(
        chroma_db_path,
        umap_n_neighbors,
        md_file_path,
        collection_name=collection_name,
        qwen_model_path=qwen_model_path,
        gemini_report_model=gemini_report_model,
        cluster_summary_path=cluster_summary_path,
    )
