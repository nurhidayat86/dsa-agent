"""
Load bank-feedback JSONL into a persistent ChromaDB collection.

Normative spec: docs/chroma_bank_feedback_ingestion.md

Embeddings (choose one via ``embedding_provider``):

- **qwen**: locally downloaded **Qwen/Qwen3-Embedding-8B** via ``transformers`` (``model_path``;
  left padding, last-token pooling, L2 normalize).
- **gemini**: **gemini-embedding-001** via ``google.genai`` (API key from arg,
  ``config.yaml`` ``gemini`` block, or ``GEMINI_API_KEY`` — see :func:`resolve_gemini_api_key`).

Chroma default: ``vector-db-writer/data/vector-db/`` (gitignored).

Call from Python::

    from ingest_bank_feedback import ingest_bank_feedback
    ingest_bank_feedback(embedding_provider="gemini", gemini_api_key="...")

Or run this file directly: edit the defaults under ``if __name__ == "__main__"``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any, Literal, Protocol

import chromadb
import yaml
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

EmbeddingProvider = Literal["qwen", "gemini"]


def resolve_gemini_api_key(
    *,
    gemini_api_key: str | None = None,
    config_path: str | Path | None = None,
) -> str | None:
    """Resolve Gemini API key: explicit arg, then ``vector-db-writer/config.yaml``, then env.

    When ``config.yaml`` exists, uses the same rules as ``ai-data-generator``:
    ``gemini.api_key`` if non-empty, else ``os.environ[gemini.api_key_env]`` (default
    ``GEMINI_API_KEY``).

    If the config file is missing or yields no key, falls back to ``GEMINI_API_KEY``.
    """
    if gemini_api_key is not None and str(gemini_api_key).strip():
        return str(gemini_api_key).strip()

    cfg = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else Path(__file__).resolve().parent / "config.yaml"
    )
    if cfg.is_file():
        with cfg.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        g = raw.get("gemini") or {}
        env_name = (g.get("api_key_env") or "GEMINI_API_KEY").strip() or "GEMINI_API_KEY"
        key = (g.get("api_key") or "").strip()
        if key:
            return key
        from_env = (os.environ.get(env_name) or "").strip()
        if from_env:
            return from_env

    return (os.environ.get("GEMINI_API_KEY") or "").strip() or None


def resolve_local_model_path(model_path: str | Path) -> Path:
    """Resolve and verify a local Hugging Face model directory (or cache layout)."""
    p = Path(model_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model path does not exist: {p}")
    return p


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Qwen3-Embedding: hidden state at last non-padding token (HF README)."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def normalize_chroma_metadata(
    row: dict[str, Any],
    *,
    source_file: str,
    omit_case_id: bool = True,
) -> dict[str, str | int | float | bool]:
    """Flat metadata: all fields except ``verbatim``; Chroma scalar types only."""
    meta: dict[str, str | int | float | bool] = {"source_file": source_file}
    for key, val in row.items():
        if key == "verbatim":
            continue
        if omit_case_id and key == "case_id":
            continue
        if val is None:
            meta[key] = ""
        elif isinstance(val, bool):
            meta[key] = val
        elif isinstance(val, (int, float)):
            meta[key] = val
        else:
            meta[key] = str(val)
    return meta


def document_text(row: dict[str, Any], *, include_subject: bool) -> str | None:
    verb = row.get("verbatim")
    if verb is None or not str(verb).strip():
        return None
    text = str(verb).strip()
    if include_subject:
        sub = row.get("subject")
        if sub and str(sub).strip():
            text = f"{str(sub).strip()}\n\n{text}"
    return text


def iter_jsonl(path: Path) -> Generator[tuple[int, dict[str, Any]], None, None]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error("%s line %s: JSON error: %s", path, line_no, e)
                continue
            if not isinstance(obj, dict):
                logger.error("%s line %s: expected object, got %s", path, line_no, type(obj))
                continue
            yield line_no, obj


def chroma_id_for_row(
    row: dict[str, Any],
    *,
    source_stem: str,
    mode: str,
) -> str | None:
    cid = row.get("case_id")
    if cid is None or str(cid).strip() == "":
        return None
    cid = str(cid).strip()
    if mode == "composite":
        return f"{source_stem}:{cid}"
    return cid


class _EmbedBatch(Protocol):
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


def _vectors_from_gemini_embed_response(resp: Any) -> list[list[float]]:
    """Parse ``client.models.embed_content`` response into float vectors."""
    embeddings = getattr(resp, "embeddings", None)
    if embeddings:
        out: list[list[float]] = []
        for item in embeddings:
            vals = getattr(item, "values", None)
            if vals is None:
                raise ValueError(f"Gemini embedding entry missing .values: {item!r}")
            out.append([float(x) for x in vals])
        return out
    single = getattr(resp, "embedding", None)
    if single is not None:
        vals = getattr(single, "values", None)
        if vals is None:
            raise ValueError(f"Gemini response.embedding missing .values: {resp!r}")
        return [[float(x) for x in vals]]
    raise ValueError(f"Unexpected Gemini embed response shape: {type(resp)!r}")


class Qwen3EmbeddingBackend:
    """Local Qwen3-Embedding-8B via Hugging Face ``transformers``."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: torch.device,
        max_length: int,
        dtype: torch.dtype,
    ) -> None:
        resolved = resolve_local_model_path(model_path)
        self.model_path = resolved
        self.device = device
        self.max_length = max_length
        local = str(resolved)
        self.tokenizer = AutoTokenizer.from_pretrained(local, padding_side="left", local_files_only=True)
        self.model = AutoModel.from_pretrained(local, torch_dtype=dtype, local_files_only=True)
        self.model.to(device)
        self.model.eval()

    def chroma_collection_metadata(self, *, max_length: int, id_mode: str, include_subject: bool) -> dict[str, str]:
        return {
            "embedding_provider": "qwen",
            "embedding_model": "Qwen/Qwen3-Embedding-8B",
            "embedding_model_path": str(self.model_path),
            "max_length": str(max_length),
            "l2_normalize": "true",
            "pooling": "last_token",
            "document_mode": "subject_plus_verbatim" if include_subject else "verbatim_only",
            "id_mode": id_mode,
        }

    @torch.inference_mode()
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        emb = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        return emb.detach().float().cpu().numpy().tolist()


class GeminiEmbeddingBackend:
    """Gemini embedding models (e.g. ``gemini-embedding-001``) via ``google.genai``."""

    def __init__(self, model: str, *, api_key: str) -> None:
        from google import genai

        self._model = model
        self._client = genai.Client(api_key=api_key)

    def chroma_collection_metadata(
        self,
        *,
        max_length: str,
        id_mode: str,
        include_subject: bool,
    ) -> dict[str, str]:
        return {
            "embedding_provider": "gemini",
            "embedding_model": self._model,
            "embedding_model_path": "",
            "max_length": max_length,
            "l2_normalize": "gemini_api_default",
            "pooling": "gemini_embed_content",
            "document_mode": "subject_plus_verbatim" if include_subject else "verbatim_only",
            "id_mode": id_mode,
        }

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.models.embed_content(
            model=self._model,
            contents=texts,
        )
        vecs = _vectors_from_gemini_embed_response(resp)
        if len(vecs) != len(texts):
            raise ValueError(
                f"Gemini returned {len(vecs)} vectors for {len(texts)} inputs (model={self._model})"
            )
        return vecs


def load_rows(
    jsonl_paths: Iterable[Path],
    *,
    id_mode: str,
    include_subject_in_document: bool,
) -> tuple[list[str], list[str], list[dict[str, str | int | float | bool]], int, int]:
    """Returns ids, documents, metadatas, n_skipped_bad, n_lines_seen."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []
    skipped = 0
    seen = 0
    for path in sorted(jsonl_paths, key=lambda p: p.as_posix().lower()):
        stem = path.stem
        source_file = path.name
        for line_no, row in iter_jsonl(path):
            seen += 1
            cid = chroma_id_for_row(row, source_stem=stem, mode=id_mode)
            if cid is None:
                logger.warning("%s line %s: missing case_id, skipping", path, line_no)
                skipped += 1
                continue
            doc = document_text(row, include_subject=include_subject_in_document)
            if doc is None:
                logger.warning("%s line %s: missing/empty verbatim, skipping", path, line_no)
                skipped += 1
                continue
            meta = normalize_chroma_metadata(row, source_file=source_file)
            ids.append(cid)
            documents.append(doc)
            metadatas.append(meta)
    return ids, documents, metadatas, skipped, seen


def _run_ingestion(
    *,
    backend: _EmbedBatch,
    coll_meta: dict[str, str],
    jsonl_dir: Path,
    chroma_path: Path,
    collection_name: str,
    batch_size: int,
    id_mode: str,
    include_subject_in_document: bool,
    reset_collection: bool,
) -> None:
    paths = sorted(jsonl_dir.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"No .jsonl files under {jsonl_dir}")

    ids, documents, metadatas, skipped_load, seen = load_rows(
        paths,
        id_mode=id_mode,
        include_subject_in_document=include_subject_in_document,
    )
    if not ids:
        raise SystemExit("No valid rows to ingest after parsing.")

    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))

    emb_dim = len(backend.embed_batch([documents[0]])[0])
    coll_meta = {**coll_meta, "embedding_dim": str(emb_dim)}

    if reset_collection:
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted existing collection %r", collection_name)
        except Exception:
            logger.info("No existing collection to delete (%r)", collection_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata=coll_meta,
    )

    total = len(ids)
    logger.info(
        "Upserting %s points (jsonl lines seen=%s, skipped=%s)",
        total,
        seen,
        skipped_load,
    )

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        b_ids = ids[start:end]
        b_docs = documents[start:end]
        b_meta = metadatas[start:end]
        b_emb = backend.embed_batch(b_docs)
        collection.upsert(
            ids=b_ids,
            embeddings=b_emb,
            documents=b_docs,
            metadatas=b_meta,
        )
        logger.info("Upserted %s/%s", end, total)

    logger.info("Done. Chroma path=%s collection=%r", chroma_path, collection_name)


def ingest_bank_feedback(
    *,
    jsonl_dir: str | Path | None = None,
    chroma_path: str | Path | None = None,
    collection_name: str = "bank_feedback",
    embedding_provider: EmbeddingProvider = "qwen",
    model_path: str | Path | None = None,
    gemini_model: str = "gemini-embedding-001",
    gemini_api_key: str | None = None,
    gemini_config_path: str | Path | None = None,
    batch_size: int = 4,
    max_length: int = 8192,
    id_mode: Literal["case_id", "composite"] = "case_id",
    include_subject_in_document: bool = False,
    reset_collection: bool = False,
    configure_logging: bool = False,
    log_level: int = logging.INFO,
) -> None:
    """Load *jsonl_dir/*.jsonl* into a Chroma collection at *chroma_path*.

    embedding_provider: ``qwen`` (local HF snapshot at model_path) or ``gemini``
    (Gemini API; default model ``gemini-embedding-001`` via gemini_model).

    For ``gemini``, the API key is resolved by :func:`resolve_gemini_api_key`:
    non-empty ``gemini_api_key`` argument, else ``gemini_config_path`` (default
    ``vector-db-writer/config.yaml``) ``gemini.api_key`` / ``api_key_env``, else
    ``GEMINI_API_KEY``.

    If configure_logging is True and the root logger has no handlers yet, applies
    ``logging.basicConfig`` with log_level.
    """
    if configure_logging and not logging.root.handlers:
        logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")

    _pkg = Path(__file__).resolve().parent
    root_jsonl = Path(jsonl_dir).resolve() if jsonl_dir is not None else _pkg / "data" / "bank_feedback"
    root_chroma = Path(chroma_path).resolve() if chroma_path is not None else _pkg / "data" / "vector-db"

    bs = max(1, batch_size)
    ml = max(32, max_length)

    if embedding_provider == "qwen":
        if model_path is None:
            raise ValueError('embedding_provider="qwen" requires model_path (local Qwen3 snapshot directory)')
        resolved_model = resolve_local_model_path(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        logger.info("Device=%s dtype=%s model_path=%s", device, dtype, resolved_model)
        backend: _EmbedBatch = Qwen3EmbeddingBackend(
            resolved_model, device=device, max_length=ml, dtype=dtype
        )
        meta = backend.chroma_collection_metadata(
            max_length=ml, id_mode=id_mode, include_subject=include_subject_in_document
        )
    elif embedding_provider == "gemini":
        api_key = resolve_gemini_api_key(
            gemini_api_key=gemini_api_key, config_path=gemini_config_path
        )
        if not api_key:
            raise ValueError(
                "embedding_provider='gemini' requires a Gemini API key: pass gemini_api_key, "
                "set gemini.api_key (or api_key_env) in vector-db-writer/config.yaml, "
                "or set GEMINI_API_KEY in the environment."
            )
        # Accept common typo / alias (user asked for gemini-embedding-001 twice)
        model_id = gemini_model.strip() or "gemini-embedding-001"
        logger.info("Gemini embedding model=%s", model_id)
        backend = GeminiEmbeddingBackend(model_id, api_key=api_key)
        meta = backend.chroma_collection_metadata(
            max_length=str(ml),
            id_mode=id_mode,
            include_subject=include_subject_in_document,
        )
    else:
        raise ValueError(f"Unknown embedding_provider: {embedding_provider!r}")

    _run_ingestion(
        backend=backend,
        coll_meta=meta,
        jsonl_dir=root_jsonl,
        chroma_path=root_chroma,
        collection_name=collection_name,
        batch_size=bs,
        id_mode=id_mode,
        include_subject_in_document=include_subject_in_document,
        reset_collection=reset_collection,
    )


__all__ = [
    "ingest_bank_feedback",
    "resolve_gemini_api_key",
    "Qwen3EmbeddingBackend",
    "GeminiEmbeddingBackend",
]


if __name__ == "__main__":
    # Paths and defaults for direct script runs — edit this block only.
    _PACKAGE_ROOT = Path(__file__).resolve().parent

    _EMBEDDING_PROVIDER: EmbeddingProvider = "qwen"
    _JSONL_DIR: Path = _PACKAGE_ROOT / "data" / "bank_feedback"
    _CHROMA_PATH: Path = _PACKAGE_ROOT / "data" / "vector-db" / "qwen"
    _COLLECTION = "bank_feedback"
    # _MODEL_PATH: Path | None = None  # r"F:\models\Qwen3-Embedding-8B" when _EMBEDDING_PROVIDER == "qwen"
    _MODEL_PATH: Path = _PACKAGE_ROOT / "model" / "Qwen3-Embedding-8B"
    _GEMINI_MODEL = "gemini-embedding-001"
    _GEMINI_API_KEY: str | None = None  # overrides yaml/env if set
    _GEMINI_CONFIG_PATH: Path | None = _PACKAGE_ROOT / "config.yaml"
    _BATCH_SIZE = 4
    _MAX_LENGTH = 8192
    _ID_MODE: Literal["case_id", "composite"] = "case_id"
    _INCLUDE_SUBJECT = False
    _RESET_COLLECTION = False

    ingest_bank_feedback(
        jsonl_dir=_JSONL_DIR,
        chroma_path=_CHROMA_PATH,
        collection_name=_COLLECTION,
        embedding_provider=_EMBEDDING_PROVIDER,
        model_path=_MODEL_PATH,
        gemini_model=_GEMINI_MODEL,
        gemini_api_key=_GEMINI_API_KEY,
        gemini_config_path=_GEMINI_CONFIG_PATH,
        batch_size=_BATCH_SIZE,
        max_length=_MAX_LENGTH,
        id_mode=_ID_MODE,
        include_subject_in_document=_INCLUDE_SUBJECT,
        reset_collection=_RESET_COLLECTION,
        configure_logging=True,
        log_level=logging.INFO,
    )
