"""
Generate synthetic raw-layer bank feedback and complaints (JSON records).

Schema aligns with docs/bank_branch_raw_feedback_context.md.
Callable from external scripts: add `ai-data-generator` to PYTHONPATH, then:
    from bank_feedback_generator import generate_bank_feedback_data
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PACKAGE_ROOT / "config.yaml"

_CHANNELS = [
    "branch_counter",
    "branch_tablet",
    "phone",
    "email",
    "mobile_app",
    "internet_banking",
    "social",
    "ombudsman_escalation",
]

_STATUS_WEIGHTS: list[tuple[str, float]] = [
    ("open", 0.15),
    ("in_progress", 0.25),
    ("resolved", 0.35),
    ("closed", 0.15),
    ("escalated", 0.05),
]

_REGION_POOL = [
    "JAVA",
    "SUMATERA",
    "KALIMANTAN",
    "SULAWESI",
    "BALI_NUSA",
    "PAPUA",
    "MALUKU",
]

_PAD_PHRASES_EN = (
    " The customer expects a timely written response.",
    " Additional documentation was offered but not accepted at the time.",
    " The issue affects everyday banking and caused inconvenience.",
    " Follow-up was requested regarding fee calculation and disclosure.",
    " The branch representative could not provide a definitive answer.",
)

_PAD_PHRASES_ID = (
    " Nasabah mengharapkan tanggapan tertulis secepatnya.",
    " Pihak cabang belum dapat memberikan penjelasan yang memuaskan.",
    " Masalah ini berdampak pada aktivitas perbankan sehari-hari.",
    " Nasabah meminta klarifikasi terkait biaya dan ketentuan produk.",
    " Dokumentasi tambahan telah ditawarkan tetapi belum diselesaikan di tempat.",
)


def _normalize_complaint_language(lang: str | None) -> tuple[str, str]:
    """Return (language_code for records, human phrase for the model prompt)."""
    s = (lang or "en").strip().lower()
    if s in ("en", "english"):
        return "en", "English"
    if s in ("id", "indonesian", "bahasa", "bahasa indonesia", "in"):
        return "id", "Bahasa Indonesia"
    if len(s) == 2 and s.isalpha():
        return s, f"the language with ISO 639-1 code {s!r} (write fluently in that language)"
    raise ValueError(
        "complaint_language must be 'en', 'id', or a two-letter ISO 639-1 code; "
        f"got {lang!r}"
    )


def _assign_target_char_counts(specs: list[dict[str, Any]], rng: random.Random) -> None:
    """Set target_chars per spec; ensure each type hits min and max when possible."""
    std_ix = [i for i, s in enumerate(specs) if s["submission_type"] == "standard_feedback_form"]
    det_ix = [i for i, s in enumerate(specs) if s["submission_type"] == "detailed_complaint_ticket"]

    def fill(indices: list[int]) -> None:
        if not indices:
            return
        for i in indices:
            lo, hi = specs[i]["min_chars"], specs[i]["max_chars"]
            if lo > hi:
                raise ValueError("min_chars cannot exceed max_chars for a row type")

        if len(indices) == 1:
            i = indices[0]
            lo, hi = specs[i]["min_chars"], specs[i]["max_chars"]
            if lo == hi:
                specs[i]["target_chars"] = lo
            else:
                specs[i]["target_chars"] = rng.choice([lo, hi, rng.randint(lo, hi)])
            return

        lo0 = specs[indices[0]]["min_chars"]
        hi1 = specs[indices[1]]["max_chars"]
        specs[indices[0]]["target_chars"] = lo0
        specs[indices[1]]["target_chars"] = hi1
        for i in indices[2:]:
            lo, hi = specs[i]["min_chars"], specs[i]["max_chars"]
            specs[i]["target_chars"] = rng.randint(lo, hi)

    fill(std_ix)
    fill(det_ix)


@dataclass(frozen=True)
class _GenConfig:
    api_key: str
    model: str
    temperature: float
    top_p: float | None
    max_output_tokens: int
    http_timeout_ms: int | None
    generator_version: str
    timezone_default: str
    chunk_size: int
    anonymous_customer_fraction: float
    max_retries: int
    retry_backoff_seconds: float


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_gen_config(raw: dict[str, Any]) -> _GenConfig:
    g = raw.get("gemini") or {}
    gen = raw.get("generator") or {}
    env_name = g.get("api_key_env") or "GEMINI_API_KEY"
    key = (g.get("api_key") or "").strip()
    if not key:
        key = os.environ.get(env_name, "").strip()
    if not key:
        raise ValueError(
            f"Missing Gemini API key: set environment variable {env_name!r} "
            f"or set gemini.api_key in config.yaml (not recommended for production)."
        )
    if "http_timeout_ms" in g:
        raw_timeout = g["http_timeout_ms"]
        if raw_timeout in (None, "", 0, "0"):
            http_timeout_ms: int | None = None
        else:
            http_timeout_ms = int(raw_timeout)
    else:
        http_timeout_ms = 900_000

    return _GenConfig(
        api_key=key,
        model=g.get("model") or "gemini-2.0-flash",
        temperature=float(g.get("temperature") if g.get("temperature") is not None else 0.85),
        top_p=float(g["top_p"]) if g.get("top_p") is not None else None,
        max_output_tokens=int(g.get("max_output_tokens") or 8192),
        http_timeout_ms=http_timeout_ms,
        generator_version=str(gen.get("generator_version") or "0.1.0"),
        timezone_default=str(gen.get("timezone_default") or "+07:00"),
        chunk_size=max(1, int(gen.get("chunk_size") or 10)),
        anonymous_customer_fraction=float(gen.get("anonymous_customer_fraction") or 0.2),
        max_retries=max(1, int(gen.get("max_retries") or 3)),
        retry_backoff_seconds=float(gen.get("retry_backoff_seconds") or 1.5),
    )


def _pick_status(rng: random.Random) -> str:
    r = rng.random()
    acc = 0.0
    for status, w in _STATUS_WEIGHTS:
        acc += w
        if r <= acc:
            return status
    return "closed"


def _branch_codes(num_branches: int, rng: random.Random) -> list[tuple[str, str]]:
    regions = list(_REGION_POOL)
    rng.shuffle(regions)
    out: list[tuple[str, str]] = []
    for i in range(num_branches):
        reg = regions[i % len(regions)]
        abbr = reg.replace("_", "")[:4]
        code = f"BR-{abbr}-{i + 1:03d}"
        out.append((code, reg))
    return out


def _parse_created_date(created_date: str | date | datetime | None) -> date | None:
    """Accept ``None``, :class:`datetime.date`, :class:`datetime.datetime`, or ISO ``YYYY-MM-DD``."""
    if created_date is None:
        return None
    if isinstance(created_date, datetime):
        return created_date.date()
    if isinstance(created_date, date):
        return created_date
    text = str(created_date).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as e:
        raise ValueError(
            f"created_date must be ISO format YYYY-MM-DD, got {created_date!r}"
        ) from e


def _random_timestamps(
    rng: random.Random,
    tz: str,
    fixed_calendar_date: date | None = None,
) -> tuple[str, str | None]:
    """Return (created_at iso, updated_at iso | None).

    If ``fixed_calendar_date`` is set, both timestamps fall on that calendar day
    with random times (full 24h). ``updated_at`` is never earlier than
    ``created_at`` on that day. Otherwise dates are randomized (legacy behaviour).
    """
    if fixed_calendar_date is not None:
        year = fixed_calendar_date.year
        month = fixed_calendar_date.month
        day = fixed_calendar_date.day
        c_sec = rng.randint(0, 24 * 60 * 60 - 1)
        hour = c_sec // 3600
        minute = (c_sec % 3600) // 60
        sec = c_sec % 60
    else:
        year = 2026
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        hour = rng.randint(8, 18)
        minute = rng.randint(0, 59)
        sec = rng.randint(0, 59)
    created = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{sec:02d}{tz}"
    if rng.random() < 0.35:
        return created, None
    if fixed_calendar_date is not None:
        u_sec = rng.randint(c_sec, 24 * 60 * 60 - 1)
        h2 = u_sec // 3600
        m2 = (u_sec % 3600) // 60
        s2 = u_sec % 60
        updated = f"{year:04d}-{month:02d}-{day:02d}T{h2:02d}:{m2:02d}:{s2:02d}{tz}"
        return created, updated
    dh = rng.randint(1, 72)
    nh = hour + (dh % 12)
    nm = (minute + dh) % 60
    updated = f"{year:04d}-{month:02d}-{day:02d}T{min(18, nh):02d}:{nm:02d}:{sec:02d}{tz}"
    return created, updated


def _assign_customers(
    n_records: int,
    min_customers: int,
    rng: random.Random,
) -> tuple[list[int], int]:
    """Return (assignments, n_distinct) with indices in 0..n_distinct-1."""
    n_distinct = rng.randint(min_customers, n_records)
    indices = list(range(n_distinct))
    assignments = [rng.choice(indices) for _ in range(n_records)]
    return assignments, n_distinct


def _build_customer_refs(n_distinct: int, branch_idx: int) -> list[str]:
    """One stable synthetic token per customer index within this branch."""
    return [
        f"SYN-CUST-B{branch_idx:02d}-U{idx:03d}-{uuid.uuid4().hex[:8].upper()}"
        for idx in range(n_distinct)
    ]


def _build_prompt_rows(
    specs: list[dict[str, Any]],
    branch_code: str,
    region: str,
    lang_instruction: str,
    lang_code: str,
    fixed_calendar_date: date | None = None,
) -> str:
    lines = []
    for i, s in enumerate(specs, start=1):
        lines.append(
            f"{i}. submission_type={s['submission_type']!r}, "
            f"min_chars_verbatim={s['min_chars']}, "
            f"max_chars_verbatim={s['max_chars']}, "
            f"target_chars_verbatim={s['target_chars']}"
        )
    rows_block = "\n".join(lines)
    channels = ", ".join(_CHANNELS)
    intake_block = ""
    if fixed_calendar_date is not None:
        iso_d = fixed_calendar_date.isoformat()
        intake_block = f"""
Intake date (must match downstream metadata — critical):
- Every record in this batch is filed on calendar date {iso_d!r}. In "subject" and "verbatim", do not assert a different submission or visit calendar date.
- Avoid month/day/year phrases that contradict {iso_d!r} (e.g. another month name, "last Tuesday" unless it is actually Tuesday on that date, "15th of this month" unless the day matches).
- Prefer neutral timing: "today", "this visit", "during this interaction", "recently", "over the past few weeks" without naming a conflicting calendar day.
"""
    return f"""You create synthetic customer complaints and feedback for a large retail bank (training data only).

Hard rules:
- No real personal data: do not use realistic full names, phone numbers, national IDs, or street addresses.
- You may say "the customer", "a teller", or use obviously fake tokens like "Customer A".
- Output must be valid JSON only, with exactly one object containing key "items" (array).
- The array length must equal {len(specs)}; order must match the numbered rows below.

Language (critical):
- Write every "subject" and "verbatim" in {lang_instruction}. Do not mix other languages unless a short product name appears.
{intake_block}
- Each item must include these keys:
  - "submission_type": string, must match exactly the value given on that row.
  - "subject": string or null (short title in {lang_instruction}; null allowed).
  - "verbatim": string, the main complaint/feedback body.
  - "verbatim" length in characters must be >= min_chars_verbatim AND <= max_chars_verbatim for that row.
  - Aim the length close to target_chars_verbatim (still staying within min and max).
  - In JSON output, "verbatim" and "subject" must be single-line strings: use \\n for line breaks and escape any " as \\" inside the text (no raw double-quotes or unescaped newlines inside JSON strings).
  - "product_bucket": one of "account", "lending", "card", "fees", "digital_channel", "mortgage", "other", or null.
  - "channel": one of: {channels}
  - "language": must be the string {lang_code!r} (the complaint language code for this dataset).
  - "has_attachments": boolean

Branch context (for tone only): branch_code={branch_code!r}, region={region!r}.

Rows (generate one item per row, same order):
{rows_block}
"""


def _parse_gemini_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object from the model response.

    Gemini sometimes returns valid JSON followed by extra prose or a second JSON
    fragment, which makes ``json.loads`` raise "Extra data". We use
    :meth:`json.JSONDecoder.raw_decode` and ignore trailing bytes after the
    first complete value.
    """
    text = text.strip()
    if text.startswith("\ufeff"):
        text = text[1:].strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found in model response", text, 0)

    decoder = json.JSONDecoder()
    obj, end = decoder.raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON must be an object with an 'items' array")
    trailing = text[start + end :].strip()
    if trailing:
        logger.debug("Ignoring %s chars after first JSON object", len(trailing))
    return obj


def _batch_items_response_schema(num_items: int, lang_code: str) -> Any:
    """Gemini structured output schema: ``{{"items": [ ... ]}}`` with fixed array length."""
    from google.genai import types

    item_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "submission_type": types.Schema(
                type=types.Type.STRING,
                enum=["standard_feedback_form", "detailed_complaint_ticket"],
            ),
            "subject": types.Schema(type=types.Type.STRING, nullable=True),
            "verbatim": types.Schema(type=types.Type.STRING),
            "product_bucket": types.Schema(type=types.Type.STRING, nullable=True),
            "channel": types.Schema(type=types.Type.STRING, enum=list(_CHANNELS)),
            "language": types.Schema(
                type=types.Type.STRING,
                enum=[lang_code],
            ),
            "has_attachments": types.Schema(type=types.Type.BOOLEAN),
        },
        required=[
            "submission_type",
            "verbatim",
            "channel",
            "language",
            "has_attachments",
        ],
    )
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "items": types.Schema(
                type=types.Type.ARRAY,
                items=item_schema,
                min_items=num_items,
                max_items=num_items,
            ),
        },
        required=["items"],
    )


def _call_gemini_batch(
    client: Any,
    cfg: _GenConfig,
    prompt: str,
    seed: int | None,
    num_items: int,
    lang_code: str,
    *,
    branch_code: str,
    chunk_index: int,
    total_chunks: int,
) -> dict[str, Any]:
    from google.genai import types

    response_schema = _batch_items_response_schema(num_items, lang_code)
    gcfg = types.GenerateContentConfig(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_output_tokens=cfg.max_output_tokens,
        response_mime_type="application/json",
        response_schema=response_schema,
        seed=seed,
    )
    last_err: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            logger.info(
                "Gemini: branch=%s chunk %s/%s (%s items, model=%s) — "
                "large max_chars / max_output_tokens can make this take many minutes",
                branch_code,
                chunk_index + 1,
                total_chunks,
                num_items,
                cfg.model,
            )
            t0 = time.perf_counter()
            response = client.models.generate_content(
                model=cfg.model,
                contents=prompt,
                config=gcfg,
            )
            elapsed = time.perf_counter() - t0
            raw = getattr(response, "text", None) or ""
            if not raw.strip():
                raise ValueError("Empty response text from Gemini")
            logger.info(
                "Gemini: branch=%s chunk %s/%s done in %.1fs (%s response chars)",
                branch_code,
                chunk_index + 1,
                total_chunks,
                elapsed,
                len(raw),
            )
            return _parse_gemini_json(raw)
        except Exception as e:
            last_err = e
            logger.warning("Gemini request failed (attempt %s/%s): %s", attempt + 1, cfg.max_retries, e)
            if attempt + 1 < cfg.max_retries:
                time.sleep(cfg.retry_backoff_seconds * (attempt + 1))
    assert last_err is not None
    raise last_err


def _clamp_verbatim(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _pad_verbatim_to_min(text: str, min_len: int, rng: random.Random, lang_code: str) -> str:
    if len(text) >= min_len:
        return text
    phrases = _PAD_PHRASES_ID if lang_code == "id" else _PAD_PHRASES_EN
    out = text.rstrip()
    guard = 0
    while len(out) < min_len and guard < 2000:
        out = (out + rng.choice(phrases)).strip()
        guard += 1
    if len(out) < min_len:
        frag = rng.choice(phrases).strip()
        while len(out) < min_len:
            need = min_len - len(out)
            out = out + frag[:need] if frag else " x"[:need]
    return out


def _finalize_verbatim(
    text: str,
    min_len: int,
    max_len: int,
    rng: random.Random,
    lang_code: str,
) -> str:
    """Ensure min <= len(verbatim) <= max after model output."""
    t = text.strip()
    t = _pad_verbatim_to_min(t, min_len, rng, lang_code)
    return _clamp_verbatim(t, max_len)


def _branch_jsonl_filename(branch_code: str) -> str:
    """Safe filename stem from branch_code (one .jsonl file per branch)."""
    forbidden = '<>:"/\\|?*'
    safe = "".join(ch if ch not in forbidden else "_" for ch in branch_code.strip())
    if not safe:
        safe = "branch_unknown"
    return f"{safe}.jsonl"


def _write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def generate_bank_feedback_data(
    num_branches: int,
    min_records_per_branch: int,
    max_records_per_branch: int,
    allow_anonymous_customer_id: bool,
    min_customers_per_branch: int,
    min_chars_standard_feedback: int,
    max_chars_standard_feedback: int,
    min_chars_detailed_complaint: int,
    max_chars_detailed_complaint: int,
    *,
    complaint_language: str = "en",
    created_date: str | date | datetime | None = None,
    config_path: str | Path | None = None,
    output_jsonl_dir: str | Path | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Generate synthetic bank_complaint_raw records using Gemini.

    Parameters
    ----------
    num_branches :
        Number of distinct branches (each gets a branch_code).
    min_records_per_branch / max_records_per_branch :
        Inclusive bounds; per branch a random count is drawn.
    allow_anonymous_customer_id :
        If True, a fraction of records may have blank ``customer_ref`` (not mappable).
    min_customers_per_branch :
        Minimum distinct synthetic customers appearing in that branch's batch
        (each non-anonymous record maps to one of these unless blanked).
    min_chars_standard_feedback / max_chars_standard_feedback :
        Inclusive character bounds for ``verbatim`` on ``standard_feedback_form`` rows.
        When there are at least two such rows in a branch, one targets the minimum length
        and one the maximum; others are random within bounds.
    min_chars_detailed_complaint / max_chars_detailed_complaint :
        Same for ``detailed_complaint_ticket`` rows.
    complaint_language :
        Language for ``subject`` and ``verbatim`` (default ``\"en\"``). Accepts
        ``\"en\"``, ``\"english\"``, ``\"id\"``, ``\"indonesian\"``, ``\"bahasa\"``, or a
        two-letter ISO 639-1 code. Stored on each record as ``language``.
    config_path :
        Path to config.yaml (defaults to ``ai-data-generator/config.yaml``).
    output_jsonl_dir :
        If set, writes **one UTF-8 JSONL file per branch** under this directory,
        named ``{branch_code}.jsonl`` (unsafe path characters in ``branch_code``
        are replaced). Each line is one JSON object.
    seed :
        Random seed for reproducible branch sizes, assignments, and Gemini seed when supported.
    created_date :
        If ``None`` (default), ``created_at`` / ``updated_at`` dates are randomized as before.
        If set (ISO ``\"YYYY-MM-DD\"`` string, :class:`datetime.date`, or :class:`datetime.datetime`),
        every record uses that calendar day with random wall-clock times on that day.
        ``case_id`` uses this date’s year (e.g. ``BK-2025-…``).

    Returns
    -------
    list[dict]
        One dict per record (same schema as written under ``output_jsonl_dir`` when set).
    """
    fixed_calendar_date = _parse_created_date(created_date)

    if num_branches < 1:
        raise ValueError("num_branches must be >= 1")
    if min_records_per_branch < 1 or max_records_per_branch < min_records_per_branch:
        raise ValueError("Invalid min_records_per_branch / max_records_per_branch")
    if min_customers_per_branch < 1:
        raise ValueError("min_customers_per_branch must be >= 1")
    if min_customers_per_branch > min_records_per_branch:
        raise ValueError(
            "min_customers_per_branch cannot exceed min_records_per_branch "
            "(otherwise not enough rows to assign distinct customers)."
        )
    if min_chars_standard_feedback < 1 or min_chars_detailed_complaint < 1:
        raise ValueError("min character limits must be >= 1")
    if min_chars_standard_feedback > max_chars_standard_feedback:
        raise ValueError(
            "min_chars_standard_feedback cannot exceed max_chars_standard_feedback"
        )
    if min_chars_detailed_complaint > max_chars_detailed_complaint:
        raise ValueError(
            "min_chars_detailed_complaint cannot exceed max_chars_detailed_complaint"
        )

    lang_code, lang_instruction = _normalize_complaint_language(complaint_language)

    case_id_year = fixed_calendar_date.year if fixed_calendar_date is not None else 2026
    if fixed_calendar_date is not None:
        logger.info(
            "All records use fixed intake date %s (created_at/updated_at on this day only; times vary).",
            fixed_calendar_date.isoformat(),
        )

    cfg_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    raw_cfg = _load_yaml_config(cfg_path)
    cfg = _resolve_gen_config(raw_cfg)

    rng = random.Random(seed)
    branches = _branch_codes(num_branches, rng)

    from google import genai
    from google.genai import types as genai_types

    http_opts: genai_types.HttpOptions | None = None
    if cfg.http_timeout_ms is not None:
        http_opts = genai_types.HttpOptions(timeout=cfg.http_timeout_ms)
        logger.info(
            "HTTP timeout is %s ms per request (set gemini.http_timeout_ms in config; use 0 for SDK default)",
            cfg.http_timeout_ms,
        )

    client = genai.Client(api_key=cfg.api_key, http_options=http_opts)

    all_records: list[dict[str, Any]] = []
    case_seq = 0

    for branch_idx, (branch_code, region) in enumerate(branches):
        n_branch = rng.randint(min_records_per_branch, max_records_per_branch)
        n_standard = rng.randint(0, n_branch)
        n_detailed = n_branch - n_standard

        specs: list[dict[str, Any]] = []
        for _ in range(n_standard):
            specs.append(
                {
                    "submission_type": "standard_feedback_form",
                    "min_chars": min_chars_standard_feedback,
                    "max_chars": max_chars_standard_feedback,
                    "target_chars": min_chars_standard_feedback,
                }
            )
        for _ in range(n_detailed):
            specs.append(
                {
                    "submission_type": "detailed_complaint_ticket",
                    "min_chars": min_chars_detailed_complaint,
                    "max_chars": max_chars_detailed_complaint,
                    "target_chars": min_chars_detailed_complaint,
                }
            )
        _assign_target_char_counts(specs, rng)
        rng.shuffle(specs)

        cust_assign, n_distinct = _assign_customers(n_branch, min_customers_per_branch, rng)
        customer_refs = _build_customer_refs(n_distinct, branch_idx)

        chunks: list[list[dict[str, Any]]] = []
        for i in range(0, len(specs), cfg.chunk_size):
            chunks.append(specs[i : i + cfg.chunk_size])

        total_chunks = len(chunks)
        logger.info(
            "Branch %s: %s records in %s API chunk(s) (chunk_size=%s)",
            branch_code,
            n_branch,
            total_chunks,
            cfg.chunk_size,
        )

        llm_items_accum: list[dict[str, Any]] = []
        chunk_seed = seed + branch_idx * 10_000 if seed is not None else None

        for c_i, chunk_specs in enumerate(chunks):
            prompt = _build_prompt_rows(
                chunk_specs,
                branch_code,
                region,
                lang_instruction,
                lang_code,
                fixed_calendar_date,
            )
            sub_seed = (chunk_seed + c_i) if chunk_seed is not None else None
            parsed = _call_gemini_batch(
                client,
                cfg,
                prompt,
                sub_seed,
                len(chunk_specs),
                lang_code,
                branch_code=branch_code,
                chunk_index=c_i,
                total_chunks=total_chunks,
            )
            items = parsed.get("items")
            if not isinstance(items, list):
                raise ValueError("Gemini JSON must contain an array 'items'")
            if len(items) != len(chunk_specs):
                raise ValueError(
                    f"Expected {len(chunk_specs)} items from model, got {len(items)}"
                )
            llm_items_accum.extend(items)

        branch_records: list[dict[str, Any]] = []
        for row_i, (spec, llm_row, cust_idx) in enumerate(
            zip(specs, llm_items_accum, cust_assign, strict=True)
        ):
            st = llm_row.get("submission_type")
            if st != spec["submission_type"]:
                logger.warning(
                    "submission_type mismatch at row %s (fixing to spec): model=%r spec=%r",
                    row_i,
                    st,
                    spec["submission_type"],
                )
                st = spec["submission_type"]

            verbatim = str(llm_row.get("verbatim") or "").strip()
            verbatim = _finalize_verbatim(
                verbatim,
                spec["min_chars"],
                spec["max_chars"],
                rng,
                lang_code,
            )

            subject = llm_row.get("subject")
            if subject is not None:
                subject = str(subject).strip() or None

            product_bucket = llm_row.get("product_bucket")
            if product_bucket is not None:
                product_bucket = str(product_bucket).strip() or None

            channel = llm_row.get("channel")
            if channel not in _CHANNELS:
                channel = rng.choice(_CHANNELS)

            lang = lang_code

            has_att = bool(llm_row.get("has_attachments"))

            cust_ref = ""
            if allow_anonymous_customer_id and rng.random() < cfg.anonymous_customer_fraction:
                cust_ref = ""
            else:
                cust_ref = customer_refs[cust_idx]

            created_at, updated_at = _random_timestamps(
                rng, cfg.timezone_default, fixed_calendar_date
            )
            case_seq += 1
            case_id = f"BK-{case_id_year}-{case_seq:08d}"

            rec: dict[str, Any] = {
                "record_type": "bank_complaint_raw",
                "submission_type": st,
                "case_id": case_id,
                "ticket_number": case_id,
                "branch_code": branch_code,
                "region": region,
                "channel": channel,
                "created_at": created_at,
                "updated_at": updated_at,
                "status": _pick_status(rng),
                "product_bucket": product_bucket,
                "subject": subject,
                "verbatim": verbatim,
                "language": lang,
                "has_attachments": has_att,
                "customer_ref": cust_ref,
                "assigned_queue": None
                if rng.random() < 0.2
                else rng.choice(["BRANCH_INTAKE", "RETAIL_CARE", "BACK_OFFICE", None]),
                "synthetic": True,
                "generator_version": cfg.generator_version,
                "model": cfg.model,
            }
            branch_records.append(rec)
            all_records.append(rec)

        if output_jsonl_dir is not None:
            out_dir = Path(output_jsonl_dir)
            out_file = out_dir / _branch_jsonl_filename(branch_code)
            _write_jsonl_records(out_file, branch_records)

    return all_records


def run_bank_feedback_generation(
    num_branches: int,
    min_records_per_branch: int,
    max_records_per_branch: int,
    allow_anonymous_customer_id: bool,
    min_customers_per_branch: int,
    complaint_language: str,
    min_chars_standard_feedback: int,
    max_chars_standard_feedback: int,
    min_chars_detailed_complaint: int,
    max_chars_detailed_complaint: int,
    seed: int | None,
    config_path: str | Path,
    output_jsonl_dir: Path | None,
    logging_level: int = logging.INFO,
    created_date: str | date | datetime | None = None,
) -> list[dict[str, Any]]:
    """Configure logging, run :func:`generate_bank_feedback_data`, print a short summary.

    Use this from ``python generator.py`` or call it from other scripts with explicit parameters.
    """
    logging.basicConfig(level=logging_level, format="%(levelname)s %(message)s")
    print(
        "Starting generation — each 'Gemini:' log is one API call; large max_chars / max_output_tokens can take many minutes.",
        flush=True,
    )

    records = generate_bank_feedback_data(
        num_branches,
        min_records_per_branch,
        max_records_per_branch,
        allow_anonymous_customer_id,
        min_customers_per_branch,
        min_chars_standard_feedback,
        max_chars_standard_feedback,
        min_chars_detailed_complaint,
        max_chars_detailed_complaint,
        complaint_language=complaint_language,
        created_date=created_date,
        config_path=config_path,
        output_jsonl_dir=output_jsonl_dir,
        seed=seed,
    )
    print(f"Generated {len(records)} record(s).")
    if output_jsonl_dir is not None:
        print(f"Wrote {num_branches} JSONL file(s) under {output_jsonl_dir.resolve()}")
    return records


if __name__ == "__main__":
    run_bank_feedback_generation(
        num_branches=27,
        min_records_per_branch=20,
        max_records_per_branch=100,
        allow_anonymous_customer_id=True,
        min_customers_per_branch=20,
        complaint_language="en",
        min_chars_standard_feedback=80,
        max_chars_standard_feedback=4000,
        min_chars_detailed_complaint=400,
        max_chars_detailed_complaint=32000,
        seed=42,
        config_path=_PACKAGE_ROOT / "config.yaml",
        output_jsonl_dir=_PACKAGE_ROOT / "out"/"bank_feedback_sample",
        logging_level=logging.INFO,
        created_date="2025-01-01",
    )


__all__ = ["generate_bank_feedback_data", "run_bank_feedback_generation"]
