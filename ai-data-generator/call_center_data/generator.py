"""
Generate synthetic call-center telesales conversations (metadata JSON + VibeVoice transcripts).

Aligns with ``docs/telesales_vibevoice_data_structure.md``. Callable from other code::

    from pathlib import Path
    import sys
    ADA = Path(".../ai-data-generator")
    sys.path.insert(0, str(ADA))
    from call_center_data import generate_call_center_data
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PACKAGE_ROOT / "config.yaml"

_REGION_POOL = [
    "JAVA",
    "SUMATERA",
    "KALIMANTAN",
    "SULAWESI",
    "BALI_NUSA",
    "PAPUA",
    "MALUKU",
]

_OUTCOMES = ["interested", "declined", "callback", "dnc", "transferred"]

# Rough TTS pacing for **word-count** limits (see ``_word_bounds_for_tts_duration``).
_WPS_FAST_FOR_MIN_WORDS_BY_LANG: dict[str, float] = {
    "en": 2.85,
    "id": 3.0,
}
_WPS_SLOW_FOR_MAX_WORDS_BY_LANG: dict[str, float] = {
    "en": 2.1,
    "id": 2.15,
}
_DEFAULT_WPS_FAST = 2.85
_DEFAULT_WPS_SLOW = 2.1

_MAX_DURATION_COMPLIANCE_RETRIES = 6

# One structured ``items`` element per Gemini request (see chunk_stride below).


@dataclass(frozen=True)
class _GeminiCfg:
    api_key: str
    model: str
    temperature: float
    top_p: float | None
    max_output_tokens: int
    http_timeout_ms: int | None
    generator_version: str
    timezone_default: str
    chunk_size: int
    max_retries: int
    retry_backoff_seconds: float


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_gemini_cfg(raw: dict[str, Any]) -> _GeminiCfg:
    g = raw.get("gemini") or {}
    gen = raw.get("generator") or {}
    env_name = g.get("api_key_env") or "GEMINI_API_KEY"
    key = (g.get("api_key") or "").strip()
    if not key:
        key = os.environ.get(env_name, "").strip()
    if not key:
        raise ValueError(
            f"Missing Gemini API key: set {env_name!r} or gemini.api_key in config.yaml."
        )
    if "http_timeout_ms" in g:
        raw_timeout = g["http_timeout_ms"]
        if raw_timeout in (None, "", 0, "0"):
            http_timeout_ms: int | None = None
        else:
            http_timeout_ms = int(raw_timeout)
    else:
        http_timeout_ms = 900_000

    return _GeminiCfg(
        api_key=key,
        model=g.get("model") or "gemini-2.0-flash",
        temperature=float(g.get("temperature") if g.get("temperature") is not None else 0.85),
        top_p=float(g["top_p"]) if g.get("top_p") is not None else None,
        max_output_tokens=int(g.get("max_output_tokens") or 16384),
        http_timeout_ms=http_timeout_ms,
        generator_version=str(gen.get("generator_version") or "0.1.0"),
        timezone_default=str(gen.get("timezone_default") or "+07:00"),
        chunk_size=max(1, int(gen.get("chunk_size") or 5)),
        max_retries=max(1, int(gen.get("max_retries") or 3)),
        retry_backoff_seconds=float(gen.get("retry_backoff_seconds") or 1.5),
    )


def _normalize_lang(lang: str) -> tuple[str, str]:
    """Return (language_code, prompt phrase)."""
    s = (lang or "en").strip().lower()
    if s in ("en", "english"):
        return "en", "English"
    if s in ("id", "indonesian", "bahasa", "bahasa indonesia", "in"):
        return "id", "Bahasa Indonesia"
    if len(s) == 2 and s.isalpha():
        return s, f"the language with ISO 639-1 code {s!r} (write fluently)"
    raise ValueError(
        "lang must be 'en', 'id', or a two-letter ISO 639-1 code; " f"got {lang!r}"
    )


def _parse_date_piece(v: str | date | datetime | None) -> date | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    t = str(v).strip()
    if not t:
        return None
    return date.fromisoformat(t)


def _effective_date_range(
    conversation_dates: Sequence[str | date | datetime | None] | None,
) -> tuple[date, date]:
    """Return inclusive (start, end) calendar range for simulated ``conversation_datetime``."""
    if conversation_dates is None:
        return date(2026, 1, 1), date(2026, 12, 31)

    if not isinstance(conversation_dates, (list, tuple)) or len(conversation_dates) != 2:
        raise ValueError(
            "conversation_dates must be None or a list/tuple of two elements [start, end] "
            "(each may be None)."
        )

    start_raw, end_raw = conversation_dates[0], conversation_dates[1]
    start_d = _parse_date_piece(start_raw)
    end_d = _parse_date_piece(end_raw)

    if start_d is None and end_d is None:
        return date(2026, 1, 1), date(2026, 12, 31)
    if start_d is None and end_d is not None:
        start_d = end_d - timedelta(days=365)
    if end_d is None and start_d is not None:
        end_d = start_d + timedelta(days=365)
    if start_d > end_d:
        raise ValueError("conversation_dates: start must be on or before end.")
    return start_d, end_d


def _random_conversation_iso(
    rng: random.Random,
    start_d: date,
    end_d: date,
    tz: str,
) -> str:
    span = (end_d - start_d).days
    day_off = rng.randint(0, max(0, span))
    d = start_d + timedelta(days=day_off)
    sec = rng.randint(0, 24 * 60 * 60 - 1)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{d.isoformat()}T{h:02d}:{m:02d}:{s:02d}{tz}"


def _random_branch_code(rng: random.Random) -> str:
    reg = rng.choice(_REGION_POOL)
    abbr = reg.replace("_", "")[:4]
    num = rng.randint(1, 999)
    return f"BR-{abbr}-{num:03d}"


def _parse_gemini_json(text: str) -> dict[str, Any]:
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
        raise ValueError("Top-level JSON must be an object")
    trailing = text[start + end :].strip()
    if trailing:
        logger.debug("Ignoring %s chars after first JSON object", len(trailing))
    return obj


def _dialog_batch_schema(
    n: int,
    outcomes: list[str],
    products: list[str],
    *,
    max_turns: int = 28,
) -> Any:
    from google.genai import types

    mt = max(6, min(64, int(max_turns)))

    turn = types.Schema(
        type=types.Type.OBJECT,
        properties={
            # google.genai Schema enum values must be strings
            "speaker": types.Schema(type=types.Type.STRING, enum=["0", "1"]),
            "text": types.Schema(type=types.Type.STRING),
        },
        required=["speaker", "text"],
    )
    item = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "row_index": types.Schema(type=types.Type.INTEGER),
            "scenario": types.Schema(type=types.Type.STRING),
            "outcome": types.Schema(type=types.Type.STRING, enum=outcomes),
            "primary_product": types.Schema(type=types.Type.STRING, enum=products),
            "secondary_products": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
            ),
            "turns": types.Schema(
                type=types.Type.ARRAY,
                items=turn,
                min_items=6,
                max_items=mt,
            ),
        },
        required=[
            "row_index",
            "scenario",
            "outcome",
            "primary_product",
            "turns",
        ],
    )
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "items": types.Schema(
                type=types.Type.ARRAY,
                items=item,
                min_items=n,
                max_items=n,
            ),
        },
        required=["items"],
    )


def _word_bounds_for_tts_duration(
    min_seconds: float,
    max_seconds: float,
    lang_code: str,
) -> tuple[int, int]:
    """Infer min/max **word counts** so typical TTS likely stays within the duration window.

    Uses a **fast** words/sec lower bound for the minimum word count (fast TTS needs more
    words to still reach ``min_seconds``) and a **slow** words/sec upper bound for the cap.
    """
    w_fast = _WPS_FAST_FOR_MIN_WORDS_BY_LANG.get(lang_code, _DEFAULT_WPS_FAST)
    w_slow = _WPS_SLOW_FOR_MAX_WORDS_BY_LANG.get(lang_code, _DEFAULT_WPS_SLOW)

    min_words = max(8, math.ceil(min_seconds * w_fast))
    max_words = max(min_words, math.floor(max_seconds * w_slow))

    if max_words < min_words:
        t_mid = (min_seconds + max_seconds) / 2.0
        w_mid = (w_fast + w_slow) / 2.0
        center = int(round(t_mid * w_mid))
        half = max(int(round((max_seconds - min_seconds) * w_mid / 2)), 15)
        min_words = max(8, center - half)
        max_words = center + half
        if max_words < min_words:
            max_words = min_words + 20

    return min_words, max_words


def _max_turns_for_chunk(chunk_rows: list[dict[str, Any]]) -> int:
    cap = max(int(r["word_count_max"]) for r in chunk_rows)
    # ~15–20 words/turn average for long scripts; cap at schema limit 64
    return min(64, max(28, cap // 17 + 14))


def _estimate_chunk_max_output_tokens(
    cfg: _GeminiCfg,
    chunk_rows: list[dict[str, Any]],
) -> int:
    """Raise token budget for long JSON dialogues (structured output)."""
    wsum = sum(int(r["word_count_max"]) for r in chunk_rows)
    est = int(wsum * 2.3 + 4096)
    return min(65536, max(int(cfg.max_output_tokens), est))


def _attach_priority_duration_targets(
    rows: list[dict[str, Any]],
    min_seconds: float,
    max_seconds: float,
    min_words: int,
    max_words: int,
    rng: random.Random,
) -> None:
    """Assign each row a duration **slot** and word-count window.

    **Guarantees** (by conversation count):

    - ``n == 1``: **maximum** duration only (highest-priority slot).
    - ``n >= 2``: **one** conversation at **maximum** duration, **one** at **minimum**;
      any others use a **random** target in ``[min_seconds, max_seconds]``.

    Row placement: **minimum** → first row, **maximum** → last row, **random** → middle rows
    (stable ordering in output files).

    If the global word band is too narrow to give non-overlapping per-slot windows, every row
    uses the **same** ``[min_words, max_words]`` while **target_spoken_seconds** and
    **duration_slot** still encode max / min / random intent for the prompt and metadata.
    """
    n = len(rows)
    if n == 0:
        return

    span_w = max_words - min_words
    span_s = max(max_seconds - min_seconds, 0.0)

    # --- duration_slot + target seconds (priority: max assured, then min, then random) ---
    if n == 1:
        slots: list[str] = ["maximum"]
        targets_s: list[float] = [float(max_seconds)]
    elif n == 2:
        slots = ["minimum", "maximum"]
        targets_s = [float(min_seconds), float(max_seconds)]
    else:
        slots = ["minimum"] + ["random"] * (n - 2) + ["maximum"]
        targets_s = [float(min_seconds)]
        if span_s <= 0:
            mid = float(min_seconds)
            targets_s.extend([mid] * (n - 2))
        else:
            targets_s.extend(
                [round(min_seconds + rng.random() * span_s, 2) for _ in range(n - 2)]
            )
        targets_s.append(float(max_seconds))

    # --- per-slot word windows vs global fallback ---
    # Need enough word span to give min-window and max-window clear separation.
    half_band = max(10, min(48, span_w // 4)) if span_w > 0 else 0
    min_separation = half_band * 2 + 20
    use_distinct_word_windows = (
        n >= 2
        and span_w >= min(70, max(48, min_separation))
        and (max_words - min_words) >= min_separation
    )

    for i, row in enumerate(rows):
        slot = slots[i]
        ts = round(targets_s[i], 2)
        row["duration_slot"] = slot
        row["target_spoken_seconds"] = ts

        if not use_distinct_word_windows or span_w <= 0:
            row["word_count_min"] = min_words
            row["word_count_max"] = max_words
            row["duration_word_band_mode"] = "global"
            continue

        if slot == "maximum":
            center_w = max_words - half_band * 0.2
        elif slot == "minimum":
            center_w = min_words + half_band * 0.2
        else:
            if span_s <= 0:
                t_frac = 0.5
            else:
                t_frac = (ts - min_seconds) / span_s
            t_frac = max(0.0, min(1.0, t_frac))
            center_w = min_words + t_frac * span_w

        lo_w = int(max(min_words, round(center_w - half_band)))
        hi_w = int(min(max_words, round(center_w + half_band)))
        if lo_w > hi_w:
            lo_w, hi_w = hi_w, lo_w
        if hi_w - lo_w < 10:
            hi_w = min(max_words, lo_w + 12)

        row["word_count_min"] = lo_w
        row["word_count_max"] = hi_w
        row["duration_word_band_mode"] = "per_slot"

    # If min-slot and max-slot word intervals overlap, distinct distribution is impossible.
    if use_distinct_word_windows and n >= 2:
        r0, r_last = rows[0], rows[-1]
        if r0["duration_slot"] == "minimum" and r_last["duration_slot"] == "maximum":
            if int(r0["word_count_max"]) > int(r_last["word_count_min"]):
                for row in rows:
                    row["word_count_min"] = min_words
                    row["word_count_max"] = max_words
                    row["duration_word_band_mode"] = "global_overlap_fallback"


def _count_spoken_words_in_turns(turns: Any) -> int:
    if not isinstance(turns, list):
        return 0
    parts: list[str] = []
    for t in turns:
        if isinstance(t, dict):
            parts.append(str(t.get("text") or ""))
    return len(re.findall(r"\S+", " ".join(parts)))


def _duration_word_violations_per_row(
    items: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
) -> list[tuple[int, int, int, int]]:
    """Return list of (row_index, word_count, word_min, word_max) for out-of-range rows."""
    bad: list[tuple[int, int, int, int]] = []
    for it, spec in zip(items, chunk_rows, strict=True):
        if not isinstance(it, dict):
            continue
        ri = int(it.get("row_index", -1))
        lo = int(spec["word_count_min"])
        hi = int(spec["word_count_max"])
        wc = _count_spoken_words_in_turns(it.get("turns"))
        # Upper slack: word-count compliance is approximate; models often overshoot trim targets.
        _hi_slack = max(35, int(round((hi - lo) * 0.28)))
        hi_eff = hi + _hi_slack
        if wc < lo or wc > hi_eff:
            bad.append((ri, wc, lo, hi))
    return bad


def _build_batch_prompt(
    rows: list[dict[str, Any]],
    lang_instruction: str,
    lang_code: str,
    min_duration: float,
    max_duration: float,
    max_turns_allowed: int,
) -> str:
    lines = []
    for r in rows:
        slot = r["duration_slot"]
        slot_note = {
            "maximum": "this dialogue must be the **longest** in the batch (~maximum TTS duration)",
            "minimum": "this dialogue must be the **shortest** in the batch (~minimum TTS duration)",
            "random": "target length is **between** min and max (~listed seconds)",
        }[slot]
        lines.append(
            f"{r['row_index']}. primary_product={r['primary_product']!r}, "
            f"customer_voice_label={r['customer_voice_label']!r} (not spoken; voices are fixed); "
            f"**duration role** = **{slot}** ({slot_note}); "
            f"**target spoken time** ~{r['target_spoken_seconds']} s (allowed range {min_duration:g}–{max_duration:g} s); "
            f"**word count** (all ``turns`` ``text`` tokens, whitespace-separated) MUST be "
            f">= {r['word_count_min']} AND <= {r['word_count_max']}"
        )
    block = "\n".join(lines)
    row_index_list = [r["row_index"] for r in rows]
    return f"""You write synthetic **phone** dialogues for **bank outbound telesales** (cross-sell / upsell). Training data only.

Hard rules:
- No real PII: no realistic full names, phone numbers, national IDs, or street addresses. Use generic roles or obviously fake tokens.
- Language: write all spoken lines in {lang_instruction}. ``language`` code for each row is {lang_code!r}.
- Speaker **0** = tele-sales agent (bank). Speaker **1** = customer.
- Natural phone conversation: alternating turns, realistic pacing, polite agent, varied customer responses matching the given ``outcome``.
- Each row must promote or discuss the given ``primary_product`` (banking product). Set ``primary_product`` in JSON **exactly** equal to the string given for that row (same spelling).
- Optional ``secondary_products``: short list of other product codes or an empty array.

Duration — **mandatory** (global TTS range **{min_duration:g}**–**{max_duration:g}** seconds):
- Every batch includes at least one **maximum**-role row (**~max seconds**, longest dialogue) when possible; with two or more rows, also a **minimum**-role row (**~min seconds**); any additional rows use **random** targets in between.
- Respect each row's **duration role** and printed word bounds. **Maximum** rows must be noticeably longer than **minimum** rows when both appear in the same batch.
- Word bounds are a TTS proxy; stay within them. Too short → add dialogue; too long → trim.

``turns`` format:
- **6–{max_turns_allowed}** items (use enough turns so long dialogues do not cram entire paragraphs into one ``text``). Each item: ``speaker`` is string ``"0"`` or ``"1"``; ``text`` is one utterance (no ``Speaker N:`` prefix).
- Start with speaker 0 (agent greeting). Do not put JSON or stage directions inside ``text``.
- Output valid JSON only: one object with key ``items`` (array). Length must equal {len(rows)}. The ``i``-th item (0-based) must have ``row_index`` equal to this list in order: {row_index_list}.
- ``scenario``: short snake_case slug, e.g. ``credit_card_upsell``, ``savings_cross_sell``.

Rows (one dialogue per row):
{block}
"""


def _call_gemini_chunk_with_duration_compliance(
    client: Any,
    cfg: _GeminiCfg,
    chunk_rows: list[dict[str, Any]],
    schema: Any,
    base_seed: int | None,
    batch_label: str,
    chunk_i: int,
    n_chunks: int,
    lang_instruction: str,
    lang_code: str,
    min_duration: float,
    max_duration: float,
) -> dict[str, Any]:
    """Call Gemini; retry with stricter feedback if word counts miss per-row proxy bounds."""
    n = len(chunk_rows)
    max_turns_allowed = _max_turns_for_chunk(chunk_rows)
    chunk_max_tokens = _estimate_chunk_max_output_tokens(cfg, chunk_rows)
    base_prompt = _build_batch_prompt(
        chunk_rows,
        lang_instruction,
        lang_code,
        min_duration,
        max_duration,
        max_turns_allowed,
    )
    feedback_suffix = ""
    last_violations: list[tuple[int, int, int, int]] = []

    for attempt in range(_MAX_DURATION_COMPLIANCE_RETRIES):
        prompt = base_prompt + feedback_suffix
        sub_seed = (
            (base_seed + attempt * 101) if base_seed is not None else None
        )
        parsed = _call_gemini_dialog_batch(
            client,
            cfg,
            prompt,
            schema,
            sub_seed,
            batch_label=batch_label,
            chunk_i=chunk_i,
            n_chunks=n_chunks,
            max_output_tokens=chunk_max_tokens,
        )
        items = parsed.get("items")
        if not isinstance(items, list) or len(items) != n:
            raise ValueError(
                f"Expected {n} items from model, got {len(items) if isinstance(items, list) else 'invalid'}"
            )
        violations = _duration_word_violations_per_row(items, chunk_rows)
        if not violations:
            if attempt > 0:
                logger.info(
                    "Duration/word bounds satisfied after %s attempt(s) (chunk %s)",
                    attempt + 1,
                    chunk_i + 1,
                )
            return parsed

        last_violations = violations
        logger.warning(
            "Chunk %s/%s: per-row word bounds violated %s — retry %s/%s",
            chunk_i + 1,
            n_chunks,
            violations,
            attempt + 1,
            _MAX_DURATION_COMPLIANCE_RETRIES,
        )
        viol_str = ", ".join(
            f"row {r}: {w} words (required {lo}-{hi})" for r, w, lo, hi in violations
        )
        row_spec = "\n".join(
            f"  row {r['row_index']}: role={r['duration_slot']}, ~{r['target_spoken_seconds']}s, "
            f"words {r['word_count_min']}-{r['word_count_max']}"
            for r in chunk_rows
        )
        feedback_suffix = f"""

*** REGENERATE REQUIRED ***
Your previous ``items`` violated per-row word counts (TTS proxy for {min_duration:g}–{max_duration:g}s). Respect each row's **duration role** (maximum / minimum / random).
Failures: {viol_str}.
Exact bounds for this chunk:
{row_spec}
Roles: {[(r["row_index"], r["duration_slot"]) for r in chunk_rows]}
Regenerate the **full** ``items`` array ({n} rows). ``row_index`` order: {[r["row_index"] for r in chunk_rows]}.
"""

    raise ValueError(
        f"Gemini output did not satisfy per-row spoken-duration proxy (~{min_duration:g}-{max_duration:g}s). "
        f"Last violations (row_index, words, min, max): {last_violations}"
    )


def _call_gemini_dialog_batch(
    client: Any,
    cfg: _GeminiCfg,
    prompt: str,
    schema: Any,
    seed: int | None,
    batch_label: str,
    chunk_i: int,
    n_chunks: int,
    *,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    from google.genai import types
    from google.genai import errors as genai_errors

    out_tok = int(max_output_tokens) if max_output_tokens is not None else int(cfg.max_output_tokens)
    if out_tok != cfg.max_output_tokens:
        logger.info(
            "Raising max_output_tokens to %s for this chunk (config has %s)",
            out_tok,
            cfg.max_output_tokens,
        )

    effective_seed = seed
    last_err: Exception | None = None
    attempt = 0
    while attempt < cfg.max_retries:
        gcfg = types.GenerateContentConfig(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_output_tokens=out_tok,
            response_mime_type="application/json",
            response_schema=schema,
            seed=effective_seed,
        )
        try:
            logger.info(
                "Gemini telesales: %s chunk %s/%s (model=%s)",
                batch_label,
                chunk_i + 1,
                n_chunks,
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
                "Gemini telesales: %s chunk %s/%s done in %.1fs (%s chars)",
                batch_label,
                chunk_i + 1,
                n_chunks,
                elapsed,
                len(raw),
            )
            return _parse_gemini_json(raw)
        except Exception as e:
            last_err = e
            if effective_seed is not None and isinstance(e, genai_errors.ClientError):
                http_code = getattr(e, "code", None) or getattr(e, "status_code", None)
                detail = str(e)
                if http_code == 400 and (
                    "INVALID_ARGUMENT" in detail or "invalid argument" in detail.lower()
                ):
                    logger.warning(
                        "Gemini 400 INVALID_ARGUMENT — retrying same request without seed"
                    )
                    effective_seed = None
                    continue
            logger.warning("Gemini request failed (attempt %s/%s): %s", attempt + 1, cfg.max_retries, e)
            attempt += 1
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_backoff_seconds * attempt)
    assert last_err is not None
    raise last_err


def _format_transcript(turns: list[dict[str, Any]]) -> str:
    lines_out: list[str] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        sp = int(str(t.get("speaker", "0")))
        text = str(t.get("text") or "").strip()
        if not text:
            continue
        lines_out.append(f"Speaker {sp}: {text}")
    return "\n".join(lines_out) + ("\n" if lines_out else "")


def _transcript_rel_path(batch_dir: Path, transcript_file: Path) -> str:
    try:
        return str(transcript_file.resolve().relative_to(batch_dir.resolve())).replace("\\", "/")
    except ValueError:
        return str(transcript_file.resolve())


def generate_call_center_data(
    number_of_conversation: int,
    featured_products: list[str],
    lang: str,
    tele_sales_speaker_name: str,
    customer_speaker_names: list[str],
    conversation_dates: Sequence[str | date | datetime | None] | None,
    output_path_json: str | Path,
    output_path_txt: str | Path,
    *,
    min_duration: float = 60.0,
    max_duration: float = 180.0,
    config_path: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Generate a telesales batch: one ``metadata.json``-shaped dict and one ``.txt`` per conversation.

    Parameters
    ----------
    number_of_conversation :
        Total dialogues to generate.
    featured_products :
        Product codes offered (e.g. ``\"credit_card_gold\"``); the model assigns one per row from your specs.
    lang :
        Dialogue language (``en``, ``id``, or two-letter ISO code).
    tele_sales_speaker_name :
        VibeVoice voice name for **Speaker 0** (agent).
    customer_speaker_names :
        Pool of VibeVoice voice names for **Speaker 1**; one is chosen per conversation.
    conversation_dates :
        ``None`` → default 2026-01-01 … 2026-12-31. Otherwise ``[start, end]`` where each
        may be ``None``: ``[None, end]`` uses a year before ``end``; ``[start, None]`` uses a year after ``start``.
    output_path_json :
        Path for the batch JSON file (e.g. ``.../batch/metadata.json``). Parent folder is the **batch root**
        for relative ``transcript_path`` values when possible.
    output_path_txt :
        Directory where transcript files are written as flat ``conv-00001.txt``, …
        (e.g. ``.../batch/conversations``).
    min_duration / max_duration :
        Allowed **spoken** length in **seconds**. At least one dialogue targets **maximum**
        duration (single-row batch) or the **last** row does when ``n >= 2``; with ``n >= 2``
        the **first** row targets **minimum**; other rows get a **random** time in
        ``[min_duration, max_duration]``. If per-slot word windows cannot be separated,
        all rows share the same word bounds while roles and target seconds stay as above.
    config_path :
        ``config.yaml`` under ``ai-data-generator`` (defaults next to package).
    seed :
        RNG seed (and Gemini seed offset when supported).
    """
    if number_of_conversation < 1:
        raise ValueError("number_of_conversation must be >= 1")
    if not featured_products:
        raise ValueError("featured_products must be a non-empty list")
    if not (cn := [x.strip() for x in customer_speaker_names if str(x).strip()]):
        raise ValueError("customer_speaker_names must contain at least one non-empty name")

    lang_code, lang_instruction = _normalize_lang(lang)
    tele_name = tele_sales_speaker_name.strip()
    if not tele_name:
        raise ValueError("tele_sales_speaker_name must be non-empty")

    products = [str(p).strip() for p in featured_products if str(p).strip()]
    if not products:
        raise ValueError("featured_products must contain at least one non-empty product string")

    lo_d = float(min_duration)
    hi_d = float(max_duration)
    if lo_d <= 0:
        raise ValueError("min_duration must be > 0 (seconds)")
    if hi_d < lo_d:
        raise ValueError("max_duration must be >= min_duration (seconds)")

    date_start, date_end = _effective_date_range(conversation_dates)

    json_path = Path(output_path_json).resolve()
    txt_root = Path(output_path_txt).resolve()
    batch_dir = json_path.parent
    batch_dir.mkdir(parents=True, exist_ok=True)
    txt_root.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    raw_cfg = _load_yaml_config(cfg_path)
    cfg = _resolve_gemini_cfg(raw_cfg)

    rng = random.Random(seed)
    year_for_ids = date_start.year

    from google import genai
    from google.genai import types as genai_types

    http_opts: genai_types.HttpOptions | None = None
    if cfg.http_timeout_ms is not None:
        http_opts = genai_types.HttpOptions(timeout=cfg.http_timeout_ms)
    client = genai.Client(api_key=cfg.api_key, http_options=http_opts)

    batch_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ") + "_synth"

    tz = cfg.timezone_default
    if not (tz.startswith("+") or tz.startswith("-")):
        tz = "+07:00"

    # Build row specs
    rows: list[dict[str, Any]] = []
    for i in range(number_of_conversation):
        prod = rng.choice(products)
        cust_voice = rng.choice(cn)
        rows.append(
            {
                "row_index": i + 1,
                "primary_product": prod,
                "customer_voice_label": cust_voice,
                "conversation_id": f"tel-{year_for_ids}-{i + 1:08d}",
                "customer_voice": cust_voice,
            }
        )

    min_words, max_words = _word_bounds_for_tts_duration(lo_d, hi_d, lang_code)
    _attach_priority_duration_targets(rows, lo_d, hi_d, min_words, max_words, rng)
    band_mode = rows[0].get("duration_word_band_mode", "?")
    logger.info(
        "TTS duration %g–%g s → global word band [%s, %s] (lang=%s); "
        "word window mode=%s; first row slot=%s ~%ss, last row slot=%s ~%ss",
        lo_d,
        hi_d,
        min_words,
        max_words,
        lang_code,
        band_mode,
        rows[0]["duration_slot"],
        rows[0]["target_spoken_seconds"],
        rows[-1]["duration_slot"],
        rows[-1]["target_spoken_seconds"],
    )

    max_w_any = max(int(r["word_count_max"]) for r in rows)
    # Always one ``items[0]`` per Gemini call — multi-item structured JSON is unreliable on flash-lite.
    chunk_stride = 1
    if cfg.chunk_size != 1:
        logger.info(
            "Telesales generator uses one conversation per request (ignoring config chunk_size=%s); "
            "longest row word cap=%s",
            cfg.chunk_size,
            max_w_any,
        )

    chunks: list[list[dict[str, Any]]] = []
    for i in range(0, len(rows), chunk_stride):
        chunks.append(rows[i : i + chunk_stride])

    all_llm_items: list[dict[str, Any]] = []

    for ci, chunk_rows in enumerate(chunks):
        n = len(chunk_rows)
        max_turns = _max_turns_for_chunk(chunk_rows)
        schema = _dialog_batch_schema(
            n, list(_OUTCOMES), products, max_turns=max_turns
        )
        sub_seed = (seed + ci * 7919) if seed is not None else None
        parsed = _call_gemini_chunk_with_duration_compliance(
            client,
            cfg,
            chunk_rows,
            schema,
            sub_seed,
            batch_id,
            ci,
            len(chunks),
            lang_instruction,
            lang_code,
            lo_d,
            hi_d,
        )
        items = parsed.get("items")
        if not isinstance(items, list):
            raise ValueError("Gemini JSON missing list 'items'")
        all_llm_items.extend(items)

    conversations_out: list[dict[str, Any]] = []
    created_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for idx, (spec, llm_item) in enumerate(zip(rows, all_llm_items, strict=True)):
        tpath = txt_root / f"conv-{idx + 1:05d}.txt"
        if int(llm_item.get("row_index", -1)) != spec["row_index"]:
            logger.warning(
                "row_index mismatch spec=%s llm=%s — using positional order",
                spec["row_index"],
                llm_item.get("row_index"),
            )
        turns_raw = llm_item.get("turns") or []
        turns = turns_raw if isinstance(turns_raw, list) else []
        body = _format_transcript(turns)
        tpath.write_text(body, encoding="utf-8")
        word_count = _count_spoken_words_in_turns(turns)

        customer_ref = f"SYN-CUST-{uuid.uuid4().hex[:10].upper()}"
        if rng.random() < 0.15:
            customer_ref = ""

        vibenames = [tele_name, spec["customer_voice"]]
        conv_dt = _random_conversation_iso(rng, date_start, date_end, tz)

        entry = {
            "conversation_id": spec["conversation_id"],
            "record_type": "bank_telesales_conversation",
            "scenario": str(llm_item.get("scenario") or "cross_sell"),
            "language": lang_code,
            "conversation_datetime": conv_dt,
            "customer_ref": customer_ref,
            "branch_code": _random_branch_code(rng),
            "primary_product": str(llm_item.get("primary_product") or spec["primary_product"]),
            "secondary_products": list(llm_item["secondary_products"])
            if isinstance(llm_item.get("secondary_products"), list)
            else [],
            "outcome": str(llm_item.get("outcome") or "declined"),
            "transcript_path": _transcript_rel_path(batch_dir, tpath),
            "speaker_map": {
                "0": {"role": "tele_sales_agent", "display_name": "Agent"},
                "1": {"role": "customer", "display_name": "Customer"},
            },
            "vibevoice_speaker_names": vibenames,
            "transcript_word_count": word_count,
            "target_spoken_seconds": spec["target_spoken_seconds"],
            "duration_slot": spec["duration_slot"],
            "duration_word_band_mode": spec.get("duration_word_band_mode", "unknown"),
            "spoken_word_count_bounds": {
                "min": spec["word_count_min"],
                "max": spec["word_count_max"],
            },
        }
        conversations_out.append(entry)

    root = {
        "batch_id": batch_id,
        "record_type": "bank_telesales_batch",
        "synthetic": True,
        "generator_version": cfg.generator_version,
        "model": cfg.model,
        "created_at": created_iso,
        "spoken_duration_target_seconds": {"min": lo_d, "max": hi_d},
        "spoken_word_count_bounds_global": {"min": min_words, "max": max_words},
        "duration_distribution": (
            "priority: at_least_one_maximum_then_minimum_then_random_between_bounds; "
            "row0=minimum_row_when_n_ge_2_last_row=maximum_row_when_n_ge_2"
        ),
        "conversation_count": len(conversations_out),
        "conversations": conversations_out,
    }

    json_path.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s conversations to %s", len(conversations_out), json_path)
    return root


def run_call_center_generation(
    number_of_conversation: int,
    featured_products: list[str],
    lang: str,
    tele_sales_speaker_name: str,
    customer_speaker_names: list[str],
    conversation_dates: Sequence[str | date | datetime | None] | None,
    output_path_json: str | Path,
    output_path_txt: str | Path,
    *,
    min_duration: float = 60.0,
    max_duration: float = 180.0,
    config_path: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Configure logging and call :func:`generate_call_center_data` (for ``python generator.py``)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    return generate_call_center_data(
        number_of_conversation,
        featured_products,
        lang,
        tele_sales_speaker_name,
        customer_speaker_names,
        conversation_dates,
        output_path_json,
        output_path_txt,
        min_duration=min_duration,
        max_duration=max_duration,
        config_path=config_path,
        seed=seed,
    )


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    _out = _root / "out" / "call_center_sample"
    run_call_center_generation(
        number_of_conversation=10,
        featured_products=["credit_card_gold", "savings_premium", "personal_loan_topup"],
        lang="en",
        tele_sales_speaker_name="Alice",
        customer_speaker_names=["Frank", "Grace"],
        conversation_dates=None,
        output_path_json=_out / "metadata.json",
        output_path_txt=_out / "conversations",
        min_duration=45.0,
        max_duration=300.0,
        config_path=_root / "config.yaml",
        seed=42,
    )
