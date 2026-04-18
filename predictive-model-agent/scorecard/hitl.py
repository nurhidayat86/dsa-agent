"""Human-in-the-loop abstraction.

The orchestrator never reads from ``stdin`` directly. Instead it asks the
``HitlInterface`` to resolve a gate. This module ships three implementations:

* ``CLIHitl`` — prompts the user in the terminal (the POC default).
* ``AutoApproveHitl`` — approves every gate with sensible defaults; used for
  smoke tests and unattended reruns.
* ``ScriptedHitl`` — replays decisions from a JSONL file, for tests.

A future GUI will subclass ``HitlInterface`` and implement ``ask_gate`` on
top of an async websocket / REST bridge — no orchestrator changes required.
"""

from __future__ import annotations

import json
import sys
import textwrap
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .schemas import HitlAction, HitlDecision


class GatePayload:
    """Everything the human needs to make a decision for one gate."""

    def __init__(
        self,
        gate_id: GateId,
        title: str,
        summary: str,
        proposal: dict[str, Any] | None = None,
        tables: dict[str, Any] | None = None,
    ) -> None:
        self.gate_id = gate_id
        self.title = title
        self.summary = summary
        self.proposal = proposal or {}
        self.tables = tables or {}


class HitlInterface(ABC):
    """Base contract: resolve a gate, return a validated ``HitlDecision``."""

    @abstractmethod
    def ask_gate(self, payload: GatePayload) -> HitlDecision: ...

    def log(self, store_dir: Path, decision: HitlDecision) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        with open(store_dir / "hitl_log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(decision.model_dump_json() + "\n")


# --------------------------------------------------------------------------- #
# CLI implementation
# --------------------------------------------------------------------------- #


def _print_table(title: str, rows: Iterable[Any], limit: int = 20) -> None:
    print(f"\n[{title}]")
    try:
        import pandas as pd  # local import
        if hasattr(rows, "head"):
            with pd.option_context("display.max_rows", limit, "display.width", 200):
                print(rows.head(limit).to_string())
            return
    except Exception:
        pass
    for i, row in enumerate(rows):
        if i >= limit:
            print(f"  ... ({i} more)")
            break
        print(f"  {row}")


class CLIHitl(HitlInterface):
    """Terminal-based HITL for the POC."""

    def ask_gate(self, payload: GatePayload) -> HitlDecision:
        print("\n" + "=" * 78)
        print(f"HITL GATE {payload.gate_id.value} — {payload.title}")
        print("=" * 78)
        print(textwrap.fill(payload.summary, width=100))
        if payload.proposal:
            print("\n[proposal]")
            print(json.dumps(payload.proposal, indent=2, default=str))
        for name, table in payload.tables.items():
            _print_table(name, table)
        print("\nActions:")
        print("  a = approve")
        print("  r = revise  (describe your change in plain English — an AI")
        print("              agent will translate it into the required config)")
        print("  x = reject  (skip this gate / abort path)")

        while True:
            raw = input("> choose [a/r/x]: ").strip().lower()
            if raw in {"a", "approve", ""}:
                action = HitlAction.APPROVE
                revise_payload: dict[str, Any] = {}
                break
            if raw in {"x", "reject"}:
                action = HitlAction.REJECT
                revise_payload = {}
                break
            if raw in {"r", "revise"}:
                text = self._read_revise_text()
                if not text:
                    print("[info] empty revision — treating as approve.")
                    action = HitlAction.APPROVE
                    revise_payload = {}
                else:
                    action = HitlAction.REVISE
                    revise_payload = {"_nl_request": text}
                break
            print(f"unrecognized input: {raw!r}")

        return HitlDecision(
            gate_id=payload.gate_id,
            action=action,
            user="cli",
            timestamp=datetime.utcnow(),
            payload=revise_payload,
        )

    @staticmethod
    def _read_revise_text() -> str:
        """Collect a free-text revision request from the user.

        The orchestrator always hands this to a per-gate reviser LLM agent,
        so no JSON / dictionary syntax is ever required from the user.
        End input with a single '.' on its own line (or EOF).
        """

        print(
            "Describe your revision in plain English (e.g. 'shrink valid to 15% "
            "and use the last 2 months as OOT', 'forbid features X and Y', "
            "'cap at 8 features with max correlation 0.4'). End with '.' on "
            "its own line:"
        )
        lines: list[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == ".":
                break
            lines.append(line)
        return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# Auto + scripted helpers (unattended / tests)
# --------------------------------------------------------------------------- #


class AutoApproveHitl(HitlInterface):
    """Approves every gate; useful for smoke tests."""

    def ask_gate(self, payload: GatePayload) -> HitlDecision:
        return HitlDecision(
            gate_id=payload.gate_id,
            action=HitlAction.APPROVE,
            user="auto",
            payload={},
        )


class ScriptedHitl(HitlInterface):
    """Replay decisions from an iterable (e.g. a JSONL file)."""

    def __init__(self, decisions: list[HitlDecision]) -> None:
        self._queue = list(decisions)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ScriptedHitl":
        p = Path(path)
        decisions: list[HitlDecision] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            decisions.append(HitlDecision.model_validate_json(line))
        return cls(decisions)

    def ask_gate(self, payload: GatePayload) -> HitlDecision:
        if not self._queue:
            raise RuntimeError(f"Scripted HITL exhausted before gate {payload.gate_id}.")
        decision = self._queue.pop(0)
        if decision.gate_id != payload.gate_id:
            print(
                f"[warn] scripted decision gate {decision.gate_id} != expected {payload.gate_id}",
                file=sys.stderr,
            )
        return decision
