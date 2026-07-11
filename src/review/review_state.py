"""Stage-6 review-state persistence (CLAUDE.md §1, T19).

Approve/reject decisions are stored in their own JSON file keyed by
``device::curve_type`` — NEVER written back into Stage 5's output files.
Stage outputs stay immutable; the review state is a separate, small,
mergeable artifact ("no silent mutation" principle).

Loading is deliberately forgiving (a missing or malformed state file starts
fresh with a logged warning instead of crashing a review session); writing
is deliberately strict (validated before write, atomic tmp+os.replace so an
interrupted save can never corrupt previous decisions).
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.common.log import get_logger

logger = get_logger(__name__)

VALID_DECISIONS = {"approve", "reject"}

ReviewState = Dict[str, Dict[str, Any]]


def decision_key(device: str, curve_type: str) -> str:
    """Build the state key for one (device, curve_type) result.

    Raises:
        ValueError: If either part is empty (mirrors the schema-level
            empty-curve_type hard error).
    """
    if not device or not curve_type:
        raise ValueError(
            f"device and curve_type must both be non-empty (got {device!r}, {curve_type!r})"
        )
    return f"{device}::{curve_type}"


def validate_state(state: ReviewState) -> None:
    """Raise ``ValueError`` if ``state`` does not conform to the state schema."""
    if not isinstance(state, dict):
        raise ValueError(f"review state must be a dict, got {type(state).__name__}")
    for key, entry in state.items():
        if not isinstance(key, str) or "::" not in key:
            raise ValueError(f"state key must look like 'device::curve_type', got {key!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"state entry for {key!r} must be a dict, got {entry!r}")
        decision = entry.get("decision")
        if decision not in VALID_DECISIONS:
            raise ValueError(
                f"state entry for {key!r} has invalid decision {decision!r} "
                f"(must be one of {sorted(VALID_DECISIONS)})"
            )


def set_decision(state: ReviewState, device: str, curve_type: str, decision: str) -> ReviewState:
    """Return a copy of ``state`` with the decision for (device, curve_type) set.

    Raises:
        ValueError: On an invalid decision value or empty key parts.
    """
    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"decision must be one of {sorted(VALID_DECISIONS)}, got {decision!r}"
        )
    new_state = dict(state)
    new_state[decision_key(device, curve_type)] = {
        "decision": decision,
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }
    return new_state


def load_state(path: Path) -> ReviewState:
    """Load a review-state file; missing/malformed files start fresh (logged).

    Never raises on bad input — a reviewer's session must not be blocked by
    a corrupt state file — but the reason is always logged so nothing is
    silently swallowed.
    """
    path = Path(path)
    if not path.exists():
        logger.info("load_state: %s does not exist, starting fresh", path)
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        validate_state(raw)
        return raw
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("load_state: %s unreadable/malformed (%s), starting fresh", path, exc)
        return {}


def save_state(state: ReviewState, path: Path) -> None:
    """Validate, then atomically write ``state`` to ``path``."""
    validate_state(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    logger.info("save_state: wrote %d decision(s) to %s", len(state), path)
