"""Stage-7 follow-up queue (CLAUDE.md §1, T20).

A simple, regenerable to-do list for humans: which devices still need
attention and why. Rebuilt from scratch on every batch run (atomic
overwrite), so re-runs can never duplicate or half-update entries.

``rejected`` and ``finalized`` are deliberately NOT queued — both are
terminal outcomes (an explicit human decision / a shipped record).
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.common.log import get_logger

logger = get_logger(__name__)

# Statuses that still need a human's attention, and what kind.
QUEUE_STATUSES = (
    "pending_review",        # awaiting a gallery decision
    "needs_review",          # Stage 5 flagged it — inspect in the gallery
    "failed_classification", # no figure found — check the datasheet/classifier
    "failed_extraction",     # Stage 5 crashed — needs a developer look
)


def build_queue(device_results: Sequence[Any]) -> List[Dict[str, str]]:
    """Build queue entries from DeviceResults, actionable statuses only.

    Sorted by (status, device) for a stable, diffable file.
    """
    entries = [
        {
            "device": r.device,
            "curve_type": r.curve_type,
            "status": r.status,
            "reason": r.reason,
        }
        for r in device_results
        if r.status in QUEUE_STATUSES
    ]
    return sorted(entries, key=lambda e: (e["status"], e["device"]))


def write_queue(entries: List[Dict[str, str]], path: Path) -> None:
    """Atomically (over)write the queue file — full regeneration, no merging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    logger.info("write_queue: %d entry(ies) -> %s", len(entries), path)
