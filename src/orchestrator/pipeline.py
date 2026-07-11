"""Stage-7 pipeline orchestrator (CLAUDE.md §1, T20).

Pure orchestration: calls Stage 4 (classification), Stage 5 (extraction),
and Stage 6 (review-state lookup) through a small adapter protocol, assigns
each device exactly one status, validates on the way to "finalized", and
writes the final records + follow-up queue. It reimplements NO stage logic.

Statuses (mutually exclusive, every device gets exactly one):
    finalized              explicit APPROVE + final validation passed
    pending_review         Stage 5 ok, no review decision yet (NO auto-pass)
    rejected               explicit REJECT (terminal, never finalized)
    needs_review           Stage 5's own status was needs_review (specific
                           reason kept visible), or an approved record that
                           failed final validation
    failed_classification  Stage 4 found no usable figure (no_match /
                           quarantined), or the classifier itself errored
    failed_extraction      Stage 5 errored/crashed (distinct from
                           needs_review, which is a valid uncertain output)

"finalized" strictly requires an explicit approval record
(``require_manual_approval=True``, the default). The flag exists so
auto-finalization of validated ok results can be enabled later without a
rebuild — flipping it is an owner decision.

CLI (precomputed-Stage-5 mode — reads a directory of Stage-5 result JSONs,
as produced by the extraction runs; live end-to-end wiring lands with the
stage 1-3 migration/GPU-box integration):
    python -m src.orchestrator.pipeline <stage5_dir> --out <output_dir>
        [--review-state <review_state.json>] [--auto-approve]
"""
import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.common.log import get_logger
from src.orchestrator.queue import build_queue, write_queue
from src.orchestrator.validation import validate_final
from src.review.review_state import decision_key, load_state

# CLI runs as __main__, which is outside the "src" logger hierarchy and
# would silently drop every log line (the known Session-1 cvat_to_coco CLI
# defect; same fix as src.review.gallery).
logger = get_logger("src.orchestrator.pipeline" if __name__ == "__main__" else __name__)

STATUSES = (
    "finalized", "pending_review", "rejected", "needs_review",
    "failed_classification", "failed_extraction",
)


@dataclass
class DeviceResult:
    device: str
    curve_type: str
    status: str
    reason: str
    stage5_result: Optional[Dict[str, Any]]
    final_record: Optional[Dict[str, Any]]


class PrecomputedStage5:
    """Stage adapter over a directory of already-produced Stage-5 JSONs.

    Classification is implicitly "matched" — a Stage-5 result exists for the
    device, which means Stage 4 already picked its figure when that result
    was produced. Live classification wiring replaces this adapter once
    stages 1-3 are migrated and the GPU box runs end-to-end.
    """

    class _Matched:
        status = "matched"

    def __init__(self, stage5_dir: Path):
        self.stage5_dir = Path(stage5_dir)

    def run_classification(self, device: str) -> Any:
        return self._Matched()

    def run_extraction(self, device: str, classification: Any) -> Dict[str, Any]:
        path = self.stage5_dir / f"{device}.json"
        return json.loads(path.read_text(encoding="utf-8"))


def _build_final_record(
    stage5: Dict[str, Any], decision_entry: Dict[str, Any]
) -> Dict[str, Any]:
    """Assemble the finalized record: Stage 5's data + provenance."""
    return {
        "device": stage5["device"],
        "curve_type": stage5["curve_type"],
        "source_image": stage5["source_image"],
        "units": stage5["units"],
        "calibration": stage5["calibration"],
        "curves": stage5["curves"],
        "provenance": {
            "stage5_status": stage5["status"],
            "duplicates_removed": stage5["duplicates_removed"],
            "review_decision": decision_entry["decision"],
            "decided_at": decision_entry.get("decided_at"),
            "finalized_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def process_device(
    device: str,
    curve_type: str,
    stages: Any,
    review_state: Dict[str, Any],
    require_manual_approval: bool = True,
) -> DeviceResult:
    """Run one device through Stage 4 -> 5 -> 6 and assign its final status.

    Args:
        device: Device identifier.
        curve_type: Target curve type (registry key).
        stages: Adapter with ``run_classification(device)`` and
            ``run_extraction(device, classification)``.
        review_state: Loaded Stage-6 review-state dict.
        require_manual_approval: When True (default), only an explicit
            APPROVE can finalize. When False, a validated ok result
            finalizes without a decision (explicit REJECT still wins).

    Returns:
        A DeviceResult with exactly one of the six statuses. Never raises
        on stage failures — they become failed_classification /
        failed_extraction (failure isolation for batch runs).
    """
    # ---- Stage 4
    try:
        classification = stages.run_classification(device)
    except Exception as exc:  # noqa: BLE001 — isolation is the contract here
        logger.error("orchestrator(%s, %s): classifier raised: %s", device, curve_type, exc)
        return DeviceResult(device, curve_type, "failed_classification",
                            f"classifier error: {exc}", None, None)
    if getattr(classification, "status", None) != "matched":
        reason = f"classification status: {getattr(classification, 'status', 'unknown')}"
        logger.info("orchestrator(%s, %s): failed_classification (%s)",
                    device, curve_type, reason)
        return DeviceResult(device, curve_type, "failed_classification", reason, None, None)

    # ---- Stage 5
    try:
        stage5 = stages.run_extraction(device, classification)
    except Exception as exc:  # noqa: BLE001 — isolation is the contract here
        logger.error("orchestrator(%s, %s): extraction raised: %s", device, curve_type, exc)
        return DeviceResult(device, curve_type, "failed_extraction",
                            f"extraction error: {exc}", None, None)

    # ---- Stage 6 (review-state lookup) + finalization rules
    decision_entry = review_state.get(decision_key(device, curve_type), {})
    decision = decision_entry.get("decision")

    if decision == "reject":
        logger.info("orchestrator(%s, %s): rejected by reviewer", device, curve_type)
        return DeviceResult(device, curve_type, "rejected",
                            "reviewer rejected", stage5, None)

    approved = decision == "approve" or (
        not require_manual_approval and stage5["status"] == "ok"
    )
    if approved:
        validation_reason = validate_final(stage5)
        if validation_reason is None:
            record = _build_final_record(
                stage5, decision_entry or {"decision": "auto", "decided_at": None})
            logger.info("orchestrator(%s, %s): finalized", device, curve_type)
            return DeviceResult(device, curve_type, "finalized", "approved + validated",
                                stage5, record)
        logger.warning("orchestrator(%s, %s): approved but validation failed: %s",
                       device, curve_type, validation_reason)
        return DeviceResult(device, curve_type, "needs_review",
                            f"final validation failed: {validation_reason}", stage5, None)

    if stage5["status"] == "needs_review":
        reason = str(stage5.get("review_reason"))
        logger.info("orchestrator(%s, %s): needs_review (%s)", device, curve_type, reason)
        return DeviceResult(device, curve_type, "needs_review", reason, stage5, None)

    logger.info("orchestrator(%s, %s): pending_review (no decision yet)",
                device, curve_type)
    return DeviceResult(device, curve_type, "pending_review",
                        "awaiting reviewer decision", stage5, None)


def run_batch(
    devices: Sequence[str],
    curve_type: str,
    stages: Any,
    review_state: Dict[str, Any],
    out_dir: Path,
    require_manual_approval: bool = True,
) -> Dict[str, Any]:
    """Process every device, write finalized records + follow-up queue + summary.

    Returns:
        ``{"counts": {status: n}, "queue_path", "summary_path",
        "finalized_devices": [...]}``
    """
    out_dir = Path(out_dir)
    results: List[DeviceResult] = []
    for device in devices:
        results.append(process_device(device, curve_type, stages, review_state,
                                      require_manual_approval))

    counts = {status: 0 for status in STATUSES}
    for result in results:
        counts[result.status] += 1
    logger.info("run_batch: %d device(s) -> %s", len(results), counts)

    final_dir = out_dir / "final"
    finalized_devices = []
    for result in results:
        if result.final_record is not None:
            path = final_dir / result.device / f"{result.curve_type}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(result.final_record, indent=2), encoding="utf-8")
            tmp.replace(path)
            finalized_devices.append(result.device)

    queue_path = out_dir / "followup_queue.json"
    write_queue(build_queue(results), queue_path)

    summary = {
        "curve_type": curve_type,
        "processed": len(results),
        "counts": counts,
        "finalized_devices": sorted(finalized_devices),
        "queue_path": str(queue_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = out_dir / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tmp.replace(summary_path)
    summary["summary_path"] = str(summary_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage-7 orchestrator over precomputed Stage-5 outputs.")
    parser.add_argument("stage5_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--review-state", type=Path, default=None,
                        help="Stage-6 review_state.json (default: none loaded)")
    parser.add_argument("--curve-type", default="capacitance_vs_vds")
    parser.add_argument("--auto-approve", action="store_true",
                        help="Finalize validated ok results without explicit "
                             "approval (default: manual approval required)")
    args = parser.parse_args(argv)

    review_state = load_state(args.review_state) if args.review_state else {}
    stages = PrecomputedStage5(args.stage5_dir)
    devices = sorted(
        f.stem for f in Path(args.stage5_dir).glob("*.json")
        if f.stem not in ("summary", "dryrun_report", "batch_summary")
    )
    logger.info("orchestrator CLI: %d device(s) from %s, review state: %s",
                len(devices), args.stage5_dir, args.review_state or "(none)")

    summary = run_batch(devices, args.curve_type, stages, review_state, args.out,
                        require_manual_approval=not args.auto_approve)
    logger.info("orchestrator CLI done: %s", summary["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
