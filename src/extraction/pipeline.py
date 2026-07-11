"""Stage-5 curve-extraction orchestrator (CLAUDE.md §1).

Orchestrates: dedup -> curve-count gating -> skeletonize -> per-curve-type
naming (looked up by registry, like Stage 4's curve_registry — never
hardcoded per curve type here) -> axis calibration -> pixel-to-engineering
conversion -> schema-validated result.

``process_detections`` is the GPU-free orchestration core (detections
already computed elsewhere) so the whole pipeline logic is unit-testable
without a GPU. ``run_pipeline`` is the thin GPU-dependent wrapper that also
runs model inference (heavy import stays lazy inside
:mod:`src.extraction.inference`, same convention as the rest of the repo).

Ambiguous/incomplete results are never guessed into an "ok" status: any
gate that fails downgrades to ``needs_review`` with an explicit reason,
mirroring Stage 4's quarantine philosophy.
"""
from typing import Any, Dict, List, Optional

from src.calibration.ticks import derive_calibration, detect_y_axis_units, pixel_to_data
from src.common.log import get_logger
from src.extraction.dedup import dedup_detections
from src.extraction.inference import Detection, run_inference
from src.extraction.naming import get_naming_fn
from src.extraction.schema import build_result
from src.extraction.skeletonize import mask_to_points

logger = get_logger(__name__)

DEFAULT_EXPECTED_CURVE_COUNT = 3

# Per-curve-type plausibility bounds (T16, revised T18). A calibration can
# be numerically valid yet physically wrong (the T15 finding: OCR-split
# log-axis exponents produced plausible-looking LINEAR fits, values off by
# orders of magnitude — since fixed at the OCR-token level in ticks.py).
# Bounds are corpus-derived, not guessed: capacitance chart y-axes across the
# training corpus are labeled from 10^0 to 10^5 pF; one decade of margin
# each side gives (0.1, 1e6).
#
# T18: dropped a hard "require_y_log" rule that used to live here. The T17
# wider sample found real capacitance charts (an International Rectifier
# datasheet template) with a genuinely linear y-axis and correctly-extracted
# values — the "capacitance y is always log" assumption itself was wrong,
# not the calibration. fit_axis already picks log vs. linear via its own
# RANSAC inlier-count comparison (see ticks.py); trust that judgment rather
# than second-guessing it here. The range check below is the real
# fit-quality signal: it checks the actual traced/converted values against
# a physical-sanity bound, which catches a bad fit (of either scale)
# end-to-end instead of assuming one scale is always right.
# Data-only, like the Stage 4 registry: new curve types add an entry, no new
# logic. A curve type without an entry is simply unchecked.
PLAUSIBILITY_SPECS: Dict[str, Dict[str, Any]] = {
    "capacitance_vs_vds": {
        "y_range": (0.1, 1_000_000.0),  # pF
    },
}


def _implausibility_reason(
    curve_type: str, calibration: Dict[str, Any], curves: List[Dict[str, Any]]
) -> Optional[str]:
    """Return a human-readable implausibility description, or None if plausible."""
    spec = PLAUSIBILITY_SPECS.get(curve_type)
    if spec is None:
        return None

    y_range = spec.get("y_range")
    if y_range is not None:
        y_values = [p["y"] for curve in curves for p in curve["points"]]
        if y_values:
            y_min, y_max = min(y_values), max(y_values)
            lo, hi = y_range
            if y_min < lo or y_max > hi:
                return (
                    f"implausible_calibration: y values span {y_min:.4g}..{y_max:.4g}, "
                    f"outside the plausible {curve_type} range {lo:g}..{hi:g}"
                )
    return None


def _placeholder_curves(kept: List[Detection]) -> List[Dict[str, Any]]:
    """Generic, schema-valid curve entries for a needs_review result whose
    count is wrong (no defensible position-based name can be assigned)."""
    return [
        {"curve_name": f"unnamed_{i}", "confidence": detection.score, "points": []}
        for i, detection in enumerate(kept)
    ]


def _needs_review(
    device: str, curve_type: str, source_image: str, reason: str,
    duplicates_removed: int, curves: List[Dict[str, Any]],
) -> Dict[str, Any]:
    logger.info("pipeline(%s, %s): needs_review - %s", device, curve_type, reason)
    return build_result(
        device=device, curve_type=curve_type, source_image=source_image,
        status="needs_review", review_reason=reason,
        duplicates_removed=duplicates_removed, calibration=None, curves=curves,
        units=None,  # not attempted — an earlier gate already failed
    )


def process_detections(
    device: str,
    curve_type: str,
    source_image: str,
    img_w: float,
    img_h: float,
    ocr_lines: List[Dict[str, Any]],
    detections: List[Detection],
    expected_curve_count: int = DEFAULT_EXPECTED_CURVE_COUNT,
) -> Dict[str, Any]:
    """Run steps 2-7 of the Stage-5 pipeline on already-computed detections.

    Args:
        device: Device identifier.
        curve_type: Registry key (must have a registered naming function).
        source_image: Figure image path/identifier, recorded in the result.
        img_w: Figure crop width in pixels (for tick zoning).
        img_h: Figure crop height in pixels (for tick zoning).
        ocr_lines: The figure's OCR lines, for axis calibration.
        detections: Score-filtered model detections (score >= 0.5 already
            applied upstream, e.g. by :func:`src.extraction.inference.run_inference`).
        expected_curve_count: Exact curve count this curve type expects
            (3 for capacitance_vs_vds; future curve types may differ).

    Returns:
        A schema-validated result dict (see :mod:`src.extraction.schema`).

    Raises:
        KeyError: If ``curve_type`` has no registered naming function —
            a configuration error, not a data problem, so it's raised
            rather than downgraded to needs_review.
    """
    logger.info(
        "pipeline(%s, %s): %s, %d raw detection(s)",
        device, curve_type, source_image, len(detections),
    )
    naming_fn = get_naming_fn(curve_type)  # fail fast on an unregistered curve_type

    n_raw = len(detections)
    if n_raw < expected_curve_count:
        reason = (
            f"only {n_raw} detection(s) after score filtering "
            f"(need exactly {expected_curve_count}) — not guessed"
        )
        return _needs_review(device, curve_type, source_image, reason, 0,
                             _placeholder_curves(detections))

    if n_raw == expected_curve_count:
        kept, n_removed = list(detections), 0
    else:
        kept, n_removed = dedup_detections(detections)
        logger.info(
            "pipeline(%s, %s): %d raw detections, dedup kept %d (%d removed)",
            device, curve_type, n_raw, len(kept), n_removed,
        )
        if len(kept) != expected_curve_count:
            reason = (
                f"{n_raw} detections, deduped to {len(kept)} "
                f"(need exactly {expected_curve_count}) — not guessed"
            )
            return _needs_review(device, curve_type, source_image, reason, n_removed,
                                 _placeholder_curves(kept))

    point_lists = [mask_to_points(detection.mask) for detection in kept]
    if any(not points for points in point_lists):
        reason = "one or more kept detections had no traceable skeleton points"
        return _needs_review(device, curve_type, source_image, reason, n_removed,
                             _placeholder_curves(kept))

    names = naming_fn(point_lists)

    calibration = derive_calibration(ocr_lines, img_w, img_h)
    if calibration is None or calibration["x_slope"] == 0 or calibration["y_slope"] == 0:
        reason = "axis calibration failed (insufficient/degenerate tick marks)"
        curves = [
            {"curve_name": name, "confidence": detection.score, "points": []}
            for detection, name in zip(kept, names)
        ]
        return _needs_review(device, curve_type, source_image, reason, n_removed, curves)

    curves = []
    for detection, points, name in zip(kept, point_lists, names):
        eng_points = [
            {"x": x, "y": y}
            for x, y in (pixel_to_data(col, row, calibration) for row, col in points)
        ]
        curves.append({"curve_name": name, "confidence": detection.score, "points": eng_points})

    implausibility = _implausibility_reason(curve_type, calibration, curves)
    if implausibility is not None:
        # Keep the traced curves AND the suspect calibration — a reviewer
        # needs to see what was computed, not an empty shell.
        logger.warning("pipeline(%s, %s): %s", device, curve_type, implausibility)
        return build_result(
            device=device, curve_type=curve_type, source_image=source_image,
            status="needs_review", review_reason=implausibility,
            duplicates_removed=n_removed, calibration=calibration, curves=curves,
            units=None,  # calibration itself is already suspect; don't also guess units
        )

    units = detect_y_axis_units(ocr_lines, img_w, img_h)
    if units is None:
        reason = "units_undetected"
        logger.warning("pipeline(%s, %s): %s (y-axis unit label not found/ambiguous)",
                       device, curve_type, reason)
        return build_result(
            device=device, curve_type=curve_type, source_image=source_image,
            status="needs_review", review_reason=reason,
            duplicates_removed=n_removed, calibration=calibration, curves=curves,
            units=None,
        )

    logger.info("pipeline(%s, %s): ok, curves=%s, units=%s, duplicates_removed=%d",
                device, curve_type, names, units, n_removed)
    return build_result(
        device=device, curve_type=curve_type, source_image=source_image,
        status="ok", review_reason=None, duplicates_removed=n_removed,
        calibration=calibration, curves=curves, units=units,
    )


def run_pipeline(
    device: str,
    curve_type: str,
    image_path: str,
    ocr_lines: List[Dict[str, Any]],
    img_w: float,
    img_h: float,
    model: Any,
    score_thr: float = 0.5,
    expected_curve_count: int = DEFAULT_EXPECTED_CURVE_COUNT,
) -> Dict[str, Any]:
    """Run inference on ``image_path`` then the full Stage-5 pipeline (GPU-only).

    Args:
        model: A loaded model, e.g. from :func:`src.extraction.inference.load_model`.
        score_thr: Minimum detection confidence to keep.
        Other args: see :func:`process_detections`.
    """
    detections = run_inference(model, image_path, score_thr=score_thr)
    return process_detections(
        device, curve_type, image_path, img_w, img_h, ocr_lines, detections,
        expected_curve_count=expected_curve_count,
    )
