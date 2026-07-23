"""Classical (non-AI) Stage-5 extraction front-end for vgsth_vs_tj.

The vgsth_vs_tj analogue of :mod:`src.extraction.classical` (rdson's own
wrapper): curves are isolated with the SAME generic detection Stage 1
provides (:func:`~src.extraction.curve_detection.detect_curve_classical`
color-first, :func:`~src.extraction.curve_detection.detect_curve_monochrome`
fallback, default tunables — no vgsth-specific override without evidence
to justify one), then handed to the frozen
:func:`src.extraction.pipeline.process_detections` for dedup,
skeletonization, calibration, and the schema-validated result. This module
reimplements none of that (CLAUDE.md §3 zero duplication; §4 frozen-stage
integrity — no frozen file is modified).

Naming/counting is entirely label-driven
(:mod:`src.extraction.naming.vgsth_vs_tj`) — unlike rdson_vs_tj, vgsth has
no meaningful position-only naming (band vs. current-value scheme, and the
curve count itself, are both unknown without OCR labels), so
``process_detections``'s own internal registry-looked-up naming is always
just a throwaway placeholder (``naming/__init__.py``'s ``curve_N`` names,
which carry no real authority — see its docstring). This wrapper ALWAYS
overrides it with the real resolved names on every non-quarantined path.

Core safety net — the expected-vs-detected curve-count comparison (never
silently merge curves at a crossing, never silently miss one): given
``N = count_expected_curves(ocr_lines)`` (how many curves the chart's own
labels imply) and ``D = len(detections)`` (how many were actually found),

    D == 0                    -> quarantine: no curves found
    N is not None and D < N   -> quarantine: likely merged at a crossing
    N is not None and D > N   -> quarantine: stray component / missed label
    N is None and D > 1       -> quarantine: no usable labels to disambiguate
    otherwise (D == N, or N is None and D == 1):
        name_curves_by_labels(...) is None -> quarantine: ambiguous naming
        else                               -> proceed, names override the
                                               registry placeholder

Units: no vgsth-specific multi-unit table (unlike rdson_vs_tj's
``RDSON_Y_PLAUSIBLE_RANGES``/``detect_rdson_units``) — every real chart
reviewed uses Volts with no ambiguity, so the frozen core's own generic
y-axis unit detector (which already recognizes a "V" token) is sufficient;
``units_undetected`` passes straight through unmodified.
"""
from typing import Any, Dict, Sequence

from src.common.log import get_logger
from src.extraction.curve_detection import (
    OcrLine,
    detect_curve_classical,
    detect_curve_monochrome,
)
from src.extraction.naming.vgsth_vs_tj import count_expected_curves, name_curves_by_labels
from src.extraction.pipeline import process_detections
from src.extraction.schema import build_result
from src.extraction.skeletonize import mask_to_points

logger = get_logger(__name__)


def _quarantine(
    device: str, curve_type: str, source_image: str, reason: str,
    detections: Sequence[Any], img_w: float, img_h: float,
    ocr_lines: Sequence[OcrLine],
) -> Dict[str, Any]:
    """Build a needs_review result for a gate failure before naming succeeds.

    Calibration doesn't depend on curve naming succeeding — it's derived
    purely from the OCR axis labels (CLAUDE.md §3: reused via
    ``process_detections``, never reimplemented here), so there's no
    reason any quarantine reason should carry less real information than
    another (mirrors rdson's own rule). ``process_detections`` is called
    with ``expected_curve_count`` set to the ACTUAL detected count, so its
    own internal count-gate is a trivial pass-through — it has no concept
    of the label-count mismatch or naming-tie this wrapper is quarantining
    for. Its status/review_reason verdict is always replaced with the
    wrapper's own (it doesn't know why THIS wrapper is unhappy), but its
    computed calibration/curves/units are kept. Curve names stay whatever
    the naming-registry placeholder produced — never fabricated here —
    exactly the case that placeholder's own docstring says it may
    legitimately surface in.
    """
    logger.warning("classical_vgsth: quarantine - %s", reason)
    core_result = process_detections(
        device, curve_type, source_image, img_w, img_h, list(ocr_lines),
        detections, expected_curve_count=len(detections),
    )
    return build_result(
        device=device, curve_type=curve_type, source_image=source_image,
        status="needs_review", review_reason=reason,
        duplicates_removed=core_result["duplicates_removed"],
        calibration=core_result["calibration"], curves=core_result["curves"],
        units=core_result["units"],
    )


def run_classical_pipeline(
    device: str,
    curve_type: str,
    source_image: str,
    image: Any,
    ocr_lines: Sequence[OcrLine],
) -> Dict[str, Any]:
    """Run classical detection then the full existing Stage-5 pipeline (GPU-free).

    The classical analogue of :func:`src.extraction.pipeline.run_pipeline`:
    :func:`~src.extraction.curve_detection.detect_curve_classical` (color)
    replaces model inference, falling back to
    :func:`~src.extraction.curve_detection.detect_curve_monochrome` when it
    finds nothing. The detected count is compared against what the chart's
    own OCR labels imply (:func:`~src.extraction.naming.vgsth_vs_tj.count_expected_curves`)
    before anything else runs — a mismatch quarantines rather than ever
    silently merging curves at a crossing or dropping one. On a match,
    :func:`~src.extraction.naming.vgsth_vs_tj.name_curves_by_labels` resolves
    the real curve names, ``process_detections`` builds everything else
    (dedup, calibration, units, schema), and the resolved names overwrite
    whatever placeholder name the (unregistered-for-real) naming registry
    entry produced.

    Args:
        device: Device identifier.
        curve_type: Registry key (``"vgsth_vs_tj"``).
        source_image: Figure image path/identifier, recorded in the result.
        image: HxWx3 uint8 BGR figure crop.
        ocr_lines: The figure's OCR lines, for counting, naming, and axis
            calibration.

    Returns:
        A schema-validated Stage-5 result dict — the exact shape Stage 6's
        gallery and Stage 7's orchestrator already consume from the AI path.

    Raises:
        ValueError: If ``image`` is not an HxWx3 array (raised by the
            reused detection functions, not re-validated here).
        KeyError: On structurally malformed inputs (e.g. an OCR line with
            no ``bounding_box``) — caller bugs are raised, never swallowed
            (CLAUDE.md §7).
    """
    img_h, img_w = image.shape[:2]

    detections = detect_curve_classical(image)
    if detections:
        logger.info(
            "classical_vgsth(%s, %s): using color detection path (%d curve[s])",
            device, curve_type, len(detections),
        )
    else:
        logger.info(
            "classical_vgsth(%s, %s): color path found no curves — "
            "falling back to monochrome detection", device, curve_type,
        )
        detections = detect_curve_monochrome(image, ocr_lines)

    detected = len(detections)
    expected = count_expected_curves(ocr_lines)

    if detected == 0:
        reason = "no curves found in the figure"
        return _quarantine(device, curve_type, source_image, reason, detections,
                           float(img_w), float(img_h), ocr_lines)

    if expected is not None and detected < expected:
        reason = (
            f"{detected} curve(s) detected but {expected} label(s) resolved "
            f"— likely merged at a crossing"
        )
        return _quarantine(device, curve_type, source_image, reason, detections,
                           float(img_w), float(img_h), ocr_lines)

    if expected is not None and detected > expected:
        reason = (
            f"{detected} curve(s) detected but only {expected} label(s) resolved "
            f"— stray component or missed label"
        )
        return _quarantine(device, curve_type, source_image, reason, detections,
                           float(img_w), float(img_h), ocr_lines)

    if expected is None and detected > 1:
        reason = "no usable labels to disambiguate multiple curves"
        return _quarantine(device, curve_type, source_image, reason, detections,
                           float(img_w), float(img_h), ocr_lines)

    # Reachable only here: (expected is None and detected == 1), or
    # (expected is not None and detected == expected).
    point_lists = [mask_to_points(d.mask) for d in detections]
    names = name_curves_by_labels(point_lists, ocr_lines)
    if names is None:
        reason = (
            "ambiguous naming despite a matched curve count "
            "(e.g. a genuine proximity tie)"
        )
        return _quarantine(device, curve_type, source_image, reason, detections,
                           float(img_w), float(img_h), ocr_lines)

    logger.info(
        "classical_vgsth(%s, %s): resolved names %s, proceeding to process_detections",
        device, curve_type, names,
    )
    result = process_detections(
        device, curve_type, source_image, float(img_w), float(img_h),
        list(ocr_lines), detections, expected_curve_count=detected,
    )
    curves = [dict(curve, curve_name=name) for curve, name in zip(result["curves"], names)]
    return build_result(
        device=device, curve_type=curve_type, source_image=source_image,
        status=result["status"], review_reason=result["review_reason"],
        duplicates_removed=result["duplicates_removed"],
        calibration=result["calibration"], curves=curves, units=result["units"],
    )
