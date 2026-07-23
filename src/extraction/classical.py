"""Classical (non-AI) Stage-5 extraction front-end for rdson_vs_tj.

An rdson_vs_tj figure is a single solid-colored curve line, so no
LineFormer/GPU is needed: the curve is isolated with plain image
processing and handed to the EXISTING Stage-5 orchestration core as
ordinary :class:`src.extraction.inference.Detection` objects. Everything
downstream — dedup, curve-count gating, skeletonization, naming (via the
registry), axis calibration (``parse_numeric_ticks``/``fit_axis``), and
the schema-validated result — is :func:`src.extraction.pipeline.process_detections`
unchanged, with ``expected_curve_count=1``. This module reimplements NONE
of it (CLAUDE.md §3 zero duplication; §4 frozen-stage integrity — no
frozen file is modified).

The generic detection functions themselves (chromatic-pixel segmentation
for the color path, the monochrome/black-curve fallback, and their shared
helpers) moved to :mod:`src.extraction.curve_detection` (2026-07-21,
owner-approved) — they contained zero rdson-specific logic (only image/
ocr_lines in, ``Detection`` list out). This module imports them and keeps
only what's genuinely rdson_vs_tj-specific: the color-then-monochrome
routing + two-curve typ/max naming + unit detection + the rdson value-
plausibility gate, all in :func:`run_classical_pipeline`.

Monochrome (black-curve) datasheets — every real rdson_vs_tj chart in the
corpus (T24/T25 survey) — are handled by
:func:`~src.extraction.curve_detection.detect_curve_monochrome`, used as a
FALLBACK: the color path runs first and, only when it finds nothing (zero
chromatic pixels, i.e. every real rdson chart), the monochrome path runs.
"""
import re
from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.common.log import get_logger
from src.extraction.curve_detection import (
    OcrLine,
    detect_curve_classical,
    detect_curve_monochrome,
    _median_col_thickness,
)
from src.extraction.naming.rdson_vs_tj import name_curves_by_labels
from src.extraction.pipeline import process_detections
from src.extraction.schema import build_result
from src.extraction.skeletonize import mask_to_points

logger = get_logger(__name__)

# rdson_vs_tj charts carry 1 curve (IR template) or 2 ("typ"/"max" on the
# Infineon "Diagram" template — owner decision 2026-07-13). Exactly-2
# detections run as the two-curve case; everything else is gated against
# this default and quarantines (never guessed).
EXPECTED_CURVE_COUNT = 1
TWO_CURVE_COUNT = 2

# Safety net (T27 follow-up, 2026-07-14, owner-approved): on the monochrome
# path, a nearby-but-not-full-width line (partial gridline, scan streak) can
# survive gridline removal (its run is under MONO_GRID_MIN_SPAN_FRAC) and
# then get fused into the curve component by the gap-bridging dilation,
# inflating its column thickness well past a normal stroke — the real defect
# found on 2/11 real charts (AUIRF7675M2TR, AUIRF7736M2TR) via overlay
# inspection: status was "ok" but the trace grew a spurious upper branch.
# MEASURED on the real 11-chart run (data/t27_mono_rdson_run/): the 9
# genuinely single-stroke extractions have MEDIAN column thickness 12-16px;
# the 2 known merged-line cases measure 21-22px. Threshold set at the
# midpoint, comfortably clear of both clusters.
MONO_MAX_MEDIAN_COL_THICKNESS_PX = 18
# Unit-aware plausible y-value ranges for rdson_vs_tj (owner-approved,
# 2026-07-14) — the counterpart to the x_range (temperature) entry in the
# core's PLAUSIBILITY_SPECS, living HERE because the bound depends on the
# detected unit, which only this wrapper knows (the core's unit detector is
# capacitance-only). Data table, same spirit as PLAUSIBILITY_SPECS.
# - normalized: owner-specified 0.3..5; the whole real 11-chart corpus
#   (all normalized) spans 0.52..2.5, comfortably inside.
# - mOhm/Ohm: no such chart exists in the tested corpus yet, so these are
#   physical bounds with the same one-decade margin idiom the capacitance
#   spec used: real MOSFET rdson spans ~0.5 mOhm (large trench FETs) to
#   ~10 Ohm (small high-voltage parts) -> mOhm (0.05, 1e5), Ohm (5e-5, 100).
#   Deliberately generous — the gate targets orders-of-magnitude calibration
#   failures, not tight physics.
RDSON_Y_PLAUSIBLE_RANGES: Dict[str, tuple] = {
    "normalized": (0.3, 5.0),
    "mOhm": (0.05, 100_000.0),
    "Ohm": (0.00005, 100.0),
}

# y-axis unit tokens for rdson charts, canonicalized to ASCII (an "Ω" does
# not round-trip reliably through Windows consoles / downstream tooling).
# First match per OCR line wins, so "mΩ" is consumed by the milli pattern
# and never double-counts as bare Ohm. The milli "m" is case-SENSITIVE:
# "MΩ" (mega) must never read as milli.
_RDSON_UNIT_PATTERNS = (
    ("mOhm", re.compile(r"m\s*(?:Ω|(?i:ohms?))")),
    ("Ohm", re.compile(r"(?<![mM])(?:Ω|(?i:\bohms?\b))")),
    ("normalized", re.compile(r"(?i)normali[sz]ed")),
)


def detect_rdson_units(ocr_lines: Sequence[OcrLine], img_w: float) -> Optional[str]:
    """Detect the rdson y-axis unit from the y-axis label zone OCR text.

    Same y-zone gate (``cx / img_w < 0.30``) as the existing
    ``parse_numeric_ticks``/``detect_y_axis_units``, so a stray unit-looking
    token elsewhere in the figure is never picked up.

    Args:
        ocr_lines: The figure's OCR lines.
        img_w: Figure crop width in pixels.

    Returns:
        ``"mOhm"``, ``"Ohm"``, or ``"normalized"`` if exactly one distinct
        unit is found in the y-zone; ``None`` if none, or more than one
        (ambiguous — never guessed).
    """
    found = set()
    for line in ocr_lines:
        text = line.get("text", "")
        bbox = line["bounding_box"]
        cx = (bbox["x1"] + bbox["x2"]) / 2
        if img_w <= 0 or cx / img_w >= 0.30:
            continue
        for unit, pattern in _RDSON_UNIT_PATTERNS:
            if pattern.search(text):
                found.add(unit)
                break

    if len(found) == 1:
        return next(iter(found))
    if found:
        logger.info("detect_rdson_units: ambiguous units %s — not guessed", sorted(found))
    return None


def run_classical_pipeline(
    device: str,
    curve_type: str,
    source_image: str,
    image: np.ndarray,
    ocr_lines: Sequence[OcrLine],
) -> Dict[str, Any]:
    """Run classical detection then the full existing Stage-5 pipeline (GPU-free).

    The classical analogue of :func:`src.extraction.pipeline.run_pipeline`:
    :func:`detect_curve_classical` (color) replaces model inference, falling
    back to :func:`detect_curve_monochrome` when it finds nothing (every real
    rdson chart), then ``process_detections`` (the frozen orchestration core)
    does everything else. 1 and 2 curves are both valid counts (IR single-curve vs.
    Infineon typ/max templates); in the two-curve case the chart's own
    "max"/"98 %"/"typ" OCR labels override the position-based names when
    they resolve unambiguously. Another addition on top: the
    core's unit detector only knows capacitance units, so when its ONLY
    complaint is ``units_undetected``, the rdson unit detector gets a
    chance to fill units in — the already-computed curves and calibration
    are reused verbatim, nothing is re-traced.

    Args:
        device: Device identifier.
        curve_type: Registry key (``"rdson_vs_tj"``).
        source_image: Figure image path/identifier, recorded in the result.
        image: HxWx3 uint8 BGR figure crop.
        ocr_lines: The figure's OCR lines, for axis calibration + units.

    Returns:
        A schema-validated Stage-5 result dict — the exact shape Stage 6's
        gallery and Stage 7's orchestrator already consume from the AI path.

    Raises:
        ValueError: If ``image`` is not an HxWx3 array.
        KeyError: On structurally malformed inputs (e.g. an OCR line with
            no ``bounding_box``, or an unregistered ``curve_type``) —
            caller bugs are raised, never swallowed (CLAUDE.md §7).
    """
    img_h, img_w = image.shape[:2]
    detections = detect_curve_classical(image)
    used_monochrome = False
    # Every real rdson_vs_tj chart is black-on-white (no chroma), so the color
    # path finds nothing; fall back to the monochrome path in that case. Color
    # first keeps colored charts (if any appear) on the simpler path.
    if not detections:
        logger.info(
            "run_classical_pipeline(%s, %s): color path found no curves — "
            "falling back to monochrome detection", device, curve_type,
        )
        detections = detect_curve_monochrome(image, ocr_lines)
        used_monochrome = True
    else:
        logger.info(
            "run_classical_pipeline(%s, %s): using color detection path (%d curve[s])",
            device, curve_type, len(detections),
        )
    # 1 or 2 curves are both valid for rdson_vs_tj; 0 or 3+ hit the core's
    # exact-count gate against the 1-curve default and quarantine.
    expected = TWO_CURVE_COUNT if len(detections) == TWO_CURVE_COUNT \
        else EXPECTED_CURVE_COUNT
    result = process_detections(
        device, curve_type, source_image, float(img_w), float(img_h),
        list(ocr_lines), detections, expected_curve_count=expected,
    )

    # Two-curve case: the registry namer assigned position-based names
    # (top = rdson_max); the chart's own "max"/"98 %"/"typ" labels take
    # precedence when they resolve (owner rule). detections order ==
    # curves order here (count matched exactly, so no dedup reordering).
    if (
        len(detections) == TWO_CURVE_COUNT
        and len(result["curves"]) == TWO_CURVE_COUNT
        and all(c["points"] for c in result["curves"])
    ):
        label_names = name_curves_by_labels(
            [mask_to_points(d.mask) for d in detections], ocr_lines)
        current = [c["curve_name"] for c in result["curves"]]
        if label_names is not None and label_names != current:
            logger.info(
                "run_classical_pipeline(%s, %s): OCR labels override "
                "position-based names %s -> %s", device, curve_type,
                current, label_names,
            )
            curves = [dict(c, curve_name=name)
                      for c, name in zip(result["curves"], label_names)]
            result = build_result(
                device=device, curve_type=curve_type, source_image=source_image,
                status=result["status"], review_reason=result["review_reason"],
                duplicates_removed=result["duplicates_removed"],
                calibration=result["calibration"], curves=curves,
                units=result["units"],
            )

    if result["status"] == "needs_review" and result["review_reason"] == "units_undetected":
        units = detect_rdson_units(ocr_lines, float(img_w))
        if units is not None:
            logger.info(
                "run_classical_pipeline(%s, %s): rdson units detected (%s), "
                "upgrading units_undetected result to ok", device, curve_type, units,
            )
            result = build_result(
                device=device, curve_type=curve_type, source_image=source_image,
                status="ok", review_reason=None,
                duplicates_removed=result["duplicates_removed"],
                calibration=result["calibration"], curves=result["curves"],
                units=units,
            )

    # Monochrome-only safety net (see MONO_MAX_MEDIAN_COL_THICKNESS_PX):
    # a merged-in parallel line inflates a detection's column thickness well
    # past a genuine stroke's. Checked against the RAW pre-dedup detections
    # (dedup only removes duplicates, never changes mask content) so it
    # applies regardless of any curve-count/naming logic above. Not applied
    # to the color path — that failure mode is specific to the monochrome
    # gridline-removal + gap-bridging mechanism.
    if used_monochrome and result["status"] == "ok":
        worst = max((_median_col_thickness(d.mask) for d in detections), default=0.0)
        if worst > MONO_MAX_MEDIAN_COL_THICKNESS_PX:
            reason = (
                f"suspiciously_thick_monochrome_trace: median column "
                f"thickness {worst:.1f}px exceeds {MONO_MAX_MEDIAN_COL_THICKNESS_PX}px "
                "— likely two lines merged during gridline-gap bridging"
            )
            logger.warning("run_classical_pipeline(%s, %s): %s", device, curve_type, reason)
            result = build_result(
                device=device, curve_type=curve_type, source_image=source_image,
                status="needs_review", review_reason=reason,
                duplicates_removed=result["duplicates_removed"],
                calibration=result["calibration"], curves=result["curves"],
                units=result["units"],
            )

    # Unit-aware y-value plausibility (see RDSON_Y_PLAUSIBLE_RANGES): runs
    # last because it needs the FINAL detected units. Applies to both color
    # and mono paths — a units/calibration mismatch isn't path-specific.
    # Curves/calibration/units are kept for the reviewer (never an empty
    # shell), same as the core's implausible_calibration gate.
    if result["status"] == "ok" and result["units"] in RDSON_Y_PLAUSIBLE_RANGES:
        lo, hi = RDSON_Y_PLAUSIBLE_RANGES[result["units"]]
        y_values = [p["y"] for c in result["curves"] for p in c["points"]]
        if y_values and (min(y_values) < lo or max(y_values) > hi):
            reason = (
                f"implausible_rdson_values: y values span "
                f"{min(y_values):.4g}..{max(y_values):.4g} {result['units']}, "
                f"outside the plausible {result['units']} range {lo:g}..{hi:g}"
            )
            logger.warning("run_classical_pipeline(%s, %s): %s", device, curve_type, reason)
            result = build_result(
                device=device, curve_type=curve_type, source_image=source_image,
                status="needs_review", review_reason=reason,
                duplicates_removed=result["duplicates_removed"],
                calibration=result["calibration"], curves=result["curves"],
                units=result["units"],
            )

    return result
