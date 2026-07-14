"""Classical (non-AI) Stage-5 extraction front-end for rdson_vs_tj.

An rdson_vs_tj figure is a single solid-colored curve line, so no
LineFormer/GPU is needed: the curve is isolated with plain image
processing (chromatic-pixel segmentation + connected components) and
handed to the EXISTING Stage-5 orchestration core as ordinary
:class:`src.extraction.inference.Detection` objects. Everything
downstream — dedup, curve-count gating, skeletonization, naming (via the
registry), axis calibration (``parse_numeric_ticks``/``fit_axis``), and
the schema-validated result — is :func:`src.extraction.pipeline.process_detections`
unchanged, with ``expected_curve_count=1``. This module reimplements NONE
of it (CLAUDE.md §3 zero duplication; §4 frozen-stage integrity — no
frozen file is modified).

Detection approach (and its deliberate limits):
- Curve pixels are the CHROMATIC ones: axes/gridlines/tick text are
  black/gray (channel spread ~0), the curve is colored. A faint,
  washed-out curve still clears the chroma threshold.
- Small print/compression dropouts along the line are bridged with a
  morphological close before component labeling, so a solid curve with
  pinhole gaps stays ONE detection.
- Components too small or too short (legend swatches, color-key dots)
  are dropped — with counts and reasons logged.
- If MORE than one credible curve survives, ALL are returned; the
  pipeline's exact-count gate then quarantines the figure
  (``needs_review``) rather than this module guessing a winner.
- Known limitation (acceptable for this curve type, logged for review
  anyway via the count gate): two same-colored curves that touch/cross
  would merge into one component. rdson_vs_tj charts are single-curve by
  definition (owner scope, 2026-07-08), so this is a quarantine-path
  concern, not a happy-path one.

Monochrome (black-curve) datasheets are NOT handled yet — a chroma-free
figure yields zero detections and quarantines. A mono fallback is future
work, to be sized against the real corpus.
"""
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.common.log import get_logger
from src.extraction.inference import Detection
from src.extraction.naming.rdson_vs_tj import name_curves_by_labels
from src.extraction.pipeline import process_detections
from src.extraction.schema import build_result
from src.extraction.skeletonize import mask_to_points

logger = get_logger(__name__)

# A pixel is "chromatic" (curve candidate) when its max-min channel spread
# is at least this. Solid datasheet colors spread 150+; faint/washed-out
# scans still spread ~70; grays/black/white spread ~0.
CHROMA_MIN_SPREAD = 40
# Components smaller than this are legend swatches / color-key marks, not
# curves (a real curve at these figure sizes is several hundred pixels).
MIN_CURVE_AREA_PX = 100
# ... and a curve must span a meaningful fraction of the image width.
MIN_COL_SPAN_FRAC = 0.08
# Morphological-close kernel (rows, cols): wide enough to bridge small
# print/compression gaps along a mostly-horizontal curve, short enough to
# never fuse vertically-separated curves.
GAP_CLOSE_KERNEL = (3, 9)
# rdson_vs_tj charts carry 1 curve (IR template) or 2 ("typ"/"max" on the
# Infineon "Diagram" template — owner decision 2026-07-13). Exactly-2
# detections run as the two-curve case; everything else is gated against
# this default and quarantines (never guessed).
EXPECTED_CURVE_COUNT = 1
TWO_CURVE_COUNT = 2

OcrLine = Dict[str, Any]  # {"text": str, "bounding_box": {"x1","y1","x2","y2"}}

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


def detect_curve_classical(image: np.ndarray) -> List[Detection]:
    """Detect solid-colored curve lines in a figure crop, classically.

    Args:
        image: HxWx3 uint8 BGR figure crop (as read by ``cv2.imread``).

    Returns:
        One :class:`Detection` per credible curve-like component (score =
        fraction of the image width the component spans, boolean HxW
        mask), in component-label order. Empty list if nothing credible
        is found — the caller quarantines, this function never guesses.

    Raises:
        ValueError: If ``image`` is not an HxWx3 array.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR image, got shape {image.shape}")

    import cv2  # lazy, same convention as skeletonize's skimage import

    img_h, img_w = image.shape[:2]
    spread = image.astype(np.int16)
    chroma = (spread.max(axis=2) - spread.min(axis=2)) >= CHROMA_MIN_SPREAD
    if not chroma.any():
        logger.info("detect_curve_classical: no chromatic pixels — nothing to detect")
        return []

    closed = cv2.morphologyEx(
        chroma.astype(np.uint8), cv2.MORPH_CLOSE, np.ones(GAP_CLOSE_KERNEL, np.uint8)
    )
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    min_span_px = MIN_COL_SPAN_FRAC * img_w
    detections: List[Detection] = []
    n_dropped = 0
    for label in range(1, n_labels):  # label 0 is background
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        if area < MIN_CURVE_AREA_PX or width < min_span_px:
            n_dropped += 1
            logger.info(
                "detect_curve_classical: dropped component %d (area=%dpx, "
                "col_span=%dpx) — too small/short to be a curve (legend "
                "swatch / stray mark)", label, area, width,
            )
            continue
        mask = labels == label
        score = min(1.0, float(np.unique(np.nonzero(mask)[1]).size) / img_w)
        detections.append(Detection(score=score, mask=mask))

    logger.info(
        "detect_curve_classical: %d component(s) found, %d kept, %d dropped",
        n_labels - 1, len(detections), n_dropped,
    )
    return detections


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
    :func:`detect_curve_classical` replaces model inference, then
    ``process_detections`` (the frozen orchestration core) does everything
    else. 1 and 2 curves are both valid counts (IR single-curve vs.
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
            return build_result(
                device=device, curve_type=curve_type, source_image=source_image,
                status="ok", review_reason=None,
                duplicates_removed=result["duplicates_removed"],
                calibration=result["calibration"], curves=result["curves"],
                units=units,
            )

    return result
