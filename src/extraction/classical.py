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

Monochrome (black-curve) datasheets — every real rdson_vs_tj chart in the
corpus (T24/T25 survey) — are handled by :func:`detect_curve_monochrome`,
a grayscale sibling of the color path used as a FALLBACK: the color path
runs first and, only when it finds nothing (zero chromatic pixels, i.e.
every real rdson chart), the monochrome path runs. It emits the same
:class:`Detection` objects, so the frozen Stage-5 core is still the only
pipeline. Design is corpus-driven
(``data/t24_mono_survey/MONO_DETECTOR_REQUIREMENTS.md``): two proven ideas
are adopted from the reviewed legacy ``cv_curve_extract.py`` (never copied)
— inpaint OCR-label boxes (not white-out, which would split a curve a label
sits on) and a density+width-span component filter to reject text — while
its documented failure mode (a flat curve segment eaten by an over-eager
gridline kernel) is guarded against: only near-full-span straight runs are
treated as gridlines, so a partially-flat curve survives.
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

# --- Monochrome (black-curve) path constants (corpus-derived, see module
# docstring / MONO_DETECTOR_REQUIREMENTS.md) ---
# A pixel is "ink" when its grayscale value is below this. Generous on
# purpose: curve ink (gray ~17-28) AND dark gridlines/axes (gray ~34-90)
# are all captured, then gridlines/axes are removed by STRUCTURE, not
# intensity (intensity alone is not a sufficient separator — Infineon axes
# can be lighter than the curve).
MONO_INK_MAX_GRAY = 128
# A straight run counts as a gridline/axis (and is removed) only if it spans
# at least this fraction of the plot dimension. This is the flat-curve
# safety bar: a curve that runs flat for LESS than this survives, while a
# genuine full-width/height gridline (spans ~0.9+) is removed.
MONO_GRID_MIN_SPAN_FRAC = 0.5
# Gap-bridge dilation after line removal (rows, cols): reconnects the nick a
# removed gridline (2-6 px + blur) leaves where a curve crosses it. A pure
# dilation (not a close) is used deliberately — a close's erosion step severs
# the thin diagonal bridge at a crossing — and the added thickness is
# harmless because `mask_to_points` skeletonizes to a centerline downstream.
# Kept small and vertically short (radius 2 rows / 4 cols) so it bridges
# gridline-thickness gaps without ever fusing two stacked typ/max curves
# (>=15 px apart -> >=11 px gap after dilation).
MONO_BRIDGE_KERNEL = (5, 9)
# Density (fill fraction of bbox) above which a component is text/logo, not
# a thin curve — adopted from the legacy density filter. Applied ONLY to
# components narrower than MONO_DENSITY_EXEMPT_SPAN_FRAC of the width, so a
# wide (flat or bendy) curve is never density-rejected.
MONO_MAX_FILL_DENSITY = 0.35
MONO_DENSITY_EXEMPT_SPAN_FRAC = 0.5
# Telea inpaint radius for OCR-label boxes (legacy used 3).
MONO_INPAINT_RADIUS = 3
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


def _inpaint_ocr_boxes(
    image: np.ndarray, ocr_lines: Sequence[OcrLine], cv2
) -> np.ndarray:
    """Return a copy of ``image`` with OCR-label boxes reconstructed by
    inpainting (Telea), so a label sitting on a curve doesn't split it.

    Adopted from legacy ``cv_curve_extract.py`` (reviewed, not copied):
    inpaint — not white-out — because white-out punches a hole that severs a
    curve running under the label; inpaint rebuilds the region from its
    surroundings. Boxes are clamped to the image; a missing ``bounding_box``
    is a caller bug and is left to raise (CLAUDE.md §7).
    """
    img_h, img_w = image.shape[:2]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    painted = 0
    for line in ocr_lines:
        bbox = line["bounding_box"]
        x1 = max(int(bbox["x1"]), 0)
        y1 = max(int(bbox["y1"]), 0)
        x2 = min(int(bbox["x2"]), img_w)
        y2 = min(int(bbox["y2"]), img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = 255
        painted += 1
    if painted == 0:
        return image
    logger.info("detect_curve_monochrome: inpainting %d OCR label box(es)", painted)
    return cv2.inpaint(image, mask, MONO_INPAINT_RADIUS, cv2.INPAINT_TELEA)


def _remove_straight_lines(ink: np.ndarray, img_w: int, img_h: int, cv2) -> np.ndarray:
    """Subtract near-full-span horizontal/vertical runs (gridlines, axes,
    table-frame borders) from an ink mask, then bridge the small gaps the
    subtraction leaves along a crossing curve.

    Only runs at least ``MONO_GRID_MIN_SPAN_FRAC`` of the plot dimension are
    removed — the flat-curve safety bar (legacy's kernel was short enough to
    eat flat curve segments; this one is not).
    """
    h_len = max(int(img_w * MONO_GRID_MIN_SPAN_FRAC), 1)
    v_len = max(int(img_h * MONO_GRID_MIN_SPAN_FRAC), 1)
    horiz = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)))
    vert = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))
    lines = cv2.bitwise_or(horiz, vert)
    cleaned = cv2.subtract(ink, lines)
    # Reconnect crossing nicks with a small dilation (skeletonize thins the
    # extra width away later); never merges stacked curves at this radius.
    bridge = cv2.getStructuringElement(cv2.MORPH_RECT, MONO_BRIDGE_KERNEL[::-1])
    return cv2.dilate(cleaned, bridge)


def detect_curve_monochrome(
    image: np.ndarray, ocr_lines: Optional[Sequence[OcrLine]] = None
) -> List[Detection]:
    """Detect solid black curve lines in a monochrome figure crop, classically.

    The grayscale fallback to :func:`detect_curve_classical` for the (universal,
    per the corpus) rdson case of black ink on white with no chromatic pixels.
    Pipeline: inpaint OCR-label boxes → threshold ink → remove near-full-span
    straight runs (gridlines/axes) by structure → bridge small crossing gaps →
    keep components that span a real fraction of the width and aren't dense
    text blobs. Emits the same :class:`Detection` objects as the color path.

    Args:
        image: HxWx3 uint8 BGR figure crop (as read by ``cv2.imread``).
        ocr_lines: Optional OCR lines; their boxes are inpainted before
            thresholding so on-curve labels don't split the curve. When
            ``None``, inpainting is skipped (detection still runs).

    Returns:
        One :class:`Detection` per credible curve-like component (score =
        fraction of image width spanned, boolean HxW mask). Empty list if
        nothing credible survives — the caller quarantines, never guessed.

    Raises:
        ValueError: If ``image`` is not an HxWx3 array.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR image, got shape {image.shape}")

    import cv2  # lazy, same convention as the color path

    img_h, img_w = image.shape[:2]
    work = _inpaint_ocr_boxes(image, ocr_lines, cv2) if ocr_lines else image
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    ink = (gray < MONO_INK_MAX_GRAY).astype(np.uint8)
    if not ink.any():
        logger.info("detect_curve_monochrome: no ink pixels — nothing to detect")
        return []

    cleaned = _remove_straight_lines(ink, img_w, img_h, cv2)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    min_span_px = MIN_COL_SPAN_FRAC * img_w
    exempt_span_px = MONO_DENSITY_EXEMPT_SPAN_FRAC * img_w
    detections: List[Detection] = []
    n_dropped = 0
    for label in range(1, n_labels):  # label 0 is background
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        density = area / max(width * height, 1)
        too_small = area < MIN_CURVE_AREA_PX or width < min_span_px
        # Dense + not-wide => text/logo block, not a thin curve. Wide
        # components (real curves, flat or bendy) are exempt so a flat curve
        # is never mistaken for a filled block.
        dense_text = width < exempt_span_px and density > MONO_MAX_FILL_DENSITY
        if too_small or dense_text:
            n_dropped += 1
            logger.info(
                "detect_curve_monochrome: dropped component %d (area=%dpx, "
                "col_span=%dpx, density=%.2f) — %s", label, area, width, density,
                "dense text/logo block" if dense_text else "too small/short",
            )
            continue
        mask = labels == label
        score = min(1.0, float(np.unique(np.nonzero(mask)[1]).size) / img_w)
        detections.append(Detection(score=score, mask=mask))

    logger.info(
        "detect_curve_monochrome: %d component(s) found, %d kept, %d dropped",
        n_labels - 1, len(detections), n_dropped,
    )
    return detections


def _median_col_thickness(mask: np.ndarray) -> float:
    """Median pixel count per occupied column of ``mask``.

    A proxy stroke-thickness measure: a genuine single stroke stays low even
    through steep segments (a handful of columns run tall, most don't), while
    two lines fused along most of their shared span pushes the MEDIAN up, not
    just a few outlier columns. Empty mask -> 0.0.
    """
    cols = np.nonzero(mask)[1]
    if cols.size == 0:
        return 0.0
    counts = np.bincount(cols)
    counts = counts[counts > 0]
    return float(np.median(counts))


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

    return result
