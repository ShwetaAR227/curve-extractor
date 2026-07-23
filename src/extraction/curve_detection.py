"""Generic classical (non-AI) curve-line detection (CLAUDE.md §3, §4).

Extracted from :mod:`src.extraction.classical` (2026-07-21, owner-approved):
these functions take only an image (+ optional OCR lines) and never a
``curve_type`` — there was zero rdson-specific logic in them. They emit the
SAME :class:`src.extraction.inference.Detection` objects the AI path
produces (score + boolean HxW mask), so any classical extraction front-end
(rdson_vs_tj today, a future curve type tomorrow) hands them straight to
the existing frozen Stage-5 orchestration core
(:func:`src.extraction.pipeline.process_detections`) unchanged. This module
reimplements nothing downstream of detection (dedup, skeletonize, naming,
calibration, schema all stay in the frozen core).

Two detection paths:
- :func:`detect_curve_classical` — chromatic-pixel segmentation for solid-
  colored curves (axes/gridlines/text are achromatic).
- :func:`detect_curve_monochrome` — a grayscale sibling for black-ink-on-
  white charts: inpaint OCR-label boxes -> threshold ink -> remove near-
  full-span straight runs (gridlines/axes) by STRUCTURE -> bridge small
  crossing gaps -> keep components that span a real fraction of the width
  and aren't dense text blobs.

Tuning constants (module-level defaults below) were corpus-calibrated
against rdson_vs_tj's real chart corpus (T24/T25/T27 surveys) — see
:mod:`src.extraction.classical` for that history. They are exposed as
KEYWORD-ONLY function parameters, defaulting to those same corpus-tuned
values, specifically so a future curve type with different chart
characteristics can override any of them without duplicating these
functions. Calling any function with no overrides reproduces rdson_vs_tj's
existing behavior exactly (this was a pure refactor — zero behavior change
for rdson_vs_tj).

Also home to :func:`nearest_curve_index` (2026-07-21) — a generic pixel-
proximity helper for curve-naming modules that anchor OCR labels to the
nearest curve (rdson_vs_tj, vgsth_vs_tj), one shared implementation
instead of each duplicating it. rdson_vs_tj.py imports this one directly
(rewired 2026-07-22, pure refactor — its own former private
``_nearest_curve_index`` was byte-for-byte identical, confirmed before
the swap; zero behavior change, its 23-test suite pins that).
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.common.log import get_logger
from src.extraction.inference import Detection

logger = get_logger(__name__)

OcrLine = Dict[str, Any]  # {"text": str, "bounding_box": {"x1","y1","x2","y2"}}
Point = Tuple[float, float]  # (row, col)

# --- detect_curve_classical (color path) defaults ---
# A pixel is "chromatic" (curve candidate) when its max-min channel spread
# is at least this. Solid datasheet colors spread 150+; faint/washed-out
# scans still spread ~70; grays/black/white spread ~0.
CHROMA_MIN_SPREAD = 40
# Morphological-close kernel (rows, cols): wide enough to bridge small
# print/compression gaps along a mostly-horizontal curve, short enough to
# never fuse vertically-separated curves.
GAP_CLOSE_KERNEL = (3, 9)

# --- shared component-filtering defaults (both color and monochrome paths) ---
# Components smaller than this are legend swatches / color-key marks, not
# curves (a real curve at these figure sizes is several hundred pixels).
MIN_CURVE_AREA_PX = 100
# ... and a curve must span a meaningful fraction of the image width.
MIN_COL_SPAN_FRAC = 0.08

# --- detect_curve_monochrome (black-curve path) defaults ---
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
MONO_BRIDGE_KERNEL = (5, 9)
# Density (fill fraction of bbox) above which a component is text/logo, not
# a thin curve. Applied ONLY to components narrower than
# MONO_DENSITY_EXEMPT_SPAN_FRAC of the width, so a wide (flat or bendy)
# curve is never density-rejected.
MONO_MAX_FILL_DENSITY = 0.35
MONO_DENSITY_EXEMPT_SPAN_FRAC = 0.5
# Telea inpaint radius for OCR-label boxes (legacy used 3).
MONO_INPAINT_RADIUS = 3


def detect_curve_classical(
    image: np.ndarray,
    *,
    chroma_min_spread: int = CHROMA_MIN_SPREAD,
    gap_close_kernel: Sequence[int] = GAP_CLOSE_KERNEL,
    min_curve_area_px: int = MIN_CURVE_AREA_PX,
    min_col_span_frac: float = MIN_COL_SPAN_FRAC,
) -> List[Detection]:
    """Detect solid-colored curve lines in a figure crop, classically.

    Args:
        image: HxWx3 uint8 BGR figure crop (as read by ``cv2.imread``).
        chroma_min_spread: Minimum per-pixel max-min channel spread to
            count as a curve-candidate (chromatic) pixel.
        gap_close_kernel: Morphological-close kernel (rows, cols) used to
            bridge small print/compression gaps along the curve.
        min_curve_area_px: Minimum connected-component area to be a
            credible curve (smaller = legend swatch / stray mark).
        min_col_span_frac: Minimum fraction of the image width a component
            must span to be a credible curve.

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
    chroma = (spread.max(axis=2) - spread.min(axis=2)) >= chroma_min_spread
    if not chroma.any():
        logger.info("detect_curve_classical: no chromatic pixels — nothing to detect")
        return []

    closed = cv2.morphologyEx(
        chroma.astype(np.uint8), cv2.MORPH_CLOSE, np.ones(gap_close_kernel, np.uint8)
    )
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    min_span_px = min_col_span_frac * img_w
    detections: List[Detection] = []
    n_dropped = 0
    for label in range(1, n_labels):  # label 0 is background
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        if area < min_curve_area_px or width < min_span_px:
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
    image: np.ndarray, ocr_lines: Sequence[OcrLine], cv2,
    *, inpaint_radius: int = MONO_INPAINT_RADIUS,
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
    return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)


def _remove_straight_lines(
    ink: np.ndarray, img_w: int, img_h: int, cv2,
    *,
    grid_min_span_frac: float = MONO_GRID_MIN_SPAN_FRAC,
    bridge_kernel: Sequence[int] = MONO_BRIDGE_KERNEL,
) -> np.ndarray:
    """Subtract near-full-span horizontal/vertical runs (gridlines, axes,
    table-frame borders) from an ink mask, then bridge the small gaps the
    subtraction leaves along a crossing curve.

    Only runs at least ``grid_min_span_frac`` of the plot dimension are
    removed — the flat-curve safety bar (legacy's kernel was short enough to
    eat flat curve segments; this one is not).
    """
    h_len = max(int(img_w * grid_min_span_frac), 1)
    v_len = max(int(img_h * grid_min_span_frac), 1)
    horiz = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)))
    vert = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))
    lines = cv2.bitwise_or(horiz, vert)
    cleaned = cv2.subtract(ink, lines)
    # Reconnect crossing nicks with a small dilation (skeletonize thins the
    # extra width away later); never merges stacked curves at this radius.
    bridge = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(bridge_kernel)[::-1])
    return cv2.dilate(cleaned, bridge)


def detect_curve_monochrome(
    image: np.ndarray,
    ocr_lines: Optional[Sequence[OcrLine]] = None,
    *,
    ink_max_gray: int = MONO_INK_MAX_GRAY,
    grid_min_span_frac: float = MONO_GRID_MIN_SPAN_FRAC,
    bridge_kernel: Sequence[int] = MONO_BRIDGE_KERNEL,
    min_curve_area_px: int = MIN_CURVE_AREA_PX,
    min_col_span_frac: float = MIN_COL_SPAN_FRAC,
    max_fill_density: float = MONO_MAX_FILL_DENSITY,
    density_exempt_span_frac: float = MONO_DENSITY_EXEMPT_SPAN_FRAC,
    inpaint_radius: int = MONO_INPAINT_RADIUS,
) -> List[Detection]:
    """Detect solid black curve lines in a monochrome figure crop, classically.

    The grayscale fallback to :func:`detect_curve_classical` for black ink
    on white with no chromatic pixels. Pipeline: inpaint OCR-label boxes ->
    threshold ink -> remove near-full-span straight runs (gridlines/axes) by
    structure -> bridge small crossing gaps -> keep components that span a
    real fraction of the width and aren't dense text blobs. Emits the same
    :class:`Detection` objects as the color path.

    Args:
        image: HxWx3 uint8 BGR figure crop (as read by ``cv2.imread``).
        ocr_lines: Optional OCR lines; their boxes are inpainted before
            thresholding so on-curve labels don't split the curve. When
            ``None``, inpainting is skipped (detection still runs).
        ink_max_gray: Grayscale value below which a pixel counts as "ink".
        grid_min_span_frac: Minimum fraction of the plot dimension a
            straight run must span to be removed as a gridline/axis.
        bridge_kernel: Gap-bridge dilation kernel (rows, cols) applied after
            line removal.
        min_curve_area_px: Minimum connected-component area to be a
            credible curve.
        min_col_span_frac: Minimum fraction of the image width a component
            must span to be a credible curve.
        max_fill_density: Bbox fill-density above which a (non-exempt)
            component is treated as text/logo, not a curve.
        density_exempt_span_frac: Components spanning at least this
            fraction of the image width are exempt from the density filter.
        inpaint_radius: Telea inpaint radius for OCR-label boxes.

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
    work = (
        _inpaint_ocr_boxes(image, ocr_lines, cv2, inpaint_radius=inpaint_radius)
        if ocr_lines else image
    )
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    ink = (gray < ink_max_gray).astype(np.uint8)
    if not ink.any():
        logger.info("detect_curve_monochrome: no ink pixels — nothing to detect")
        return []

    cleaned = _remove_straight_lines(
        ink, img_w, img_h, cv2,
        grid_min_span_frac=grid_min_span_frac, bridge_kernel=bridge_kernel,
    )
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    min_span_px = min_col_span_frac * img_w
    exempt_span_px = density_exempt_span_frac * img_w
    detections: List[Detection] = []
    n_dropped = 0
    for label in range(1, n_labels):  # label 0 is background
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        density = area / max(width * height, 1)
        too_small = area < min_curve_area_px or width < min_span_px
        # Dense + not-wide => text/logo block, not a thin curve. Wide
        # components (real curves, flat or bendy) are exempt so a flat curve
        # is never mistaken for a filled block.
        dense_text = width < exempt_span_px and density > max_fill_density
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


def nearest_curve_index(curves: Sequence[Sequence[Point]], cx: float, cy: float) -> int:
    """Index of the curve with the closest point to pixel ``(cx, cy)``.

    Generic proximity helper shared by curve-naming modules that anchor
    OCR labels to the nearest curve by pixel distance (moved out of
    rdson_vs_tj.py's private ``_nearest_curve_index``, same algorithm, so
    it can be reused instead of duplicated by every naming module).

    Args:
        curves: One or more point lists (``(row, col)`` pixel points each).
        cx: Query point's x (column) pixel coordinate.
        cy: Query point's y (row) pixel coordinate.

    Returns:
        Index into ``curves`` of the nearest curve. On a genuine distance
        tie, deterministically favors the LOWER curve index (the first
        curve encountered wins — never randomly broken); callers that need
        to detect and refuse a tie rather than silently pick one should
        compare distances themselves rather than relying on this return
        value alone.
    """
    best_index, best_d2 = 0, float("inf")
    for i, points in enumerate(curves):
        for row, col in points:
            d2 = (row - cy) ** 2 + (col - cx) ** 2
            if d2 < best_d2:
                best_index, best_d2 = i, d2
    return best_index
