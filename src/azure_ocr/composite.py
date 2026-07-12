"""Stage-3 composite-figure splitting: one image, N charts -> N images.

Root cause of the known legacy bug (two charts shipped merged as one image):
legacy pixel-gap detection required columns to be ~97% white over the FULL
image height, so any caption/title text spanning under both charts crossed
the gap and defeated detection; the OCR-cluster fallback then often failed
too (garbled labels), and the merged figure passed through silently.

Fixes in this rebuild:

1. **OCR-text masking** — every OCR line's bounding box is painted white
   before gap detection. Spanning captions are OCR text, so they can no
   longer mask a real gap; chart ink (borders, curves, unrecognized ticks)
   is not OCR text, so it still blocks false gaps inside a chart.
2. **Stricter whiteness (0.995)** — with text masked, a true inter-chart
   gap is essentially perfectly white; chart-interior columns still cross
   border/curve pixels and fail. (Legacy's 0.97 tolerance existed to absorb
   caption text — masking removes the need and the false-negative source.)
3. **Candidate recall** — two or more "Figure N" sub-captions now qualify a
   figure as a composite candidate regardless of size/aspect (legacy missed
   squarish 2-ups below its 2.2 aspect threshold).
4. **No silent failures** — a candidate that cannot be split is flagged
   ``composite_suspect: True`` so downstream review can see it (legacy
   passed it through indistinguishable from a normal figure).
"""
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --- thresholds (rationale inline; config stays in one place) -----------------

_SUBFIG_CAPTION_RE = re.compile(r"(?:figure|fig\.?)\s*\d+", re.IGNORECASE)

_MIN_SUB_WIDTH = 200      # px — smaller crops can't be a usable chart
_MIN_SUB_HEIGHT = 150
_MAX_SUB_FIGURES = 8

_WHITE_PX = 230           # gray value above which a pixel counts as white
_GAP_WHITE_FRACTION = 0.995   # near-perfect white required once text is masked
_MIN_GAP_CORE_PX = 5
_GAP_MERGE_DISTANCE = 30
_EDGE_MARGIN_FRACTION = 0.12  # ignore split positions this close to the edge
_MAX_SPLITS_PER_AXIS = 3

# OCR-cluster fallback (columns): axis-title keywords + unit pattern
_X_AXIS_KEYWORDS = [
    "drain current", "gate-source voltage", "gate-to-source voltage",
    "drain-source voltage", "drain-to-source voltage", "gate charge",
    "junction temperature", "vds", "vgs", "qg", "tj", "time", "frequency",
    "capacitance",
]
_UNIT_RE = re.compile(r"\((?:V|A|mA|pF|nF|nC|°C|℃|C|Ω|mΩ|m[Ωo]?hm|mQ|W|mW|MHz|kHz|ns|%)\)", re.IGNORECASE)
_CLUSTER_SEP_FRACTION = 0.30
_OCR_SPLIT_WHITE_FRACTION = 0.55  # legacy band-validation threshold
_OCR_SPLIT_BAND_PX = 12


def find_composite_candidates(figures: List[dict]) -> List[dict]:
    """Return the figures worth attempting to split.

    Signals (any one qualifies): area > 2x median, width > 1.8x median,
    aspect > 2.2 (side-by-side) or < 1/2.2 (stacked), a generic composite
    caption, or >= 2 sub-figure captions in the OCR text. Package outlines,
    figures without OCR lines, and already-split figures never qualify.
    """
    dims = []
    for fig in figures:
        bb = fig.get("bounding_box") or {}
        w, h = bb.get("x2", 0) - bb.get("x1", 0), bb.get("y2", 0) - bb.get("y1", 0)
        if w > 0 and h > 0:
            dims.append((w * h, w))
    if not dims:
        return []
    median_area = sorted(a for a, _ in dims)[len(dims) // 2]
    median_width = sorted(w for _, w in dims)[len(dims) // 2]

    candidates = []
    for fig in figures:
        if fig.get("is_package_outline"):
            continue
        if fig.get("detection_method") == "composite_split":
            continue
        ocr_lines = fig.get("ocr_lines") or []
        if not ocr_lines:
            continue

        bb = fig.get("bounding_box") or {}
        w, h = bb.get("x2", 0) - bb.get("x1", 0), bb.get("y2", 0) - bb.get("y1", 0)
        if w <= 0 or h <= 0:
            continue
        aspect = w / h

        is_large = median_area > 0 and (w * h) > 2 * median_area
        is_wide = median_width > 0 and w > 1.8 * median_width
        is_extreme_aspect = aspect > 2.2 or aspect < 1 / 2.2
        caption = (fig.get("caption") or "").lower()
        is_generic_caption = any(
            kw in caption for kw in
            ("typical characteristics", "electrical characteristics", "performance characteristics")
        )
        n_sub_captions = sum(
            1 for ln in ocr_lines if _SUBFIG_CAPTION_RE.search(ln.get("text", ""))
        )
        has_sub_captions = n_sub_captions >= 2

        if is_large or is_wide or is_extreme_aspect or is_generic_caption or has_sub_captions:
            candidates.append(fig)
    return candidates


def split_composite_figures(
    figures: List[dict],
    base_dir: Union[str, Path],
) -> List[dict]:
    """Split composite figures into sub-figures; non-composites pass through.

    Must run AFTER OCR attachment (uses ``ocr_lines`` for text masking,
    candidate signals, and OCR reassignment). Candidates that cannot be
    split are kept with ``composite_suspect=True``. All indices are
    re-assigned sequentially at the end.
    """
    base_dir = Path(base_dir)
    if not figures:
        return figures

    candidate_ids = {id(f) for f in find_composite_candidates(figures)}
    result: List[dict] = []
    split_count = 0

    for fig in figures:
        if id(fig) not in candidate_ids:
            result.append(fig)
            continue
        sub_figures = _try_split_figure(fig, base_dir)
        if sub_figures is None:
            fig["composite_suspect"] = True
            logger.warning(
                "Composite candidate %s could not be split — flagged composite_suspect",
                fig.get("image_path"),
            )
            result.append(fig)
        else:
            result.extend(sub_figures)
            split_count += 1
            logger.info(
                "Split %s into %d sub-figure(s)", fig.get("image_path"), len(sub_figures)
            )

    for idx, fig in enumerate(result):
        fig["index"] = idx
    logger.info(
        "Composite pass: %d candidate(s), %d split, %d figure(s) total",
        len(candidate_ids), split_count, len(result),
    )
    return result


# --- single-figure split pipeline ---------------------------------------------


def _try_split_figure(fig: dict, base_dir: Path) -> Optional[List[dict]]:
    """Attempt one figure. Returns sub-figure dicts, or None if unsplittable."""
    image_path = base_dir / fig.get("image_path", "")
    img = cv2.imread(str(image_path)) if image_path.exists() else None
    if img is None:
        logger.warning("Composite candidate image missing/unreadable: %s", image_path)
        return None

    img_h, img_w = img.shape[:2]
    ocr_lines = fig.get("ocr_lines") or []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked = _mask_ocr_text(gray, ocr_lines)

    v_splits = _detect_gaps(masked, axis="vertical")
    h_splits = _detect_gaps(masked, axis="horizontal")

    if not v_splits:
        v_splits = _ocr_column_splits(ocr_lines, masked, img_w, img_h)

    if not v_splits and not h_splits:
        logger.debug("%s: no split evidence found", fig.get("image_path"))
        return None

    regions = _crop_regions(v_splits, h_splits, img_w, img_h)
    if len(regions) < 2:
        return None
    if len(regions) > _MAX_SUB_FIGURES:
        logger.warning(
            "%s: %d sub-regions exceeds max %d — truncating",
            fig.get("image_path"), len(regions), _MAX_SUB_FIGURES,
        )
        regions = regions[:_MAX_SUB_FIGURES]

    sub_captions = [
        ln for ln in ocr_lines if _SUBFIG_CAPTION_RE.search(ln.get("text", ""))
    ]

    figures_dir = image_path.parent
    stem = image_path.stem
    sub_figures: List[dict] = []

    for sub_idx, (cx1, cy1, cx2, cy2) in enumerate(regions):
        if cx2 - cx1 < _MIN_SUB_WIDTH or cy2 - cy1 < _MIN_SUB_HEIGHT:
            logger.debug("Sub-region %d too small (%dx%d) — dropped",
                         sub_idx, cx2 - cx1, cy2 - cy1)
            continue
        sub_name = f"{stem}_sub{sub_idx}.png"
        if not cv2.imwrite(str(figures_dir / sub_name), img[cy1:cy2, cx1:cx2]):
            logger.warning("Failed to write sub-figure %s", sub_name)
            continue

        sub_text, sub_lines = _reassign_ocr(ocr_lines, cx1, cy1, cx2, cy2)
        caption = _region_caption(sub_captions, cx1, cy1, cx2, cy2)
        if caption is None:
            caption = f"{fig.get('caption') or 'Unknown'} (sub {sub_idx + 1})"

        sub_figures.append({
            "index": 0,  # re-assigned by caller
            "page": fig.get("page"),
            "image_path": f"figures/{sub_name}",
            "caption": caption,
            "bounding_box": {"x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2},
            "ocr_text": sub_text,
            "ocr_lines": sub_lines,
            "detection_method": "composite_split",
            "is_package_outline": False,
        })

    if len(sub_figures) < 2:
        # Split degenerated — remove any partial files, report unsplittable.
        for sf in sub_figures:
            p = base_dir / sf["image_path"]
            if p.exists():
                p.unlink()
        return None
    return sub_figures


def _mask_ocr_text(gray: np.ndarray, ocr_lines: List[dict]) -> np.ndarray:
    """Paint every OCR line's bounding box white.

    THE bug fix: caption/title text spanning across charts is OCR text and
    disappears here, so it can no longer break gap detection; chart ink is
    not OCR'd and still blocks false gaps.
    """
    masked = gray.copy()
    img_h, img_w = masked.shape[:2]
    for ln in ocr_lines:
        bb = ln.get("bounding_box") or {}
        x1 = max(0, int(bb.get("x1", 0)) - 2)
        y1 = max(0, int(bb.get("y1", 0)) - 2)
        x2 = min(img_w, int(bb.get("x2", 0)) + 2)
        y2 = min(img_h, int(bb.get("y2", 0)) + 2)
        if x2 > x1 and y2 > y1:
            masked[y1:y2, x1:x2] = 255
    return masked


def _detect_gaps(masked: np.ndarray, axis: str) -> List[int]:
    """Find near-perfectly-white bands separating charts along one axis.

    Returns split positions (x for vertical, y for horizontal). Runs of
    white columns/rows are merged when close, must clear the edge margin,
    and are selected center-outward subject to minimum sub-figure sizes.
    """
    img_h, img_w = masked.shape[:2]
    if axis == "vertical":
        length, min_sub = img_w, _MIN_SUB_WIDTH
        white_fraction = np.mean(masked > _WHITE_PX, axis=0)
    else:
        length, min_sub = img_h, _MIN_SUB_HEIGHT
        white_fraction = np.mean(masked > _WHITE_PX, axis=1)

    if length < 2 * min_sub:
        return []
    margin = int(length * _EDGE_MARGIN_FRACTION)
    is_white = white_fraction >= _GAP_WHITE_FRACTION

    # Contiguous white runs -> (start, end)
    runs: List[Tuple[int, int]] = []
    start = None
    for i in range(length):
        if is_white[i] and start is None:
            start = i
        elif not is_white[i] and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, length))

    # Merge nearby runs (a thin gridline between two white runs is noise)
    merged: List[Tuple[int, int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] < _GAP_MERGE_DISTANCE:
            merged[-1] = (merged[-1][0], run[1])
        else:
            merged.append(run)

    # Split 1/3 into the gap so the next chart's leading labels stay with it
    # (legacy-verified positioning rationale).
    candidates = []
    for run_start, run_end in merged:
        width = run_end - run_start
        pos = run_start + width // 3
        if width >= _MIN_GAP_CORE_PX and margin <= pos <= length - margin:
            candidates.append(pos)
    if not candidates:
        return []

    max_splits = min((length // min_sub) - 1, _MAX_SPLITS_PER_AXIS)
    center = length / 2
    candidates.sort(key=lambda p: abs(p - center))

    selected: List[int] = []
    for pos in candidates:
        if len(selected) >= max_splits:
            break
        edges = [0] + sorted(selected + [pos]) + [length]
        if all(edges[i + 1] - edges[i] >= min_sub for i in range(len(edges) - 1)):
            selected.append(pos)
    return sorted(selected)


def _ocr_column_splits(
    ocr_lines: List[dict],
    masked: np.ndarray,
    img_w: int,
    img_h: int,
) -> List[int]:
    """Fallback: cluster x-axis title positions to find column boundaries.

    An OCR line counts as an x-axis title when it has an axis keyword AND
    (a parenthesised unit, length >= 20, or sits in the bottom quarter).
    Midpoints between clusters become splits, validated against a
    whitespace band on the masked image (legacy 0.55 threshold).
    """
    centers = []
    for ln in ocr_lines:
        text = (ln.get("text") or "").lower()
        bb = ln.get("bounding_box") or {}
        has_kw = any(kw in text for kw in _X_AXIS_KEYWORDS)
        if not has_kw:
            continue
        cy = (bb.get("y1", 0) + bb.get("y2", 0)) / 2
        if _UNIT_RE.search(ln.get("text") or "") or len(text) >= 20 or cy > img_h * 0.75:
            centers.append((bb.get("x1", 0) + bb.get("x2", 0)) / 2)

    if len(centers) < 2:
        return []
    centers.sort()
    clusters: List[List[float]] = [[centers[0]]]
    for c in centers[1:]:
        if c - clusters[-1][-1] > img_w * _CLUSTER_SEP_FRACTION:
            clusters.append([c])
        else:
            clusters[-1].append(c)
    if len(clusters) < 2:
        return []

    cluster_centers = sorted(sum(c) / len(c) for c in clusters)
    splits = [
        int((cluster_centers[i] + cluster_centers[i + 1]) / 2)
        for i in range(len(cluster_centers) - 1)
    ]
    return [s for s in splits if _band_is_whitish(masked, s)]


def _band_is_whitish(masked: np.ndarray, x: int) -> bool:
    half = _OCR_SPLIT_BAND_PX // 2
    x1, x2 = max(0, x - half), min(masked.shape[1], x + half)
    if x2 <= x1:
        return False
    band = masked[:, x1:x2]
    return float(np.mean(band > _WHITE_PX)) >= _OCR_SPLIT_WHITE_FRACTION


def _crop_regions(
    v_splits: List[int], h_splits: List[int], img_w: int, img_h: int,
) -> List[Tuple[int, int, int, int]]:
    """Row-major (x1, y1, x2, y2) regions from the split grid."""
    col_edges = [0] + sorted(v_splits) + [img_w]
    row_edges = [0] + sorted(h_splits) + [img_h]
    return [
        (col_edges[c], row_edges[r], col_edges[c + 1], row_edges[r + 1])
        for r in range(len(row_edges) - 1)
        for c in range(len(col_edges) - 1)
    ]


def _reassign_ocr(
    ocr_lines: List[dict], cx1: int, cy1: int, cx2: int, cy2: int,
) -> Tuple[List[str], List[dict]]:
    """Assign OCR lines whose center falls in the region; localize coords."""
    sub_text: List[str] = []
    sub_lines: List[dict] = []
    for ln in ocr_lines:
        bb = ln.get("bounding_box") or {}
        cx = (bb.get("x1", 0) + bb.get("x2", 0)) / 2
        cy = (bb.get("y1", 0) + bb.get("y2", 0)) / 2
        if cx1 <= cx <= cx2 and cy1 <= cy <= cy2:
            sub_text.append(ln.get("text", ""))
            sub_lines.append({
                "text": ln.get("text", ""),
                "bounding_box": {
                    "x1": max(0, bb.get("x1", 0) - cx1),
                    "y1": max(0, bb.get("y1", 0) - cy1),
                    "x2": max(0, bb.get("x2", 0) - cx1),
                    "y2": max(0, bb.get("y2", 0) - cy1),
                },
            })
    return sub_text, sub_lines


def _region_caption(
    sub_captions: List[dict], cx1: int, cy1: int, cx2: int, cy2: int,
) -> Optional[str]:
    """First sub-figure caption whose center lies inside the region."""
    for cap in sub_captions:
        bb = cap.get("bounding_box") or {}
        cx = (bb.get("x1", 0) + bb.get("x2", 0)) / 2
        cy = (bb.get("y1", 0) + bb.get("y2", 0)) / 2
        if cx1 <= cx <= cx2 and cy1 <= cy <= cy2:
            return cap.get("text")
    return None
