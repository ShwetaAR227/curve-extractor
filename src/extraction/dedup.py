"""Duplicate-detection removal (Stage 5), a named/tested function per the brainstorm.

The model sometimes emits a near-duplicate second mask, most often on
flat/low-texture curves (documented in T8a/T8b) — this greedily keeps the
higher-confidence detection out of each duplicate group and drops the rest.
Two detections are duplicates if EITHER their masks have high pixel IoU
(reused from :func:`src.training.eval_lineformer.mask_iou`, not
reimplemented) OR — when ``use_flat_curve_heuristic`` is enabled (the
default) — they sit in nearly the same vertical band with a strongly
overlapping x-span (catches the flat-curve case, where a 1-2px vertical
offset can drag IoU below a useful threshold even though it's visibly the
same curve).

``use_flat_curve_heuristic=False`` opts out of that second check, using
mask IoU alone. Added for the ``zth_multicurve_run1`` dedup investigation
(2026-07-17): the flat-band/x-span heuristic was tuned for
``capacitance_vs_vds``'s flat-plateau duplicates and over-merges genuinely
distinct, near-parallel curves on multi-curve-family charts (thermal
impedance's duty-cycle families) — measured on the 51-image
``split_zth_multicurve_batch1`` set: default heuristic gave 66.7% exact
curve-count match with 4 regressions (real curves dropped), IoU-only gave
78.4% with 2. Default stays ``True`` so the already-approved
``capacitance_vs_vds`` Stage-5 pipeline (the only current caller) is
unaffected.
"""
from typing import List, Optional, Tuple

import numpy as np

from src.common.log import get_logger
from src.extraction.inference import Detection
from src.training.eval_lineformer import mask_iou

logger = get_logger(__name__)

IOU_DUPLICATE_THRESHOLD = 0.5
VERTICAL_BAND_PX = 6.0
X_SPAN_OVERLAP_RATIO = 0.7


def _mean_y(mask: np.ndarray) -> Optional[float]:
    rows, _ = np.nonzero(mask)
    return float(rows.mean()) if rows.size else None


def _x_span(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    _, cols = np.nonzero(mask)
    return (int(cols.min()), int(cols.max())) if cols.size else None


def _x_span_overlap_ratio(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    overlap = min(a[1], b[1]) - max(a[0], b[0])
    if overlap <= 0:
        return 0.0
    shorter = min(a[1] - a[0], b[1] - b[0])
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def _is_duplicate(
    a: Detection, b: Detection, use_flat_curve_heuristic: bool = True
) -> bool:
    if mask_iou(a.mask, b.mask) >= IOU_DUPLICATE_THRESHOLD:
        return True

    if not use_flat_curve_heuristic:
        return False

    y_a, y_b = _mean_y(a.mask), _mean_y(b.mask)
    span_a, span_b = _x_span(a.mask), _x_span(b.mask)
    if y_a is None or y_b is None or span_a is None or span_b is None:
        return False

    same_band = abs(y_a - y_b) <= VERTICAL_BAND_PX
    same_shape = _x_span_overlap_ratio(span_a, span_b) >= X_SPAN_OVERLAP_RATIO
    return same_band and same_shape


def dedup_detections(
    detections: List[Detection], use_flat_curve_heuristic: bool = True
) -> Tuple[List[Detection], int]:
    """Greedily drop near-duplicate detections, keeping the higher-confidence one.

    Args:
        detections: Kept model detections (already score-filtered).
        use_flat_curve_heuristic: When True (default, preserves existing
            behavior for ``capacitance_vs_vds``), also treats same-vertical-
            band + overlapping-x-span pairs as duplicates. Set False to
            compare on mask IoU alone — see module docstring for why
            ``zth_multicurve_run1`` needs this off.

    Returns:
        ``(kept, n_removed)`` — ``kept`` in descending-score order, with at
        most one representative per duplicate group.
    """
    ordered = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    for candidate in ordered:
        if any(_is_duplicate(candidate, existing, use_flat_curve_heuristic)
               for existing in kept):
            logger.info(
                "dedup_detections: dropped detection (score=%.3f) as a duplicate",
                candidate.score,
            )
            continue
        kept.append(candidate)

    n_removed = len(detections) - len(kept)
    if n_removed:
        logger.info("dedup_detections: %d/%d detections removed as duplicates",
                     n_removed, len(detections))
    return kept, n_removed
