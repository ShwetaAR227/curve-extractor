"""Mask -> ordered centerline points (Stage 5).

Uses ``skimage.morphology.skeletonize`` to collapse a predicted curve mask
to a single-pixel-wide centerline, consistent with LineFormer's own
``infer.py`` mask->centerline approach (per LEGACY_REVIEW.md §5), then
walks it column-by-column into an ordered ``(row, col)`` point list.

Deliberately avoids two documented legacy flaws (LEGACY_REVIEW.md §5):
averaging happens on the already-thin SKELETON, not the raw thick mask
(legacy's ``trace_curve`` averaging window ran on the full mask and
produced bogus mid-lines on crossing/multi-branch curves); and every
column is kept (legacy's ``interpolate`` step discarded any x that repeated
in its point list before resampling — here each column simply contributes
one point, no data is silently dropped column-by-column).
"""
from typing import List, Tuple

import numpy as np

from src.common.log import get_logger

logger = get_logger(__name__)

Point = Tuple[float, float]  # (row, col) in pixel space


def mask_to_points(mask: np.ndarray) -> List[Point]:
    """Skeletonize ``mask`` and return ordered ``(row, col)`` centerline points.

    Args:
        mask: Boolean HxW mask for one curve instance.

    Returns:
        One point per skeleton column, ordered by ascending column. A
        column with more than one skeleton pixel (e.g. a near-vertical
        segment) contributes the mean row of that column's pixels. Empty
        for an all-``False`` mask.
    """
    if not mask.any():
        return []

    from skimage.morphology import skeletonize

    skeleton = skeletonize(mask)
    rows, cols = np.nonzero(skeleton)
    if rows.size == 0:
        logger.info("mask_to_points: skeletonize produced no pixels from a non-empty mask")
        return []

    rows_by_col: dict = {}
    for row, col in zip(rows.tolist(), cols.tolist()):
        rows_by_col.setdefault(col, []).append(row)

    points = [
        (float(np.mean(rows_by_col[col])), float(col)) for col in sorted(rows_by_col)
    ]
    return points
