"""Position-based curve naming for capacitance_vs_vds.

Sorts the 3 kept curves by mean y-pixel row (image-space: smaller row =
higher on the chart) and names them top -> bottom Ciss -> Coss -> Crss.
Pixel space, not engineering units — this runs BEFORE calibration.
"""
from typing import List, Sequence, Tuple

CURVE_ORDER_TOP_TO_BOTTOM = ["Ciss", "Coss", "Crss"]

Point = Tuple[float, float]  # (row, col)


def name_curves(curves: Sequence[Sequence[Point]]) -> List[str]:
    """Name exactly 3 traced curves Ciss/Coss/Crss by top-to-bottom pixel position.

    Args:
        curves: Exactly 3 point lists (each a traced curve's ``(row, col)``
            pixel points, e.g. from :func:`src.extraction.skeletonize.mask_to_points`).

    Returns:
        Names aligned to ``curves``' input order (not sorted order).

    Raises:
        ValueError: If ``curves`` does not have exactly 3 entries, or any
            entry has no points (no position to sort by).
    """
    if len(curves) != len(CURVE_ORDER_TOP_TO_BOTTOM):
        raise ValueError(
            f"capacitance_vs_vds naming needs exactly {len(CURVE_ORDER_TOP_TO_BOTTOM)} "
            f"curves, got {len(curves)}"
        )

    mean_rows = []
    for i, points in enumerate(curves):
        if not points:
            raise ValueError(f"curve at index {i} has no points to name by position")
        mean_rows.append(sum(row for row, _ in points) / len(points))

    # Stable sort: ties keep original relative order, so a tie never crashes
    # and always produces a deterministic (if arbitrary) assignment.
    order = sorted(range(len(curves)), key=lambda i: mean_rows[i])

    names = [""] * len(curves)
    for rank, curve_index in enumerate(order):
        names[curve_index] = CURVE_ORDER_TOP_TO_BOTTOM[rank]
    return names
