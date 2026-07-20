"""Curve naming for rdson_vs_tj (1- or 2-curve chart).

Two real templates exist in the corpus (T24/T25 surveys, 2026-07-13):

- IR/AUIRF: ONE solid curve -> named ``"rdson"``.
- Infineon "Diagram": TWO identical black solid curves distinguished only
  by a floating text label near each ("max"/"typ", or "98 %"/"typ" on the
  older BSB template) and by vertical position (upper = max/98 %, lower =
  typ on every observed chart) -> named ``"rdson_max"``/``"rdson_typ"``.

Owner rules (2026-07-13): label proximity is the primary naming signal for
the 2-curve case (:func:`name_curves_by_labels`); top/bottom position
(:func:`name_curves`) is the fallback when labels are absent/ambiguous.
"98 %" is treated as max. Any curve count other than 1 or 2 is an error
here — the pipeline's count gate quarantines those before naming.

Absolute-vs-normalized charts are distinguished by the result's ``units``
field, never by curve name.
"""
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.common.log import get_logger

logger = get_logger(__name__)

CURVE_NAMES = ["rdson"]
TWO_CURVE_NAMES_TOP_TO_BOTTOM = ["rdson_max", "rdson_typ"]

# Floating label text -> curve name. Normalized before lookup: lowercased,
# spaces collapsed away, trailing punctuation stripped ("98 %" == "98%",
# "TYP." == "typ").
_LABEL_TO_NAME = {
    "max": "rdson_max",
    "98%": "rdson_max",  # older BSB template: 98th-percentile upper curve
    "typ": "rdson_typ",
}

Point = Tuple[float, float]  # (row, col)
OcrLine = Dict[str, Any]  # {"text": str, "bounding_box": {"x1","y1","x2","y2"}}


def name_curves(curves: Sequence[Sequence[Point]]) -> List[str]:
    """Position-based naming: 1 curve -> ["rdson"]; 2 -> max on top, typ below.

    This is the registry naming function (and the 2-curve fallback when
    :func:`name_curves_by_labels` finds no usable labels). Names align to
    ``curves``' input order, mirroring the capacitance namer's contract.

    Args:
        curves: 1 or 2 point lists (``(row, col)`` pixel points, e.g. from
            :func:`src.extraction.skeletonize.mask_to_points`).

    Returns:
        ``["rdson"]``, or the 2-curve names aligned to input order.

    Raises:
        ValueError: If ``curves`` has neither 1 nor 2 entries, or any entry
            has no points (no position to name by).
    """
    if len(curves) not in (1, 2):
        raise ValueError(
            f"rdson_vs_tj naming needs exactly 1 or 2 curves, got {len(curves)}"
        )
    mean_rows = []
    for i, points in enumerate(curves):
        if not points:
            raise ValueError(f"curve at index {i} has no points to name by position")
        mean_rows.append(sum(row for row, _ in points) / len(points))

    if len(curves) == 1:
        return list(CURVE_NAMES)

    # Stable sort by mean pixel row (smaller row = higher on the chart):
    # ties keep input order, so the assignment is always deterministic.
    order = sorted(range(2), key=lambda i: mean_rows[i])
    names = [""] * 2
    for rank, curve_index in enumerate(order):
        names[curve_index] = TWO_CURVE_NAMES_TOP_TO_BOTTOM[rank]
    return names


def _normalize_label(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower()).rstrip(".:,;-")


def _nearest_curve_index(
    curves: Sequence[Sequence[Point]], cx: float, cy: float
) -> int:
    """Index of the curve with the closest point to pixel ``(cx, cy)``."""
    best_index, best_d2 = 0, float("inf")
    for i, points in enumerate(curves):
        for row, col in points:
            d2 = (row - cy) ** 2 + (col - cx) ** 2
            if d2 < best_d2:
                best_index, best_d2 = i, d2
    return best_index


def name_curves_by_labels(
    curves: Sequence[Sequence[Point]], ocr_lines: Sequence[OcrLine]
) -> Optional[List[str]]:
    """Name 2 curves from nearby "max"/"98 %"/"typ" OCR labels, or None.

    Each recognized label is anchored to the curve with the nearest traced
    point (same proximity idea as the position-based namers, but driven by
    the chart's own text). A single resolved label fixes the other curve to
    the remaining name — the labels always come in complementary pairs on
    the real charts, so one is enough.

    Args:
        curves: Exactly 2 point lists (``(row, col)`` pixel points).
        ocr_lines: The figure's OCR lines (pixel-space bounding boxes).

    Returns:
        Names aligned to ``curves``' input order, or ``None`` when no
        recognized label is present or the labels are contradictory /
        ambiguous (both nearest to the same curve, duplicate conflicting
        labels) — the caller falls back to position, never guesses here.

    Raises:
        ValueError: If ``curves`` does not have exactly 2 entries or any
            entry is empty.
    """
    if len(curves) != 2:
        raise ValueError(f"label naming needs exactly 2 curves, got {len(curves)}")
    if any(not points for points in curves):
        raise ValueError("label naming needs non-empty point lists")

    assigned: Dict[int, str] = {}  # curve index -> name
    for line in ocr_lines:
        name = _LABEL_TO_NAME.get(_normalize_label(line.get("text", "")))
        if name is None:
            continue
        bbox = line["bounding_box"]
        cx = (bbox["x1"] + bbox["x2"]) / 2
        cy = (bbox["y1"] + bbox["y2"]) / 2
        index = _nearest_curve_index(curves, cx, cy)
        if assigned.get(index, name) != name:
            logger.info(
                "rdson label naming: conflicting labels anchored to curve %d "
                "(%s vs %s) — falling back to position", index, assigned[index], name,
            )
            return None
        assigned[index] = name

    if not assigned:
        return None
    if len(assigned) == 1:
        [(index, name)] = assigned.items()
        other = 1 - index
        remaining = next(n for n in TWO_CURVE_NAMES_TOP_TO_BOTTOM if n != name)
        assigned[other] = remaining
    if len(set(assigned.values())) != 2:
        # Both labels landed on the same curve name for different curves —
        # contradictory; never guessed.
        logger.info("rdson label naming: ambiguous label anchoring %s — "
                    "falling back to position", assigned)
        return None

    logger.info("rdson label naming: %s", assigned)
    return [assigned[0], assigned[1]]
