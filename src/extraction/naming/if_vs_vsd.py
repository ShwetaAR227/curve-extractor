"""Curve naming for if_vs_vsd (body-diode forward/reverse current vs.
source-drain voltage).

Curves are distinguished by the junction temperature the measurement was
taken at, expressed on real charts as either a bare label ("25°C",
"175°C") or a "TJ = ..." prefixed one ("TJ = 25°C") — a single scheme,
unlike vgsth_vs_tj's two (band vs. current-value).

:func:`count_expected_curves` reads a chart's labels up front (before
detection/naming) to predict how many curves it should have — the
expected-count gate for this curve type, mirroring
:func:`~src.extraction.naming.vgsth_vs_tj.count_expected_curves`'s role.

DUPLICATE-VALUE RULE (same owner rule as vgsth_vs_tj): duplicate
normalized temperature values are ALWAYS ambiguous -> ``None``, never
silently trusted as harmless redundant notation.

COMPOUND-LABEL RULE (owner instruction, 2026-07-22): a label combining a
temperature with a percentile (e.g. "150°C, 98%" — seen once in the
reviewed corpus) is NOT parsed and is ALWAYS treated as ambiguous. Telling
apart "this chart genuinely has an unresolvable compound label" from "OCR
mangled a percentile onto the wrong line" would need scope this module
does not attempt — every compound label unconditionally quarantines,
never guessed.

:func:`name_curves_by_labels` mirrors
:func:`~src.extraction.naming.vgsth_vs_tj.name_curves_by_labels`'s shape
(proximity-anchored labels, no elimination-completion, contradictions ->
``None``) with ONE key difference: labels are anchored using only each
curve's LOW-V_SD-region point subset (roughly the first 25% of its own
pixel-column span), never the whole curve. Every real if_vs_vsd chart
reviewed converges/crosses at high current (forward voltage becomes less
temperature-dependent as current rises), so a curve's high-V_SD points
can end up geometrically closer to a label that's actually anchored to a
DIFFERENT curve's low-V_SD segment than that other curve's own (correct)
low-V_SD points are — an unrestricted whole-curve nearest-point search (or
a whole-curve average, as rdson_vs_tj's/capacitance's positional namers
use) would misattribute the label in exactly that case. Restricting the
search to each curve's own low-V_SD region, where every chart reviewed
stays well-separated, avoids this.
"""
import re
from typing import Dict, List, Optional, Sequence

from src.common.log import get_logger
from src.extraction.curve_detection import OcrLine, Point, nearest_curve_index

logger = get_logger(__name__)

CURVE_NAMES = ["if"]

# Fraction of each curve's own pixel-column span treated as its "low-V_SD
# region" for label-anchoring purposes (see module docstring). 0.25 leaves
# comfortable room for a floating label placed near a curve's start while
# staying well clear of the high-V_SD convergence zone on every chart
# reviewed.
LOW_VSD_REGION_FRAC = 0.25

# "TJ = <value>[degree-sign optional]C" -- prefix required, degree sign
# optional (OCR sometimes drops it when a "TJ ="/"Tj=" prefix is present).
_PREFIXED_TEMP_RE = re.compile(
    r"(?i)t[_,]?j\s*=\s*(-?\d+(?:\.\d+)?)\s*(?:°|º|deg\.?)?\s*c\b"
)
# "<value><degree-sign required>C" -- no prefix needed, but the degree
# token IS required (otherwise a bare "25C" is too easy to false-positive
# on unrelated OCR text).
_BARE_TEMP_RE = re.compile(
    r"(?i)(-?\d+(?:\.\d+)?)\s*(?:°|º|deg\.?)\s*c\b"
)


def _parse_temp_c(text: str) -> Optional[float]:
    """Parse a "TJ = 25°C" or bare "25°C"-style label into a float, or None."""
    match = _PREFIXED_TEMP_RE.search(text)
    if match is None:
        match = _BARE_TEMP_RE.search(text)
    if match is None:
        return None
    return float(match.group(1))


def _is_compound_temp_label(text: str) -> bool:
    """True iff ``text`` carries a temperature value AND a percentile
    (e.g. "150°C, 98%") -- the COMPOUND-LABEL RULE case, never parsed."""
    return _parse_temp_c(text) is not None and "%" in text


def count_expected_curves(ocr_lines: Sequence[OcrLine]) -> Optional[int]:
    """Predict an if_vs_vsd chart's curve count from its own OCR labels.

    Args:
        ocr_lines: The figure's OCR lines (pixel-space bounding boxes).

    Returns:
        The number of distinct temperature values the label set implies,
        or ``None`` when the labels don't resolve to exactly one safe
        count: no labels at all, any duplicate normalized value (see the
        module's DUPLICATE-VALUE RULE), or any compound label (see the
        COMPOUND-LABEL RULE) -- never guessed.

    Raises:
        KeyError: If any ``ocr_lines`` entry has no ``bounding_box`` — a
            caller bug, never silently swallowed (CLAUDE.md §7).
    """
    values: List[float] = []
    for line in ocr_lines:
        _ = line["bounding_box"]  # structural validation, raises if missing
        text = line.get("text", "")
        if _is_compound_temp_label(text):
            logger.info(
                "count_expected_curves: compound label %r (temperature + "
                "percentile) — not parsed, ambiguous", text,
            )
            return None
        value = _parse_temp_c(text)
        if value is not None:
            values.append(value)

    if not values:
        return None
    if len(set(values)) != len(values):
        logger.info(
            "count_expected_curves: duplicate temperature value(s) %s — ambiguous",
            values,
        )
        return None
    return len(values)


def _format_c(value: float) -> str:
    if value == int(value):
        return f"{int(value)}C"
    return f"{value:g}C"


def _low_vsd_region(points: Sequence[Point], frac: float = LOW_VSD_REGION_FRAC) -> List[Point]:
    """Return the subset of ``points`` in this curve's own low-V_SD region.

    "Low V_SD" = smallest pixel-column values -- ``col <= min_col +
    frac * (max_col - min_col)``, computed from THIS curve's own column
    span (never a whole-chart/other-curve span), so it works regardless
    of how much of the plotted width any one curve's detection actually
    covers. Always non-empty for a non-empty input (the point(s) at
    ``min_col`` always satisfy ``col <= min_col`` trivially).
    """
    cols = [col for _, col in points]
    min_col, max_col = min(cols), max(cols)
    threshold = min_col + frac * (max_col - min_col)
    return [p for p in points if p[1] <= threshold]


def _min_dist2(points: Sequence[Point], cx: float, cy: float) -> float:
    return min((row - cy) ** 2 + (col - cx) ** 2 for row, col in points)


def _resolve_curve_index(
    curves: Sequence[Sequence[Point]], cx: float, cy: float
) -> Optional[int]:
    """``nearest_curve_index``, but None on a genuine distance tie.

    Mirrors :func:`src.extraction.naming.vgsth_vs_tj._resolve_curve_index`
    (same rationale: a label anchor must refuse to guess on a genuine
    tie, unlike ``nearest_curve_index``'s own deterministic tie-break for
    callers that need SOME answer regardless).
    """
    winner = nearest_curve_index(curves, cx, cy)
    winner_d2 = _min_dist2(curves[winner], cx, cy)
    ties = sum(1 for points in curves if _min_dist2(points, cx, cy) == winner_d2)
    return winner if ties == 1 else None


def name_curves_by_labels(
    curves: Sequence[Sequence[Point]], ocr_lines: Sequence[OcrLine]
) -> Optional[List[str]]:
    """Name curves from nearby temperature OCR labels, or None.

    1 curve always names ``["if"]``, regardless of any labels present. For
    2+ curves, every curve must get its own independently-resolved
    temperature label (no elimination-completion) or the whole result is
    unresolved. Label anchoring uses ONLY each curve's low-V_SD-region
    point subset (see :func:`_low_vsd_region` and the module docstring) —
    never the whole curve.

    Args:
        curves: One or more point lists (``(row, col)`` pixel points).
        ocr_lines: The figure's OCR lines (pixel-space bounding boxes).

    Returns:
        Names aligned to ``curves``' input order (``if_25C``, ``if_175C``,
        ...), or ``None`` when the labels don't safely resolve every
        curve — the caller quarantines, never guessed here.

    Raises:
        ValueError: If ``curves`` is empty or any entry has no points.
    """
    if not curves:
        raise ValueError("if_vs_vsd naming needs at least 1 curve, got 0")
    if any(not points for points in curves):
        raise ValueError("naming needs a non-empty point list for every curve")

    if len(curves) == 1:
        return list(CURVE_NAMES)

    low_region_curves = [_low_vsd_region(points) for points in curves]

    assigned: Dict[int, float] = {}
    for line in ocr_lines:
        text = line.get("text", "")
        if _is_compound_temp_label(text):
            logger.info(
                "if_vs_vsd label naming: compound label %r — ambiguous, not parsed", text,
            )
            return None
        value = _parse_temp_c(text)
        if value is None:
            continue  # unrelated text, ignored

        bbox = line["bounding_box"]
        cx = (bbox["x1"] + bbox["x2"]) / 2
        cy = (bbox["y1"] + bbox["y2"]) / 2
        index = _resolve_curve_index(low_region_curves, cx, cy)
        if index is None:
            logger.info(
                "if_vs_vsd label naming: tied nearest-curve distance for "
                "%r — ambiguous, not guessed", text,
            )
            return None

        if assigned.get(index, value) != value:
            logger.info(
                "if_vs_vsd label naming: conflicting temperature labels "
                "anchored to curve %d — ambiguous", index,
            )
            return None
        assigned[index] = value

    if len(assigned) != len(curves) or len(set(assigned.values())) != len(assigned):
        logger.info(
            "if_vs_vsd label naming: labels don't cover every curve "
            "uniquely (%s of %d) — ambiguous", assigned, len(curves),
        )
        return None

    names = [""] * len(curves)
    for idx, value in assigned.items():
        names[idx] = f"if_{_format_c(value)}"
    logger.info("if_vs_vsd label naming: %s", names)
    return names
