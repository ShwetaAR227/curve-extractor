"""Curve naming for vgsth_vs_tj (gate-threshold voltage vs. junction
temperature).

Curves are distinguished by which bias current the measurement was taken
at, expressed on real charts via one of two label vocabularies (owner-
specified, 2026-07-21):

- BAND scheme: "max"/"typ"/"min" (or the percentile variant "98%"/"typ"/
  "2%", extending rdson_vs_tj's own "98%"==max convention with a "2%"==min
  counterpart) — 1, 2, or 3 curves.
- CURRENT-VALUE scheme: the bias current stated directly ("I_D = 250uA",
  "I_D = 1.0mA", ...) — any curve count, disambiguated by the label's own
  normalized numeric value rather than a fixed vocabulary.

:func:`count_expected_curves` reads a chart's labels up front (before
detection/naming) to predict how many curves it should have — the
expected-count gate for this curve type, data-driven per chart instead of
a fixed constant (unlike rdson_vs_tj's ``EXPECTED_CURVE_COUNT``), since
vgsth's curve count isn't fixed.

:func:`name_curves_by_labels` mirrors rdson_vs_tj's function of the same
name (proximity-anchored labels via :func:`~src.extraction.curve_detection.nearest_curve_index`,
contradictions -> ``None``, caller quarantines) generalized to variable
curve count and a second naming scheme. UNLIKE rdson_vs_tj's 2-curve case,
there is no elimination-completion here: every curve must have its own
independently-resolved label, at any curve count — a partial resolution
(e.g. 2 of 3 labels found) quarantines rather than inferring the rest.

DUPLICATE-VALUE RULE (owner-decided, 2026-07-21): duplicate normalized
values are ALWAYS ambiguous -> ``None``/quarantine. Applies uniformly to
both schemes (a repeated band word and two current-value labels that
normalize to the identical value are treated identically) — a duplicate is
never silently trusted as harmless redundant notation, regardless of what
else is present on the chart.

KNOWN, DELIBERATELY OUT-OF-SCOPE LIMITATION: a duplicate could legitimately
arise from OCR re-detecting the same physical text region twice, rather
than from two genuinely distinct curves. Telling that apart from a real
ambiguous collision would need label-POSITION reasoning (are the duplicate
labels' bounding boxes suspiciously close together?) that this module does
not attempt — it always quarantines on any duplicate rather than guessing
which case it is.
"""
import re
from typing import Dict, List, Optional, Sequence

from src.common.log import get_logger
from src.extraction.curve_detection import OcrLine, Point, nearest_curve_index

logger = get_logger(__name__)

CURVE_NAMES = ["vgsth"]

# Band label text -> canonical role. Normalized before lookup (same
# normalization as rdson_vs_tj's own _normalize_label): lowercased,
# whitespace collapsed away, trailing punctuation stripped.
_BAND_LABEL_TO_ROLE: Dict[str, str] = {
    "max": "max",
    "98%": "max",   # percentile variant, mirrors rdson's 98%==max
    "typ": "typ",
    "min": "min",
    "2%": "min",    # percentile variant counterpart to 98%
}

# "I_D = <value><unit>" (or "ID=...", case-insensitive). Unit prefix is
# optional: bare "A" (amps), "m" (milli), or "u"/"µ"/"μ" (micro — ASCII
# fallback plus both real-world micro-sign glyphs, U+00B5 and U+03BC).
_ID_LABEL_RE = re.compile(r"(?i)I_?D\s*=\s*([\d.]+)\s*(u|µ|μ|m)?A\b")

_UNIT_MULTIPLIER_TO_UA: Dict[Optional[str], float] = {
    None: 1_000_000.0,
    "u": 1.0,
    "µ": 1.0,
    "μ": 1.0,
    "m": 1_000.0,
}


def _normalize_band_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower()).rstrip(".:,;-")


def _classify_band_label(text: str) -> Optional[str]:
    """Normalized ``text`` -> canonical role ("max"/"typ"/"min"), or None."""
    return _BAND_LABEL_TO_ROLE.get(_normalize_band_text(text))


def _parse_id_value_uA(text: str) -> Optional[float]:
    """Parse an "I_D = <value><unit>" label into a normalized µA float.

    Returns None if ``text`` doesn't match the pattern at all (not a
    current-value label).
    """
    match = _ID_LABEL_RE.search(text)
    if match is None:
        return None
    raw_value, unit_prefix = match.groups()
    return float(raw_value) * _UNIT_MULTIPLIER_TO_UA[unit_prefix]


def _format_ua(value: float) -> str:
    if value == int(value):
        return f"{int(value)}uA"
    return f"{value:g}uA"


def count_expected_curves(ocr_lines: Sequence[OcrLine]) -> Optional[int]:
    """Predict a vgsth_vs_tj chart's curve count from its own OCR labels.

    Args:
        ocr_lines: The figure's OCR lines (pixel-space bounding boxes).

    Returns:
        The number of distinct curves the label set implies, or ``None``
        when the labels don't resolve to exactly one safe count: no
        labels at all, both schemes present at once, or any duplicate
        normalized value/role (see the module's DUPLICATE-VALUE RULE) —
        never guessed.

    Raises:
        KeyError: If any ``ocr_lines`` entry has no ``bounding_box`` — a
            caller bug, never silently swallowed (CLAUDE.md §7).
    """
    band_roles: List[str] = []
    current_values: List[float] = []
    for line in ocr_lines:
        _ = line["bounding_box"]  # structural validation, raises if missing
        text = line.get("text", "")
        role = _classify_band_label(text)
        if role is not None:
            band_roles.append(role)
            continue
        value = _parse_id_value_uA(text)
        if value is not None:
            current_values.append(value)

    if band_roles and current_values:
        logger.info(
            "count_expected_curves: both band and current-value labels "
            "present — ambiguous, not guessed",
        )
        return None
    if band_roles:
        if len(set(band_roles)) != len(band_roles):
            logger.info(
                "count_expected_curves: duplicate band label(s) %s — ambiguous",
                band_roles,
            )
            return None
        return len(band_roles)
    if current_values:
        if len(set(current_values)) != len(current_values):
            logger.info(
                "count_expected_curves: duplicate current value(s) %s — ambiguous",
                current_values,
            )
            return None
        return len(current_values)
    return None


def _min_dist2_to_curve(points: Sequence[Point], cx: float, cy: float) -> float:
    return min((row - cy) ** 2 + (col - cx) ** 2 for row, col in points)


def _resolve_curve_index(
    curves: Sequence[Sequence[Point]], cx: float, cy: float
) -> Optional[int]:
    """``nearest_curve_index``, but None on a genuine distance tie.

    ``nearest_curve_index`` always returns SOME answer (deterministic,
    lower-index-wins tie-break) for callers that need one regardless; a
    label anchor is different — a genuine tie means we can't tell which
    curve the label actually belongs to, so naming must refuse to guess
    rather than silently accept the shared function's tie-break.
    """
    winner = nearest_curve_index(curves, cx, cy)
    winner_d2 = _min_dist2_to_curve(curves[winner], cx, cy)
    ties = sum(
        1 for points in curves if _min_dist2_to_curve(points, cx, cy) == winner_d2
    )
    return winner if ties == 1 else None


def name_curves_by_labels(
    curves: Sequence[Sequence[Point]], ocr_lines: Sequence[OcrLine]
) -> Optional[List[str]]:
    """Name curves from nearby band or current-value OCR labels, or None.

    1 curve always names ``["vgsth"]``, regardless of any labels present
    (no disambiguation needed). For 2+ curves, every curve must get its
    own independently-resolved label from EXACTLY ONE of the two schemes
    (no elimination-completion, no scheme mixing) or the whole result is
    unresolved.

    Args:
        curves: One or more point lists (``(row, col)`` pixel points).
        ocr_lines: The figure's OCR lines (pixel-space bounding boxes).

    Returns:
        Names aligned to ``curves``' input order, or ``None`` when the
        labels don't safely resolve every curve — the caller quarantines,
        never guesses here.

    Raises:
        ValueError: If ``curves`` is empty or any entry has no points.
    """
    if not curves:
        raise ValueError("vgsth_vs_tj naming needs at least 1 curve, got 0")
    if any(not points for points in curves):
        raise ValueError("naming needs a non-empty point list for every curve")

    if len(curves) == 1:
        return list(CURVE_NAMES)

    band_assigned: Dict[int, str] = {}
    value_assigned: Dict[int, float] = {}

    for line in ocr_lines:
        text = line.get("text", "")
        role = _classify_band_label(text)
        value = None if role is not None else _parse_id_value_uA(text)
        if role is None and value is None:
            continue  # unrelated text, ignored

        bbox = line["bounding_box"]
        cx = (bbox["x1"] + bbox["x2"]) / 2
        cy = (bbox["y1"] + bbox["y2"]) / 2
        index = _resolve_curve_index(curves, cx, cy)
        if index is None:
            logger.info(
                "vgsth label naming: tied nearest-curve distance for %r — "
                "ambiguous, not guessed", text,
            )
            return None

        if role is not None:
            if band_assigned.get(index, role) != role:
                logger.info(
                    "vgsth label naming: conflicting band labels anchored "
                    "to curve %d — ambiguous", index,
                )
                return None
            band_assigned[index] = role
        else:
            if value_assigned.get(index, value) != value:
                logger.info(
                    "vgsth label naming: conflicting current-value labels "
                    "anchored to curve %d — ambiguous", index,
                )
                return None
            value_assigned[index] = value

    if band_assigned and value_assigned:
        logger.info(
            "vgsth label naming: both band and current-value labels "
            "resolved — ambiguous, not guessed",
        )
        return None

    if band_assigned:
        if len(band_assigned) != len(curves) or len(set(band_assigned.values())) != len(band_assigned):
            logger.info(
                "vgsth label naming: band labels don't cover every curve "
                "uniquely (%s of %d) — ambiguous", band_assigned, len(curves),
            )
            return None
        names = [""] * len(curves)
        for idx, role in band_assigned.items():
            names[idx] = f"vgsth_{role}"
        logger.info("vgsth label naming (band scheme): %s", names)
        return names

    if value_assigned:
        if len(value_assigned) != len(curves) or len(set(value_assigned.values())) != len(value_assigned):
            logger.info(
                "vgsth label naming: current-value labels don't cover "
                "every curve uniquely (%s of %d) — ambiguous",
                value_assigned, len(curves),
            )
            return None
        names = [""] * len(curves)
        for idx, value in value_assigned.items():
            names[idx] = f"vgsth_id_{_format_ua(value)}"
        logger.info("vgsth label naming (current-value scheme): %s", names)
        return names

    logger.info("vgsth label naming: no recognized labels — ambiguous")
    return None
