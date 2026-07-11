"""Text/OCR-only figure-vs-curve-type scoring (CLAUDE.md §1, stage 4).

One shared :func:`score_figure` scores ANY figure against ANY
:class:`~src.classification.curve_registry.CurveTypeSpec` — there is no
curve-type-specific code path here, only data-driven matching.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.common.log import get_logger
from src.classification.curve_registry import CurveTypeSpec

logger = get_logger(__name__)

# Scoring weights. Not owner-tunable config yet (no real data has been run
# through this yet) - deliberately simple, named constants so they're easy
# to find and adjust once real classification runs expose bad calls.
CAPTION_KEYWORD_WEIGHT = 3.0
AXIS_CORRECT_ZONE_WEIGHT = 2.5
AXIS_UNKNOWN_ZONE_WEIGHT = 1.0
AXIS_WRONG_ZONE_WEIGHT = 0.5

# An OCR line is "tall & narrow" or "wide & short" if one dimension is at
# least this many times the other.
ZONE_ASPECT_RATIO = 2.0
# A line counts as "near the left edge" / "near the bottom edge" if its
# center falls within this fraction of the figure's width/height.
ZONE_EDGE_FRACTION = 0.2

BoundingBox = Tuple[float, float, float, float]


@dataclass
class OcrLine:
    """One OCR-detected line of text within a figure, with its pixel bbox."""

    text: str
    bbox: Optional[BoundingBox] = None


@dataclass
class FigureCandidate:
    """One figure extracted from a page (stage 3 output), ready to score."""

    figure_id: str
    page: int
    figure_index: int
    image_path: str
    caption: Optional[str] = None
    ocr_lines: List[OcrLine] = field(default_factory=list)
    figure_width: Optional[float] = None
    figure_height: Optional[float] = None


@dataclass
class MatchedSignal:
    """One scoring contribution, kept for human auditability."""

    source: str  # "caption_keyword" | "axis_x" | "axis_y" | "positive_phrase" | "negative_phrase"
    text: str
    weight: float


@dataclass
class ScoreResult:
    total_score: float
    matched_signals: List[MatchedSignal]


def _classify_zone(
    bbox: Optional[Tuple], figure_width: Optional[float], figure_height: Optional[float]
) -> Optional[str]:
    """Return "x", "y", or None (unknown) for where a bbox sits in a figure.

    "y" = tall & narrow, near the left edge (a rotated y-axis label).
    "x" = wide & short, near the bottom edge (an x-axis label).
    Anything else (missing/malformed geometry, ambiguous shape/position) is
    unknown, not an error — callers give partial credit instead of crashing.
    """
    if bbox is None or figure_width is None or figure_height is None:
        return None
    if figure_width <= 0 or figure_height <= 0:
        return None
    try:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    except (TypeError, ValueError):
        logger.debug("Malformed bbox %r treated as unknown zone", bbox)
        return None

    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    is_tall = height > width * ZONE_ASPECT_RATIO
    is_wide = width > height * ZONE_ASPECT_RATIO
    near_left = center_x < figure_width * ZONE_EDGE_FRACTION
    near_bottom = center_y > figure_height * (1 - ZONE_EDGE_FRACTION)

    if is_tall and near_left:
        return "y"
    if is_wide and near_bottom:
        return "x"
    return None


def figure_has_complete_axes(figure: FigureCandidate) -> bool:
    """Does this figure have OCR lines in BOTH the x-axis and y-axis zones?

    Reuses the same :func:`_classify_zone` bbox-position heuristic as
    :func:`score_figure` — purely structural (any line in each zone counts,
    regardless of its text). A figure that fails this is very likely a
    partial crop (e.g. one panel of a composite-split multi-figure page)
    even if it scored well on content, since a missing axis label means the
    corresponding tick marks are probably missing too.

    Args:
        figure: The figure to check.

    Returns:
        True only if at least one OCR line classifies as "x" zone and at
        least one classifies as "y" zone.
    """
    zones = {
        _classify_zone(line.bbox, figure.figure_width, figure.figure_height)
        for line in figure.ocr_lines
    }
    return "x" in zones and "y" in zones


def score_figure(figure: FigureCandidate, spec: CurveTypeSpec) -> ScoreResult:
    """Score one figure against one curve-type spec.

    Combines caption keyword hits, position-aware axis-label keyword hits,
    and weighted positive/negative phrase matches. Deterministic: the same
    figure+spec always produces the same score and signal list, and
    duplicate OCR lines never inflate the score beyond the single best
    match per (zone, keyword).

    Args:
        figure: The extracted figure candidate to score.
        spec: The curve-type spec to score it against.

    Returns:
        ScoreResult with the total score and every contributing signal.
    """
    raw_signals: List[MatchedSignal] = []
    caption = figure.caption or ""
    caption_lower = caption.lower()

    for keyword in spec.caption_keywords:
        if keyword.lower() in caption_lower:
            raw_signals.append(MatchedSignal("caption_keyword", keyword, CAPTION_KEYWORD_WEIGHT))

    for line in figure.ocr_lines:
        text_lower = (line.text or "").lower()
        if not text_lower:
            continue
        zone = _classify_zone(line.bbox, figure.figure_width, figure.figure_height)
        for axis_name, keywords in spec.axis_keywords.items():
            for keyword in keywords:
                if keyword.lower() not in text_lower:
                    continue
                if zone == axis_name:
                    weight = AXIS_CORRECT_ZONE_WEIGHT
                elif zone is None:
                    weight = AXIS_UNKNOWN_ZONE_WEIGHT
                else:
                    weight = AXIS_WRONG_ZONE_WEIGHT
                raw_signals.append(MatchedSignal(f"axis_{axis_name}", keyword, weight))

    combined_text = " ".join([caption] + [line.text or "" for line in figure.ocr_lines]).lower()
    for phrase, weight in spec.positive_phrases:
        if phrase.lower() in combined_text:
            raw_signals.append(MatchedSignal("positive_phrase", phrase, weight))
    for phrase, weight in spec.negative_phrases:
        if phrase.lower() in combined_text:
            raw_signals.append(MatchedSignal("negative_phrase", phrase, -abs(weight)))

    # Dedupe by (source, text): keep only the strongest instance of each
    # keyword/phrase match so repeated OCR lines can't inflate the score.
    best: Dict[Tuple[str, str], MatchedSignal] = {}
    for signal in raw_signals:
        key = (signal.source, signal.text)
        if key not in best or signal.weight > best[key].weight:
            best[key] = signal

    signals = list(best.values())
    total = sum(s.weight for s in signals)
    return ScoreResult(total_score=total, matched_signals=signals)
