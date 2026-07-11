"""Per-page and per-device curve-type classification (CLAUDE.md §1, stage 4).

Ranks every unclaimed figure on a page against a target curve-type spec and
returns a matched / quarantined / no_match verdict. Mutual exclusion across
curve types is explicit state (a ``claimed`` set passed in and returned) —
never global mutation or monkey-patching, so callers (a future orchestrator)
fully control and can audit claim order.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from src.common.log import get_logger
from src.classification.curve_registry import get_spec
from src.classification.scoring import FigureCandidate, figure_has_complete_axes, score_figure

logger = get_logger(__name__)

# A figure "matches" only if its score clears this floor AND beats the
# runner-up by at least this margin; otherwise it's ambiguous -> quarantine.
MATCH_THRESHOLD = 5.0
MATCH_MARGIN = 2.0


class ClassificationStatus(str, Enum):
    MATCHED = "matched"
    QUARANTINED = "quarantined"
    NO_MATCH = "no_match"


@dataclass
class ClassificationResult:
    target_curve_type: str
    status: ClassificationStatus
    figure: Optional[FigureCandidate]
    score: float
    runner_up_score: float
    margin: float
    page: Optional[int]
    reason: str
    all_scores: List[Tuple[str, float]]


def classify_page(
    figures: List[FigureCandidate], target_curve_type: str, claimed: Optional[Set[str]] = None
) -> ClassificationResult:
    """Score every unclaimed figure on one page against ``target_curve_type``.

    Args:
        figures: Figures extracted from a single page.
        target_curve_type: Registry key to score against.
        claimed: figure_ids already claimed by another (or the same) curve
            type; these are excluded from consideration entirely.

    Returns:
        A ClassificationResult naming the best candidate (if any) and why
        it was matched, quarantined, or rejected as no_match.
    """
    claimed = claimed or set()
    spec = get_spec(target_curve_type)  # raises KeyError early on unknown types
    page = figures[0].page if figures else None

    candidates = []
    for figure in figures:
        if figure.figure_id in claimed:
            logger.debug("Skipping claimed figure %s on page %s", figure.figure_id, page)
            continue
        candidates.append((figure, score_figure(figure, spec)))

    if not candidates:
        reason = "no unclaimed figures available on this page"
        logger.info("classify_page(%s, page=%s): %s", target_curve_type, page, reason)
        return ClassificationResult(
            target_curve_type, ClassificationStatus.NO_MATCH, None, 0.0, 0.0, 0.0, page, reason, []
        )

    candidates.sort(key=lambda pair: pair[1].total_score, reverse=True)
    all_scores = [(fig.figure_id, res.total_score) for fig, res in candidates]
    best_figure, best_result = candidates[0]
    runner_up_score = candidates[1][1].total_score if len(candidates) > 1 else 0.0
    margin = best_result.total_score - runner_up_score

    logger.info(
        "classify_page(%s, page=%s): considered %d figure(s), scores=%s",
        target_curve_type, page, len(candidates), all_scores,
    )

    if best_result.total_score <= 0:
        reason = f"best candidate {best_figure.figure_id} scored {best_result.total_score:.2f} (<=0), no meaningful match"
        status = ClassificationStatus.NO_MATCH
    elif best_result.total_score >= MATCH_THRESHOLD and margin >= MATCH_MARGIN:
        reason = (
            f"{best_figure.figure_id} scored {best_result.total_score:.2f} "
            f"(>= threshold {MATCH_THRESHOLD}) with margin {margin:.2f} "
            f"(>= {MATCH_MARGIN}) over runner-up"
        )
        status = ClassificationStatus.MATCHED
    else:
        reason = (
            f"{best_figure.figure_id} scored {best_result.total_score:.2f} but "
            f"threshold ({MATCH_THRESHOLD}) or margin ({margin:.2f} vs required "
            f"{MATCH_MARGIN}) not met — ambiguous, needs human review"
        )
        status = ClassificationStatus.QUARANTINED

    if status == ClassificationStatus.MATCHED and not figure_has_complete_axes(best_figure):
        reason = (
            f"{best_figure.figure_id} scored {best_result.total_score:.2f} (would match) but is "
            f"missing an x-axis or y-axis OCR signal (incomplete_axes) — likely a partial/composite "
            f"crop, quarantined for review instead of trusted"
        )
        status = ClassificationStatus.QUARANTINED

    logger.info("classify_page(%s, page=%s): %s -> %s", target_curve_type, page, status.value, reason)
    figure_for_result = best_figure if status != ClassificationStatus.NO_MATCH else None
    return ClassificationResult(
        target_curve_type, status, figure_for_result, best_result.total_score,
        runner_up_score, margin, page, reason, all_scores,
    )


def classify_device(
    figures_by_page: Dict[int, List[FigureCandidate]],
    target_curve_type: str,
    claimed: Optional[Set[str]] = None,
) -> Tuple[ClassificationResult, Set[str]]:
    """Classify across every page of a device, picking the single best result.

    Prefers any "matched" page result (highest score wins); if none match,
    falls back to the best "quarantined" result; otherwise "no_match". Only
    a "matched" result claims its figure — quarantined figures stay
    available for human review / re-classification.

    Args:
        figures_by_page: Every extracted figure for one device, keyed by
            page number.
        target_curve_type: Registry key to score against.
        claimed: figure_ids already claimed by earlier calls (other curve
            types, or this one on a prior device). Not mutated in place.

    Returns:
        (result, new_claimed) — new_claimed is ``claimed`` plus the winning
        figure_id if (and only if) the result is "matched".
    """
    claimed = set(claimed) if claimed else set()
    get_spec(target_curve_type)  # raises KeyError early on unknown types

    page_results: List[ClassificationResult] = []
    for page in sorted(figures_by_page):
        figures = figures_by_page[page]
        if not figures:
            continue
        page_results.append(classify_page(figures, target_curve_type, claimed))

    if not page_results:
        reason = "device has no figures on any page"
        logger.info("classify_device(%s): %s", target_curve_type, reason)
        return (
            ClassificationResult(
                target_curve_type, ClassificationStatus.NO_MATCH, None, 0.0, 0.0, 0.0, None, reason, []
            ),
            claimed,
        )

    matched = [r for r in page_results if r.status == ClassificationStatus.MATCHED]
    if matched:
        best = max(matched, key=lambda r: r.score)
        new_claimed = set(claimed)
        new_claimed.add(best.figure.figure_id)
        logger.info(
            "classify_device(%s): claimed %s on page %s (score %.2f)",
            target_curve_type, best.figure.figure_id, best.page, best.score,
        )
        return best, new_claimed

    quarantined = [r for r in page_results if r.status == ClassificationStatus.QUARANTINED]
    if quarantined:
        best = max(quarantined, key=lambda r: r.score)
        logger.info(
            "classify_device(%s): quarantined, best candidate %s on page %s (score %.2f)",
            target_curve_type, best.figure.figure_id if best.figure else None, best.page, best.score,
        )
        return best, claimed

    best = max(page_results, key=lambda r: r.score)
    logger.info("classify_device(%s): no_match across all pages", target_curve_type)
    return best, claimed
