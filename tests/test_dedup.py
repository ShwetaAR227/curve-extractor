"""Tests for src.extraction.dedup — written FIRST (CLAUDE.md §2).

Named, tested duplicate-removal function (not inline guesswork, per the
brainstorm). Reuses mask_iou from src.training.eval_lineformer rather than
reimplementing IoU.
"""
import numpy as np
import pytest

from src.extraction.dedup import dedup_detections
from src.extraction.inference import Detection


def rect_mask(shape, y0, y1, x0, x1):
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_exact_duplicate_masks_keeps_higher_confidence():
    mask = rect_mask((100, 100), 40, 50, 10, 90)
    a = Detection(score=0.9, mask=mask)
    b = Detection(score=0.6, mask=mask.copy())
    kept, n_removed = dedup_detections([a, b])
    assert len(kept) == 1
    assert kept[0].score == pytest.approx(0.9)
    assert n_removed == 1


def test_high_iou_masks_deduplicated():
    a = Detection(score=0.9, mask=rect_mask((100, 100), 40, 50, 10, 90))
    b = Detection(score=0.7, mask=rect_mask((100, 100), 41, 51, 10, 90))  # 1px shift
    kept, n_removed = dedup_detections([a, b])
    assert len(kept) == 1
    assert kept[0].score == pytest.approx(0.9)
    assert n_removed == 1


def test_low_iou_but_same_vertical_band_and_x_span_deduplicated():
    # Simulates the documented T8a/T8b near-duplicate-on-flat-curve case:
    # two thin bands a few px apart vertically, same x-span, low mask IoU.
    a = Detection(score=0.85, mask=rect_mask((100, 100), 40, 43, 10, 90))
    b = Detection(score=0.55, mask=rect_mask((100, 100), 45, 48, 10, 90))
    kept, n_removed = dedup_detections([a, b])
    assert len(kept) == 1
    assert kept[0].score == pytest.approx(0.85)
    assert n_removed == 1


def test_genuinely_distinct_curves_both_kept():
    a = Detection(score=0.9, mask=rect_mask((100, 100), 10, 15, 10, 90))
    b = Detection(score=0.8, mask=rect_mask((100, 100), 70, 75, 10, 90))
    kept, n_removed = dedup_detections([a, b])
    assert len(kept) == 2
    assert n_removed == 0


def test_no_duplicates_among_three_is_a_no_op():
    a = Detection(score=0.9, mask=rect_mask((100, 100), 10, 15, 10, 90))
    b = Detection(score=0.8, mask=rect_mask((100, 100), 45, 50, 10, 90))
    c = Detection(score=0.7, mask=rect_mask((100, 100), 80, 85, 10, 90))
    kept, n_removed = dedup_detections([a, b, c])
    assert len(kept) == 3
    assert n_removed == 0


def test_three_detections_two_duplicates_one_distinct_reduces_to_two():
    a = Detection(score=0.9, mask=rect_mask((100, 100), 40, 50, 10, 90))
    dup = Detection(score=0.6, mask=rect_mask((100, 100), 41, 51, 10, 90))
    distinct = Detection(score=0.75, mask=rect_mask((100, 100), 80, 85, 10, 90))
    kept, n_removed = dedup_detections([a, dup, distinct])
    assert len(kept) == 2
    assert n_removed == 1
    kept_scores = {round(d.score, 2) for d in kept}
    assert kept_scores == {0.9, 0.75}


def test_single_detection_is_a_no_op():
    a = Detection(score=0.9, mask=rect_mask((100, 100), 10, 15, 10, 90))
    kept, n_removed = dedup_detections([a])
    assert kept == [a]
    assert n_removed == 0


def test_empty_list_is_a_no_op():
    kept, n_removed = dedup_detections([])
    assert kept == []
    assert n_removed == 0


def test_different_x_span_not_deduplicated_even_at_same_y():
    # Same vertical band, but disjoint x-span -> genuinely different curves
    # (e.g. two separate segments), not a duplicate.
    a = Detection(score=0.9, mask=rect_mask((100, 100), 40, 45, 5, 30))
    b = Detection(score=0.8, mask=rect_mask((100, 100), 40, 45, 70, 95))
    kept, n_removed = dedup_detections([a, b])
    assert len(kept) == 2
    assert n_removed == 0
