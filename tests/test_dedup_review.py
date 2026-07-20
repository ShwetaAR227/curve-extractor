"""Tests for src.training.dedup_review — written FIRST (CLAUDE.md §2).

Only the pure summarization logic is unit-tested here (no GPU/mmdet
required); the inference-driving path reuses src.extraction.inference and
src.extraction.dedup directly (both already tested) rather than
reimplementing anything, so it isn't retested here.
"""
import pytest

from src.training.dedup_review import build_report_row, summarize_dedup_report


def test_build_report_row_exact_match_both_before_and_after():
    row = build_report_row("img.png", "val", gt_count=5, raw_count=5,
                           dedup_count=5, n_removed=0)
    assert row["raw_match"] is True
    assert row["dedup_match"] is True


def test_build_report_row_over_detected_before_fixed_after():
    row = build_report_row("img.png", "val", gt_count=5, raw_count=7,
                           dedup_count=5, n_removed=2)
    assert row["raw_match"] is False
    assert row["dedup_match"] is True


def test_build_report_row_regression_dedup_removed_real_curve():
    row = build_report_row("img.png", "val", gt_count=5, raw_count=5,
                           dedup_count=4, n_removed=1)
    assert row["raw_match"] is True
    assert row["dedup_match"] is False


def test_summarize_empty_report_is_all_zero():
    summary = summarize_dedup_report([])
    assert summary["n_images"] == 0
    assert summary["exact_before"] == 0
    assert summary["exact_after"] == 0
    assert summary["regressions"] == []


def test_summarize_counts_exact_before_and_after():
    rows = [
        build_report_row("a.png", "val", 5, 5, 5, 0),  # exact both
        build_report_row("b.png", "val", 5, 7, 5, 2),  # fixed by dedup
        build_report_row("c.png", "val", 5, 6, 6, 0),  # still over after
    ]
    summary = summarize_dedup_report(rows)
    assert summary["n_images"] == 3
    assert summary["exact_before"] == 1
    assert summary["exact_after"] == 2


def test_summarize_flags_regressions_not_just_counts_them():
    rows = [
        build_report_row("a.png", "val", 5, 5, 5, 0),   # exact, unaffected
        build_report_row("b.png", "val", 5, 5, 4, 1),   # regression!
    ]
    summary = summarize_dedup_report(rows)
    assert summary["regressions"] == ["b.png"]


def test_summarize_under_detected_before_never_gets_worse_flagged_separately():
    # raw already under GT; dedup can't add detections, only remove, so a
    # further drop here is also worth surfacing distinctly from "regression"
    # (which specifically means "was exact, now isn't").
    rows = [build_report_row("a.png", "train", gt_count=7, raw_count=6,
                             dedup_count=6, n_removed=0)]
    summary = summarize_dedup_report(rows)
    assert summary["regressions"] == []
    assert summary["under_detected_worsened"] == []


def test_summarize_under_detected_worsened_by_dedup_is_flagged():
    rows = [build_report_row("a.png", "train", gt_count=7, raw_count=6,
                             dedup_count=5, n_removed=1)]
    summary = summarize_dedup_report(rows)
    assert summary["under_detected_worsened"] == ["a.png"]


def test_summarize_over_and_exact_after_partition_all_rows():
    rows = [
        build_report_row("a.png", "val", 5, 5, 5, 0),
        build_report_row("b.png", "val", 5, 6, 6, 0),
        build_report_row("c.png", "val", 7, 6, 6, 0),
    ]
    summary = summarize_dedup_report(rows)
    assert summary["exact_after"] == 1
    assert summary["over_after"] == 1
    assert summary["under_after"] == 1


def test_summarize_total_removed_sums_n_removed():
    rows = [
        build_report_row("a.png", "val", 5, 7, 5, 2),
        build_report_row("b.png", "val", 5, 6, 5, 1),
    ]
    summary = summarize_dedup_report(rows)
    assert summary["total_removed"] == 3


def test_summarize_exact_match_pct_computed_correctly():
    rows = [build_report_row(f"{i}.png", "val", 5, 5, 5, 0) for i in range(3)] + \
           [build_report_row("x.png", "val", 5, 6, 6, 0)]
    summary = summarize_dedup_report(rows)
    assert summary["exact_after_pct"] == pytest.approx(75.0)
