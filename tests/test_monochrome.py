"""Tests for the monochrome (black-and-white) rdson_vs_tj FALLBACK WIRING —
written FIRST (CLAUDE.md §2, red phase).

Real rdson_vs_tj datasheet charts are black ink on white with NO chromatic
pixels, so the existing color path (`detect_curve_classical`) finds nothing on
100% of the corpus (T24/T25 survey: 11/11 matched charts quarantined with 0
detections). `run_classical_pipeline` wires `detect_curve_monochrome` in as a
FALLBACK: color runs first, and only if that returns nothing does it fall
back to monochrome. This file covers that routing + the rdson-specific
suspiciously-thick-trace safety gate.

(`detect_curve_monochrome` itself — generic, zero rdson-specific logic —
moved to :mod:`src.extraction.curve_detection`; its own tests moved to
``tests/test_curve_detection.py`` (2026-07-21, owner-approved refactor). The
mono-chart drawing fixtures below (`mono_chart`, `draw_mono_curve`, etc.)
stay HERE since `test_curve_detection.py` imports them from this module.)

All images are synthetic numpy arrays (no fixtures on disk, no GPU, no
network — CLAUDE.md §2).
"""
import numpy as np
import pytest

from src.extraction.classical import detect_curve_classical, run_classical_pipeline
from src.extraction.schema import validate_result
from tests.test_classical import (
    X_AXIS_ROWS,
    Y_AXIS_COLS,
    blank_chart,
    curve_row,
    draw_axes,
    good_ocr_lines,
)

BLACK = (0, 0, 0)


# ---------------------------------------------------------------- fixtures

def draw_mono_curve(img, col_start=60, col_end=350, gaps=(), thickness=3,
                    row_fn=curve_row):
    """Draw a solid BLACK curve (row_fn maps column -> row)."""
    for col in range(col_start, col_end):
        if any(g0 <= col < g1 for g0, g1 in gaps):
            continue
        row = row_fn(col)
        img[row:row + thickness, col] = BLACK


def draw_hline(img, row, col0=Y_AXIS_COLS[1], col1=355, gray=0, thickness=2):
    img[row:row + thickness, col0:col1] = gray


def draw_vline(img, col, row0=40, row1=X_AXIS_ROWS[0], gray=0, thickness=2):
    img[row0:row1, col:col + thickness] = gray


def mono_chart(gridlines=False, **curve_kwargs):
    """Axes + one solid black curve (optionally dark gridlines)."""
    img = blank_chart()
    draw_axes(img)
    if gridlines:
        for row in (90, 140, 190):
            draw_hline(img, row, gray=0)
    draw_mono_curve(img, **curve_kwargs)
    return img


def run_mono(img, ocr_lines=None):
    return run_classical_pipeline(
        device="DEVM", curve_type="rdson_vs_tj", source_image="fig.png",
        image=img, ocr_lines=good_ocr_lines() if ocr_lines is None else ocr_lines,
    )


# ------------------------------------------------- fallback wiring

class TestColorToMonoFallback:
    def test_color_path_empty_triggers_monochrome_fallback_to_ok(self):
        # The whole point: a black chart yields ZERO color detections, so the
        # pipeline must fall back to mono and still produce a clean result.
        img = mono_chart(gridlines=True)
        assert detect_curve_classical(img) == []  # color path finds nothing
        result = run_mono(img)
        validate_result(result)
        assert result["status"] == "ok"
        assert len(result["curves"]) == 1
        assert result["curves"][0]["curve_name"] == "rdson"
        assert result["curves"][0]["points"]

    def test_fallback_result_detects_mohm_units(self):
        result = run_mono(mono_chart())
        assert result["status"] == "ok"
        assert result["units"] == "mOhm"

    def test_fallback_values_within_axis_ranges(self):
        result = run_mono(mono_chart())
        assert result["status"] == "ok"
        for p in result["curves"][0]["points"]:
            assert 25 - 2 <= p["x"] <= 150 + 2
            assert 0 - 2 <= p["y"] <= 80 + 2

    def test_fallback_no_curve_still_quarantines(self):
        img = blank_chart()
        draw_axes(img)
        for row in (90, 140, 190):
            draw_hline(img, row, gray=0)
        result = run_mono(img)
        validate_result(result)
        assert result["status"] == "needs_review"
        assert result["review_reason"]

    def test_two_mono_curves_via_fallback_return_both_named(self):
        img = blank_chart()
        draw_axes(img)
        draw_mono_curve(img, row_fn=curve_row)
        draw_mono_curve(img, row_fn=lambda c: curve_row(c) + 30)
        result = run_mono(img)
        validate_result(result)
        assert result["status"] == "ok"
        assert len(result["curves"]) == 2
        names = {c["curve_name"] for c in result["curves"]}
        assert names == {"rdson_max", "rdson_typ"}

    def test_color_success_does_not_use_mono_path(self):
        # Regression guard: a genuinely colored curve is still handled by the
        # color path (mono never consulted), result unchanged.
        from tests.test_classical import standard_chart
        img = standard_chart()
        assert len(detect_curve_classical(img)) == 1  # color path fires
        result = run_mono(img)
        assert result["status"] == "ok"
        assert result["curves"][0]["curve_name"] == "rdson"


# --------------------------------- suspiciously-thick-trace safety check
#
# Real-data finding (T27 follow-up, 2026-07-14): on 2/11 real charts
# (AUIRF7675M2TR, AUIRF7736M2TR) a nearby partial-width streak survived
# gridline removal and got fused into the curve component by the bridging
# dilation, growing a spurious upper branch while status stayed "ok" — a
# silently-wrong result. These tests cover the corpus-calibrated median
# column-thickness gate added to catch that failure mode.

class TestSuspiciouslyThickTraceCheck:
    def test_merged_parallel_streak_downgrades_ok_to_needs_review(self):
        # Reproduces the diagnosed mechanism: a partial-width (45% of plot
        # width, under the gridline-removal span bar) streak 10px above the
        # curve gets bridged into one fat component (median thickness 20px,
        # measured above threshold 18px).
        img = blank_chart()
        draw_axes(img)
        draw_mono_curve(img)
        for col in range(150, 330):
            r = curve_row(col) - 10
            img[r:r + 3, col] = 0
        result = run_mono(img)
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "thick" in result["review_reason"].lower()
        # Never an empty shell: the reviewer still sees what was traced.
        assert result["curves"][0]["points"]
        assert result["calibration"] is not None

    def test_clean_single_stroke_trace_is_not_downgraded(self):
        # Control: the same fixture family without the merged streak stays ok
        # (guards against an over-eager threshold).
        result = run_mono(mono_chart(gridlines=True))
        assert result["status"] == "ok"

    def test_color_path_thick_trace_is_not_subject_to_the_mono_gate(self):
        # The gate is specific to the monochrome merge-risk mechanism; a
        # colored curve (color path succeeds, mono never runs) is exempt
        # even though the same helper would flag equivalent thickness.
        from tests.test_classical import standard_chart
        img = standard_chart()
        assert len(detect_curve_classical(img)) == 1
        result = run_mono(img)
        assert result["status"] == "ok"
