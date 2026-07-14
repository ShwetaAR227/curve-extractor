"""Tests for the monochrome (black-and-white) rdson_vs_tj detector — written
FIRST (CLAUDE.md §2, red phase).

Real rdson_vs_tj datasheet charts are black ink on white with NO chromatic
pixels, so the existing color path (`detect_curve_classical`) finds nothing on
100% of the corpus (T24/T25 survey: 11/11 matched charts quarantined with 0
detections). This module adds `detect_curve_monochrome(image, ocr_lines=None)`
— a grayscale path that emits the SAME `src.extraction.inference.Detection`
objects the color path does, so the frozen Stage-5 core stays the only
pipeline — and wires it in as a FALLBACK: `run_classical_pipeline` tries color
first, and only if that returns nothing falls back to monochrome.

Design facts baked into these tests come from the real corpus
(`data/t24_mono_survey/MONO_DETECTOR_REQUIREMENTS.md`): curves are solid
black 3-10 px strokes spanning ~75%+ of width; gridlines/axes are long
straight runs (dark OR light) that must be removed by STRUCTURE, not intensity;
a flat curve segment must survive gridline removal; two close typ/max curves
must never be merged.

All images are synthetic numpy arrays (no fixtures on disk, no GPU, no
network — CLAUDE.md §2).
"""
import numpy as np
import pytest

from src.extraction.classical import (
    detect_curve_classical,
    detect_curve_monochrome,
    run_classical_pipeline,
)
from src.extraction.schema import validate_result
from tests.test_classical import (
    IMG_H,
    IMG_W,
    X_AXIS_ROWS,
    Y_AXIS_COLS,
    blank_chart,
    curve_row,
    draw_axes,
    good_ocr_lines,
    ocr_line,
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


# ------------------------------------------------ detect_curve_monochrome

class TestDetectCurveMonochrome:
    def test_clean_black_curve_yields_single_detection(self):
        img = mono_chart()
        dets = detect_curve_monochrome(img)
        assert len(dets) == 1
        det = dets[0]
        assert det.mask.dtype == np.bool_
        assert det.mask.shape == (IMG_H, IMG_W)
        assert 0.0 < det.score <= 1.0
        for col in (61, 150, 250, 348):
            assert det.mask[:, col].any(), f"curve col {col} missing from mask"

    def test_dark_gridlines_removed_not_returned_as_curves(self):
        img = mono_chart(gridlines=True)
        dets = detect_curve_monochrome(img)
        assert len(dets) == 1
        det = dets[0]
        # A dark gridline pixel far from the curve must not be in the mask.
        # Gridline at row 90 spans full width; the curve at col 300 is ~row 88..
        # test a gridline location where the curve is nowhere near (col 70,
        # curve there is ~row 197).
        assert not det.mask[90, 70:74].any()

    def test_flat_curve_segment_survives_gridline_removal(self):
        # THE critical safety case (legacy warning): a curve that runs nearly
        # flat for PART of its length must NOT be deleted as if it were a
        # gridline. Flat run here is 140 px (35% of width) — below the
        # gridline-length bar — while a real full-width gridline at row 100
        # IS removed.
        def flat_then_rise(col):
            if col < 240:
                return 150            # flat across cols 100..240 (140 px)
            return 150 - (col - 240)  # then rises
        img = blank_chart()
        draw_axes(img)
        draw_hline(img, 100, gray=0)                 # full-width gridline
        draw_mono_curve(img, col_start=100, col_end=340, row_fn=flat_then_rise)
        dets = detect_curve_monochrome(img)
        assert len(dets) == 1
        det = dets[0]
        # The flat middle survives:
        assert det.mask[148:154, 120:220].any(), "flat curve segment was deleted"
        assert det.mask[:, 110].any() and det.mask[:, 230].any()
        # The gridline (row 100), away from the curve, is gone:
        assert not det.mask[100, 110:220].any(), "full-width gridline not removed"

    def test_curve_crossing_gridline_small_gap_is_bridged(self):
        # A dark gridline the curve crosses nicks the curve when removed; a
        # small, conservative close bridges it back into ONE detection.
        img = blank_chart()
        draw_axes(img)
        draw_hline(img, 135, gray=0)  # the curve passes through ~row 135 near col 168
        draw_mono_curve(img)
        dets = detect_curve_monochrome(img)
        assert len(dets) == 1
        det = dets[0]
        assert det.mask[:, 61].any() and det.mask[:, 348].any()

    def test_two_parallel_curves_stay_separate_never_merged(self):
        img = blank_chart()
        draw_axes(img)
        draw_mono_curve(img, row_fn=curve_row)                 # upper
        draw_mono_curve(img, row_fn=lambda c: curve_row(c) + 30)  # lower, 30 px below
        dets = detect_curve_monochrome(img)
        assert len(dets) == 2, "typ/max curves must not be fused into one blob"

    def test_two_close_parallel_curves_still_separate(self):
        # Even a modest vertical gap (well beyond the tiny bridge kernel) must
        # not merge — the close is horizontal-biased on purpose.
        img = blank_chart()
        draw_axes(img)
        draw_mono_curve(img, row_fn=curve_row)
        draw_mono_curve(img, row_fn=lambda c: curve_row(c) + 18)
        dets = detect_curve_monochrome(img)
        assert len(dets) == 2

    def test_blank_image_returns_empty_no_crash(self):
        assert detect_curve_monochrome(blank_chart()) == []

    def test_axes_and_gridlines_but_no_curve_returns_empty(self):
        img = blank_chart()
        draw_axes(img)
        for row in (90, 140, 190):
            draw_hline(img, row, gray=0)
        for col in (118, 176, 234, 292):
            draw_vline(img, col, gray=0)
        assert detect_curve_monochrome(img) == []

    def test_thick_low_quality_curve_still_detected(self):
        # Scan-quality outlier: an 8-10 px bolder stroke still reads as ONE curve.
        img = mono_chart(thickness=9)
        dets = detect_curve_monochrome(img)
        assert len(dets) == 1
        assert dets[0].mask[:, 200].any()

    def test_curve_touching_border_detected_without_crash(self):
        img = blank_chart()
        for col in range(0, IMG_W):
            row = 5 + int(round((250 - 5) * (1 - col / (IMG_W - 1))))
            img[row:min(row + 3, IMG_H), col] = BLACK
        dets = detect_curve_monochrome(img)
        assert len(dets) == 1
        assert dets[0].mask[:, 0].any() and dets[0].mask[:, IMG_W - 1].any()

    def test_text_on_curve_inpaint_prevents_split_vs_whiteout_baseline(self):
        # Inpainting an OCR label box that sits ON the curve reconstructs the
        # curve underneath; a naive white-out would punch a hole and split it.
        cl = 200
        r = curve_row(cl)
        box_l, box_r = cl - 7, cl + 7  # 14 px wide, wider than the bridge kernel
        img = mono_chart()
        # a dark label blob sitting tightly on the curve stroke
        img[r - 1:r + 4, box_l:box_r] = 0
        label = [ocr_line("max", box_l, r - 1, box_r, r + 4)]

        # Inpaint reconstructs the stroke under the label -> ONE detection that
        # spans clear across the box.
        inpainted = detect_curve_monochrome(img, ocr_lines=label)
        assert len(inpainted) == 1
        det = inpainted[0]
        assert det.mask[:, box_l - 6].any() and det.mask[:, box_r + 6].any()

        # White-out baseline: same box blanked, no inpaint -> the curve splits
        # into two components (the 14 px hole exceeds the bridge kernel).
        whiteout = img.copy()
        whiteout[:, box_l:box_r] = 255
        base = detect_curve_monochrome(whiteout)
        assert len(base) >= 2, "white-out baseline should split the curve"

    def test_ocr_masking_suppresses_in_plot_text_blob(self):
        # A dark in-plot caption ("I_D = 84A") that is wide enough to worry
        # about is removed by OCR-box masking, leaving only the curve.
        # Text band in the upper-left, clear of the curve (which is low/right
        # there: curve is at rows ~146-195 across cols 70-180).
        img = mono_chart()
        img[60:78, 70:180] = 0  # dark in-plot caption band
        lines = good_ocr_lines() + [ocr_line("I_D = 84A", 70, 60, 180, 78)]
        dets = detect_curve_monochrome(img, ocr_lines=lines)
        assert len(dets) == 1
        assert not dets[0].mask[60:78, 70:180].any()

    def test_dense_text_blob_without_ocr_filtered_by_density(self):
        # No OCR available: a dense, narrow-ish text/logo block must still be
        # rejected by the density filter, not returned as a curve.
        img = blank_chart()
        draw_axes(img)
        img[60:120, 120:210] = 0  # 90x60 solid dark block: dense, sub-half-width
        assert detect_curve_monochrome(img) == []

    def test_tick_mark_residue_not_returned_only_curve(self):
        img = mono_chart()
        for col in (118, 176, 234, 292):        # short tick stubs under the axis
            img[X_AXIS_ROWS[1]:X_AXIS_ROWS[1] + 6, col:col + 2] = 0
        dets = detect_curve_monochrome(img)
        assert len(dets) == 1

    def test_dense_infineon_style_grid_removed(self):
        img = blank_chart()
        draw_axes(img)
        for row in range(50, 235, 12):          # ~15 dense 2 px gridlines
            draw_hline(img, row, gray=0, thickness=2)
        draw_mono_curve(img)
        dets = detect_curve_monochrome(img)
        assert len(dets) == 1

    def test_score_tracks_width_span_fraction(self):
        img = mono_chart()
        det = detect_curve_monochrome(img)[0]
        cols = np.unique(np.nonzero(det.mask)[1])
        span_frac = cols.size / IMG_W
        assert abs(det.score - span_frac) < 0.05

    def test_non_3channel_image_raises(self):
        with pytest.raises(ValueError):
            detect_curve_monochrome(np.zeros((IMG_H, IMG_W), dtype=np.uint8))


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
