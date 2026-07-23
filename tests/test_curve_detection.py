"""Tests for src.extraction.curve_detection — generic classical curve
detection, extracted from src.extraction.classical (2026-07-21,
owner-approved). This is a PURE REFACTOR: the two test classes below were
moved verbatim (same bodies, same assertions) from
``tests/test_classical.py::TestDetectCurveClassical`` and
``tests/test_monochrome.py::TestDetectCurveMonochrome`` — only the import
location changed. The tuning constants these functions used as module-level
globals in ``classical.py`` are now keyword-only parameters on the moved
functions, defaulting to the exact same corpus-tuned values, so calling
either function with no overrides reproduces rdson_vs_tj's existing
behavior exactly (verified by the fact that these moved tests, unchanged,
still pass — plus rdson_vs_tj's own full suite, checked separately).

``TestParameterOverridesActuallyApply`` at the bottom is NOT migrated from
anywhere — it's new coverage proving the actual point of the refactor: a
future curve type can override any tunable and get different behavior,
without touching (or duplicating) these functions.

All images are synthetic numpy arrays (no fixtures on disk, no GPU, no
network — CLAUDE.md §2).
"""
import numpy as np
import pytest

from src.extraction.curve_detection import (
    detect_curve_classical,
    detect_curve_monochrome,
)
from tests.test_classical import (
    CURVE_BGR,
    FAINT_BGR,
    IMG_H,
    IMG_W,
    X_AXIS_ROWS,
    blank_chart,
    curve_row,
    draw_axes,
    draw_curve,
    draw_gridlines,
    good_ocr_lines,
    ocr_line,
    standard_chart,
)
from tests.test_monochrome import BLACK, draw_hline, draw_mono_curve, draw_vline, mono_chart


# ------------------------------------------------- detect_curve_classical
# Moved verbatim from tests/test_classical.py::TestDetectCurveClassical.

class TestDetectCurveClassical:
    def test_clean_curve_yields_single_detection_with_valid_score_and_mask(self):
        # Scenario 1: clean solid curve line.
        img = standard_chart()
        detections = detect_curve_classical(img)
        assert len(detections) == 1
        det = detections[0]
        assert det.mask.dtype == np.bool_
        assert det.mask.shape == (IMG_H, IMG_W)
        assert 0.0 < det.score <= 1.0
        # The mask must actually cover the drawn curve, end to end.
        for col in (60, 150, 250, 349):
            assert det.mask[:, col].any(), f"curve pixel at col {col} not in mask"

    def test_gridline_crossing_keeps_one_detection_and_excludes_gridline_pixels(self):
        # Scenario 2: the curve crosses gray gridlines; the gridlines are
        # background, not curve — they must be neither absorbed nor allowed
        # to split the detection in two.
        img = standard_chart()
        detections = detect_curve_classical(img)
        assert len(detections) == 1
        det = detections[0]
        # Gridline-only pixels far from the curve stay out of the mask:
        # top horizontal gridline (row 40) where the curve is near row 182,
        assert not det.mask[40, 100:110].any()
        # and a vertical gridline (col 118) well below the curve (row ~174).
        assert not det.mask[225:239, 118].any()

    def test_curve_near_axis_edge_does_not_absorb_axis_pixels(self):
        # Scenario 3: a low, flat curve hugging the x-axis (2 px above it).
        img = blank_chart()
        draw_axes(img)
        img[236:238, 60:350] = CURVE_BGR
        detections = detect_curve_classical(img)
        assert len(detections) == 1
        det = detections[0]
        # The black axis lines must not leak into the curve mask.
        black = img.sum(axis=2) < 100
        assert not (det.mask & black).any()
        assert not det.mask[X_AXIS_ROWS[0]:X_AXIS_ROWS[1], :].any()

    def test_small_gaps_still_one_detection_spanning_full_extent(self):
        # Scenario 4: small breaks (compression/anti-aliasing dropouts) in a
        # solid curve are bridged — one curve, not three fragments.
        img = standard_chart(gaps=((150, 153), (220, 223), (280, 283)))
        detections = detect_curve_classical(img)
        assert len(detections) == 1
        det = detections[0]
        assert det.mask[:, 65].any()    # before the first gap
        assert det.mask[:, 345].any()   # after the last gap

    def test_legend_swatch_and_text_blob_are_not_detected_as_curves(self):
        # Scenario 5: a short legend swatch (same color as the curve) and a
        # dark text blob must not become detections or join the curve mask.
        img = standard_chart()
        img[50:53, 280:292] = CURVE_BGR   # legend swatch: 12 px, way too short
        img[48:58, 296:330] = 60          # dark "Tj = 25degC"-style text blob
        detections = detect_curve_classical(img)
        assert len(detections) == 1
        assert not detections[0].mask[45:62, 275:335].any()

    def test_faint_low_contrast_curve_is_still_detected(self):
        # Scenario 6: washed-out scan/print colors still count as a curve.
        img = blank_chart()
        draw_axes(img)
        draw_gridlines(img)
        draw_curve(img, color=FAINT_BGR)
        detections = detect_curve_classical(img)
        assert len(detections) == 1
        assert detections[0].mask[:, 200].any()

    def test_curve_touching_image_border_is_detected_without_crash(self):
        # Scenario 7: the crop sometimes clips the plot; a curve running to
        # the image edge must not crash border-sensitive morphology.
        img = blank_chart()
        for col in range(0, IMG_W):
            row = 5 + int(round((250 - 5) * (1 - col / (IMG_W - 1))))
            img[row:min(row + 3, IMG_H), col] = CURVE_BGR
        detections = detect_curve_classical(img)
        assert len(detections) == 1
        assert detections[0].mask[:, 0].any()
        assert detections[0].mask[:, IMG_W - 1].any()

    def test_blank_image_returns_no_detections_without_crash(self):
        # Scenario 8 (detector half): nothing to find -> empty list, no throw.
        assert detect_curve_classical(blank_chart()) == []

    def test_two_distinct_curves_are_both_returned_never_silently_picked(self):
        # Never-guess rule: if the chart actually has two long curves (e.g.
        # two Vgs conditions), BOTH come back — the pipeline's count gate
        # quarantines the figure; the detector must not pick a winner.
        img = standard_chart()
        for col in range(60, 350):  # second, blue curve well below the first
            row = curve_row(col) + 60
            img[row:row + 3, col] = (220, 40, 40)
        detections = detect_curve_classical(img)
        assert len(detections) == 2


# ------------------------------------------------ detect_curve_monochrome
# Moved verbatim from tests/test_monochrome.py::TestDetectCurveMonochrome.

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


# ------------------------------------- parameter overrides actually apply
#
# NEW (not migrated): the point of turning the tuning constants into
# keyword-only parameters is that a future curve type (different chart
# characteristics than rdson_vs_tj) can override any of them without
# duplicating these functions. These confirm an override actually changes
# the outcome, not just that it exists in the signature.

class TestParameterOverridesActuallyApply:
    def test_raising_chroma_min_spread_can_make_a_faint_curve_invisible(self):
        img = blank_chart()
        draw_axes(img)
        draw_curve(img, color=FAINT_BGR)  # channel spread ~70
        assert len(detect_curve_classical(img)) == 1  # default (40) sees it
        assert detect_curve_classical(img, chroma_min_spread=200) == []

    def test_lowering_min_curve_area_px_admits_a_previously_dropped_swatch(self):
        img = blank_chart()
        draw_axes(img)
        img[50:53, 280:292] = CURVE_BGR  # ~36px swatch: too small by default
        assert detect_curve_classical(img) == []
        dets = detect_curve_classical(img, min_curve_area_px=5, min_col_span_frac=0.0)
        assert len(dets) >= 1

    def test_raising_ink_max_gray_detects_a_lighter_gray_curve(self):
        img = blank_chart()
        draw_axes(img)
        gray = 150  # lighter than the default ink_max_gray=128 threshold
        for col in range(60, 350):
            row = curve_row(col)
            img[row:row + 3, col] = (gray, gray, gray)
        assert detect_curve_monochrome(img) == []  # default threshold misses it
        dets = detect_curve_monochrome(img, ink_max_gray=200)
        assert len(dets) == 1
