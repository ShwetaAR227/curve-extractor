"""Tests for src.calibration.ticks — written FIRST (CLAUDE.md §2).

Ported from legacy D:\\Extractor\\5_opencv_extract\\cv_curve_extract.py
(parse_numeric_ticks lines 116-197, fit_axis lines 200-293, pixel_to_data
lines 391-402), per LEGACY_REVIEW.md §3 — this is the one piece of legacy
code deliberately lifted (CLAUDE.md §6). Behavior is verified against known
tick examples, not trusted blindly: linear axis, log axis auto-detection,
RANSAC outlier rejection, degenerate/insufficient ticks -> None.
"""
import math

import pytest

from src.calibration.ticks import (
    data_to_pixel,
    derive_calibration,
    detect_y_axis_units,
    fit_axis,
    parse_numeric_ticks,
    pixel_to_data,
)


def ocr_line(text, x1, y1, x2, y2):
    return {"text": text, "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


# --------------------------------------------------------------- parse_numeric_ticks

def test_parse_ticks_x_zone_by_bottom_position():
    # img 800x600; bottom 30% (cy/h > 0.70) is x-zone. cx=310 is well past
    # the 15%-width tight-corner cutoff (120px), so this isn't a baseline label.
    lines = [ocr_line("10", 300, 560, 320, 580)]
    x_ticks, y_ticks = parse_numeric_ticks(lines, 800, 600)
    assert x_ticks == [(10.0, 310.0)]
    assert y_ticks == []


def test_parse_ticks_y_zone_by_left_position():
    # left 30% (cx/w < 0.30) is y-zone, but not in x-zone (cy/h <= 0.70).
    lines = [ocr_line("100", 50, 100, 90, 120)]
    x_ticks, y_ticks = parse_numeric_ticks(lines, 800, 600)
    assert y_ticks == [(100.0, 110.0)]
    assert x_ticks == []


def test_parse_ticks_tight_corner_nonnegative_goes_to_y():
    # cx/w < 0.15 AND cy/h > 0.70, value >= 0 -> y-axis baseline label.
    lines = [ocr_line("0", 20, 560, 40, 580)]
    x_ticks, y_ticks = parse_numeric_ticks(lines, 800, 600)
    assert y_ticks == [(0.0, 570.0)]
    assert x_ticks == []


def test_parse_ticks_tight_corner_negative_goes_to_x():
    lines = [ocr_line("-40", 20, 560, 60, 580)]
    x_ticks, y_ticks = parse_numeric_ticks(lines, 800, 600)
    assert x_ticks == [(-40.0, 40.0)]
    assert y_ticks == []


def test_parse_ticks_repairs_unicode_minus_and_split_negative():
    lines = [ocr_line("−40", 100, 560, 140, 580)]  # unicode minus
    x_ticks, _ = parse_numeric_ticks(lines, 800, 600)
    assert x_ticks == [(-40.0, 120.0)]


def test_parse_ticks_strips_comma_formatted_numbers():
    lines = [ocr_line("10,000", 100, 560, 160, 580)]
    x_ticks, _ = parse_numeric_ticks(lines, 800, 600)
    assert x_ticks == [(10000.0, 130.0)]


def test_parse_ticks_compound_token_splits_evenly_across_bbox():
    lines = [ocr_line("0 20 40 60", 100, 560, 260, 580)]
    x_ticks, _ = parse_numeric_ticks(lines, 800, 600)
    assert len(x_ticks) == 4
    vals = [v for v, _ in x_ticks]
    assert vals == [0.0, 20.0, 40.0, 60.0]
    pxs = [p for _, p in x_ticks]
    assert pxs == sorted(pxs)


def test_parse_ticks_rejects_ocr_merged_huge_integer():
    lines = [ocr_line("100120140160180", 100, 560, 260, 580)]
    x_ticks, _ = parse_numeric_ticks(lines, 800, 600)
    assert x_ticks == []


def test_parse_ticks_ignores_non_numeric_text():
    lines = [ocr_line("VGS [V]", 100, 560, 200, 580)]
    x_ticks, y_ticks = parse_numeric_ticks(lines, 800, 600)
    assert x_ticks == [] and y_ticks == []


# --------------------------------------------------------------------------- fit_axis

def test_fit_axis_linear_recovers_known_slope_intercept():
    # pixel = 10*value + 500 (contrived linear mapping)
    ticks = [(v, 10 * v + 500) for v in [0, 10, 20, 30]]
    result = fit_axis(ticks)
    assert result is not None
    slope, intercept, used, is_log = result
    assert slope == pytest.approx(10.0, abs=0.01)
    assert intercept == pytest.approx(500.0, abs=0.5)
    assert is_log is False
    assert len(used) == 4


def test_fit_axis_log_scale_auto_detected():
    # pixel decreases linearly with log10(value); values span >=1 decade.
    values = [1, 10, 100, 1000]
    ticks = [(v, 600 - 100 * math.log10(v)) for v in values]
    result = fit_axis(ticks)
    assert result is not None
    slope, intercept, used, is_log = result
    assert is_log is True
    assert slope == pytest.approx(-100.0, abs=0.5)


def test_fit_axis_ransac_rejects_single_outlier():
    ticks = [(v, 10 * v + 500) for v in [0, 10, 20, 30]]
    ticks.append((40, 9999))  # wild outlier
    result = fit_axis(ticks)
    assert result is not None
    slope, intercept, used, is_log = result
    assert slope == pytest.approx(10.0, abs=0.5)
    used_vals = {v for v, _ in used}
    assert 40 not in used_vals


def test_fit_axis_insufficient_ticks_returns_none():
    assert fit_axis([(1.0, 100.0)]) is None
    assert fit_axis([]) is None


def test_fit_axis_degenerate_single_value_returns_none():
    # All ticks share the same value -> no slope can be fit.
    ticks = [(5.0, 100.0), (5.0, 200.0), (5.0, 300.0)]
    assert fit_axis(ticks) is None


def test_fit_axis_dedups_near_duplicate_pixel_positions():
    ticks = [(10.0, 100.0), (10.4, 102.0), (20.0, 300.0), (30.0, 500.0)]
    result = fit_axis(ticks)
    assert result is not None
    _, _, used, _ = result
    pixel_positions = [p for _, p in used]
    # The two near-duplicate pixels (100, 102) collapse to one tick.
    assert len(pixel_positions) <= 3


# ----------------------------------------------------------------------- pixel_to_data

def test_pixel_to_data_linear_round_trip():
    cal = {"x_slope": 10.0, "x_intercept": 500.0, "x_log": False,
           "y_slope": -5.0, "y_intercept": 300.0, "y_log": False}
    x, y = pixel_to_data(600.0, 250.0, cal)
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(10.0)


def test_pixel_to_data_log_axis():
    cal = {"x_slope": 1.0, "x_intercept": 0.0, "x_log": False,
           "y_slope": -100.0, "y_intercept": 600.0, "y_log": True}
    # log10(y) = (py - 600) / -100 -> at py=500, log10(y)=1 -> y=10
    x, y = pixel_to_data(50.0, 500.0, cal)
    assert y == pytest.approx(10.0, rel=0.01)


def test_pixel_to_data_clamps_extreme_log_exponent():
    cal = {"x_slope": 1.0, "x_intercept": 0.0, "x_log": False,
           "y_slope": 1.0, "y_intercept": 0.0, "y_log": True}
    x, y = pixel_to_data(0.0, 1000.0, cal)  # log10(y) = 1000 -> clamp to 12
    assert y == pytest.approx(1e12)
    assert math.isfinite(y)


# ------------------------------------------------------------------ data_to_pixel (T19)
# Inverse projection for Stage 6's overlay drawing: engineering units back to
# pixels using Stage 5's STORED calibration. Lives here, next to
# pixel_to_data, so viewer code can never grow its own drifted copy of the
# calibration math (a real legacy bug — see the T19 task spec).

def test_data_to_pixel_round_trips_linear_calibration():
    cal = {"x_slope": 10.0, "x_intercept": 500.0, "x_log": False,
           "y_slope": -5.0, "y_intercept": 300.0, "y_log": False}
    x, y = pixel_to_data(600.0, 250.0, cal)
    px, py = data_to_pixel(x, y, cal)
    assert px == pytest.approx(600.0)
    assert py == pytest.approx(250.0)


def test_data_to_pixel_round_trips_log_calibration():
    cal = {"x_slope": 200.0, "x_intercept": 100.0, "x_log": True,
           "y_slope": -100.0, "y_intercept": 600.0, "y_log": True}
    x, y = pixel_to_data(300.0, 400.0, cal)
    px, py = data_to_pixel(x, y, cal)
    assert px == pytest.approx(300.0)
    assert py == pytest.approx(400.0)


def test_data_to_pixel_log_axis_direct_value():
    cal = {"x_slope": 1.0, "x_intercept": 0.0, "x_log": False,
           "y_slope": -100.0, "y_intercept": 600.0, "y_log": True}
    # y=10 -> log10=1 -> py = -100*1 + 600 = 500
    px, py = data_to_pixel(5.0, 10.0, cal)
    assert px == pytest.approx(5.0)
    assert py == pytest.approx(500.0)


def test_data_to_pixel_nonpositive_value_on_log_axis_returns_none():
    cal = {"x_slope": 1.0, "x_intercept": 0.0, "x_log": False,
           "y_slope": -100.0, "y_intercept": 600.0, "y_log": True}
    assert data_to_pixel(5.0, 0.0, cal) is None
    assert data_to_pixel(5.0, -3.0, cal) is None


# ------------------------------------------------------------------- derive_calibration

def test_derive_calibration_combines_both_axes():
    lines = [
        ocr_line("0", 100, 560, 120, 580),
        ocr_line("10", 300, 560, 320, 580),
        ocr_line("20", 500, 560, 520, 580),
        ocr_line("100", 50, 400, 90, 420),
        ocr_line("200", 50, 200, 90, 220),
    ]
    cal = derive_calibration(lines, 800, 600)
    assert cal is not None
    assert cal["x_log"] is False
    assert "plot_bbox" in cal
    assert set(cal["plot_bbox"]) == {"left", "right", "top", "bottom"}


def test_derive_calibration_returns_none_when_one_axis_fails():
    # Only x-zone ticks provided; y-axis has < min_n ticks -> None overall.
    lines = [
        ocr_line("0", 100, 560, 120, 580),
        ocr_line("10", 300, 560, 320, 580),
    ]
    assert derive_calibration(lines, 800, 600) is None


def test_derive_calibration_returns_none_on_empty_ocr():
    assert derive_calibration([], 800, 600) is None


# ------------------------------------------------- log-axis exponent OCR repair (T16)
# Real failing examples from the T15 sample run: Azure OCR renders log-axis
# superscript labels ("10^3") three ways — "10 3" (one token, space),
# "104"/"102"/"101" (concatenated), "10º" (ordinal char). All from real
# full_extraction.json data, devices named per case.

def test_exponent_space_form_parsed_as_power_of_ten():
    # BSP125H6327XTSA1 y-axis: '10 3' bbox (87,22,129,49) on a 679x784 crop
    lines = [ocr_line("10 3", 87, 22, 129, 49)]
    _, y_ticks = parse_numeric_ticks(lines, 679, 784)
    assert y_ticks == [(1000.0, 35.5)]


def test_exponent_space_form_zero_exponent():
    lines = [ocr_line("10 0", 89, 300, 127, 320)]
    _, y_ticks = parse_numeric_ticks(lines, 679, 784)
    assert y_ticks == [(1.0, 310.0)]


def test_exponent_ordinal_char_form_parsed_as_power_of_ten():
    # BSS138IXTSA1 y-axis: '10º' (ordinal indicator = superscript 0)
    lines = [ocr_line("10º", 78, 742, 115, 765)]
    _, y_ticks = parse_numeric_ticks(lines, 796, 869)
    assert y_ticks == [(1.0, 753.5)]


def test_exponent_unicode_superscript_form():
    lines = [ocr_line("10³", 87, 22, 129, 49)]
    _, y_ticks = parse_numeric_ticks(lines, 679, 784)
    assert y_ticks == [(1000.0, 35.5)]


def test_bare_concatenated_exponents_reinterpreted_when_two_or_more():
    # BSF050N03LQ3G y-axis: '104', '102', '101' = 10^4, 10^2, 10^1
    lines = [
        ocr_line("104", 70, 38, 101, 60),
        ocr_line("102", 68, 396, 100, 421),
        ocr_line("101", 72, 578, 98, 598),
    ]
    _, y_ticks = parse_numeric_ticks(lines, 576, 691)
    values = sorted(v for v, _ in y_ticks)
    assert values == [10.0, 100.0, 10000.0]


def test_single_bare_10x_value_not_reinterpreted():
    # One lone "104" among normal ticks must stay 104 (a real value).
    lines = [
        ocr_line("104", 50, 100, 90, 120),
        ocr_line("50", 50, 300, 90, 320),
        ocr_line("200", 50, 500, 90, 520),
    ]
    _, y_ticks = parse_numeric_ticks(lines, 800, 600)
    values = {v for v, _ in y_ticks}
    assert 104.0 in values


def test_zero_tick_dropped_from_exponent_labeled_axis():
    # BSP88H6327XTSA1 y-axis: exponent labels plus a stray '0' — a 0 cannot
    # be a tick on a log-decade axis and blocks fit_axis's log detection.
    lines = [
        ocr_line("10 3", 86, 22, 128, 48),
        ocr_line("10 2", 87, 237, 128, 265),
        ocr_line("0", 28, 304, 49, 322),
        ocr_line("10 1", 87, 452, 127, 477),
    ]
    _, y_ticks = parse_numeric_ticks(lines, 678, 782)
    values = sorted(v for v, _ in y_ticks)
    assert values == [10.0, 100.0, 1000.0]


def test_genuine_compound_token_still_splits_normally():
    # Regression: the exponent fix must not break real compound tick rows.
    lines = [ocr_line("0 20 40 60", 100, 560, 260, 580)]
    x_ticks, _ = parse_numeric_ticks(lines, 800, 600)
    assert [v for v, _ in x_ticks] == [0.0, 20.0, 40.0, 60.0]


def test_plain_ten_tick_unaffected_by_exponent_handling():
    lines = [ocr_line("10", 245, 607, 268, 624)]
    x_ticks, _ = parse_numeric_ticks(lines, 576, 691)
    assert x_ticks == [(10.0, 256.5)]


def test_real_bsp125_y_axis_calibrates_as_log():
    # End-to-end on the real BSP125H6327XTSA1 label geometry that silently
    # produced a wrong LINEAR fit in T15.
    lines = [
        ocr_line("10 3", 87, 22, 129, 49),
        ocr_line("10 2", 87, 238, 129, 266),
        ocr_line("10 1", 87, 452, 127, 478),
        # x-axis ticks so derive_calibration can fit both axes
        ocr_line("0", 125, 697, 138, 715),
        ocr_line("8", 202, 695, 218, 715),
        ocr_line("16", 278, 695, 302, 715),
        ocr_line("24", 354, 695, 382, 714),
        ocr_line("32", 433, 696, 459, 716),
    ]
    cal = derive_calibration(lines, 679, 784)
    assert cal is not None
    assert cal["y_log"] is True
    # pixel_to_data at the 10^2 label's pixel center should give ~100 pF
    _, y = pixel_to_data(300.0, 252.0, cal)
    assert y == pytest.approx(100.0, rel=0.05)


# ------------------------------------------------- T18: negative exponents,
# bare "100"->10^0, and stray-zero-tick dropping on log-labeled axes.
# Real failing examples from the T17 wider sample: BSS127H6327XTSA2 /
# BSS127IXTSA1 / BTS132E3129NKSA1 use "100"/"10-1"/"10-2" bare-digit decade
# labels (no space, no superscript); AUIRL3705N / 94-3316 / IRFL014NTRPBF
# have a genuinely linear y-axis but a log x-axis (1..100 V) polluted by a
# stray "0" tick that blocks fit_axis's all-positive log detection.

def test_negative_exponent_dash_one_parsed_as_tenth():
    lines = [ocr_line("10-1", 97, 678, 128, 698)]
    _, y_ticks = parse_numeric_ticks(lines, 681, 793)
    assert y_ticks == [(0.1, 688.0)]


def test_negative_exponent_dash_two_parsed_as_hundredth():
    lines = [ocr_line("10-2", 86, 676, 114, 690)]
    _, y_ticks = parse_numeric_ticks(lines, 658, 775)
    assert y_ticks == [(0.01, 683.0)]


def test_bare_100_reinterpreted_as_ten_to_zero_with_supporting_evidence():
    # BSS127H6327XTSA2 y-axis: '102', '101', '100' (bare digits) -> 10^2/10^1/10^0.
    lines = [
        ocr_line("102", 100, 38, 140, 58),
        ocr_line("101", 100, 252, 140, 272),
        ocr_line("100", 100, 464, 140, 484),
    ]
    _, y_ticks = parse_numeric_ticks(lines, 681, 793)
    values = sorted(v for v, _ in y_ticks)
    assert values == [1.0, 10.0, 100.0]


def test_bare_100_and_negative_exponent_together_real_bss127h_case():
    # Full real BSS127H6327XTSA2 y-axis: 102/101/100 bare + 10-1 negative form.
    lines = [
        ocr_line("102", 100, 38, 140, 58),
        ocr_line("101", 100, 252, 140, 272),
        ocr_line("100", 100, 464, 140, 484),
        ocr_line("10-1", 97, 678, 128, 698),
    ]
    _, y_ticks = parse_numeric_ticks(lines, 681, 793)
    values = sorted(v for v, _ in y_ticks)
    assert values == [0.1, 1.0, 10.0, 100.0]


def test_single_bare_100_not_reinterpreted_among_normal_linear_ticks():
    # A lone genuine "100" tick (e.g. 0/50/100/150/200 volt ticks) must stay
    # 100 — reinterpretation requires >=2 corroborating bare-exponent ticks.
    lines = [
        ocr_line("0", 50, 690, 90, 710),
        ocr_line("50", 200, 690, 240, 710),
        ocr_line("100", 350, 690, 390, 710),
        ocr_line("150", 500, 690, 540, 710),
    ]
    x_ticks, _ = parse_numeric_ticks(lines, 800, 720)
    values = {v for v, _ in x_ticks}
    assert 100.0 in values
    assert 1.0 not in values


def test_zero_tick_dropped_when_remaining_x_ticks_span_wide_log_range():
    # AUIRL3705N x-axis: '0','1','10','100' -- the printed axis is log
    # 1..100 V; the '0' tick is OCR noise (or an origin marker) that blocks
    # fit_axis's all-positive log detection.
    lines = [
        ocr_line("0", 106, 558, 125, 578),
        ocr_line("1", 124, 584, 143, 604),
        ocr_line("10", 386, 584, 406, 604),
        ocr_line("100", 646, 583, 666, 603),
    ]
    x_ticks, _ = parse_numeric_ticks(lines, 688, 655)
    values = sorted(v for v, _ in x_ticks)
    assert values == [1.0, 10.0, 100.0]


def test_zero_tick_kept_when_remaining_range_is_narrow_linear():
    # A genuinely linear axis (0..30, ratio 3x) must NOT have its 0 dropped.
    lines = [
        ocr_line("0", 60, 270, 80, 290),
        ocr_line("10", 160, 270, 180, 290),
        ocr_line("20", 260, 270, 280, 290),
        ocr_line("30", 330, 270, 350, 290),
    ]
    x_ticks, _ = parse_numeric_ticks(lines, 400, 300)
    values = sorted(v for v, _ in x_ticks)
    assert values == [0.0, 10.0, 20.0, 30.0]


def test_zero_tick_kept_when_negative_ticks_present():
    # Mixed positive/negative range is never a log axis -- 0 must survive.
    lines = [
        ocr_line("-10", 60, 270, 90, 290),
        ocr_line("0", 160, 270, 180, 290),
        ocr_line("10", 260, 270, 280, 290),
        ocr_line("20", 330, 270, 350, 290),
    ]
    x_ticks, _ = parse_numeric_ticks(lines, 400, 300)
    values = sorted(v for v, _ in x_ticks)
    assert values == [-10.0, 0.0, 10.0, 20.0]


def test_real_auirl3705n_x_axis_calibrates_as_log_after_zero_drop():
    lines = [
        ocr_line("0", 106, 558, 125, 578),
        ocr_line("1", 124, 584, 143, 604),
        ocr_line("10", 386, 584, 406, 604),
        ocr_line("100", 646, 583, 666, 603),
        # y-axis ticks (genuinely linear, unaffected)
        ocr_line("6000", 86, 36, 126, 56),
        ocr_line("5000", 86, 124, 126, 144),
        ocr_line("1000", 86, 470, 126, 490),
    ]
    cal = derive_calibration(lines, 688, 655)
    assert cal is not None
    assert cal["x_log"] is True
    assert cal["y_log"] is False


def test_real_bss127h_y_axis_calibrates_as_log_with_full_decade_range():
    lines = [
        ocr_line("102", 100, 38, 140, 58),
        ocr_line("101", 100, 252, 140, 272),
        ocr_line("100", 100, 464, 140, 484),
        ocr_line("10-1", 97, 678, 128, 698),
        ocr_line("0", 138, 706, 152, 720),
        ocr_line("5", 238, 704, 252, 720),
        ocr_line("10", 340, 706, 354, 720),
        ocr_line("15", 440, 705, 454, 720),
    ]
    cal = derive_calibration(lines, 681, 793)
    assert cal is not None
    assert cal["y_log"] is True


# ------------------------------------------------------- T18: units detection

def test_detects_bare_pf_label():
    # BSP125H6327XTSA1 style: standalone "pF" line in the y-axis zone.
    lines = [ocr_line("pF", 74, 104, 113, 134)]
    assert detect_y_axis_units(lines, 679, 784) == "pF"


def test_detects_uppercase_pf_label():
    lines = [ocr_line("PF", 85, 109, 116, 136)]
    assert detect_y_axis_units(lines, 678, 782) == "pF"


def test_detects_pf_embedded_in_bracket_axis_label():
    # BSF050N03LQ3G / IQD005N04NM6ATMA1 style: "C [pF]".
    lines = [ocr_line("C [pF]", 28, 287, 52, 352)]
    assert detect_y_axis_units(lines, 576, 691) == "pF"


def test_detects_pf_embedded_in_parenthesized_axis_label():
    # AUIRL3705N / 94-3316 / IRFL014NTRPBF style: "C, Capacitance (pF)".
    lines = [ocr_line("C, Capacitance (pF)", 38, 291, 60, 400)]
    assert detect_y_axis_units(lines, 688, 655) == "pF"


def test_detects_bare_nf_label():
    # BTS132E3129NKSA1 / BTS247ZE3062ANTMA1 style: standalone "nF" line.
    lines = [ocr_line("nF", 84, 85, 97, 95)]
    assert detect_y_axis_units(lines, 658, 775) == "nF"


def test_detects_uf_and_micro_sign_variants():
    lines_u = [ocr_line("uF", 84, 85, 100, 105)]
    lines_mu = [ocr_line("µF", 84, 85, 100, 105)]
    assert detect_y_axis_units(lines_u, 658, 775) == "uF"
    assert detect_y_axis_units(lines_mu, 658, 775) == "uF"


def test_returns_none_when_no_unit_text_present():
    lines = [
        ocr_line("1000", 20, 30, 55, 45),
        ocr_line("100", 20, 90, 55, 105),
    ]
    assert detect_y_axis_units(lines, 400, 300) is None


def test_returns_none_when_multiple_conflicting_units_found():
    # Ambiguous -- never guess between two different detected units.
    lines = [
        ocr_line("pF", 74, 104, 113, 134),
        ocr_line("nF", 74, 400, 113, 430),
    ]
    assert detect_y_axis_units(lines, 679, 784) is None


def test_unit_text_outside_y_zone_ignored():
    # A "pF" token sitting in the x-axis / caption zone (right/bottom of the
    # figure) must not count -- only the y-axis label column is scanned.
    lines = [ocr_line("pF", 600, 700, 630, 720)]
    assert detect_y_axis_units(lines, 679, 784) is None


def test_formula_frequency_variable_not_mistaken_for_bare_farad():
    # Real axis annotation "f = 1 MHz" uses lowercase f as a frequency
    # variable, not a Farad unit -- must not produce a false "F" detection.
    lines = [ocr_line("f = 1 MHz", 30, 100, 200, 120)]
    assert detect_y_axis_units(lines, 679, 784) is None


def test_empty_ocr_lines_returns_none():
    assert detect_y_axis_units([], 679, 784) is None


def test_real_bts247_detects_nf_not_pf():
    # The exact T17 finding: BTS247ZE3062ANTMA1's axis is in nF, and a
    # pF-assuming consumer would be silently 1000x off.
    lines = [
        ocr_line("10 1", 87, 22, 129, 49),
        ocr_line("nF", 84, 75, 97, 95),
        ocr_line("C 5", 33, 100, 99, 126),
    ]
    assert detect_y_axis_units(lines, 658, 775) == "nF"


def test_lone_bare_100_with_duplicate_pixel_position_not_reinterpreted():
    # Real AUIRLU3114Z x-axis regression (found while wiring T18 into the
    # T17 re-run): two literal "100" tokens at different pixel positions
    # (one a genuine tick, one an OCR duplicate/outlier) plus "1" and "10".
    # No 101-109 sibling and no explicit exponent notation -- reinterpreting
    # "100" here would corrupt a tick set fit_axis's own RANSAC already
    # handles correctly by rejecting the outlier duplicate.
    lines = [
        ocr_line("100", 83, 476, 123, 496),
        ocr_line("1", 124, 509, 164, 529),
        ocr_line("10", 348, 509, 388, 529),
        ocr_line("100", 572, 509, 612, 529),
    ]
    x_ticks, _ = parse_numeric_ticks(lines, 639, 602)
    values = sorted(v for v, _ in x_ticks)
    assert values == [1.0, 10.0, 100.0, 100.0]
    x_fit = fit_axis(x_ticks)
    assert x_fit is not None
    assert x_fit[3] is True  # is_log -- RANSAC rejects the duplicate outlier
