"""Tests for rdson_vs_tj two-curve (typ/max) support — written FIRST (CLAUDE.md §2).

Owner decision (2026-07-13): Infineon "Diagram" rdson charts plot TWO curves
("max"/"typ", or "98 %"/"typ" on the older BSB template) as identical black
solid lines, distinguished ONLY by a floating text label near each curve and
by vertical position (upper = max/98 %, lower = typ — consistent across all
observed charts). The pipeline must:

- accept 1 OR 2 curves for rdson_vs_tj (anything else quarantines, never guessed);
- name the 2-curve case from the nearby OCR labels ("max"/"98 %" -> rdson_max,
  "typ" -> rdson_typ), anchored by proximity to each curve;
- fall back to top/bottom position (top = rdson_max) only when no label is found;
- keep the 1-curve IR case named "rdson" exactly as before.

Label geometry below is REAL OCR data from the four devices where the
two-curve template was found (D:/Extractor/data/OCR1-OCR13, T25 survey):
BSC010N04LSATMA1 fig_p9_020, BSC012N06NSATMA1 fig_p8_021,
BSB028N06NN3GXUMA2 fig_p9_021, BSB053N03LPG fig_p6_012.
"""
import numpy as np
import pytest

from src.extraction.naming import get_naming_fn
from src.extraction.naming.rdson_vs_tj import name_curves, name_curves_by_labels
from src.extraction.schema import validate_result

from tests.test_classical import (
    CURVE_BGR, IMG_H, IMG_W, blank_chart, curve_row, draw_axes, draw_curve,
    draw_gridlines, good_ocr_lines, ocr_line, run_standard, standard_chart,
)

# ---------------------------------------------------------------- helpers

def trace(rows_by_col):
    """[(col_start, col_end, row_fn)] -> ordered (row, col) point list."""
    return [(float(row), float(col)) for col, row in rows_by_col]


def line_trace(col0, col1, row0, row1, step=4):
    """Straight-line pixel trace from (col0,row0) to (col1,row1)."""
    pts = []
    for col in range(col0, col1, step):
        frac = (col - col0) / (col1 - col0)
        pts.append((row0 + frac * (row1 - row0), float(col)))
    return pts


def raw_ocr(text, x1, y1, x2, y2):
    return {"text": text, "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


# ------------------------------------------------- positional naming (registry fn)

class TestPositionalNaming:
    def test_one_curve_still_named_rdson(self):
        assert name_curves([line_trace(60, 350, 200, 70)]) == ["rdson"]

    def test_two_curves_named_max_top_typ_bottom(self):
        upper = line_trace(100, 660, 440, 60)
        lower = line_trace(100, 660, 470, 170)
        assert name_curves([upper, lower]) == ["rdson_max", "rdson_typ"]

    def test_two_curve_names_align_to_input_order(self):
        upper = line_trace(100, 660, 440, 60)
        lower = line_trace(100, 660, 470, 170)
        assert name_curves([lower, upper]) == ["rdson_typ", "rdson_max"]

    def test_three_curves_raise(self):
        pts = line_trace(60, 350, 200, 70)
        with pytest.raises(ValueError):
            name_curves([pts, pts, pts])

    def test_zero_curves_raise(self):
        with pytest.raises(ValueError):
            name_curves([])

    def test_empty_curve_raises(self):
        with pytest.raises(ValueError):
            name_curves([line_trace(60, 350, 200, 70), []])

    def test_registry_still_serves_rdson_vs_tj(self):
        assert get_naming_fn("rdson_vs_tj") is name_curves


# ------------------------------------------- label-anchored naming (real OCR data)

# Approximate pixel traces of the two curves in each real chart, consistent
# with the rendered images; label bboxes are VERBATIM from Stage-3 OCR.
REAL_CASES = [
    # (device, upper trace, lower trace, max-ish label, typ label)
    ("BSC010N04LSATMA1",
     line_trace(100, 690, 440, 55), line_trace(100, 690, 470, 165),
     raw_ocr("max", 300, 315, 342, 330), raw_ocr("typ", 431, 362, 465, 382)),
    ("BSC012N06NSATMA1",
     line_trace(80, 720, 530, 110), line_trace(80, 720, 590, 270),
     raw_ocr("max", 427, 244, 471, 261), raw_ocr("typ", 475, 447, 506, 468)),
    ("BSB028N06NN3GXUMA2",
     line_trace(100, 660, 450, 210), line_trace(100, 660, 500, 290),
     raw_ocr("max", 348, 309, 393, 324), raw_ocr("typ", 454, 390, 488, 408)),
    ("BSB053N03LPG",
     line_trace(90, 560, 420, 140), line_trace(90, 560, 460, 210),
     raw_ocr("98 %", 298, 239, 334, 254), raw_ocr("typ", 404, 334, 425, 350)),
]


class TestLabelAnchoredNaming:
    @pytest.mark.parametrize(
        "device,upper,lower,max_label,typ_label", REAL_CASES,
        ids=[c[0] for c in REAL_CASES])
    def test_real_device_labels_name_upper_max_lower_typ(
            self, device, upper, lower, max_label, typ_label):
        names = name_curves_by_labels([upper, lower], [max_label, typ_label])
        assert names == ["rdson_max", "rdson_typ"]

    def test_names_follow_labels_and_align_to_input_order(self):
        _, upper, lower, max_label, typ_label = REAL_CASES[0]
        assert name_curves_by_labels([lower, upper], [max_label, typ_label]) \
            == ["rdson_typ", "rdson_max"]

    def test_labels_win_over_position_when_they_disagree(self):
        # Owner rule: labels take precedence; position is only the fallback.
        upper = line_trace(100, 690, 440, 55)
        lower = line_trace(100, 690, 470, 165)
        # Deliberately swapped: "typ" near the UPPER curve, "max" near the lower.
        typ_near_upper = raw_ocr("typ", 300, 240, 342, 258)
        max_near_lower = raw_ocr("max", 431, 400, 465, 420)
        assert name_curves_by_labels([upper, lower], [typ_near_upper, max_near_lower]) \
            == ["rdson_typ", "rdson_max"]

    def test_case_and_spacing_variants_are_recognized(self):
        _, upper, lower, _, _ = REAL_CASES[0]
        labels = [raw_ocr("Max", 300, 315, 342, 330), raw_ocr("TYP.", 431, 362, 465, 382)]
        assert name_curves_by_labels([upper, lower], labels) == ["rdson_max", "rdson_typ"]
        labels = [raw_ocr("98%", 300, 315, 342, 330), raw_ocr("typ", 431, 362, 465, 382)]
        assert name_curves_by_labels([upper, lower], labels) == ["rdson_max", "rdson_typ"]

    def test_single_label_resolves_and_other_curve_gets_remaining_name(self):
        _, upper, lower, _, typ_label = REAL_CASES[0]
        assert name_curves_by_labels([upper, lower], [typ_label]) \
            == ["rdson_max", "rdson_typ"]

    def test_no_labels_returns_none_for_positional_fallback(self):
        _, upper, lower, _, _ = REAL_CASES[0]
        lines = [raw_ocr("RDS(on)[m2]", 48, 276, 80, 424), raw_ocr("2.00", 60, 50, 95, 66)]
        assert name_curves_by_labels([upper, lower], lines) is None

    def test_both_labels_nearest_same_curve_is_ambiguous_returns_none(self):
        _, upper, lower, _, _ = REAL_CASES[0]
        # Both labels sit right on the upper curve -> ambiguous, never guessed.
        labels = [raw_ocr("max", 300, 315, 342, 330), raw_ocr("typ", 310, 300, 350, 318)]
        assert name_curves_by_labels([upper, lower], labels) is None

    def test_unknown_floating_text_is_ignored(self):
        _, upper, lower, max_label, typ_label = REAL_CASES[0]
        lines = [max_label, typ_label, raw_ocr("min", 500, 500, 540, 520)]
        assert name_curves_by_labels([upper, lower], lines) == ["rdson_max", "rdson_typ"]


# ------------------------------------ end-to-end classical pipeline (2-curve case)

def two_curve_chart():
    """Standard synthetic chart plus a second curve 35 px below the first.

    35 px (not more) keeps the lower curve's cold end ABOVE the x-axis /
    0-mOhm tick row (240): the original +60 offset drew it down to row 260,
    i.e. NEGATIVE resistance — physically impossible values that the
    owner-approved rdson y-range gate now correctly rejects. The fixture was
    the unphysical part; the assertions are unchanged.
    """
    img = standard_chart()
    for col in range(60, 350):
        row = curve_row(col) + 35
        img[row:row + 3, col] = CURVE_BGR
    return img


def two_curve_ocr_lines():
    # "max" floats near the upper curve (col ~200, row ~137), "typ" near the
    # lower (col ~200, row ~172) — same geometry as the real Infineon charts.
    return good_ocr_lines() + [
        ocr_line("max", 180, 110, 215, 125),
        ocr_line("typ", 180, 180, 215, 195),
    ]


class TestClassicalPipelineTwoCurves:
    def test_two_labeled_curves_produce_ok_result_named_typ_and_max(self):
        result = run_standard(two_curve_chart(), ocr_lines=two_curve_ocr_lines())
        validate_result(result)
        assert result["status"] == "ok"
        names = {c["curve_name"] for c in result["curves"]}
        assert names == {"rdson_max", "rdson_typ"}
        # The upper curve (smaller rows -> larger engineering y) must be max.
        by_name = {c["curve_name"]: c for c in result["curves"]}
        max_mean_y = np.mean([p["y"] for p in by_name["rdson_max"]["points"]])
        typ_mean_y = np.mean([p["y"] for p in by_name["rdson_typ"]["points"]])
        assert max_mean_y > typ_mean_y

    def test_two_unlabeled_curves_fall_back_to_position(self):
        result = run_standard(two_curve_chart())  # no max/typ OCR labels
        assert result["status"] == "ok"
        assert {c["curve_name"] for c in result["curves"]} == {"rdson_max", "rdson_typ"}

    def test_single_curve_ir_chart_still_named_rdson(self):
        result = run_standard(standard_chart())
        assert result["status"] == "ok"
        assert [c["curve_name"] for c in result["curves"]] == ["rdson"]

    def test_three_curves_still_quarantined_never_guessed(self):
        img = two_curve_chart()
        for col in range(60, 350):  # a third distinct curve, blue, further down
            # +70: well clear of the lower red curve at +35 so the three stay
            # three components (quarantine fires at the count gate, before
            # calibration, so this one running below the axis is irrelevant).
            row = curve_row(col) + 70
            img[row:row + 3, col] = (220, 40, 40)
        result = run_standard(img)
        validate_result(result)
        assert result["status"] == "needs_review"

    def test_zero_curves_still_quarantined(self):
        img = blank_chart()
        draw_axes(img)
        draw_gridlines(img)
        result = run_standard(img)
        assert result["status"] == "needs_review"
