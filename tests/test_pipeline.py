"""Tests for src.extraction.pipeline — written FIRST (CLAUDE.md §2).

Orchestrates dedup -> curve-count gating -> skeletonize -> naming ->
calibration -> engineering-unit conversion -> schema-validated result, for
capacitance_vs_vds as the reference curve type. process_detections is the
GPU-free orchestration core (detections already computed) so it's fully
unit-testable; run_pipeline is the thin GPU-dependent wrapper that also
calls inference (tested by injecting a fake mmdet.apis, same technique as
test_inference.py).
"""
import sys
import types

import numpy as np
import pytest

from src.extraction.inference import Detection
from src.extraction.pipeline import process_detections, run_pipeline


def ocr_line(text, x1, y1, x2, y2):
    return {"text": text, "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


IMG_W, IMG_H = 400, 300


def band_mask(row, thickness=2, col_start=60, col_end=340):
    mask = np.zeros((IMG_H, IMG_W), dtype=bool)
    mask[row : row + thickness, col_start:col_end] = True
    return mask


def good_ocr_lines():
    # x-axis ticks (bottom zone, cy/h > 0.70): linear 0..30
    x = [
        ocr_line("0", 60, 270, 80, 290),
        ocr_line("10", 160, 270, 180, 290),
        ocr_line("20", 260, 270, 280, 290),
        ocr_line("30", 330, 270, 350, 290),
    ]
    # y-axis ticks (left zone, cx/w < 0.30, not tight-corner): log 1/10/100/1000
    y = [
        ocr_line("1000", 20, 30, 55, 45),
        ocr_line("100", 20, 90, 55, 105),
        ocr_line("10", 20, 150, 55, 165),
        ocr_line("1", 20, 210, 55, 225),
        ocr_line("pF", 20, 240, 45, 255),  # y-axis unit label
    ]
    return x + y


def good_ocr_lines_no_units():
    return [line for line in good_ocr_lines() if line["text"] != "pF"]


def three_good_detections():
    return [
        Detection(score=0.95, mask=band_mask(30)),   # top
        Detection(score=0.9, mask=band_mask(140)),   # middle
        Detection(score=0.85, mask=band_mask(250)),  # bottom
    ]


# ---------------------------------------------------------------- process_detections

def test_exactly_three_detections_produces_ok_result_with_correct_naming():
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines(), three_good_detections(),
    )
    assert result["status"] == "ok"
    assert result["duplicates_removed"] == 0
    names = [c["curve_name"] for c in result["curves"]]
    assert names == ["Ciss", "Coss", "Crss"]  # top -> bottom


def test_ok_result_points_are_finite_and_nonempty():
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines(), three_good_detections(),
    )
    for curve in result["curves"]:
        assert curve["points"]
        for point in curve["points"]:
            assert np.isfinite(point["x"])
            assert np.isfinite(point["y"])


def test_four_detections_with_real_duplicate_dedups_to_three_and_logs():
    detections = three_good_detections()
    dup = Detection(score=0.4, mask=band_mask(31))  # near-duplicate of the top band
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines(), detections + [dup],
    )
    assert result["status"] == "ok"
    assert result["duplicates_removed"] == 1
    assert len(result["curves"]) == 3


def test_four_genuinely_distinct_detections_stays_needs_review():
    detections = three_good_detections()
    distinct = Detection(score=0.6, mask=band_mask(190))  # not a duplicate of anything
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines(), detections + [distinct],
    )
    assert result["status"] == "needs_review"
    assert result["duplicates_removed"] == 0
    assert "4" in result["review_reason"]


def test_fewer_than_three_detections_is_needs_review_never_guessed():
    detections = three_good_detections()[:2]
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines(), detections,
    )
    assert result["status"] == "needs_review"
    assert result["duplicates_removed"] == 0
    assert len(result["curves"]) == 2
    assert result["calibration"] is None


def test_zero_detections_is_needs_review_with_empty_curves():
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H, good_ocr_lines(), [],
    )
    assert result["status"] == "needs_review"
    assert result["curves"] == []


def test_calibration_failure_still_names_curves_but_leaves_points_empty():
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        [],  # no OCR ticks at all -> calibration must fail
        three_good_detections(),
    )
    assert result["status"] == "needs_review"
    assert "calibrat" in result["review_reason"].lower()
    names = {c["curve_name"] for c in result["curves"]}
    assert names == {"Ciss", "Coss", "Crss"}
    for curve in result["curves"]:
        assert curve["points"] == []


def test_degenerate_zero_slope_calibration_treated_as_failure(monkeypatch):
    import src.extraction.pipeline as pipeline_mod

    def fake_derive_calibration(ocr_lines, w, h):
        return {"x_slope": 0.0, "x_intercept": 0.0, "y_slope": -10.0,
                "y_intercept": 100.0, "x_log": False, "y_log": False}

    monkeypatch.setattr(pipeline_mod, "derive_calibration", fake_derive_calibration)
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines(), three_good_detections(),
    )
    assert result["status"] == "needs_review"
    assert "calibrat" in result["review_reason"].lower()


def test_unregistered_curve_type_raises():
    with pytest.raises(KeyError):
        process_detections(
            "DEV1", "not_a_real_curve_type", "fig.png", IMG_W, IMG_H,
            good_ocr_lines(), three_good_detections(),
        )


def test_result_passes_schema_validation_implicitly():
    # build_result calls validate_result internally; if this doesn't raise,
    # the result is schema-valid.
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines(), three_good_detections(),
    )
    assert result["device"] == "DEV1"
    assert result["curve_type"] == "capacitance_vs_vds"
    assert result["source_image"] == "fig.png"


# ------------------------------------------------------- plausibility check
# (T16, revised T18: require_y_log removed — the T17 wider sample found real
# capacitance charts (International Rectifier template) with a genuinely
# linear y-axis; the T16 "y must be log" assumption was itself wrong. The
# range check (a physical-sanity check on the actual traced values, which
# indirectly exercises fit_axis's own log-vs-linear judgment rather than
# second-guessing it) is now the sole plausibility gate.

def test_linear_y_axis_no_longer_auto_rejected_when_plausible(monkeypatch):
    # T18 regression test: a genuinely linear y-axis whose resulting values
    # land in-range must now pass as "ok" -- this was wrongly force-rejected
    # by the T16 require_y_log rule (the exact IR-template finding from T17).
    import src.extraction.pipeline as pipeline_mod

    def fake_derive_calibration(ocr_lines, w, h):
        return {"x_slope": 10.0, "x_intercept": 50.0, "y_slope": -1.0,
                "y_intercept": 400.0, "x_log": False, "y_log": False}

    monkeypatch.setattr(pipeline_mod, "derive_calibration", fake_derive_calibration)
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines(), three_good_detections(),
    )
    assert result["status"] == "ok"
    assert result["calibration"]["y_log"] is False


def test_implausible_y_values_out_of_range_downgrades(monkeypatch):
    # Log y-axis but values wildly outside the plausible capacitance range.
    import src.extraction.pipeline as pipeline_mod

    def fake_derive_calibration(ocr_lines, w, h):
        # log10(y) = (py - 0) / 10 -> at py=30..250, y = 10^3 .. 10^25: insane
        return {"x_slope": 10.0, "x_intercept": 50.0, "y_slope": 10.0,
                "y_intercept": 0.0, "x_log": False, "y_log": True}

    monkeypatch.setattr(pipeline_mod, "derive_calibration", fake_derive_calibration)
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines(), three_good_detections(),
    )
    assert result["status"] == "needs_review"
    assert "implausible_calibration" in result["review_reason"]


def test_implausible_linear_y_values_out_of_range_also_downgrades(monkeypatch):
    # The range check must catch a bad LINEAR fit too, not just log ones --
    # removing require_y_log must not make linear fits unconditionally
    # trusted.
    import src.extraction.pipeline as pipeline_mod

    def fake_derive_calibration(ocr_lines, w, h):
        # y = (py - 0) / 0.01 -> at py=30..250, y = 3000..25000... actually
        # make it clearly negative/huge: slope tiny + intercept far off.
        return {"x_slope": 10.0, "x_intercept": 50.0, "y_slope": 0.001,
                "y_intercept": -10000.0, "x_log": False, "y_log": False}

    monkeypatch.setattr(pipeline_mod, "derive_calibration", fake_derive_calibration)
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines(), three_good_detections(),
    )
    assert result["status"] == "needs_review"
    assert "implausible_calibration" in result["review_reason"]


def test_implausible_result_keeps_curves_and_calibration_for_review(monkeypatch):
    import src.extraction.pipeline as pipeline_mod

    def fake_derive_calibration(ocr_lines, w, h):
        return {"x_slope": 10.0, "x_intercept": 50.0, "y_slope": 10.0,
                "y_intercept": 0.0, "x_log": False, "y_log": True}

    monkeypatch.setattr(pipeline_mod, "derive_calibration", fake_derive_calibration)
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines(), three_good_detections(),
    )
    # Human reviewers need the traced curves + suspect calibration, not an
    # empty shell.
    assert result["status"] == "needs_review"
    names = {c["curve_name"] for c in result["curves"]}
    assert names == {"Ciss", "Coss", "Crss"}
    assert all(c["points"] for c in result["curves"])


def test_plausible_log_calibration_stays_ok():
    # The realistic good_ocr_lines() fixture has a genuine log y-axis
    # (1..1000) -> values land within the plausible capacitance range.
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines(), three_good_detections(),
    )
    assert result["status"] == "ok"


def test_curve_type_without_plausibility_spec_is_unchecked(monkeypatch):
    # A future curve type with a naming fn but no plausibility entry must not
    # crash or be spuriously flagged.
    import src.extraction.naming as naming_mod
    import src.extraction.pipeline as pipeline_mod

    monkeypatch.setitem(naming_mod._NAMING_REGISTRY, "test_type",
                        naming_mod._NAMING_REGISTRY["capacitance_vs_vds"])
    result = process_detections(
        "DEV1", "test_type", "fig.png", IMG_W, IMG_H,
        good_ocr_lines(), three_good_detections(),
    )
    assert result["status"] == "ok"


# ------------------------------------------------------------ units detection (T18)

def test_units_detected_and_populated_on_ok_result():
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines(), three_good_detections(),
    )
    assert result["status"] == "ok"
    assert result["units"] == "pF"


def test_missing_unit_label_downgrades_to_units_undetected():
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines_no_units(), three_good_detections(),
    )
    assert result["status"] == "needs_review"
    assert result["review_reason"] == "units_undetected"
    assert result["units"] is None


def test_units_undetected_result_still_keeps_curves_and_calibration():
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines_no_units(), three_good_detections(),
    )
    names = {c["curve_name"] for c in result["curves"]}
    assert names == {"Ciss", "Coss", "Crss"}
    assert all(c["points"] for c in result["curves"])
    assert result["calibration"] is not None


def test_ambiguous_units_downgrades_to_units_undetected():
    lines = good_ocr_lines_no_units() + [
        ocr_line("pF", 20, 240, 45, 255),
        ocr_line("nF", 20, 260, 45, 275),
    ]
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        lines, three_good_detections(),
    )
    assert result["status"] == "needs_review"
    assert result["review_reason"] == "units_undetected"
    assert result["units"] is None


def test_early_needs_review_results_carry_units_none_not_a_crash():
    # Curve-count gate fires before units detection is ever attempted.
    result = process_detections(
        "DEV1", "capacitance_vs_vds", "fig.png", IMG_W, IMG_H,
        good_ocr_lines(), three_good_detections()[:2],
    )
    assert result["status"] == "needs_review"
    assert result["units"] is None


# --------------------------------------------------------------------- run_pipeline

def test_run_pipeline_calls_inference_then_processes(monkeypatch):
    bbox_result = [[
        np.array([0, 0, 10, 10, 0.95]), np.array([0, 0, 10, 10, 0.9]),
        np.array([0, 0, 10, 10, 0.85]),
    ]]
    segm_result = [[band_mask(30), band_mask(140), band_mask(250)]]

    def fake_inference_detector(model, image_path):
        return bbox_result, segm_result

    fake_apis = types.SimpleNamespace(inference_detector=fake_inference_detector)
    fake_mmdet = types.ModuleType("mmdet")
    fake_mmdet.apis = fake_apis
    monkeypatch.setitem(sys.modules, "mmdet", fake_mmdet)
    monkeypatch.setitem(sys.modules, "mmdet.apis", fake_apis)

    result = run_pipeline(
        "DEV1", "capacitance_vs_vds", "fig.png", good_ocr_lines(), IMG_W, IMG_H,
        model="fake_model",
    )
    assert result["status"] == "ok"
    assert [c["curve_name"] for c in result["curves"]] == ["Ciss", "Coss", "Crss"]
