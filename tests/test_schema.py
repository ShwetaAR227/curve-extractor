"""Tests for src.extraction.schema — written FIRST (CLAUDE.md §2).

Fresh output schema (NOT legacy's flat/keyed cv_curves.json fork —
CLAUDE.md §6). curve_type is ALWAYS present and non-empty — this directly
prevents the legacy "" curve_type overwrite bug. One file per device per
curve type (no merge logic needed at all, so the merge-corruption bug class
is structurally impossible) with an atomic tmp+os.replace write, validated
before any disk write.
"""
import json

import pytest

from src.extraction.schema import build_result, result_path, validate_result, write_result


def make_curve(name="Ciss", confidence=0.9, points=None):
    if points is None:
        points = [{"x": 0.0, "y": 1.0}, {"x": 1.0, "y": 2.0}]
    return {"curve_name": name, "confidence": confidence, "points": points}


def make_calibration():
    return {
        "x_slope": 10.0, "x_intercept": 100.0,
        "y_slope": -50.0, "y_intercept": 500.0,
        "x_log": False, "y_log": True,
    }


def make_ok_result(**overrides):
    result = {
        "device": "BSF050N03LQ3G",
        "curve_type": "capacitance_vs_vds",
        "source_image": "fig_p8_021.png",
        "status": "ok",
        "review_reason": None,
        "duplicates_removed": 0,
        "calibration": make_calibration(),
        "curves": [make_curve("Ciss"), make_curve("Coss"), make_curve("Crss")],
        "units": "pF",
    }
    result.update(overrides)
    return result


# ------------------------------------------------------------------- validate_result

def test_valid_ok_result_passes_validation():
    validate_result(make_ok_result())  # no raise


def test_valid_needs_review_result_passes_validation():
    result = make_ok_result(status="needs_review", review_reason="only 2 detections found",
                            calibration=None, curves=[make_curve("Ciss")])
    validate_result(result)


def test_empty_curve_type_rejected():
    with pytest.raises(ValueError, match="curve_type"):
        validate_result(make_ok_result(curve_type=""))


def test_missing_curve_type_key_rejected():
    result = make_ok_result()
    del result["curve_type"]
    with pytest.raises(ValueError):
        validate_result(result)


def test_invalid_status_rejected():
    with pytest.raises(ValueError, match="status"):
        validate_result(make_ok_result(status="maybe"))


def test_ok_status_with_non_none_review_reason_rejected():
    with pytest.raises(ValueError):
        validate_result(make_ok_result(review_reason="should be none"))


def test_needs_review_status_requires_a_reason():
    with pytest.raises(ValueError):
        validate_result(make_ok_result(status="needs_review", review_reason=None,
                                       curves=[]))


def test_duplicate_curve_names_rejected():
    result = make_ok_result(curves=[make_curve("Ciss"), make_curve("Ciss"), make_curve("Crss")])
    with pytest.raises(ValueError, match="duplicate"):
        validate_result(result)


def test_nan_point_rejected():
    result = make_ok_result(curves=[
        make_curve("Ciss", points=[{"x": float("nan"), "y": 1.0}]),
        make_curve("Coss"), make_curve("Crss"),
    ])
    with pytest.raises(ValueError):
        validate_result(result)


def test_inf_point_rejected():
    result = make_ok_result(curves=[
        make_curve("Ciss", points=[{"x": 1.0, "y": float("inf")}]),
        make_curve("Coss"), make_curve("Crss"),
    ])
    with pytest.raises(ValueError):
        validate_result(result)


def test_confidence_out_of_range_rejected():
    result = make_ok_result(curves=[
        make_curve("Ciss", confidence=1.5), make_curve("Coss"), make_curve("Crss"),
    ])
    with pytest.raises(ValueError):
        validate_result(result)


def test_negative_duplicates_removed_rejected():
    with pytest.raises(ValueError):
        validate_result(make_ok_result(duplicates_removed=-1))


def test_calibration_missing_key_rejected():
    cal = make_calibration()
    del cal["y_log"]
    with pytest.raises(ValueError, match="calibration"):
        validate_result(make_ok_result(calibration=cal))


def test_calibration_non_finite_slope_rejected():
    cal = make_calibration()
    cal["x_slope"] = float("nan")
    with pytest.raises(ValueError):
        validate_result(make_ok_result(calibration=cal))


def test_calibration_non_bool_log_flag_rejected():
    cal = make_calibration()
    cal["x_log"] = "false"
    with pytest.raises(ValueError):
        validate_result(make_ok_result(calibration=cal))


def test_calibration_none_is_allowed_for_needs_review():
    result = make_ok_result(status="needs_review", review_reason="calibration failed",
                            calibration=None, curves=[])
    validate_result(result)  # no raise


# ------------------------------------------------------------- units field (T18)

def test_missing_units_key_rejected():
    result = make_ok_result()
    del result["units"]
    with pytest.raises(ValueError, match="units"):
        validate_result(result)


def test_units_non_string_rejected():
    with pytest.raises(ValueError, match="units"):
        validate_result(make_ok_result(units=123))


def test_units_none_is_allowed():
    result = make_ok_result(status="needs_review", review_reason="units_undetected",
                            units=None)
    validate_result(result)  # no raise


def test_units_undetected_reason_requires_units_none():
    with pytest.raises(ValueError, match="units"):
        validate_result(make_ok_result(
            status="needs_review", review_reason="units_undetected", units="pF",
        ))


def test_units_valid_string_passes():
    validate_result(make_ok_result(units="nF"))  # no raise


# ---------------------------------------------------------------------- build_result

def test_build_result_produces_valid_dict():
    result = build_result(
        device="BSF050N03LQ3G", curve_type="capacitance_vs_vds",
        source_image="fig_p8_021.png", status="ok", review_reason=None,
        duplicates_removed=0, calibration=make_calibration(),
        curves=[make_curve("Ciss"), make_curve("Coss"), make_curve("Crss")],
        units="pF",
    )
    validate_result(result)  # no raise
    assert result["curve_type"] == "capacitance_vs_vds"
    assert result["units"] == "pF"


def test_build_result_rejects_empty_curve_type():
    with pytest.raises(ValueError):
        build_result(device="d", curve_type="", source_image="i.png", status="ok",
                     review_reason=None, duplicates_removed=0, calibration=None, curves=[],
                     units=None)


# --------------------------------------------------------------------------- result_path

def test_result_path_rejects_empty_curve_type(tmp_path):
    with pytest.raises(ValueError):
        result_path(tmp_path, "device", "")


def test_result_path_is_one_file_per_device_per_curve_type(tmp_path):
    p1 = result_path(tmp_path, "DEV1", "capacitance_vs_vds")
    p2 = result_path(tmp_path, "DEV1", "id_vs_vgs")
    assert p1 != p2
    assert "DEV1" in str(p1)


# ----------------------------------------------------------------------------- write_result

def test_write_result_round_trips(tmp_path):
    result = make_ok_result()
    path = result_path(tmp_path, result["device"], result["curve_type"])
    write_result(result, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["curve_type"] == "capacitance_vs_vds"
    assert loaded["curves"][0]["curve_name"] == "Ciss"


def test_write_result_is_atomic_no_tmp_file_left(tmp_path):
    result = make_ok_result()
    path = result_path(tmp_path, result["device"], result["curve_type"])
    write_result(result, path)
    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert tmp_files == []


def test_write_result_raises_before_touching_disk_when_invalid(tmp_path):
    result = make_ok_result(curve_type="")
    path = tmp_path / "device" / "capacitance_vs_vds.json"
    with pytest.raises(ValueError):
        write_result(result, path)
    assert not path.exists()
    assert not path.parent.exists()


def test_write_result_refuses_to_overwrite_different_curve_type_at_same_path(tmp_path):
    result_a = make_ok_result(curve_type="capacitance_vs_vds")
    path = tmp_path / "device" / "shared.json"
    write_result(result_a, path)

    result_b = make_ok_result(curve_type="id_vs_vgs")
    with pytest.raises(ValueError, match="curve_type"):
        write_result(result_b, path)


def test_write_result_allows_rewriting_same_curve_type_at_same_path(tmp_path):
    result = make_ok_result()
    path = result_path(tmp_path, result["device"], result["curve_type"])
    write_result(result, path)
    result["duplicates_removed"] = 1
    write_result(result, path)  # no raise, same curve_type
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["duplicates_removed"] == 1
