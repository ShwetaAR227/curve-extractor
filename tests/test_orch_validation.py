"""Tests for src.orchestrator.validation — written FIRST (CLAUDE.md §2).

Final validation is the last cheap safety gate before "finalized". It
REUSES Stage 5's schema validator (never reimplements those checks) and
adds only what schema validation can't know: expected curve names per the
naming registry, and units/calibration presence (both optional at the
Stage-5 level but mandatory for a finalized record).
"""
import pytest

from src.extraction.naming import get_expected_names
from src.orchestrator.validation import validate_final


def make_ok_result(**overrides):
    result = {
        "device": "BSF050N03LQ3G",
        "curve_type": "capacitance_vs_vds",
        "source_image": "fig.png",
        "status": "ok",
        "review_reason": None,
        "duplicates_removed": 0,
        "calibration": {"x_slope": 10.0, "x_intercept": 100.0, "y_slope": -50.0,
                        "y_intercept": 500.0, "x_log": False, "y_log": True},
        "curves": [
            {"curve_name": "Ciss", "confidence": 0.9,
             "points": [{"x": 1.0, "y": 100.0}]},
            {"curve_name": "Coss", "confidence": 0.9,
             "points": [{"x": 1.0, "y": 50.0}]},
            {"curve_name": "Crss", "confidence": 0.9,
             "points": [{"x": 1.0, "y": 10.0}]},
        ],
        "units": "pF",
    }
    result.update(overrides)
    return result


def test_get_expected_names_for_capacitance():
    assert get_expected_names("capacitance_vs_vds") == ["Ciss", "Coss", "Crss"]


def test_get_expected_names_unknown_type_raises():
    with pytest.raises(KeyError):
        get_expected_names("not_a_type")


def test_valid_result_passes():
    assert validate_final(make_ok_result()) is None


def test_missing_curve_name_caught():
    result = make_ok_result()
    result["curves"] = result["curves"][:2]  # Crss missing
    reason = validate_final(result)
    assert reason is not None and "Crss" in reason


def test_unexpected_extra_curve_name_caught():
    result = make_ok_result()
    result["curves"][0]["curve_name"] = "Cxyz"
    reason = validate_final(result)
    assert reason is not None


def test_nan_point_caught_via_schema_validator():
    result = make_ok_result()
    result["curves"][0]["points"][0]["y"] = float("nan")
    reason = validate_final(result)
    assert reason is not None


def test_missing_units_caught():
    reason = validate_final(make_ok_result(units=None))
    assert reason is not None and "units" in reason.lower()


def test_missing_calibration_caught():
    reason = validate_final(make_ok_result(calibration=None))
    assert reason is not None and "calibration" in reason.lower()


def test_empty_curve_type_caught():
    reason = validate_final(make_ok_result(curve_type=""))
    assert reason is not None


def test_validate_final_never_raises_on_garbage():
    assert validate_final({"curve_type": "capacitance_vs_vds"}) is not None
