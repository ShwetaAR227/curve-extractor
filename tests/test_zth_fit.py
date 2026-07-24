"""Tests for src.extraction.zth_fit — written FIRST (CLAUDE.md §2, red
phase). RED PHASE ONLY — the module under test does not exist yet.

Pulled out of classical_zth.py (2026-07-23, owner-approved, mirrors the
curve_detection.py precedent — see that module's own docstring for the
original extraction this one repeats the shape of): the "we have the
curve's pixel points -> convert to engineering units -> sanity-check the
shape -> cross-check against a printed Rth value -> fit the Foster physics
model -> build the final result" recipe wasn't its own reusable piece
before this — it was inline inside classical_zth.py's
``run_classical_pipeline``, mixed with the OLD rule-based
clustering/picker logic that stays there, untouched. This is a PURE
EXTRACTION: same math, same gates, same thresholds, same messages,
byte-for-byte — proven by classical_zth.py's own 67-test suite passing
UNCHANGED, same count, before and after the refactor that wires this
module in (see PROGRESS.md for that before/after re-run).

``fit_foster`` and ``pick_rth_constraint`` are deliberately NOT owned by
this module — they stay defined in classical_zth.py (its existing tests
monkeypatch them directly by name: ``monkeypatch.setattr(classical_zth,
"fit_foster", ...)``, etc. — moving them would silently break that) and
are passed into :func:`fit_and_validate_curve` as plain parameters
instead. Every test below that needs them passes a FAKE directly, so this
module's own suite has zero import-time dependency on classical_zth.py.

No GPU, no network — every fixture is a synthetic numpy array or a plain
dict; the Rth-table-file tests use tmp_path (same convention as
test_classical_zth.py's own TestRthTableFileRead, which this mirrors).
"""
import json

import numpy as np
import pytest

from src.extraction.zth_fit import (
    CURVE_NAME,
    STAGE3_ROOT_ENV_VAR,
    UNITS,
    build_needs_review_result,
    calibration_with_bonus_fields,
    clamp_confidence,
    fit_and_validate_curve,
    pixel_to_data,
    read_full_extraction_for_rth,
)
from src.extraction.schema import validate_result

LOG_CAL = {
    "x_scale": "log", "x_slope": 100.0, "x_intercept": 0.0,
    "y_scale": "log", "y_slope": -100.0, "y_intercept": 1000.0,
    "x_log": True, "y_log": True,
}


def _to_px(cal, x, y):
    px = cal["x_slope"] * np.log10(x) + cal["x_intercept"]
    py = cal["y_slope"] * np.log10(y) + cal["y_intercept"]
    return (px, py)


def _rising_foster_points(cal=LOG_CAL, r=2.0, tau=1e-3, n=30):
    """A realistic rising Foster-shape trace (asymptote ~r), in pixel space
    under ``cal`` — the "everything is normal" fixture most tests below
    reuse, varying only the injected fit_foster/pick_rth_constraint."""
    ts = np.logspace(-6, -1.5, n)
    zs = np.clip(r * (1.0 - np.exp(-ts / tau)), 1e-4, None)
    return [_to_px(cal, t, z) for t, z in zip(ts, zs)]


def _falling_points(cal=LOG_CAL, n=10):
    """A physically-implausible (falling) trace, for the calibration_disaster
    gate — rise_ratio well under 1.0."""
    xs = np.logspace(-6, -1, n)
    ys = np.logspace(1, -2, n)
    return [_to_px(cal, x, y) for x, y in zip(xs, ys)]


def _fake(return_value):
    """A callable that records every call and always returns return_value."""
    calls = []

    def _fn(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return return_value

    _fn.calls = calls
    return _fn


# ------------------------------------------------------------- pixel_to_data

class TestPixelToData:
    def test_linear_axis_round_trip(self):
        cal = {"x_scale": "linear", "x_slope": 10.0, "x_intercept": 0.0,
               "y_scale": "linear", "y_slope": -10.0, "y_intercept": 100.0}
        x, y = pixel_to_data(50.0, 50.0, cal)
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(5.0)

    def test_log_axis_conversion(self):
        x, y = pixel_to_data(200.0, 800.0, LOG_CAL)
        assert x == pytest.approx(100.0)  # log10(x)=(200-0)/100=2 -> x=100
        assert y == pytest.approx(100.0)  # log10(y)=(800-1000)/-100=2 -> y=100

    def test_log_axis_extreme_pixel_clamped_not_inf(self):
        # Wildly extrapolated pixel -> clamped to +/-12 decades, never inf/nan.
        x, _ = pixel_to_data(1e9, 0.0, LOG_CAL)
        assert np.isfinite(x)


# ------------------------------------------------------------ clamp_confidence

class TestClampConfidence:
    def test_none_clamps_to_zero(self):
        assert clamp_confidence(None) == 0.0

    def test_negative_clamps_to_zero(self):
        assert clamp_confidence(-0.7) == 0.0

    def test_above_one_clamps_to_one(self):
        assert clamp_confidence(1.5) == 1.0

    def test_normal_value_passes_through(self):
        assert clamp_confidence(0.83) == pytest.approx(0.83)


# ------------------------------------------------------- build_needs_review_result

class TestBuildNeedsReviewResult:
    def test_basic_shape(self):
        result = build_needs_review_result("DEV1", "zth_vs_time", "fig.png", "some reason")
        validate_result(result)
        assert result["status"] == "needs_review"
        assert result["review_reason"] == "some reason"
        assert result["curves"][0]["confidence"] == 0.0
        assert result["curves"][0]["curve_name"] == CURVE_NAME

    def test_keeps_calibration_when_given(self):
        cal = dict(LOG_CAL, plot_bbox={"left": 0, "right": 1, "top": 0, "bottom": 1})
        result = build_needs_review_result("DEV1", "zth_vs_time", "fig.png", "reason", calibration=cal)
        assert result["calibration"] == cal

    def test_keeps_points_when_given(self):
        pts = [{"x": 1.0, "y": 2.0}]
        result = build_needs_review_result("DEV1", "zth_vs_time", "fig.png", "reason", points=pts)
        assert result["curves"][0]["points"] == pts


# --------------------------------------------------- read_full_extraction_for_rth

class TestReadFullExtractionForRth:
    def test_valid_file_found_via_stage3_root(self, tmp_path):
        device_dir = tmp_path / "DEV1"
        device_dir.mkdir()
        (device_dir / "full_extraction.json").write_text(json.dumps({"tables": []}), encoding="utf-8")
        result = read_full_extraction_for_rth("DEV1", str(tmp_path))
        assert result == {"tables": []}

    def test_missing_root_returns_none_not_a_crash(self):
        assert read_full_extraction_for_rth("DEV1", None) is None

    def test_missing_device_folder_returns_none(self, tmp_path):
        assert read_full_extraction_for_rth("DEV1", str(tmp_path)) is None

    def test_malformed_json_returns_none_not_a_crash(self, tmp_path):
        device_dir = tmp_path / "DEV1"
        device_dir.mkdir()
        (device_dir / "full_extraction.json").write_text("{not valid json", encoding="utf-8")
        assert read_full_extraction_for_rth("DEV1", str(tmp_path)) is None

    def test_env_var_name_matches_module_constant(self):
        assert STAGE3_ROOT_ENV_VAR == "LINEFORMER_STAGE3_ROOT"


# ------------------------------------------------------- calibration_with_bonus_fields

class TestCalibrationWithBonusFields:
    def test_adds_x_log_y_log_from_scale_strings(self):
        cal_zth = {"x_scale": "log", "y_scale": "linear", "x_slope": 1.0}
        out = calibration_with_bonus_fields(cal_zth)
        assert out["x_log"] is True
        assert out["y_log"] is False
        assert out["x_slope"] == 1.0  # original fields preserved


# ------------------------------------------------------------ fit_and_validate_curve

class TestFitAndValidateCurve:
    def _cal(self):
        return dict(LOG_CAL)

    def test_too_few_points_needs_review(self):
        pts = _rising_foster_points()[:3]  # < 6
        result = fit_and_validate_curve(
            "DEV1", "zth_vs_time", "fig.png", pts, self._cal(), None,
            fit_foster=_fake((None, None)), pick_rth_constraint=_fake((None, None)),
        )
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "too_few_points" in result["review_reason"]

    def test_calibration_disaster_needs_review(self):
        pts = _falling_points()
        result = fit_and_validate_curve(
            "DEV1", "zth_vs_time", "fig.png", pts, self._cal(), None,
            fit_foster=_fake((None, None)), pick_rth_constraint=_fake((None, None)),
        )
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "calibration_disaster" in result["review_reason"]

    def test_skip_fit_calibration_broken_needs_review(self):
        # Curve asymptotes near 2.0; a wildly mismatched rth_constraint=50.0
        # gives scale_ratio ~0.04, outside [0.05, 3.0] -> skip_fit.
        pts = _rising_foster_points(r=2.0)
        result = fit_and_validate_curve(
            "DEV1", "zth_vs_time", "fig.png", pts, self._cal(), None,
            fit_foster=_fake((None, None)), pick_rth_constraint=_fake((50.0, "table_max")),
        )
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "calibration_broken" in result["review_reason"]

    def test_foster_fit_failure_needs_review_keeps_points(self):
        pts = _rising_foster_points(r=2.0)
        result = fit_and_validate_curve(
            "DEV1", "zth_vs_time", "fig.png", pts, self._cal(), None,
            fit_foster=_fake((None, None)), pick_rth_constraint=_fake((None, None)),
        )
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "foster_fit_failed" in result["review_reason"]
        assert len(result["curves"][0]["points"]) == len(pts)

    def test_low_r_squared_needs_review(self):
        pts = _rising_foster_points(r=2.0)
        result = fit_and_validate_curve(
            "DEV1", "zth_vs_time", "fig.png", pts, self._cal(), None,
            fit_foster=_fake(({"r1": 1.0, "tau1": 1e-3}, 0.2)),
            pick_rth_constraint=_fake((None, None)),
        )
        assert result["status"] == "needs_review"
        assert "foster_fit_failed" in result["review_reason"]

    def test_unconstrained_success_is_ok(self):
        pts = _rising_foster_points(r=2.0)
        result = fit_and_validate_curve(
            "DEV1", "zth_vs_time", "fig.png", pts, self._cal(), None,
            fit_foster=_fake(({"r1": 2.0, "tau1": 1e-3}, 0.95)),
            pick_rth_constraint=_fake((None, None)),
        )
        validate_result(result)
        assert result["status"] == "ok"
        curve = result["curves"][0]
        assert curve["confidence"] == pytest.approx(0.95)
        assert curve["extraction_source"] == "curve_fit_v3"
        assert curve["r_fixed_at_rth_jc"] is False
        assert curve["rth_jc_steady_state_source"] == "foster_unconstrained"
        assert curve["rth_jc_steady_state"] == pytest.approx(2.0)
        assert curve["curve_name"] == CURVE_NAME
        assert result["units"] == UNITS

    def test_constrained_success_is_ok_and_passes_constraint_through(self):
        pts = _rising_foster_points(r=2.0)
        fake_fit = _fake(({"r1": 2.0, "tau1": 1e-3}, 0.97))
        result = fit_and_validate_curve(
            "DEV1", "zth_vs_time", "fig.png", pts, self._cal(), None,
            fit_foster=fake_fit, pick_rth_constraint=_fake((2.0, "table_max")),
        )
        validate_result(result)
        assert result["status"] == "ok"
        curve = result["curves"][0]
        assert curve["extraction_source"] == "curve_fit_v3_constrained"
        assert curve["r_fixed_at_rth_jc"] is True
        assert curve["rth_jc_steady_state_source"] == "table_max"
        assert curve["rth_jc_steady_state"] == pytest.approx(2.0)
        # The constraint must actually reach fit_foster, not get lost:
        assert fake_fit.calls[0]["kwargs"]["rth_constraint"] == pytest.approx(2.0)

    def test_injected_fit_foster_is_the_one_actually_called(self):
        # Proves fit_and_validate_curve calls the INJECTED function, not
        # some import-time-bound copy of its own.
        pts = _rising_foster_points(r=2.0)
        fake_fit = _fake(({"r1": 2.0, "tau1": 1e-3}, 0.9))
        fit_and_validate_curve(
            "DEV1", "zth_vs_time", "fig.png", pts, self._cal(), None,
            fit_foster=fake_fit, pick_rth_constraint=_fake((None, None)),
        )
        assert len(fake_fit.calls) == 1

    def test_injected_pick_rth_constraint_is_the_one_actually_called(self):
        pts = _rising_foster_points(r=2.0)
        fake_pick = _fake((None, None))
        fit_and_validate_curve(
            "DEV1", "zth_vs_time", "fig.png", pts, self._cal(), None,
            fit_foster=_fake(({"r1": 2.0, "tau1": 1e-3}, 0.9)),
            pick_rth_constraint=fake_pick,
        )
        assert len(fake_pick.calls) == 1

    def test_real_fit_foster_and_pick_rth_constraint_end_to_end(self):
        # Integration proof using the REAL functions (imported directly from
        # classical_zth.py, not reimplemented here) -- confirms the whole
        # recipe genuinely works, not just against fakes.
        from src.extraction.classical_zth import fit_foster as real_fit_foster
        from src.extraction.classical_zth import pick_rth_constraint as real_pick_rth_constraint

        pts = _rising_foster_points(r=2.0, tau=1e-3)
        result = fit_and_validate_curve(
            "DEV1", "zth_vs_time", "fig.png", pts, self._cal(), None,
            fit_foster=real_fit_foster, pick_rth_constraint=real_pick_rth_constraint,
        )
        validate_result(result)
        assert result["status"] == "ok"
        assert result["curves"][0]["fitted_params"]["r1"] == pytest.approx(2.0, rel=0.2)
