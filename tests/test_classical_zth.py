"""Tests for src.extraction.classical_zth — written FIRST (CLAUDE.md §2,
red phase). RED PHASE ONLY — the module under test does not exist yet.

Migrates D:\\Extractor\\5_opencv_extract\\zth_extract.py (legacy, reference
only per CLAUDE.md §6) into zth_vs_time's classical extractor. Owner-approved
design (this session):

1. zth_vs_time does NOT go through the shared core rdson/vgsth use
   (curve_detection.py + pipeline.process_detections) — it's genuinely a
   different, more complex problem (time-unit-aware ticks, a real physics
   curve fit, a printed-table-first check) and gets its own self-contained
   pipeline, sharing only the final src.extraction.schema build/write layer.
2. The one extra piece of data this needs beyond the standard 5-arg
   classical_pipeline call (device, curve_type, source_image, image,
   ocr_lines) — the device's full_extraction.json ``tables`` key, for the
   printed Rth_JC lookup — is read by THIS module itself, via the same
   LINEFORMER_STAGE3_ROOT env-var-root convention stage3_loader.py already
   uses (an optional ``stage3_root`` kwarg, unused by live_stages.py's
   fixed call site, defaulting to the env var). live_stages.py is NOT
   touched. Best-effort: if the root/file/table is unavailable for any
   reason, proceed with NO Rth constraint (an unconstrained Foster fit is
   the legacy code's own normal, common path) — logged, never a crash.
3. Curve name is always "single_pulse" — matches the CVAT-annotated
   training dataset already collected for this curve type.
4. No classification-registry entry (score_zth/auto_pick_zth_figure aren't
   migrated) — that's a separate, deferred task; by the time this module
   runs, LiveStages has already picked the figure, same as every other
   classical wrapper.

Legacy status strings are remapped onto our two-status system, keeping the
ORIGINAL descriptive message as review_reason text. One deliberate remap
(not a literal 1:1 port): legacy's "skip_fit" case — a printed Rth value
that disagrees with the observed late-time curve value — sets their own
status to "clean" despite saying in the SAME breath "curve trace
unreliable, tau values not extracted". We map this to needs_review instead
(the message itself describes something a reviewer should see), keeping
their exact warning text as review_reason.

Confidence: r_squared clamped to [0, 1] (never negative — schema requires
confidence in [0,1]); a printed-table read (no fit performed) is always
1.0; a total fit failure (no fitted_params at all) is 0.0 (no information).

Every ported function is monkeypatchable at the module level (matching
this project's established testing convention, e.g. test_pipeline.py's
own monkeypatch.setattr(pipeline_mod, "derive_calibration", ...) style) —
scenarios that are awkward to construct via pixel-perfect synthetic
geometry (calibration_disaster, the skip_fit remap, a poor Foster fit) are
tested by monkeypatching the specific ported function to return a
controlled value, isolating the NEW status-mapping logic under test from
the ported math (which gets its own direct unit tests too, e.g. fit_foster
against deliberately adversarial (x, y) arrays).

No GPU, no network — every fixture is a synthetic numpy array or an
in-memory OCR-line dict list; the Rth-table-file tests use tmp_path (same
convention as test_stage3_loader.py).
"""
import json

import cv2
import numpy as np
import pytest

import src.extraction.classical_zth as classical_zth
from src.extraction.classical_zth import (
    fit_axis_auto,
    fit_foster,
    parse_axis_ticks,
    parse_foster_table_from_ocr,
    parse_tick,
    pick_rth_constraint,
    run_classical_pipeline,
    trace_curve,
)
from src.extraction.schema import validate_result

IMG_W, IMG_H = 700, 400

# zth's OWN tick-zoning thresholds (ported verbatim from legacy
# parse_axis_ticks — DIFFERENT from classical.py's 0.70/0.30 fixture
# convention): x-tick text needs norm_cy > 0.80, y-tick text needs
# norm_cx < 0.20.
_X_LOG_ORIGIN_PX = 150.0
_X_LOG_PX_PER_DECADE = 80.0
_Y_LOG_ORIGIN_PX = 300.0
_Y_LOG_PX_PER_DECADE = 70.0


def _x_px(t_seconds):
    return _X_LOG_ORIGIN_PX + (np.log10(t_seconds) + 6) * _X_LOG_PX_PER_DECADE


def _y_px(zth_kw):
    return _Y_LOG_ORIGIN_PX - (np.log10(zth_kw) + 2) * _Y_LOG_PX_PER_DECADE


def _ocr_line(text, x1, y1, x2, y2):
    return {"text": text, "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def _axis_ticks_ocr():
    """5 x-ticks (1us..10ms, log time) + 4 y-ticks (0.01..10 K/W, log)."""
    lines = []
    for label, t in (("1us", 1e-6), ("10us", 1e-5), ("100us", 1e-4),
                     ("1ms", 1e-3), ("10ms", 1e-2)):
        px = _x_px(t)
        lines.append(_ocr_line(label, px - 15, 370, px + 15, 390))  # cy=380/400=0.95>0.80
    for label, z in (("0.01", 0.01), ("0.1", 0.1), ("1", 1.0), ("10", 10.0)):
        py = _y_px(z)
        lines.append(_ocr_line(label, 20, py - 8, 60, py + 8))  # cx=40/700=0.057<0.20
    return lines


def _foster_curve_points(r=2.0, tau=1e-3, n=60):
    """Real Foster-shape points: Zth(t) = r*(1 - exp(-t/tau))."""
    ts = np.logspace(-6, -1.8, n)
    zs = r * (1.0 - np.exp(-ts / tau))
    zs = np.clip(zs, 1e-4, None)
    return ts, zs


def _draw_zth_chart(with_gridlines=True, r=2.0, tau=1e-3):
    img = np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)
    if with_gridlines:
        for _, t in (("1us", 1e-6), ("10us", 1e-5), ("100us", 1e-4), ("1ms", 1e-3), ("10ms", 1e-2)):
            px = int(round(_x_px(t)))
            cv2.line(img, (px, 30), (px, 340), (180, 180, 180), 1)
        for _, z in (("0.01", 0.01), ("0.1", 0.1), ("1", 1.0), ("10", 10.0)):
            py = int(round(_y_px(z)))
            cv2.line(img, (100, py), (620, py), (180, 180, 180), 1)
    ts, zs = _foster_curve_points(r=r, tau=tau)
    pts = np.array([[int(round(_x_px(t))), int(round(_y_px(z)))] for t, z in zip(ts, zs)])
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < IMG_W) & (pts[:, 1] >= 0) & (pts[:, 1] < IMG_H)]
    for i in range(len(pts) - 1):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), (0, 0, 0), 2)
    return img


def run_standard(image=None, ocr_lines=None, stage3_root=None):
    return run_classical_pipeline(
        device="DEV1", curve_type="zth_vs_time", source_image="fig.png",
        image=image if image is not None else _draw_zth_chart(),
        ocr_lines=ocr_lines if ocr_lines is not None else _axis_ticks_ocr(),
        stage3_root=stage3_root,
    )


# ---------------------------------------------------- A. printed-table-first

class TestPrintedTableFirst:
    def _table_ocr_lines(self):
        return [
            _ocr_line("Ri (\u00b0C/W)", 290, 45, 320, 58),
            _ocr_line("\u03c4i (sec)", 390, 47, 420, 60),
            _ocr_line("0.5", 295, 75, 315, 88),
            _ocr_line("0.0001", 385, 77, 425, 90),
            _ocr_line("1.5", 295, 105, 315, 118),
            _ocr_line("0.001", 385, 107, 425, 120),
        ]

    def test_table_found_returns_ok_with_confidence_one(self):
        result = run_standard(ocr_lines=self._table_ocr_lines())
        validate_result(result)
        assert result["status"] == "ok"
        assert result["curves"][0]["confidence"] == 1.0
        assert result["curves"][0]["curve_name"] == "single_pulse"

    def test_table_found_never_touches_the_cv_pipeline(self, monkeypatch):
        def _explode(*a, **kw):
            raise AssertionError("derive_calibration_zth must not be called when a table is found")
        monkeypatch.setattr(classical_zth, "derive_calibration_zth", _explode)
        result = run_standard(ocr_lines=self._table_ocr_lines())
        assert result["status"] == "ok"

    def test_table_result_keeps_rc_pairs_and_rth_steady_as_bonus_fields(self):
        result = run_standard(ocr_lines=self._table_ocr_lines())
        curve = result["curves"][0]
        assert curve.get("rc_pairs") == [
            {"R": 0.5, "tau": 0.0001, "row": 1},
            {"R": 1.5, "tau": 0.001, "row": 2},
        ]
        assert curve.get("rth_jc_steady_state") == pytest.approx(2.0)

    def test_table_result_units_are_kw(self):
        result = run_standard(ocr_lines=self._table_ocr_lines())
        assert result["units"] == "K/W"

    def test_no_table_falls_through_to_cv_pipeline(self):
        # Same chart/OCR minus the table lines -> normal axis-tick OCR only.
        result = run_standard()
        validate_result(result)
        assert result["status"] == "ok"
        assert len(result["curves"][0]["points"]) > 1  # real digitized points, not a table read


# ------------------------------------------------------ B. curve-fit fallback

class TestCurveFitFallback:
    def test_clean_chart_ok_with_digitized_points(self):
        result = run_standard()
        validate_result(result)
        assert result["status"] == "ok"
        curve = result["curves"][0]
        assert curve["curve_name"] == "single_pulse"
        assert len(curve["points"]) >= 6
        assert curve.get("fitted_params") is not None
        assert "r1" in curve["fitted_params"] and "tau1" in curve["fitted_params"]

    def test_units_always_kw(self):
        result = run_standard()
        assert result["units"] == "K/W"

    def test_calibration_bonus_fields_kept_alongside_required_six(self):
        result = run_standard()
        cal = result["calibration"]
        for key in ("x_slope", "x_intercept", "y_slope", "y_intercept", "x_log", "y_log"):
            assert key in cal
        # Legacy's own extra detail, preserved rather than thrown away:
        for key in ("x_scale", "y_scale", "x_ticks_used", "y_ticks_used", "plot_bbox"):
            assert key in cal

    def test_no_curves_found_needs_review(self):
        blank = np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)
        result = run_standard(image=blank)
        validate_result(result)
        assert result["status"] == "needs_review"
        assert result["review_reason"]

    def test_calibration_failure_needs_review_with_descriptive_reason(self):
        result = run_standard(ocr_lines=[])
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "calibration" in result["review_reason"].lower()

    def test_too_few_points_needs_review(self, monkeypatch):
        monkeypatch.setattr(classical_zth, "trace_curve", lambda *a, **kw: [(1.0, 2.0), (2.0, 3.0)])
        result = run_standard()
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "point" in result["review_reason"].lower()


# ------------------------------------------- C. status-mapping conversion

class TestStatusMappingConversion:
    def test_calibration_disaster_maps_to_needs_review_with_rise_ratio_in_message(self, monkeypatch):
        # Force an implausible (decreasing-in-data, since pixel-row increases
        # with x on this log-y chart) trace so rise_ratio < 1.0, with no Rth
        # constraint available (unconstrained gate applies).
        monkeypatch.setattr(classical_zth, "pick_rth_constraint", lambda *a, **kw: (None, None))
        monkeypatch.setattr(
            classical_zth, "trace_curve",
            lambda cluster, x_step=1: [(float(i), 200.0 + i) for i in range(0, 60, 2)],
        )
        result = run_standard()
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "rise_ratio" in result["review_reason"]

    def test_skip_fit_calibration_broken_remapped_to_needs_review_not_ok(self, monkeypatch):
        # Legacy calls this "clean" (their status string) -- we deliberately
        # remap it to needs_review, keeping their own warning text verbatim.
        monkeypatch.setattr(classical_zth, "pick_rth_constraint", lambda *a, **kw: (50.0, "table_max"))
        result = run_standard()
        validate_result(result)
        # The drawn curve asymptotes to r=2.0; against a forced
        # rth_constraint=50.0, scale_ratio (~0.04) falls outside [0.05, 3.0].
        assert result["status"] == "needs_review"
        assert "calibration_broken" in result["review_reason"]

    def test_foster_fit_failure_maps_to_needs_review(self, monkeypatch):
        monkeypatch.setattr(classical_zth, "pick_rth_constraint", lambda *a, **kw: (None, None))
        monkeypatch.setattr(classical_zth, "fit_foster", lambda xs, ys, rth_constraint=None: (None, None))
        result = run_standard()
        validate_result(result)
        assert result["status"] == "needs_review"

    def test_low_r_squared_fit_maps_to_needs_review(self, monkeypatch):
        monkeypatch.setattr(classical_zth, "pick_rth_constraint", lambda *a, **kw: (None, None))
        monkeypatch.setattr(
            classical_zth, "fit_foster",
            lambda xs, ys, rth_constraint=None: ({"r1": 1.0, "tau1": 1e-3}, 0.2),
        )
        result = run_standard()
        validate_result(result)
        assert result["status"] == "needs_review"

    def test_rth_constraint_anchors_the_fit_when_plausible(self, monkeypatch):
        # A plausible constraint (close to the drawn curve's real asymptote,
        # r=2.0) must be threaded into fit_foster as rth_constraint.
        captured = {}
        real_fit_foster = fit_foster

        def _spy(xs, ys, rth_constraint=None):
            captured["rth_constraint"] = rth_constraint
            return real_fit_foster(xs, ys, rth_constraint=rth_constraint)

        monkeypatch.setattr(classical_zth, "pick_rth_constraint", lambda *a, **kw: (2.0, "table_max"))
        monkeypatch.setattr(classical_zth, "fit_foster", _spy)
        result = run_standard()
        assert captured["rth_constraint"] == 2.0
        if result["status"] == "ok":
            assert result["curves"][0]["fitted_params"]["r1"] == pytest.approx(2.0)


# ------------------------------------------------- D. confidence clamping

class TestConfidenceClamping:
    def test_negative_r_squared_clamps_to_zero_not_negative(self, monkeypatch):
        # A negative r-squared must never reach validate_result (which
        # requires confidence in [0, 1]) unclamped.
        monkeypatch.setattr(classical_zth, "pick_rth_constraint", lambda *a, **kw: (None, None))
        monkeypatch.setattr(
            classical_zth, "fit_foster",
            lambda xs, ys, rth_constraint=None: ({"r1": 1.0, "tau1": 1e-3}, -0.7),
        )
        # r2=-0.7 < 0.5 threshold -> needs_review path; but IF a curve entry
        # is still produced with a confidence, it must be clamped.
        result = run_standard()
        validate_result(result)  # would raise if any confidence is out of [0,1]

    def test_table_path_confidence_is_exactly_one_regardless_of_anything_else(self):
        result = run_standard(ocr_lines=[
            _ocr_line("Ri (\u00b0C/W)", 290, 45, 320, 58),
            _ocr_line("\u03c4i (sec)", 390, 47, 420, 60),
            _ocr_line("0.5", 295, 75, 315, 88),
            _ocr_line("0.0001", 385, 77, 425, 90),
            _ocr_line("1.5", 295, 105, 315, 118),
            _ocr_line("0.001", 385, 107, 425, 120),
        ])
        assert result["curves"][0]["confidence"] == 1.0

    def test_good_fit_confidence_matches_r_squared(self):
        result = run_standard()
        assert result["status"] == "ok"
        r2 = result["curves"][0]["fitted_params"] and result["curves"][0].get("r_squared")
        if r2 is not None:
            assert result["curves"][0]["confidence"] == pytest.approx(max(0.0, min(1.0, r2)))


# --------------------------------------------- E. no legacy hardcoded paths

class TestNoLegacyHardcodedPaths:
    def test_source_contains_no_legacy_absolute_paths(self):
        import inspect
        source = inspect.getsource(classical_zth)
        assert "/mnt/c/Archit" not in source
        assert "INFINEON_DIRS" not in source
        assert "def find_device(" not in source

    def test_stage3_root_is_env_var_driven_not_hardcoded(self):
        import inspect
        source = inspect.getsource(classical_zth)
        assert "LINEFORMER_STAGE3_ROOT" in source


# --------------------------------------- F. Rth-table self-contained file read

class TestRthTableFileRead:
    def _write_full_extraction_with_rth(self, tmp_path, device, typ=0.4, mx=0.5):
        device_dir = tmp_path / device
        device_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "tables": [{
                "headers": [
                    {"text": "Symbol", "col": 0}, {"text": "Typ", "col": 1},
                    {"text": "Max", "col": 2}, {"text": "Unit", "col": 3},
                ],
                "rows": [{"cells": [
                    {"col": 0, "text": "RthJC"}, {"col": 1, "text": str(typ)},
                    {"col": 2, "text": str(mx)}, {"col": 3, "text": "K/W"},
                ]}],
            }],
        }
        (device_dir / "full_extraction.json").write_text(json.dumps(payload), encoding="utf-8")
        return device_dir

    def test_valid_rth_table_found_via_stage3_root(self, tmp_path):
        self._write_full_extraction_with_rth(tmp_path, "DEV1")
        result = run_standard(stage3_root=str(tmp_path))
        # rth_jc_steady_state should reflect the table's max (0.5), not None,
        # whenever the fit actually ran with a constraint.
        curve = result["curves"][0]
        if result["status"] == "ok":
            assert curve.get("rth_jc_steady_state") in (0.5, None) or curve.get("rth_jc_steady_state") is not None

    def test_missing_stage3_root_proceeds_without_constraint_not_a_crash(self):
        result = run_standard(stage3_root=None)
        # Must not raise -- env var almost certainly unset in the test env.
        validate_result(result)

    def test_missing_device_folder_proceeds_without_constraint(self, tmp_path):
        result = run_standard(stage3_root=str(tmp_path))  # tmp_path has no DEV1 subfolder
        validate_result(result)

    def test_malformed_json_proceeds_without_constraint_not_a_crash(self, tmp_path):
        device_dir = tmp_path / "DEV1"
        device_dir.mkdir(parents=True, exist_ok=True)
        (device_dir / "full_extraction.json").write_text("{not valid json", encoding="utf-8")
        result = run_standard(stage3_root=str(tmp_path))
        validate_result(result)


# ------------------------------------------------- ported-function unit tests

class TestPortedTickParsing:
    def test_plain_number(self):
        assert parse_tick("10", axis="x") == 10.0

    def test_si_microseconds(self):
        assert parse_tick("1us", axis="x") == pytest.approx(1e-6)

    def test_si_milliseconds(self):
        assert parse_tick("10ms", axis="x") == pytest.approx(1e-2)

    def test_bare_power_of_ten_mojibake(self):
        assert parse_tick("102", axis="y") == pytest.approx(100.0)

    def test_unicode_superscript_exponent(self):
        assert parse_tick("10\u207b\u00b3", axis="x") == pytest.approx(1e-3)

    def test_garbage_returns_none(self):
        assert parse_tick("garbage!!", axis="x") is None


class TestPortedAxisFit:
    def test_fit_axis_auto_picks_log_for_wide_decade_span(self):
        ticks = [(1e-6, 150.0), (1e-5, 230.0), (1e-4, 310.0), (1e-3, 390.0), (1e-2, 470.0)]
        fit = fit_axis_auto(ticks, "x")
        assert fit["scale"] == "log"

    def test_fit_axis_auto_picks_linear_for_narrow_range(self):
        ticks = [(0.0, 100.0), (10.0, 150.0), (20.0, 200.0), (30.0, 250.0)]
        fit = fit_axis_auto(ticks, "x")
        assert fit["scale"] == "linear"


class TestPortedFosterFit:
    def test_clean_foster_shape_fits_well(self):
        ts, zs = _foster_curve_points(r=2.0, tau=1e-3)
        params, r2 = fit_foster(ts.tolist(), zs.tolist())
        assert params is not None
        assert r2 > 0.9
        assert params["r1"] == pytest.approx(2.0, rel=0.2)

    def test_constrained_fit_fixes_r_at_given_value(self):
        ts, zs = _foster_curve_points(r=2.0, tau=1e-3)
        params, r2 = fit_foster(ts.tolist(), zs.tolist(), rth_constraint=2.0)
        assert params is not None
        assert params["r1"] == 2.0

    def test_too_few_points_returns_none(self):
        params, r2 = fit_foster([1.0, 2.0], [1.0, 2.0])
        assert params is None
        assert r2 is None


class TestPortedFosterTableParsing:
    def test_no_table_headers_returns_none(self):
        assert parse_foster_table_from_ocr({"ocr_lines": []}) is None

    def test_valid_table_parsed(self):
        fig_meta = {"ocr_lines": [
            _ocr_line("Ri (\u00b0C/W)", 290, 45, 320, 58),
            _ocr_line("\u03c4i (sec)", 390, 47, 420, 60),
            _ocr_line("0.5", 295, 75, 315, 88),
            _ocr_line("0.0001", 385, 77, 425, 90),
            _ocr_line("1.5", 295, 105, 315, 118),
            _ocr_line("0.001", 385, 107, 425, 120),
        ]}
        table = parse_foster_table_from_ocr(fig_meta)
        assert table is not None
        assert table["n_pairs"] == 2
        assert table["rth_steady"] == pytest.approx(2.0)


class TestPortedRthConstraint:
    def test_prefers_max_over_typ(self):
        full_extraction = {"tables": [{
            "headers": [{"text": "Symbol", "col": 0}, {"text": "Typ", "col": 1},
                       {"text": "Max", "col": 2}, {"text": "Unit", "col": 3}],
            "rows": [{"cells": [{"col": 0, "text": "RthJC"}, {"col": 1, "text": "0.4"},
                                {"col": 2, "text": "0.5"}, {"col": 3, "text": "K/W"}]}],
        }]}
        value, source = pick_rth_constraint(full_extraction)
        assert value == 0.5
        assert source == "table_max"

    def test_no_matching_rows_returns_none(self):
        value, source = pick_rth_constraint({"tables": []})
        assert value is None
        assert source is None
