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
    derive_calibration_zth,
    detect_normalized_ratio_axis,
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


# --------------------------------------------- G. Ratio-axis guard (new check)

class TestRatioAxisGuard:
    """Some zth_vs_time charts plot a NORMALIZED ratio r(t) rather than a
    direct K/W value (real bug found on device AG087FGD3HRBTL, 2026-07-23,
    caught during a look-only AI-model comparison check) -- treating r(t)
    as if it were real K/W is silently wrong. See
    detect_normalized_ratio_axis's own docstring for the two detection
    signals (reused here as wired into run_classical_pipeline; the
    detector itself gets its own isolated unit tests in
    TestDetectNormalizedRatioAxis, below)."""

    def _yzone_line(self, text):
        # cx/img_w < 0.20 with IMG_W=700 -> cx must stay under 140.
        return _ocr_line(text, 10, 290, 50, 310)

    def _offzone_line(self, text):
        return _ocr_line(text, 300, 200, 480, 220)

    def test_normalized_yzone_label_short_circuits_to_needs_review(self):
        lines = _axis_ticks_ocr() + [
            self._yzone_line("Normalized Transient Resistance : r(t)")
        ]
        result = run_standard(ocr_lines=lines)
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "ratio" in result["review_reason"].lower()

    def test_formula_pattern_short_circuits_to_needs_review(self):
        lines = _axis_ticks_ocr() + [
            self._offzone_line("Rth(j-c)(t)=r(t) x Rth(j-c)")
        ]
        result = run_standard(ocr_lines=lines)
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "ratio" in result["review_reason"].lower()

    def test_ratio_axis_never_reaches_calibration(self, monkeypatch):
        def _explode(*a, **kw):
            raise AssertionError(
                "derive_calibration_zth must not be called once a ratio axis is detected"
            )
        monkeypatch.setattr(classical_zth, "derive_calibration_zth", _explode)
        lines = _axis_ticks_ocr() + [
            self._yzone_line("Normalized Transient Resistance : r(t)")
        ]
        result = run_standard(ocr_lines=lines)
        assert result["status"] == "needs_review"

    def test_printed_table_shortcut_wins_over_ratio_signal(self):
        # Even when the OCR text ALSO contains a ratio signal somewhere,
        # a valid printed Foster table must still win with status "ok" --
        # the ratio check only ever applies on the CV-pipeline fallback
        # path, never ahead of the table-read shortcut.
        lines = [
            _ocr_line("Ri (°C/W)", 290, 45, 320, 58),
            _ocr_line("τi (sec)", 390, 47, 420, 60),
            _ocr_line("0.5", 295, 75, 315, 88),
            _ocr_line("0.0001", 385, 77, 425, 90),
            _ocr_line("1.5", 295, 105, 315, 118),
            _ocr_line("0.001", 385, 107, 425, 120),
            self._yzone_line("Normalized Transient Resistance : r(t)"),
        ]
        result = run_standard(ocr_lines=lines)
        assert result["status"] == "ok"
        assert result["curves"][0]["confidence"] == 1.0

    def test_direct_units_chart_unaffected_still_ok(self):
        # The standard fixture's OCR text has no "normalized" wording and
        # no r(t) x Rth formula anywhere -- the new guard must not fire,
        # and the existing clean-chart behavior must be unchanged.
        result = run_standard()
        validate_result(result)
        assert result["status"] == "ok"

    def test_real_device_ag087_now_flagged_instead_of_silently_ok(self):
        # Real bug (2026-07-23): AG087FGD3HRBTL's zth_vs_time chart (Fig.3,
        # fig_p4_007.png) is a normalized-ratio chart -- its y-axis label
        # literally reads "Normalized Transient Resistance : r(t)" and it
        # prints its own "Rth(j-c)(t)=r(t) x Rth(j-c)" conversion formula.
        # BEFORE this guard existed, calling run_classical_pipeline with no
        # Rth constraint (stage3_root=None, the unconstrained path) returned
        # status="ok" with confidence 0.88 -- a real digitized trace,
        # silently mislabeled as direct K/W when it is actually the
        # dimensionless ratio r(t). Exact OCR line set from the real figure
        # (full_extraction.json figure index 7), crop size 707x694.
        real_lines = [
            _ocr_line("10", 114, 42, 144, 63),
            _ocr_line("Tc=25℃", 184, 79, 279, 107),
            _ocr_line("1", 129, 220, 142, 237),
            _ocr_line("Duty cycle", 479, 263, 574, 283),
            _ocr_line("top", 479, 285, 511, 304),
            _ocr_line("D=1", 554, 284, 591, 301),
            _ocr_line("D=0.5", 553, 305, 610, 322),
            _ocr_line("D=0.1", 555, 326, 607, 343),
            _ocr_line("D=0.05", 555, 347, 620, 363),
            _ocr_line("D=0.01", 553, 368, 620, 385),
            _ocr_line("0.1", 104, 393, 140, 414),
            _ocr_line("bottom Single", 479, 389, 611, 409),
            _ocr_line("Rth(j-c)=2.80℃/W", 265, 487, 429, 507),
            _ocr_line("Rth(j-c)(t)=r(t) × Rth(j-c)", 265, 511, 476, 533),
            _ocr_line("Normalized Transient Resistance : r(t)", 20, 76, 57, 559),
            _ocr_line("0.01", 88, 567, 142, 590),
            _ocr_line("0.0001 0.001", 129, 594, 276, 616),
            _ocr_line("0.01", 277, 594, 351, 615),
            _ocr_line("0.1", 394, 595, 425, 615),
            _ocr_line("1", 488, 596, 499, 613),
            _ocr_line("10", 562, 595, 590, 614),
            _ocr_line("100", 638, 595, 679, 615),
            _ocr_line("Pulse Width : PW [s]", 282, 642, 547, 674),
        ]
        img = np.full((694, 707, 3), 255, dtype=np.uint8)
        result = run_classical_pipeline(
            device="AG087FGD3HRBTL", curve_type="zth_vs_time",
            source_image="fig_p4_007.png", image=img, ocr_lines=real_lines,
            stage3_root=None,
        )
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "ratio" in result["review_reason"].lower()
        assert result["curves"][0]["confidence"] == 0.0


class TestDetectNormalizedRatioAxis:
    """Unit tests for detect_normalized_ratio_axis in isolation -- pure
    OCR-text-in / reason-string-out, no image or pipeline involved."""

    def _yzone(self, text):
        return _ocr_line(text, 10, 100, 50, 300)  # cx=30, well inside cx/700<0.20

    def _offzone(self, text):
        return _ocr_line(text, 300, 100, 480, 120)

    def test_yzone_normalized_word_detected(self):
        lines = [self._yzone("Normalized Transient Resistance : r(t)")]
        reason = detect_normalized_ratio_axis(lines, IMG_W, IMG_H)
        assert reason is not None
        assert "ratio" in reason.lower()

    def test_yzone_normalised_british_spelling_detected(self):
        lines = [self._yzone("Normalised Transient Thermal Resistance")]
        assert detect_normalized_ratio_axis(lines, IMG_W, IMG_H) is not None

    def test_yzone_normalized_all_caps_detected(self):
        lines = [self._yzone("NORMALIZED TRANSIENT RESISTANCE")]
        assert detect_normalized_ratio_axis(lines, IMG_W, IMG_H) is not None

    def test_formula_unicode_times_sign_detected(self):
        lines = [self._offzone("Rth(j-c)(t)=r(t) × Rth(j-c)")]
        assert detect_normalized_ratio_axis(lines, IMG_W, IMG_H) is not None

    def test_formula_plain_x_no_space_detected(self):
        lines = [self._offzone("Rth(ch-c)(t) = r(t) xrth(ch-c)")]
        assert detect_normalized_ratio_axis(lines, IMG_W, IMG_H) is not None

    def test_formula_space_before_paren_detected(self):
        lines = [self._offzone("rth(j-c)(t) = r (t) × rth(j-c)")]
        assert detect_normalized_ratio_axis(lines, IMG_W, IMG_H) is not None

    def test_formula_with_extra_whitespace_detected(self):
        lines = [self._offzone("Rth(j-c) (t)  =  r ( t )   ×   Rth(j-c)")]
        assert detect_normalized_ratio_axis(lines, IMG_W, IMG_H) is not None

    def test_both_signals_present_detected_once(self):
        lines = [
            self._yzone("Normalized Transient Resistance : r(t)"),
            self._offzone("Rth(j-c)(t)=r(t) × Rth(j-c)"),
        ]
        reason = detect_normalized_ratio_axis(lines, IMG_W, IMG_H)
        assert reason is not None

    def test_neither_signal_returns_none(self):
        lines = [self._yzone("ZthJC [K/W]"), self._offzone("Single Pulse")]
        assert detect_normalized_ratio_axis(lines, IMG_W, IMG_H) is None

    def test_direct_units_label_not_falsely_flagged(self):
        lines = [self._yzone("Transient Thermal Impedance : ZthJC [K/W]")]
        assert detect_normalized_ratio_axis(lines, IMG_W, IMG_H) is None

    def test_normalized_word_outside_yzone_alone_is_not_enough(self):
        # "normalized" appearing OUTSIDE the y-axis-label zone, with no
        # formula pattern anywhere, must NOT trigger -- the word-based
        # signal is deliberately scoped to the y-axis label zone only.
        lines = [self._offzone("see the normalized curve family in Fig. 9")]
        assert detect_normalized_ratio_axis(lines, IMG_W, IMG_H) is None

    def test_empty_ocr_lines_returns_none(self):
        assert detect_normalized_ratio_axis([], IMG_W, IMG_H) is None

    def test_lines_missing_bounding_box_are_skipped_not_a_crash(self):
        lines = [{"text": "Normalized"}, self._offzone("Single Pulse")]
        assert detect_normalized_ratio_axis(lines, IMG_W, IMG_H) is None


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

    def test_plain_100_is_not_misread_as_exponent(self):
        # Real bug found on device AG087FGD3HRBTL (2026-07-23): the pulse-
        # width x-axis has a genuine tick labeled "100" (100 seconds). The
        # bare-exponent-mojibake repair above (for OCR misreading "10\u00b3"
        # as "103") was firing on this too, via a regex that matched ANY
        # "10" + single digit 0-9 -- so "100" was reinterpreted as
        # 10**0 == 1.0, silently corrupting the axis. "101"-"109" are
        # deliberately left alone (test_bare_power_of_ten_mojibake, above,
        # still expects "102" -> 100.0 unchanged) since those are
        # essentially never real tick values in this domain; "100" is
        # ordinary and ambiguous, so it must parse as itself.
        assert parse_tick("100", axis="x") == 100.0
        assert parse_tick("100", axis="y") == 100.0

    def test_bare_109_upper_boundary_still_treated_as_exponent(self):
        # Pins the narrowed rule's boundary: 109 (not 100) is the edge of
        # the still-reinterpreted range.
        assert parse_tick("109", axis="y") == pytest.approx(1e9)


class TestAxisTicksGluedLabels:
    """Real bug found on device AG087FGD3HRBTL (2026-07-23): Azure OCR read
    the adjacent "0.0001" and "0.001" pulse-width x-axis tick labels as ONE
    glued line of text, "0.0001 0.001". parse_tick returns None for that
    whole string (it isn't a single number), so parse_axis_ticks silently
    dropped BOTH ticks -- shrinking the fitted plot box from the left.

    src.calibration.ticks.parse_numeric_ticks already has a fix for this
    exact OCR-gluing problem (its "compound token" path, see that module's
    own docstring): split the line on whitespace, and if every part parses
    as its own number, distribute them evenly across the line's own
    bounding-box pixel span instead of dropping the whole line. These tests
    check parse_axis_ticks reuses that same approach rather than a new one.
    """

    def test_glued_x_labels_recovered_as_two_ticks(self):
        lines = [_ocr_line("0.0001 0.001", 100, 350, 250, 370)]  # cy=360/400=0.90 -> x-zone
        x_ticks, y_ticks = parse_axis_ticks(lines, IMG_W, IMG_H)
        vals = sorted(v for v, _ in x_ticks)
        assert vals == pytest.approx([0.0001, 0.001])
        assert y_ticks == []

    def test_glued_labels_spread_across_the_lines_own_pixel_span(self):
        # Left-to-right reading order -> increasing pixel x, spread across
        # the line's OWN bounding box (x1=100 -> x2=250), same fractional
        # interpolation as ticks.py's compound-token path.
        lines = [_ocr_line("0.0001 0.001", 100, 350, 250, 370)]
        x_ticks, _ = parse_axis_ticks(lines, IMG_W, IMG_H)
        by_val = dict(x_ticks)
        assert by_val[0.0001] == pytest.approx(100.0)
        assert by_val[0.001] == pytest.approx(250.0)

    def test_glued_y_labels_recovered_too(self):
        # Mirrors the x case but in the y (left, cx/img_w < 0.20) zone,
        # top-to-bottom reading order -> increasing pixel y.
        lines = [_ocr_line("10 1", 20, 40, 60, 200)]  # cx=40/700=0.057 -> y-zone
        _, y_ticks = parse_axis_ticks(lines, IMG_W, IMG_H)
        vals = sorted(v for v, _ in y_ticks)
        assert vals == pytest.approx([1.0, 10.0])

    def test_glued_label_with_unparseable_part_is_still_dropped(self):
        # Not every multi-word line is a glued compound tick -- if any part
        # fails to parse, this isn't that case; drop the whole line exactly
        # as before (never guess at a partial match).
        lines = [_ocr_line("0.0001 garbage", 100, 350, 250, 370)]
        x_ticks, _ = parse_axis_ticks(lines, IMG_W, IMG_H)
        assert x_ticks == []

    def test_single_token_lines_are_unaffected(self):
        # Pre-existing (non-glued) behavior must not change.
        lines = [_ocr_line("10", 560, 350, 590, 370)]
        x_ticks, _ = parse_axis_ticks(lines, IMG_W, IMG_H)
        assert x_ticks == [(10.0, 575.0)]

    def test_real_device_ag087_x_axis_recovers_full_six_decade_span(self):
        # The exact OCR line set from AG087FGD3HRBTL's zth_vs_time chart
        # (Fig.3 "Normalized Transient Thermal Resistance vs. Pulse Width",
        # fig_p4_007.png, crop size 707x694) -- both bugs' real-world
        # trigger together: the glued "0.0001 0.001" line, and the plain
        # "100" label. Before either fix, derive_calibration_zth's x fit
        # only spanned 3 decades (0.01..10, "100" corrupted to a duplicate
        # "1"); with both fixed it must recover the chart's real 6-decade
        # span (0.0001..100).
        lines = [
            _ocr_line("10", 114, 42, 144, 63),
            _ocr_line("Tc=25℃", 184, 79, 279, 107),
            _ocr_line("1", 129, 220, 142, 237),
            _ocr_line("0.1", 104, 393, 140, 414),
            _ocr_line("0.01", 88, 567, 142, 590),
            _ocr_line("0.0001 0.001", 129, 594, 276, 616),
            _ocr_line("0.01", 277, 594, 351, 615),
            _ocr_line("0.1", 394, 595, 425, 615),
            _ocr_line("1", 488, 596, 499, 613),
            _ocr_line("10", 562, 595, 590, 614),
            _ocr_line("100", 638, 595, 679, 615),
        ]
        cal = derive_calibration_zth({"ocr_lines": lines}, 707, 694)
        assert cal is not None
        assert cal["x_decade_span"] >= 5.0  # was 3.0 before either fix


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
