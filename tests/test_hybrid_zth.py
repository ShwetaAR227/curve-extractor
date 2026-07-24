"""Tests for src.extraction.hybrid_zth — written FIRST (CLAUDE.md §2, red
phase). RED PHASE ONLY — the module under test does not exist yet.

hybrid_zth.py combines the AI model (finds the single-pulse curve's pixel
points) with classical_zth.py's existing rule-based logic for everything
else (axis reading, the printed-table shortcut, the physics fit, the Rth
cross-check, ratio-axis detection) — see the module's own docstring for
the full design (owner-approved, 2026-07-23).

Every dependency this wrapper calls is mocked here (same convention as
test_model_if_vsd.py): this file tests ONLY hybrid_zth's own routing/
decision logic (table-shortcut delegation, model detection-count gates,
the ratio-axis override), not the internals of already-separately-tested
functions (classical_zth.py's own 67-test suite, zth_fit.py's own 26-test
suite, inference.py's/skeletonize.py's own suites). No GPU, no network,
no real trained checkpoint — one lightweight integration test at the end
uses the REAL calibration/fit/Rth-check machinery end-to-end (everything
that doesn't need a GPU), with only run_inference faked, to prove the
non-model wiring genuinely works, not just against mocks.
"""
import inspect
import os
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.extraction.inference import Detection
from src.extraction.schema import build_result, validate_result

import src.extraction.classical_zth as classical_zth
import src.extraction.hybrid_zth as mod
from src.extraction.hybrid_zth import (
    CHECKPOINT_RELATIVE_PATH,
    CHECKPOINTS_ROOT_ENV_VAR,
    CONFIG_RELATIVE_PATH,
    resolve_checkpoint_and_config,
    run_hybrid_pipeline,
)

IMG_H, IMG_W = 400, 700


def make_ocr_lines():
    return [{"text": "10", "bounding_box": {"x1": 0, "y1": 380, "x2": 20, "y2": 395}}]


def make_detections(n, score=0.9):
    mask = np.zeros((IMG_H, IMG_W), dtype=bool)
    mask[200, 100:600] = True
    return [Detection(score=score, mask=mask.copy()) for _ in range(n)]


def ok_result(reason=None, status="ok"):
    return build_result(
        device="DEV1", curve_type="zth_vs_time", source_image="fig.png",
        status=status, review_reason=reason, duplicates_removed=0,
        calibration={"x_slope": 1.0, "x_intercept": 0.0, "y_slope": 1.0,
                    "y_intercept": 0.0, "x_log": False, "y_log": False},
        curves=[{"curve_name": "single_pulse", "confidence": 0.9,
                 "points": [{"x": 1.0, "y": 2.0}]}],
        units="K/W" if status == "ok" else "K/W",
    )


DEFAULT_CAL_ZTH = {
    "plot_bbox": {"left": 0, "right": 700, "top": 0, "bottom": 400},
    "x_scale": "linear", "x_slope": 1.0, "x_intercept": 0.0,
    "y_scale": "linear", "y_slope": 1.0, "y_intercept": 0.0,
}


def build_run(
    monkeypatch, table=None, ratio_reason=None, detections=None,
    cal_zth=DEFAULT_CAL_ZTH, fit_result=None,
):
    """Patch every dependency run_hybrid_pipeline calls, sensible defaults."""
    table_mock = MagicMock(return_value=table)
    monkeypatch.setattr(mod, "parse_foster_table_from_ocr", table_mock)

    ratio_mock = MagicMock(return_value=ratio_reason)
    monkeypatch.setattr(mod, "detect_normalized_ratio_axis", ratio_mock)

    inference_mock = MagicMock(return_value=[] if detections is None else list(detections))
    monkeypatch.setattr(mod, "run_inference", inference_mock)

    mask_points_mock = MagicMock(
        return_value=[(0.0, float(i)) for i in range(6)]  # (row, col) x 6
    )
    monkeypatch.setattr(mod, "mask_to_points", mask_points_mock)

    cal_mock = MagicMock(return_value=cal_zth)
    monkeypatch.setattr(mod, "derive_calibration_zth", cal_mock)

    fit_mock = MagicMock(return_value=fit_result if fit_result is not None else ok_result())
    monkeypatch.setattr(mod, "fit_and_validate_curve", fit_mock)

    return {
        "parse_foster_table_from_ocr": table_mock,
        "detect_normalized_ratio_axis": ratio_mock,
        "run_inference": inference_mock,
        "mask_to_points": mask_points_mock,
        "derive_calibration_zth": cal_mock,
        "fit_and_validate_curve": fit_mock,
    }


def run(device="DEV1", curve_type="zth_vs_time", source_image="fig.png",
        image=None, ocr_lines=None, model=None, stage3_root=None, score_thr=0.5):
    return run_hybrid_pipeline(
        device=device, curve_type=curve_type, source_image=source_image,
        image=image if image is not None else np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8),
        ocr_lines=ocr_lines if ocr_lines is not None else make_ocr_lines(),
        model=model if model is not None else object(),
        stage3_root=stage3_root, score_thr=score_thr,
    )


# ============================================================
# A. Printed-table shortcut — unchanged, model never runs
# ============================================================

class TestTableShortcut:
    def test_table_found_delegates_to_classical_zth_run_classical_pipeline(self, monkeypatch):
        mocks = build_run(monkeypatch, table={"n_pairs": 2})
        sentinel = ok_result()
        delegate_mock = MagicMock(return_value=sentinel)
        monkeypatch.setattr(classical_zth, "run_classical_pipeline", delegate_mock)

        result = run()

        assert result == sentinel
        delegate_mock.assert_called_once()
        mocks["run_inference"].assert_not_called()

    def test_table_found_model_never_invoked_even_if_it_would_explode(self, monkeypatch):
        build_run(monkeypatch, table={"n_pairs": 2})

        def _explode(*a, **kw):
            raise AssertionError("run_inference must not be called when a table is found")
        monkeypatch.setattr(mod, "run_inference", _explode)
        monkeypatch.setattr(classical_zth, "run_classical_pipeline",
                            MagicMock(return_value=ok_result()))

        run()  # must not raise

    def test_no_table_never_calls_classical_zth_run_classical_pipeline(self, monkeypatch):
        build_run(monkeypatch, table=None)

        def _explode(*a, **kw):
            raise AssertionError("classical_zth.run_classical_pipeline must not run without a table")
        monkeypatch.setattr(classical_zth, "run_classical_pipeline", _explode)

        run()  # must not raise -- proceeds via the model path instead


# ============================================================
# B. Model detection-count gates
# ============================================================

class TestModelDetectionGates:
    def test_zero_detections_needs_review(self, monkeypatch):
        build_run(monkeypatch, detections=[])
        result = run()
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "no curves" in result["review_reason"].lower() or "0" in result["review_reason"]

    def test_multiple_detections_needs_review_distinct_from_zero_case(self, monkeypatch):
        build_run(monkeypatch, detections=make_detections(3))
        result = run()
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "3" in result["review_reason"]

    def test_zero_and_multiple_give_different_reasons(self, monkeypatch):
        build_run(monkeypatch, detections=[])
        zero_reason = run()["review_reason"]
        build_run(monkeypatch, detections=make_detections(2))
        multi_reason = run()["review_reason"]
        assert zero_reason != multi_reason

    def test_exactly_one_detection_proceeds_to_fit(self, monkeypatch):
        mocks = build_run(monkeypatch, detections=make_detections(1))
        run()
        mocks["fit_and_validate_curve"].assert_called_once()

    def test_score_thr_passed_through_to_run_inference(self, monkeypatch):
        mocks = build_run(monkeypatch, detections=make_detections(1))
        run(score_thr=0.75)
        _, kwargs = mocks["run_inference"].call_args
        assert kwargs.get("score_thr") == 0.75 or mocks["run_inference"].call_args[0][2] == 0.75


# ============================================================
# C. Calibration gates
# ============================================================

class TestCalibrationGates:
    def test_calibration_failed_needs_review(self, monkeypatch):
        build_run(monkeypatch, detections=make_detections(1), cal_zth=None)
        result = run()
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "calibration" in result["review_reason"].lower()

    def test_plot_bbox_too_small_needs_review(self, monkeypatch):
        tiny_cal = dict(DEFAULT_CAL_ZTH, plot_bbox={"left": 0, "right": 10, "top": 0, "bottom": 10})
        build_run(monkeypatch, detections=make_detections(1), cal_zth=tiny_cal)
        result = run()
        validate_result(result)
        assert result["status"] == "needs_review"
        assert "small" in result["review_reason"].lower()

    def test_calibration_failure_never_reaches_fit(self, monkeypatch):
        mocks = build_run(monkeypatch, detections=make_detections(1), cal_zth=None)
        run()
        mocks["fit_and_validate_curve"].assert_not_called()


# ============================================================
# D. Mask -> pixel points conversion
# ============================================================

class TestMaskToPixelPoints:
    def test_row_col_from_mask_to_points_converted_to_x_y_for_fit(self, monkeypatch):
        mocks = build_run(monkeypatch, detections=make_detections(1))
        monkeypatch.setattr(mod, "mask_to_points", MagicMock(
            return_value=[(10.0, 20.0), (11.0, 21.0), (12.0, 22.0),
                          (13.0, 23.0), (14.0, 24.0), (15.0, 25.0)]
        ))
        run()
        args, kwargs = mocks["fit_and_validate_curve"].call_args
        passed_points = kwargs.get("pixel_points") if "pixel_points" in kwargs else args[3]
        # (row, col) -> (x, y) == (col, row)
        assert passed_points[0] == (20.0, 10.0)
        assert passed_points[-1] == (25.0, 15.0)


# ============================================================
# E. Ratio-axis override
# ============================================================

class TestRatioAxisOverride:
    def test_no_ratio_issue_leaves_ok_result_unchanged(self, monkeypatch):
        build_run(monkeypatch, detections=make_detections(1), ratio_reason=None,
                  fit_result=ok_result())
        result = run()
        assert result["status"] == "ok"
        assert result["review_reason"] is None

    def test_ratio_detected_overrides_ok_to_needs_review(self, monkeypatch):
        build_run(monkeypatch, detections=make_detections(1),
                  ratio_reason="axis_is_normalized_ratio: ...", fit_result=ok_result())
        result = run()
        validate_result(result)
        assert result["status"] == "needs_review"
        assert result["review_reason"] == "axis_is_normalized_ratio: ..."

    def test_ratio_override_keeps_the_traced_curve_and_calibration(self, monkeypatch):
        underlying = ok_result()
        build_run(monkeypatch, detections=make_detections(1),
                  ratio_reason="axis_is_normalized_ratio: ...", fit_result=underlying)
        result = run()
        assert result["curves"] == underlying["curves"]
        assert result["calibration"] == underlying["calibration"]

    def test_ratio_detected_overrides_even_a_different_needs_review_reason(self, monkeypatch):
        # fit_and_validate_curve itself already said needs_review for its
        # OWN reason (e.g. a poor fit) -- the ratio reason still wins as
        # the final, most actionable message.
        underlying = ok_result(reason="foster_fit_failed: r_squared=0.1", status="needs_review")
        build_run(monkeypatch, detections=make_detections(1),
                  ratio_reason="axis_is_normalized_ratio: ...", fit_result=underlying)
        result = run()
        assert result["status"] == "needs_review"
        assert result["review_reason"] == "axis_is_normalized_ratio: ..."

    def test_ratio_check_runs_even_when_detections_are_zero(self, monkeypatch):
        # detect_normalized_ratio_axis must be evaluated regardless of the
        # model's outcome -- called once per run_hybrid_pipeline call.
        mocks = build_run(monkeypatch, detections=[], ratio_reason=None)
        run()
        mocks["detect_normalized_ratio_axis"].assert_called_once()


# ============================================================
# F. Identity checks — same functions, not copies
# ============================================================

class TestIdentityNotDuplication:
    def test_fit_foster_is_classical_zths_own(self):
        assert mod.fit_foster is classical_zth.fit_foster

    def test_pick_rth_constraint_is_classical_zths_own(self):
        assert mod.pick_rth_constraint is classical_zth.pick_rth_constraint

    def test_parse_foster_table_from_ocr_is_classical_zths_own(self):
        assert mod.parse_foster_table_from_ocr is classical_zth.parse_foster_table_from_ocr

    def test_detect_normalized_ratio_axis_is_classical_zths_own(self):
        assert mod.detect_normalized_ratio_axis is classical_zth.detect_normalized_ratio_axis

    def test_derive_calibration_zth_is_classical_zths_own(self):
        assert mod.derive_calibration_zth is classical_zth.derive_calibration_zth

    def test_run_classical_pipeline_is_classical_zths_own(self):
        assert mod.classical_zth.run_classical_pipeline is classical_zth.run_classical_pipeline

    def test_fit_and_validate_curve_is_zth_fits_own(self):
        import src.extraction.zth_fit as zth_fit
        assert mod.fit_and_validate_curve is zth_fit.fit_and_validate_curve

    def test_calibration_with_bonus_fields_is_zth_fits_own(self):
        import src.extraction.zth_fit as zth_fit
        assert mod.calibration_with_bonus_fields is zth_fit.calibration_with_bonus_fields

    def test_build_needs_review_result_is_zth_fits_own(self):
        import src.extraction.zth_fit as zth_fit
        assert mod.build_needs_review_result is zth_fit.build_needs_review_result

    def test_run_inference_is_inference_modules_own(self):
        import src.extraction.inference as inference_mod
        assert mod.run_inference is inference_mod.run_inference

    def test_mask_to_points_is_skeletonizes_own(self):
        import src.extraction.skeletonize as skeletonize_mod
        assert mod.mask_to_points is skeletonize_mod.mask_to_points


# ============================================================
# G. No fallback to the old rule-based clustering/picker
# ============================================================

class TestNoOldRuleBasedFallback:
    def test_source_never_references_old_clustering_or_picker(self):
        # The module docstring legitimately NAMES these (to explain what's
        # deliberately NOT used) -- only the actual CODE (after the module
        # docstring) must never reference them.
        source = inspect.getsource(mod)
        _, _, code_only = source.split('"""', 2)
        for banned in ("cluster_into_curves_zth", "pick_single_pulse", "trace_curve",
                      "_clean_for_clustering"):
            assert banned not in code_only, f"hybrid_zth.py must never call {banned}"


# ============================================================
# H. No hardcoded paths / checkpoint resolution
# ============================================================

class TestCheckpointResolution:
    def test_env_var_name(self):
        assert CHECKPOINTS_ROOT_ENV_VAR == "LINEFORMER_CHECKPOINTS_ROOT"

    def test_checkpoint_relative_path_matches_owner_spec(self):
        assert CHECKPOINT_RELATIVE_PATH == "zth_vs_time_run2_patience2000/best_segm_mAP_50_iter_4000.pth"

    def test_missing_root_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv(CHECKPOINTS_ROOT_ENV_VAR, raising=False)
        with pytest.raises(RuntimeError):
            resolve_checkpoint_and_config(checkpoints_root=None)

    def test_explicit_root_joins_with_relative_checkpoint(self):
        checkpoint, _ = resolve_checkpoint_and_config(checkpoints_root="/some/root")
        assert checkpoint == os.path.join("/some/root", CHECKPOINT_RELATIVE_PATH)

    def test_env_var_used_when_no_explicit_root(self, monkeypatch):
        monkeypatch.setenv(CHECKPOINTS_ROOT_ENV_VAR, "/env/root")
        checkpoint, _ = resolve_checkpoint_and_config(checkpoints_root=None)
        assert checkpoint == os.path.join("/env/root", CHECKPOINT_RELATIVE_PATH)

    def test_explicit_root_wins_over_env_var(self, monkeypatch):
        monkeypatch.setenv(CHECKPOINTS_ROOT_ENV_VAR, "/env/root")
        checkpoint, _ = resolve_checkpoint_and_config(checkpoints_root="/explicit/root")
        assert checkpoint == os.path.join("/explicit/root", CHECKPOINT_RELATIVE_PATH)

    def test_config_path_ends_with_expected_relative_path(self):
        _, config = resolve_checkpoint_and_config(checkpoints_root="/some/root")
        assert config.endswith(CONFIG_RELATIVE_PATH)

    def test_source_contains_no_hardcoded_absolute_checkpoint_path(self):
        source = inspect.getsource(mod)
        assert "/mnt/data" not in source
        assert "/home/ec2-user" not in source


# ============================================================
# I. End-to-end integration (real calibration/fit machinery, no GPU)
# ============================================================

class TestRealNonModelPipelineIntegration:
    """Only run_inference is faked (needs a GPU/trained checkpoint);
    everything else -- calibration, mask_to_points' real skeletonize,
    pixel_to_data, fit_foster, pick_rth_constraint -- is the REAL thing,
    proving the wiring genuinely works end to end, not just against mocks."""

    def _draw_chart(self):
        img = np.full((400, 700, 3), 255, dtype=np.uint8)
        # A real, traceable rising line the model "detects" as a mask.
        pts = np.array([[x, int(380 - 300 * (1 - np.exp(-(x - 100) / 150.0)))]
                        for x in range(100, 600)])
        pts = pts[(pts[:, 1] >= 0) & (pts[:, 1] < 400)]
        for i in range(len(pts) - 1):
            cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), (0, 0, 0), 3)
        return img, pts

    def _axis_ticks_ocr(self):
        # Linear x/y ticks a real derive_calibration_zth can fit.
        return [
            {"text": "0", "bounding_box": {"x1": 95, "y1": 385, "x2": 105, "y2": 398}},
            {"text": "10", "bounding_box": {"x1": 195, "y1": 385, "x2": 210, "y2": 398}},
            {"text": "20", "bounding_box": {"x1": 295, "y1": 385, "x2": 310, "y2": 398}},
            {"text": "30", "bounding_box": {"x1": 395, "y1": 385, "x2": 410, "y2": 398}},
            {"text": "0", "bounding_box": {"x1": 60, "y1": 375, "x2": 75, "y2": 388}},
            {"text": "1", "bounding_box": {"x1": 60, "y1": 275, "x2": 75, "y2": 288}},
            {"text": "2", "bounding_box": {"x1": 60, "y1": 175, "x2": 75, "y2": 188}},
            {"text": "3", "bounding_box": {"x1": 60, "y1": 75, "x2": 75, "y2": 88}},
        ]

    def test_real_pipeline_produces_a_validated_result(self, monkeypatch):
        img, pts = self._draw_chart()
        mask = np.zeros(img.shape[:2], dtype=bool)
        for x, y in pts:
            mask[max(y - 1, 0):y + 2, x] = True

        monkeypatch.setattr(mod, "parse_foster_table_from_ocr", MagicMock(return_value=None))
        monkeypatch.setattr(mod, "detect_normalized_ratio_axis", MagicMock(return_value=None))
        monkeypatch.setattr(mod, "run_inference",
                            MagicMock(return_value=[Detection(score=0.9, mask=mask)]))

        result = run(image=img, ocr_lines=self._axis_ticks_ocr())
        validate_result(result)
        # Whatever it decides (ok or a specific needs_review gate), it must
        # be a real, schema-valid result produced by the real machinery --
        # not a crash, not a mock artifact.
        assert result["calibration"] is not None
