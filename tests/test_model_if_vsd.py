"""Tests for src.extraction.model_if_vsd — written FIRST (CLAUDE.md §2, red
phase). RED PHASE ONLY — the module under test does not exist yet.

The if_vs_vsd analogue of classical_vgsth.py's rdson-style wrapper, but on
the MODEL (LineFormer) side rather than classical OpenCV detection:
``run_model_pipeline(device, curve_type, image_path, ocr_lines, img_w,
img_h, model, score_thr, expected_curve_count) -> Dict[str, Any]``.
Detection is :func:`src.extraction.inference.run_inference` (the same
model-inference primitive :func:`src.extraction.pipeline.run_pipeline`
itself calls — reused, never reimplemented). Naming/counting is
:mod:`src.extraction.naming.if_vs_vsd`'s ``count_expected_curves``/
``name_curves_by_labels``. The final result is built by the frozen
:func:`src.extraction.pipeline.process_detections`, never reimplemented.

CORE LOGIC under test — the SAME expected-vs-detected count comparison
already proven for vgsth_vs_tj's classical wrapper (see
test_classical_vgsth.py), applied on the model side: given
``N = count_expected_curves(ocr_lines)`` and ``D = len(detections)``,

    D == 0                    -> quarantine, "no curves found"
    N is not None and D < N   -> quarantine, "likely merged at a crossing"
    N is not None and D > N   -> quarantine, "stray component / missed label"
    N is None and D > 1       -> quarantine, "no usable labels"
    (N is None and D == 1) or (D == N):
        names = name_curves_by_labels(point_lists, ocr_lines)
        names is None -> quarantine, "ambiguous naming"
        else          -> process_detections(..., expected_curve_count=D),
                          then curve_name fields are overridden with `names`
                          unconditionally (if_vs_vsd, like vgsth_vs_tj, has
                          no meaningful position-only naming to start from)

Every dependency (inference, counting, naming, the frozen core) is MOCKED
here — this file tests ONLY the wrapper's own routing/decision logic, not
the internals of already-separately-tested functions (inference.py's own
suite, naming/if_vs_vsd's own suite, pipeline.py's own suite). No GPU, no
network, no real images.

Note ``ExtractionSpec.expected_curve_count`` for if_vs_vsd is ``None`` —
mirrors vgsth_vs_tj's own registry field (which its classical wrapper
never reads directly either): this wrapper's REAL expected count always
comes from ``count_expected_curves(ocr_lines)``, computed dynamically per
chart. ``run_model_pipeline`` accepts
``expected_curve_count`` as a parameter purely for call-signature
symmetry with the generic ``run_pipeline`` (so live_stages.py's dispatch
can call either uniformly) — it is never read inside the function body.
"""
import logging
import re
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.extraction.inference import Detection
from src.extraction.schema import build_result, validate_result

import src.extraction.model_if_vsd as mod
from src.extraction.model_if_vsd import run_model_pipeline


# ---------------------------------------------------------------- fixtures

IMG_H, IMG_W = 50, 80


def make_ocr_lines():
    return [{"text": "25°C", "bounding_box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}]


def make_detections(n, score=0.9):
    return [Detection(score=score, mask=np.zeros((4, 4), dtype=bool)) for _ in range(n)]


def ok_result(n=1, names=None, units="V"):
    names = names or [f"placeholder_{i}" for i in range(n)]
    return build_result(
        device="DEV1", curve_type="if_vs_vsd", source_image="fig.png",
        status="ok", review_reason=None, duplicates_removed=0,
        calibration={"x_slope": 1.0, "x_intercept": 0.0, "y_slope": 1.0,
                    "y_intercept": 0.0, "x_log": False, "y_log": False},
        curves=[{"curve_name": name, "confidence": 0.9,
                 "points": [{"x": 1.0, "y": 2.0}]} for name in names],
        units=units,
    )


def build_run(
    monkeypatch,
    detections=None,
    expected_count=None,
    resolved_names=None,
    pipeline_result=None,
):
    """Patch every dependency model_if_vsd's run_model_pipeline calls, with
    sensible defaults; returns the mocks for assertion."""
    detections = [] if detections is None else list(detections)

    inference_mock = MagicMock(return_value=detections)
    monkeypatch.setattr(mod, "run_inference", inference_mock)

    point_lists = [[(0.0, float(i))] for i in range(len(detections))]
    mask_to_points_mock = MagicMock(side_effect=list(point_lists))
    monkeypatch.setattr(mod, "mask_to_points", mask_to_points_mock)

    count_mock = MagicMock(return_value=expected_count)
    monkeypatch.setattr(mod, "count_expected_curves", count_mock)

    names_mock = MagicMock(return_value=resolved_names)
    monkeypatch.setattr(mod, "name_curves_by_labels", names_mock)

    default_result = pipeline_result if pipeline_result is not None else ok_result(
        n=len(detections), names=resolved_names or [f"if_{i}" for i in range(len(detections))])
    pipeline_mock = MagicMock(return_value=default_result)
    monkeypatch.setattr(mod, "process_detections", pipeline_mock)

    return {
        "run_inference": inference_mock, "mask_to_points": mask_to_points_mock,
        "count_expected_curves": count_mock, "name_curves_by_labels": names_mock,
        "process_detections": pipeline_mock,
    }


def run(device="DEV1", curve_type="if_vs_vsd", image_path="fig.png",
        ocr_lines=None, img_w=IMG_W, img_h=IMG_H, model=None, score_thr=0.5,
        expected_curve_count=None):
    return run_model_pipeline(
        device=device, curve_type=curve_type, image_path=image_path,
        ocr_lines=ocr_lines if ocr_lines is not None else make_ocr_lines(),
        img_w=img_w, img_h=img_h, model=model if model is not None else object(),
        score_thr=score_thr, expected_curve_count=expected_curve_count,
    )


# =================================================================
# A. Detection routing (model inference delegation)
# =================================================================

class TestDetectionRouting:
    def test_run_inference_called_with_model_image_path_and_score_thr(self, monkeypatch):
        mocks = build_run(monkeypatch, detections=make_detections(1), expected_count=1,
                          resolved_names=["if_25C"])
        sentinel_model = object()
        run(model=sentinel_model, image_path="figures/fig.png", score_thr=0.7)
        mocks["run_inference"].assert_called_once_with(sentinel_model, "figures/fig.png", score_thr=0.7)

    def test_run_inference_is_the_literal_inference_module_object(self):
        # No monkeypatching -- confirms model_if_vsd imports (not
        # reimplements) the shared inference primitive.
        import src.extraction.inference as inference_mod
        assert mod.run_inference is inference_mod.run_inference


# =================================================================
# B. Happy path -- single curve
# =================================================================

class TestSingleCurveHappyPath:
    def test_one_curve_no_labels_named_if_ok(self, monkeypatch):
        build_run(monkeypatch, detections=make_detections(1), expected_count=None,
                  resolved_names=["if"], pipeline_result=ok_result(n=1, names=["placeholder"]))
        result = run()
        assert result["status"] == "ok"
        assert result["curves"][0]["curve_name"] == "if"

    def test_one_curve_with_temp_label_present_still_if(self, monkeypatch):
        build_run(monkeypatch, detections=make_detections(1), expected_count=1,
                  resolved_names=["if"], pipeline_result=ok_result(n=1, names=["placeholder"]))
        result = run(ocr_lines=[{"text": "25°C",
                                 "bounding_box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}])
        assert result["status"] == "ok"
        assert result["curves"][0]["curve_name"] == "if"


# =================================================================
# C. Happy path -- matched count, 2 curves
# =================================================================

class TestTwoCurveHappyPath:
    def test_two_curves_two_distinct_temps_ok(self, monkeypatch):
        names = ["if_25C", "if_175C"]
        build_run(monkeypatch, detections=make_detections(2), expected_count=2,
                  resolved_names=names, pipeline_result=ok_result(n=2, names=["p0", "p1"]))
        result = run()
        assert result["status"] == "ok"
        assert [c["curve_name"] for c in result["curves"]] == names


# =================================================================
# D. Happy path -- matched count, 4 curves (clean, no compound label)
# =================================================================

class TestFourCurveHappyPath:
    def test_four_curves_four_distinct_temps_ok(self, monkeypatch):
        names = ["if_-40C", "if_25C", "if_125C", "if_175C"]
        build_run(monkeypatch, detections=make_detections(4), expected_count=4,
                  resolved_names=names, pipeline_result=ok_result(n=4, names=["p0", "p1", "p2", "p3"]))
        result = run()
        assert result["status"] == "ok"
        assert [c["curve_name"] for c in result["curves"]] == names


# =================================================================
# E. Quarantine -- no usable labels
# =================================================================

class TestQuarantineNoUsableLabels:
    def test_no_labels_multi_curve_quarantines(self, monkeypatch):
        mocks = build_run(monkeypatch, detections=make_detections(2), expected_count=None)
        result = run()
        assert result["status"] == "needs_review"
        assert "label" in result["review_reason"].lower()
        mocks["process_detections"].assert_called_once()
        assert result["calibration"] is not None

    def test_compound_label_ambiguity_takes_same_path(self, monkeypatch):
        # count_expected_curves returning None because of a compound
        # temp+percentile label is indistinguishable to the wrapper from
        # "no labels at all" -- same code path, same message class.
        mocks = build_run(monkeypatch, detections=make_detections(3), expected_count=None)
        result = run()
        assert result["status"] == "needs_review"
        assert "label" in result["review_reason"].lower()
        mocks["process_detections"].assert_called_once()
        assert result["calibration"] is not None


# =================================================================
# F. Quarantine -- detected < N (crossing/merge safety net)
# =================================================================

class TestQuarantineFewerThanExpected:
    def test_two_labels_one_detected_quarantines_crossing_reason(self, monkeypatch):
        mocks = build_run(monkeypatch, detections=make_detections(1), expected_count=2)
        result = run()
        assert result["status"] == "needs_review"
        reason = result["review_reason"].lower()
        assert "crossing" in reason or "merge" in reason
        assert "1" in result["review_reason"] and "2" in result["review_reason"]
        mocks["process_detections"].assert_called_once()
        assert result["calibration"] is not None

    def test_four_labels_three_detected_generalizes(self, monkeypatch):
        mocks = build_run(monkeypatch, detections=make_detections(3), expected_count=4)
        result = run()
        assert result["status"] == "needs_review"
        reason = result["review_reason"].lower()
        assert "crossing" in reason or "merge" in reason
        mocks["process_detections"].assert_called_once()
        assert result["calibration"] is not None

    def test_zero_curves_detected_quarantines_with_distinct_message(self, monkeypatch):
        for n_value in (2, None):
            mocks = build_run(monkeypatch, detections=[], expected_count=n_value)
            result = run()
            assert result["status"] == "needs_review"
            reason = result["review_reason"].lower()
            assert "no curve" in reason or "0 curve" in reason or "found" in reason
            assert "crossing" not in reason and "merge" not in reason
            mocks["process_detections"].assert_called_once()
            assert result["calibration"] is not None


# =================================================================
# G. Quarantine -- detected > N
# =================================================================

class TestQuarantineMoreThanExpected:
    def test_two_labels_three_detected_quarantines_distinct_reason(self, monkeypatch):
        mocks = build_run(monkeypatch, detections=make_detections(3), expected_count=2)
        result = run()
        assert result["status"] == "needs_review"
        reason = result["review_reason"].lower()
        assert "stray" in reason or "missed" in reason or "unexpected" in reason
        assert "crossing" not in reason and "merge" not in reason
        mocks["process_detections"].assert_called_once()
        assert result["calibration"] is not None

    def test_exceeds_by_one_still_quarantines_no_leniency(self, monkeypatch):
        mocks = build_run(monkeypatch, detections=make_detections(5), expected_count=4)
        result = run()
        assert result["status"] == "needs_review"
        mocks["process_detections"].assert_called_once()
        assert result["calibration"] is not None


# =================================================================
# H. Quarantine -- matched count, naming still fails
# =================================================================

class TestQuarantineNamingAmbiguous:
    def test_matched_count_but_naming_returns_none_quarantines(self, monkeypatch):
        mocks = build_run(monkeypatch, detections=make_detections(2), expected_count=2,
                          resolved_names=None)
        result = run()
        assert result["status"] == "needs_review"
        mocks["process_detections"].assert_called_once()
        assert result["calibration"] is not None

    def test_naming_ambiguous_reason_distinguishable_from_count_mismatch(self, monkeypatch):
        build_run(monkeypatch, detections=make_detections(2), expected_count=2,
                  resolved_names=None)
        ambiguous_reason = run()["review_reason"].lower()

        build_run(monkeypatch, detections=make_detections(2), expected_count=3)
        mismatch_reason = run()["review_reason"].lower()

        assert ambiguous_reason != mismatch_reason
        assert "crossing" not in ambiguous_reason and "merge" not in ambiguous_reason
        assert ("ambiguous" in ambiguous_reason or "tie" in ambiguous_reason
                or "naming" in ambiguous_reason)


# =================================================================
# I. Units
# =================================================================

class TestUnits:
    def test_v_token_resolves_units_v_ok(self, monkeypatch):
        build_run(monkeypatch, detections=make_detections(1), expected_count=1,
                  resolved_names=["if"],
                  pipeline_result=ok_result(n=1, names=["placeholder"], units="V"))
        result = run()
        assert result["status"] == "ok"
        assert result["units"] == "V"

    def test_no_unit_token_falls_to_generic_units_undetected(self, monkeypatch):
        undetected = build_result(
            device="DEV1", curve_type="if_vs_vsd", source_image="fig.png",
            status="needs_review", review_reason="units_undetected",
            duplicates_removed=0,
            calibration={"x_slope": 1.0, "x_intercept": 0.0, "y_slope": 1.0,
                        "y_intercept": 0.0, "x_log": False, "y_log": False},
            curves=[{"curve_name": "placeholder", "confidence": 0.9,
                     "points": [{"x": 1.0, "y": 2.0}]}],
            units=None,
        )
        build_run(monkeypatch, detections=make_detections(1), expected_count=1,
                  resolved_names=["if"], pipeline_result=undetected)
        result = run()
        assert result["status"] == "needs_review"
        assert result["review_reason"] == "units_undetected"
        assert result["units"] is None


# =================================================================
# J. Plausibility
# =================================================================

class TestPlausibility:
    def test_implausible_calibration_passes_through_from_frozen_core(self, monkeypatch):
        implausible = build_result(
            device="DEV1", curve_type="if_vs_vsd", source_image="fig.png",
            status="needs_review",
            review_reason="implausible_calibration: x values span -120..250, "
                          "outside the plausible if_vs_vsd range -75..200",
            duplicates_removed=0,
            calibration={"x_slope": 1.0, "x_intercept": 0.0, "y_slope": 1.0,
                        "y_intercept": 0.0, "x_log": False, "y_log": False},
            curves=[{"curve_name": "placeholder", "confidence": 0.9,
                     "points": [{"x": -120.0, "y": 2.0}, {"x": 250.0, "y": 2.0}]}],
            units=None,
        )
        build_run(monkeypatch, detections=make_detections(1), expected_count=1,
                  resolved_names=["if"], pipeline_result=implausible)
        result = run()
        assert result["status"] == "needs_review"
        assert "implausible_calibration" in result["review_reason"]


# =================================================================
# K. Delegation / zero duplication
# =================================================================

class TestDelegation:
    def test_process_detections_is_the_frozen_core_not_reimplemented(self, monkeypatch):
        mocks = build_run(monkeypatch, detections=make_detections(2), expected_count=2,
                          resolved_names=["if_25C", "if_175C"])
        ocr_lines = make_ocr_lines()
        run(ocr_lines=ocr_lines, image_path="fig.png")
        mocks["process_detections"].assert_called_once()
        args, kwargs = mocks["process_detections"].call_args
        all_args = list(args) + list(kwargs.values())
        assert "DEV1" in all_args
        assert "if_vs_vsd" in all_args
        assert "fig.png" in all_args
        assert float(IMG_W) in all_args
        assert float(IMG_H) in all_args

    def test_process_detections_is_the_literal_pipeline_module_object(self):
        import src.extraction.pipeline as pipeline_mod
        assert mod.process_detections is pipeline_mod.process_detections


# =================================================================
# L. Schema / output contract
# =================================================================

class TestSchemaContract:
    def test_result_shape_matches_stage6_stage7_contract(self, monkeypatch):
        build_run(monkeypatch, detections=make_detections(2), expected_count=2,
                  resolved_names=["if_25C", "if_175C"],
                  pipeline_result=ok_result(n=2, names=["p0", "p1"], units="V"))
        result = run()
        validate_result(result)  # raises if anything is off-schema
        assert set(result.keys()) == {
            "device", "curve_type", "source_image", "status", "review_reason",
            "duplicates_removed", "calibration", "curves", "units",
        }

    def test_quarantined_result_carries_curves_not_an_empty_shell(self, monkeypatch):
        build_run(monkeypatch, detections=make_detections(2), expected_count=3)
        result = run()
        assert result["status"] == "needs_review"
        assert len(result["curves"]) > 0
        assert result["calibration"] is not None


# =================================================================
# M. Input validation
# =================================================================

class TestInputValidation:
    def test_malformed_ocr_line_missing_bounding_box_raises(self, monkeypatch):
        monkeypatch.setattr(mod, "run_inference", MagicMock(return_value=make_detections(1)))
        monkeypatch.setattr(mod, "count_expected_curves",
                            MagicMock(side_effect=KeyError("bounding_box")))
        with pytest.raises((KeyError, ValueError)):
            run(ocr_lines=[{"text": "25°C"}])  # no bounding_box


# =================================================================
# N. Logging
# =================================================================

class TestLogging:
    def test_quarantine_branches_log_distinct_identifiable_reasons(self, monkeypatch, caplog):
        caplog.set_level(logging.INFO)
        build_run(monkeypatch, detections=make_detections(2), expected_count=3)
        run()
        assert any("crossing" in r.message.lower() or "merge" in r.message.lower()
                  for r in caplog.records)

        caplog.clear()
        build_run(monkeypatch, detections=make_detections(3), expected_count=2)
        run()
        assert any("stray" in r.message.lower() or "missed" in r.message.lower()
                  or "unexpected" in r.message.lower() for r in caplog.records)


# =================================================================
# O. Placeholder-leak safety net
# =================================================================
#
# The naming registry carries a throwaway placeholder for if_vs_vsd
# ("curve_0", "curve_1", ... — see src/extraction/naming/__init__.py) that
# exists ONLY so process_detections doesn't KeyError; it carries no real
# naming authority. This wrapper is REQUIRED to override it on every
# ok-status path.

_OK_SCENARIOS = [
    dict(n=1, expected_count=None, resolved_names=["if"]),
    dict(n=1, expected_count=1, resolved_names=["if"],
        ocr_lines=[{"text": "25°C",
                    "bounding_box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}]),
    dict(n=2, expected_count=2, resolved_names=["if_25C", "if_175C"]),
    dict(n=4, expected_count=4, resolved_names=[
        "if_-40C", "if_25C", "if_125C", "if_175C"]),
]


class TestPlaceholderNeverLeaksIntoTrustedResult:
    def test_no_ok_result_curve_name_matches_placeholder_pattern(self, monkeypatch):
        placeholder_pattern = re.compile(r"^curve_\d+$")
        for scenario in _OK_SCENARIOS:
            n = scenario["n"]
            placeholder_names = [f"curve_{i}" for i in range(n)]
            build_run(
                monkeypatch, detections=make_detections(n),
                expected_count=scenario["expected_count"],
                resolved_names=scenario["resolved_names"],
                pipeline_result=ok_result(n=n, names=placeholder_names),
            )
            result = run(ocr_lines=scenario.get("ocr_lines"))
            assert result["status"] == "ok", f"scenario {scenario} did not reach ok"
            for curve in result["curves"]:
                assert not placeholder_pattern.match(curve["curve_name"]), (
                    f"placeholder name {curve['curve_name']!r} leaked into an "
                    f"ok result for scenario {scenario}"
                )
