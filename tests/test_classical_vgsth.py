"""Tests for src.extraction.classical_vgsth — written FIRST (CLAUDE.md §2,
red phase). RED PHASE ONLY — the module under test does not exist yet.

The vgsth_vs_tj analogue of classical.py's rdson wrapper:
``run_classical_pipeline(device, curve_type, source_image, image,
ocr_lines) -> Dict[str, Any]``. Detection is the SAME shared functions
rdson's wrapper uses (:func:`src.extraction.curve_detection.detect_curve_classical`
color-first, :func:`~...curve_detection.detect_curve_monochrome` fallback,
default tunables — no vgsth-specific override without evidence). Naming/
counting is :mod:`src.extraction.naming.vgsth_vs_tj`'s
``count_expected_curves``/``name_curves_by_labels``. The final result is
built by the frozen :func:`src.extraction.pipeline.process_detections`,
never reimplemented.

CORE LOGIC under test — the expected-vs-detected count comparison (Option
A's actual safety net against silently merging curves at a crossing):
given ``N = count_expected_curves(ocr_lines)`` and ``D = len(detections)``,

    D == 0                    -> quarantine, "no curves found" (distinct
                                  message, fires regardless of N)
    N is not None and D < N   -> quarantine, "likely merged at a crossing"
    N is not None and D > N   -> quarantine, "stray component / missed label"
    N is None and D > 1       -> quarantine, "no usable labels"
    (N is None and D == 1) or (D == N):
        names = name_curves_by_labels(point_lists, ocr_lines)
        names is None -> quarantine, "ambiguous naming" (distinct from the
                          count-mismatch reasons above — matched count,
                          different failure)
        else          -> process_detections(..., expected_curve_count=D),
                          then curve_name fields are overridden with
                          `names` (mirrors rdson's own post-hoc label
                          override, but unconditional here since vgsth has
                          no meaningful position-only naming to start from)

Every dependency (detection, counting, naming, the frozen core) is MOCKED
here — this file tests ONLY the wrapper's own routing/decision logic, not
the internals of already-separately-tested functions (curve_detection.py's
own suite, naming/vgsth_vs_tj's own suite, pipeline.py's own suite). No
GPU, no network, no real images.

============================================================================
FLAGGING FOR OWNER REVIEW — two dependencies this module needs that do NOT
currently exist in the codebase, discovered while designing these tests:

1. PLAUSIBILITY_SPECS["vgsth_vs_tj"] (src/extraction/pipeline.py) — does
   NOT exist yet (confirmed by reading pipeline.py directly: only
   "capacitance_vs_vds" and "rdson_vs_tj" are registered). Per instruction,
   J.22/23 below are written ASSUMING it will exist (a separate approved
   frozen-file addition, rdson's exact x_range=(-75.0, 200.0), no y_range)
   — but since it doesn't exist yet, those two tests mock
   `process_detections`'s return value directly rather than exercising the
   real plausibility gate, so they don't depend on that addition landing
   first. The real gate itself is pipeline.py's own test responsibility,
   not this wrapper's.

2. The naming registry (src/extraction/naming/__init__.py's
   `_NAMING_REGISTRY`) has no "vgsth_vs_tj" entry, and `process_detections`
   unconditionally calls `get_naming_fn(curve_type)` internally (raising
   KeyError if unregistered) BEFORE this wrapper would ever get a chance to
   apply its own label-based override. Unlike rdson_vs_tj (registered with
   a real position-based `name_curves`), vgsth has no meaningful
   position-only naming — its only real naming function,
   `name_curves_by_labels`, needs `ocr_lines` as a second argument, which
   the registry's `NamingFn` type (`Callable[[Sequence[Sequence[Point]]],
   List[str]]`) doesn't support. This means `process_detections` cannot
   currently succeed for curve_type="vgsth_vs_tj" AT ALL, in any code path,
   until *something* is registered for it (most likely a throwaway
   placeholder naming function that this wrapper always overrides
   afterward — mirroring rdson's own override pattern, but starting from
   a meaningless default instead of a meaningful one). That is an explicit
   architecture decision this red-phase session does NOT make: every test
   below mocks `process_detections` itself rather than assuming any
   specific registry entry, so it stays valid regardless of how that gets
   resolved. Flagging so it's resolved (or explicitly deferred) before the
   green/implementation phase — the wrapper cannot work end-to-end without it.
============================================================================
"""
import logging
import re
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from src.extraction.inference import Detection
from src.extraction.schema import build_result, validate_result

import src.extraction.classical_vgsth as mod
from src.extraction.classical_vgsth import run_classical_pipeline


# ---------------------------------------------------------------- fixtures

IMG_H, IMG_W = 50, 80


def make_image():
    return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)


def make_ocr_lines():
    return [{"text": "max", "bounding_box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}]


def make_detections(n, score=0.9):
    return [Detection(score=score, mask=np.zeros((4, 4), dtype=bool)) for _ in range(n)]


def ok_result(n=1, names=None, units="V"):
    names = names or [f"placeholder_{i}" for i in range(n)]
    return build_result(
        device="DEV1", curve_type="vgsth_vs_tj", source_image="fig.png",
        status="ok", review_reason=None, duplicates_removed=0,
        calibration={"x_slope": 1.0, "x_intercept": 0.0, "y_slope": 1.0,
                    "y_intercept": 0.0, "x_log": False, "y_log": False},
        curves=[{"curve_name": name, "confidence": 0.9,
                 "points": [{"x": 1.0, "y": 2.0}]} for name in names],
        units=units,
    )


def build_run(
    monkeypatch,
    color_detections=None,
    mono_detections=None,
    expected_count=None,
    resolved_names=None,
    pipeline_result=None,
):
    """Patch every dependency classical_vgsth's run_classical_pipeline
    calls, with sensible defaults; returns the mocks for assertion.
    """
    color_detections = [] if color_detections is None else list(color_detections)
    mono_detections = [] if mono_detections is None else list(mono_detections)
    used = color_detections if color_detections else mono_detections

    color_mock = MagicMock(return_value=color_detections)
    monkeypatch.setattr(mod, "detect_curve_classical", color_mock)
    mono_mock = MagicMock(return_value=mono_detections)
    monkeypatch.setattr(mod, "detect_curve_monochrome", mono_mock)

    point_lists = [[(0.0, float(i))] for i in range(len(used))]
    mask_to_points_mock = MagicMock(side_effect=list(point_lists))
    monkeypatch.setattr(mod, "mask_to_points", mask_to_points_mock)

    count_mock = MagicMock(return_value=expected_count)
    monkeypatch.setattr(mod, "count_expected_curves", count_mock)

    names_mock = MagicMock(return_value=resolved_names)
    monkeypatch.setattr(mod, "name_curves_by_labels", names_mock)

    default_result = pipeline_result if pipeline_result is not None else ok_result(
        n=len(used), names=resolved_names or [f"vgsth_{i}" for i in range(len(used))])
    pipeline_mock = MagicMock(return_value=default_result)
    monkeypatch.setattr(mod, "process_detections", pipeline_mock)

    return {
        "detect_curve_classical": color_mock, "detect_curve_monochrome": mono_mock,
        "mask_to_points": mask_to_points_mock, "count_expected_curves": count_mock,
        "name_curves_by_labels": names_mock, "process_detections": pipeline_mock,
    }


def run(device="DEV1", curve_type="vgsth_vs_tj", source_image="fig.png",
        image=None, ocr_lines=None):
    return run_classical_pipeline(
        device=device, curve_type=curve_type, source_image=source_image,
        image=image if image is not None else make_image(),
        ocr_lines=ocr_lines if ocr_lines is not None else make_ocr_lines(),
    )


# =================================================================
# A. Detection routing
# =================================================================

class TestDetectionRouting:
    def test_color_curves_present_mono_never_invoked(self, monkeypatch):
        mocks = build_run(monkeypatch, color_detections=make_detections(2),
                          expected_count=2, resolved_names=["vgsth_max", "vgsth_typ"])
        run()
        mocks["detect_curve_classical"].assert_called_once()
        mocks["detect_curve_monochrome"].assert_not_called()

    def test_no_chromatic_pixels_falls_back_to_mono(self, monkeypatch):
        mocks = build_run(monkeypatch, color_detections=[], mono_detections=make_detections(1),
                          expected_count=1, resolved_names=["vgsth"])
        image, ocr_lines = make_image(), make_ocr_lines()
        run(image=image, ocr_lines=ocr_lines)
        mocks["detect_curve_classical"].assert_called_once()
        mocks["detect_curve_monochrome"].assert_called_once()

    def test_default_tunables_no_vgsth_specific_override(self, monkeypatch):
        # No keyword overrides without evidence to justify one -- exact
        # call args, no extra kwargs.
        mocks = build_run(monkeypatch, color_detections=[], mono_detections=make_detections(1),
                          expected_count=1, resolved_names=["vgsth"])
        image, ocr_lines = make_image(), make_ocr_lines()
        run(image=image, ocr_lines=ocr_lines)
        mocks["detect_curve_classical"].assert_called_once_with(image)
        mocks["detect_curve_monochrome"].assert_called_once_with(image, ocr_lines)


# =================================================================
# B. Happy path -- single curve
# =================================================================

class TestSingleCurveHappyPath:
    def test_one_curve_no_labels_named_vgsth_ok(self, monkeypatch):
        build_run(monkeypatch, color_detections=make_detections(1), expected_count=None,
                  resolved_names=["vgsth"], pipeline_result=ok_result(n=1, names=["placeholder"]))
        result = run()
        assert result["status"] == "ok"
        assert result["curves"][0]["curve_name"] == "vgsth"

    def test_one_curve_with_id_label_present_still_vgsth(self, monkeypatch):
        # count_expected_curves resolving N=1 from the I_D= label (matching
        # Stage 2's own D.17 rule) takes the D==N path instead of the
        # N-is-None path -- both must converge on the same "vgsth" result.
        build_run(monkeypatch, color_detections=make_detections(1), expected_count=1,
                  resolved_names=["vgsth"], pipeline_result=ok_result(n=1, names=["placeholder"]))
        result = run(ocr_lines=[{"text": "I_D = 250uA",
                                 "bounding_box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}])
        assert result["status"] == "ok"
        assert result["curves"][0]["curve_name"] == "vgsth"


# =================================================================
# C. Happy path -- matched count, band scheme
# =================================================================

class TestBandSchemeHappyPath:
    def test_three_curves_three_band_labels_ok(self, monkeypatch):
        names = ["vgsth_max", "vgsth_typ", "vgsth_min"]
        build_run(monkeypatch, color_detections=make_detections(3), expected_count=3,
                  resolved_names=names, pipeline_result=ok_result(n=3, names=["p0", "p1", "p2"]))
        result = run()
        assert result["status"] == "ok"
        assert [c["curve_name"] for c in result["curves"]] == names

    def test_two_curves_two_band_labels_ok(self, monkeypatch):
        names = ["vgsth_max", "vgsth_typ"]
        build_run(monkeypatch, color_detections=make_detections(2), expected_count=2,
                  resolved_names=names, pipeline_result=ok_result(n=2, names=["p0", "p1"]))
        result = run()
        assert result["status"] == "ok"
        assert [c["curve_name"] for c in result["curves"]] == names


# =================================================================
# D. Happy path -- matched count, current-value scheme
# =================================================================

class TestCurrentValueSchemeHappyPath:
    def test_two_curves_two_distinct_id_values_ok(self, monkeypatch):
        names = ["vgsth_id_250uA", "vgsth_id_1000uA"]
        build_run(monkeypatch, color_detections=make_detections(2), expected_count=2,
                  resolved_names=names, pipeline_result=ok_result(n=2, names=["p0", "p1"]))
        result = run()
        assert result["status"] == "ok"
        assert [c["curve_name"] for c in result["curves"]] == names

    def test_four_curves_four_distinct_id_values_ok(self, monkeypatch):
        names = ["vgsth_id_10uA", "vgsth_id_250uA", "vgsth_id_1000uA", "vgsth_id_2500uA"]
        build_run(monkeypatch, color_detections=make_detections(4), expected_count=4,
                  resolved_names=names,
                  pipeline_result=ok_result(n=4, names=["p0", "p1", "p2", "p3"]))
        result = run()
        assert result["status"] == "ok"
        assert [c["curve_name"] for c in result["curves"]] == names


# =================================================================
# E. Quarantine -- no usable labels
# =================================================================

class TestQuarantineNoUsableLabels:
    def test_no_labels_multi_curve_quarantines(self, monkeypatch):
        mocks = build_run(monkeypatch, color_detections=make_detections(2), expected_count=None)
        result = run()
        assert result["status"] == "needs_review"
        assert "label" in result["review_reason"].lower()
        # Calibration doesn't depend on naming succeeding (rule 27) --
        # process_detections IS called here, just its status/reason are
        # overridden by the wrapper's own verdict.
        mocks["process_detections"].assert_called_once()
        assert result["calibration"] is not None

    def test_ambiguous_scheme_multi_curve_quarantines_same_path(self, monkeypatch):
        # count_expected_curves returning None because the schemes were
        # mixed/ambiguous is indistinguishable to the wrapper from "no
        # labels at all" -- same code path, same message class.
        mocks = build_run(monkeypatch, color_detections=make_detections(3), expected_count=None)
        result = run()
        assert result["status"] == "needs_review"
        assert "label" in result["review_reason"].lower()
        mocks["process_detections"].assert_called_once()
        assert result["calibration"] is not None


# =================================================================
# F. Quarantine -- detected < N (crossing/merge safety net)
# =================================================================

class TestQuarantineFewerThanExpected:
    def test_three_labels_two_detected_quarantines_crossing_reason(self, monkeypatch):
        mocks = build_run(monkeypatch, color_detections=make_detections(2), expected_count=3)
        result = run()
        assert result["status"] == "needs_review"
        reason = result["review_reason"].lower()
        assert "crossing" in reason or "merge" in reason
        assert "2" in result["review_reason"] and "3" in result["review_reason"]
        mocks["process_detections"].assert_called_once()
        assert result["calibration"] is not None

    def test_four_labels_three_detected_generalizes(self, monkeypatch):
        mocks = build_run(monkeypatch, color_detections=make_detections(3), expected_count=4)
        result = run()
        assert result["status"] == "needs_review"
        reason = result["review_reason"].lower()
        assert "crossing" in reason or "merge" in reason
        mocks["process_detections"].assert_called_once()
        assert result["calibration"] is not None

    def test_two_labels_one_detected_quarantines_at_smallest_gap(self, monkeypatch):
        # D=1 here, but N=2 is NOT None -- must NOT be mistaken for the
        # "always-safe single curve" case (that only applies when N is None).
        mocks = build_run(monkeypatch, color_detections=make_detections(1), expected_count=2)
        result = run()
        assert result["status"] == "needs_review"
        reason = result["review_reason"].lower()
        assert "crossing" in reason or "merge" in reason
        mocks["process_detections"].assert_called_once()
        assert result["calibration"] is not None

    def test_zero_curves_detected_quarantines_with_distinct_message(self, monkeypatch):
        for n_value in (3, None):
            mocks = build_run(monkeypatch, color_detections=[], mono_detections=[],
                              expected_count=n_value)
            result = run()
            assert result["status"] == "needs_review"
            reason = result["review_reason"].lower()
            assert "no curve" in reason or "0 curve" in reason or "found" in reason
            assert "crossing" not in reason and "merge" not in reason
            # Even with zero curves, axis calibration is derived purely
            # from the OCR tick labels (img_w/img_h/ocr_lines) -- still
            # computed, not skipped.
            mocks["process_detections"].assert_called_once()
            assert result["calibration"] is not None


# =================================================================
# G. Quarantine -- detected > N
# =================================================================

class TestQuarantineMoreThanExpected:
    def test_two_labels_three_detected_quarantines_distinct_reason(self, monkeypatch):
        mocks = build_run(monkeypatch, color_detections=make_detections(3), expected_count=2)
        result = run()
        assert result["status"] == "needs_review"
        reason = result["review_reason"].lower()
        assert "stray" in reason or "missed" in reason or "unexpected" in reason
        assert "crossing" not in reason and "merge" not in reason
        mocks["process_detections"].assert_called_once()
        assert result["calibration"] is not None

    def test_exceeds_by_one_still_quarantines_no_leniency(self, monkeypatch):
        mocks = build_run(monkeypatch, color_detections=make_detections(6), expected_count=5)
        result = run()
        assert result["status"] == "needs_review"
        mocks["process_detections"].assert_called_once()
        assert result["calibration"] is not None


# =================================================================
# H. Quarantine -- matched count, naming still fails
# =================================================================

class TestQuarantineNamingAmbiguous:
    def test_matched_count_but_naming_returns_none_quarantines(self, monkeypatch):
        mocks = build_run(monkeypatch, color_detections=make_detections(2), expected_count=2,
                          resolved_names=None)
        result = run()
        assert result["status"] == "needs_review"
        # Wrapper never invents curve names when naming refuses to guess --
        # but calibration is independent of naming, so process_detections
        # still runs and its calibration is still attached (rule 27).
        mocks["process_detections"].assert_called_once()
        assert result["calibration"] is not None

    def test_naming_ambiguous_reason_distinguishable_from_count_mismatch(self, monkeypatch):
        build_run(monkeypatch, color_detections=make_detections(2), expected_count=2,
                  resolved_names=None)
        ambiguous_reason = run()["review_reason"].lower()

        build_run(monkeypatch, color_detections=make_detections(2), expected_count=3)
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
        # Unit detection itself lives in the frozen core; the wrapper must
        # not mangle/strip whatever units process_detections resolved.
        build_run(monkeypatch, color_detections=make_detections(1), expected_count=1,
                  resolved_names=["vgsth"],
                  pipeline_result=ok_result(n=1, names=["placeholder"], units="V"))
        result = run()
        assert result["status"] == "ok"
        assert result["units"] == "V"

    def test_no_unit_token_falls_to_generic_units_undetected(self, monkeypatch):
        # No vgsth-specific multi-unit table (unlike rdson's
        # RDSON_Y_PLAUSIBLE_RANGES/detect_rdson_units) -- the frozen core's
        # own units_undetected outcome passes straight through unmodified.
        undetected = build_result(
            device="DEV1", curve_type="vgsth_vs_tj", source_image="fig.png",
            status="needs_review", review_reason="units_undetected",
            duplicates_removed=0,
            calibration={"x_slope": 1.0, "x_intercept": 0.0, "y_slope": 1.0,
                        "y_intercept": 0.0, "x_log": False, "y_log": False},
            curves=[{"curve_name": "placeholder", "confidence": 0.9,
                     "points": [{"x": 1.0, "y": 2.0}]}],
            units=None,
        )
        build_run(monkeypatch, color_detections=make_detections(1), expected_count=1,
                  resolved_names=["vgsth"], pipeline_result=undetected)
        result = run()
        assert result["status"] == "needs_review"
        assert result["review_reason"] == "units_undetected"
        assert result["units"] is None


# =================================================================
# J. Plausibility
# =================================================================
#
# The plausibility gate itself lives in the frozen core
# (PLAUSIBILITY_SPECS / _implausibility_reason in pipeline.py) -- these
# tests confirm the WRAPPER passes such a result through unchanged, via a
# mocked process_detections return value, so they don't depend on whether
# PLAUSIBILITY_SPECS["vgsth_vs_tj"] has actually been added yet (flagged
# at the top of this file -- it hasn't, as of this session).

class TestPlausibility:
    def test_implausible_x_range_passes_through_from_frozen_core(self, monkeypatch):
        implausible = build_result(
            device="DEV1", curve_type="vgsth_vs_tj", source_image="fig.png",
            status="needs_review",
            review_reason="implausible_calibration: x values span -120..250, "
                          "outside the plausible vgsth_vs_tj range -75..200",
            duplicates_removed=0,
            calibration={"x_slope": 1.0, "x_intercept": 0.0, "y_slope": 1.0,
                        "y_intercept": 0.0, "x_log": False, "y_log": False},
            curves=[{"curve_name": "placeholder", "confidence": 0.9,
                     "points": [{"x": -120.0, "y": 2.0}, {"x": 250.0, "y": 2.0}]}],
            units=None,
        )
        build_run(monkeypatch, color_detections=make_detections(1), expected_count=1,
                  resolved_names=["vgsth"], pipeline_result=implausible)
        result = run()
        assert result["status"] == "needs_review"
        assert "implausible_calibration" in result["review_reason"]

    def test_negative_y_values_not_flagged_no_y_range_check_yet(self, monkeypatch):
        # Real negative-value vgsth charts exist in the sample corpus;
        # deliberately no y_range gate exists for this curve type yet.
        negative_y_ok = ok_result(n=1, names=["placeholder"], units="V")
        negative_y_ok["curves"][0]["points"] = [{"x": 25.0, "y": -3.5}]
        build_run(monkeypatch, color_detections=make_detections(1), expected_count=1,
                  resolved_names=["vgsth"], pipeline_result=negative_y_ok)
        result = run()
        assert result["status"] == "ok"


# =================================================================
# K. Delegation / zero duplication
# =================================================================

class TestDelegation:
    def test_process_detections_is_the_frozen_core_not_reimplemented(self, monkeypatch):
        mocks = build_run(monkeypatch, color_detections=make_detections(2), expected_count=2,
                          resolved_names=["vgsth_max", "vgsth_typ"])
        image, ocr_lines = make_image(), make_ocr_lines()
        run(image=image, ocr_lines=ocr_lines)
        mocks["process_detections"].assert_called_once()
        args, kwargs = mocks["process_detections"].call_args
        all_args = list(args) + list(kwargs.values())
        assert "DEV1" in all_args
        assert "vgsth_vs_tj" in all_args
        assert "fig.png" in all_args
        assert float(IMG_W) in all_args
        assert float(IMG_H) in all_args

    def test_detection_functions_are_the_literal_curve_detection_objects(self):
        # No monkeypatching here -- checked BEFORE any mocking, confirming
        # classical_vgsth imports (not copies/reimplements) Stage 1's
        # shared detection functions.
        import src.extraction.curve_detection as curve_detection_mod
        assert mod.detect_curve_classical is curve_detection_mod.detect_curve_classical
        assert mod.detect_curve_monochrome is curve_detection_mod.detect_curve_monochrome


# =================================================================
# L. Schema / output contract
# =================================================================

class TestSchemaContract:
    def test_result_shape_matches_stage6_stage7_contract(self, monkeypatch):
        build_run(monkeypatch, color_detections=make_detections(2), expected_count=2,
                  resolved_names=["vgsth_max", "vgsth_typ"],
                  pipeline_result=ok_result(n=2, names=["p0", "p1"], units="V"))
        result = run()
        validate_result(result)  # raises if anything is off-schema
        assert set(result.keys()) == {
            "device", "curve_type", "source_image", "status", "review_reason",
            "duplicates_removed", "calibration", "curves", "units",
        }

    def test_quarantined_result_carries_curves_not_an_empty_shell(self, monkeypatch):
        build_run(monkeypatch, color_detections=make_detections(2), expected_count=3)
        result = run()
        assert result["status"] == "needs_review"
        assert len(result["curves"]) > 0
        # Calibration doesn't depend on curve naming succeeding -- it's
        # still computed and attached even at this early count-mismatch
        # gate (rule 27: no quarantine reason should carry less real
        # information than another).
        assert result["calibration"] is not None


# =================================================================
# M. Input validation
# =================================================================

class TestInputValidation:
    def test_non_hxwx3_image_raises_value_error(self):
        # Real detect_curve_classical (not mocked) supplies this
        # validation -- reused, not reimplemented (CLAUDE.md §3).
        bad_image = np.zeros((IMG_H, IMG_W), dtype=np.uint8)  # missing channel dim
        with pytest.raises(ValueError):
            run(image=bad_image)

    def test_malformed_ocr_line_missing_bounding_box_raises(self, monkeypatch):
        monkeypatch.setattr(mod, "count_expected_curves",
                            MagicMock(side_effect=KeyError("bounding_box")))
        monkeypatch.setattr(mod, "detect_curve_classical",
                            MagicMock(return_value=make_detections(1)))
        with pytest.raises((KeyError, ValueError)):
            run(ocr_lines=[{"text": "max"}])  # no bounding_box


# =================================================================
# N. Logging & path-independence
# =================================================================

class TestLoggingAndPathIndependence:
    def test_quarantine_branches_log_distinct_identifiable_reasons(self, monkeypatch, caplog):
        caplog.set_level(logging.INFO)
        build_run(monkeypatch, color_detections=make_detections(2), expected_count=3)
        run()
        assert any("crossing" in r.message.lower() or "merge" in r.message.lower()
                  for r in caplog.records)

        caplog.clear()
        build_run(monkeypatch, color_detections=make_detections(3), expected_count=2)
        run()
        assert any("stray" in r.message.lower() or "missed" in r.message.lower()
                  or "unexpected" in r.message.lower() for r in caplog.records)

    def test_count_mismatch_logic_identical_for_color_and_mono_paths(self, monkeypatch):
        # Same D<N scenario, once via color, once via mono fallback --
        # identical outcome, no path-specific special-casing.
        build_run(monkeypatch, color_detections=make_detections(2), expected_count=3)
        color_result = run()

        build_run(monkeypatch, color_detections=[], mono_detections=make_detections(2),
                  expected_count=3)
        mono_result = run()

        assert color_result["status"] == mono_result["status"] == "needs_review"
        color_reason = color_result["review_reason"].lower()
        mono_reason = mono_result["review_reason"].lower()
        assert ("crossing" in color_reason or "merge" in color_reason)
        assert ("crossing" in mono_reason or "merge" in mono_reason)


# =================================================================
# O. Placeholder-leak safety net (owner-approved addition, 2026-07-21)
# =================================================================
#
# The naming registry now carries a throwaway placeholder for vgsth_vs_tj
# ("curve_0", "curve_1", ... — see src/extraction/naming/__init__.py) that
# exists ONLY so process_detections doesn't KeyError; it carries no real
# naming authority. This wrapper is REQUIRED to override it on every
# ok-status path. Rather than trust that by inspection, this locks it in
# directly: across every one of this file's own "status == ok" scenarios,
# the pipeline_result mock is configured with EXACTLY that placeholder
# pattern (mirroring what the real registry entry actually produces), and
# no resulting curve_name may match it.

_OK_SCENARIOS = [
    # 1 curve, no labels (B.4)
    dict(n=1, expected_count=None, resolved_names=["vgsth"]),
    # 1 curve, an I_D= label present anyway (B.5)
    dict(n=1, expected_count=1, resolved_names=["vgsth"],
        ocr_lines=[{"text": "I_D = 250uA",
                    "bounding_box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}]),
    # 3 curves, band scheme (C.6)
    dict(n=3, expected_count=3, resolved_names=["vgsth_max", "vgsth_typ", "vgsth_min"]),
    # 2 curves, band scheme (C.7)
    dict(n=2, expected_count=2, resolved_names=["vgsth_max", "vgsth_typ"]),
    # 2 curves, current-value scheme (D.8)
    dict(n=2, expected_count=2, resolved_names=["vgsth_id_250uA", "vgsth_id_1000uA"]),
    # 4 curves, current-value scheme (D.9)
    dict(n=4, expected_count=4, resolved_names=[
        "vgsth_id_10uA", "vgsth_id_250uA", "vgsth_id_1000uA", "vgsth_id_2500uA"]),
]


class TestPlaceholderNeverLeaksIntoTrustedResult:
    def test_no_ok_result_curve_name_matches_placeholder_pattern(self, monkeypatch):
        placeholder_pattern = re.compile(r"^curve_\d+$")
        for scenario in _OK_SCENARIOS:
            n = scenario["n"]
            # The mocked process_detections result uses the REAL registry
            # placeholder's exact naming pattern -- this is what
            # process_detections would actually hand back before the
            # wrapper's override runs.
            placeholder_names = [f"curve_{i}" for i in range(n)]
            build_run(
                monkeypatch, color_detections=make_detections(n),
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
