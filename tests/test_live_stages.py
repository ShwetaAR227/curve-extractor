"""Tests for src.orchestrator.live_stages — written FIRST (CLAUDE.md §2, red
phase). Module does not exist yet.

``LiveStages`` replaces ``PrecomputedStage5`` (src/orchestrator/pipeline.py)
with a real Stage 4 -> Stage 5 wiring adapter matching the SAME protocol
(``run_classification(device)`` / ``run_extraction(device, classification)``)
so ``process_device``/``run_batch`` (frozen, unmodified) work unchanged.

Everything Stage 4/5 already does is reused, never reimplemented:
``classify_device`` (classification), ``run_classical_pipeline`` (classical
extraction), ``run_pipeline`` (model extraction), ``load_model`` (mmdet),
``load_figures_by_page`` (new Stage-3 loader, its own tests in
test_stage3_loader.py) — every one of those is MOCKED here; this file tests
ONLY the adapter's wiring/routing logic. No GPU, no network, no real Azure
OCR calls anywhere in this file.

``ocr_lines`` conversion (confirmed by reading the actual body code of
``run_classical_pipeline``, ``run_pipeline``, and ``parse_numeric_ticks``,
not guessed from type hints): ``FigureCandidate.ocr_lines`` holds
``scoring.OcrLine`` dataclass instances (``.text``, ``.bbox`` attributes) —
classification's own output shape. But every extraction-side consumer
(``classical.py``, ``pipeline.py``, ``ticks.py``, ``naming/rdson_vs_tj.py``)
does dict access (``line["bounding_box"]``) against a locally-defined
``OcrLine = Dict[str, Any]`` (``{"text": str, "bounding_box": {"x1","y1",
"x2","y2"}}``). So ``run_extraction`` must convert dataclass -> dict BEFORE
calling EITHER extraction path — this is not split between classical and
model, both need the identical conversion. Tested below via
``TestOcrLineConversion``: the same figure run through a classical-routed
curve type and a model-routed curve type must produce byte-identical
converted ocr_lines at each mocked call site.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.classification.classify import ClassificationResult, ClassificationStatus
from src.classification.scoring import FigureCandidate, OcrLine
from src.extraction.extraction_registry import ExtractionSpec
from src.extraction.inference import Detection
from src.extraction.schema import build_result
import src.orchestrator.live_stages as live_stages_mod
from src.orchestrator.live_stages import ClaimTracker, LiveStages, NoExtractorAvailable


# ---------------------------------------------------------------- fixtures

def make_figure(figure_id="figures/fig_p3_000.png", page=3, caption="Fig 3. Typical Transfer Characteristics",
                ocr_lines=None, width=800, height=650):
    return FigureCandidate(
        figure_id=figure_id, page=page, figure_index=0, image_path=figure_id,
        caption=caption, ocr_lines=ocr_lines or [
            OcrLine(text="VGS, Gate-to-Source Voltage (V)", bbox=(177, 547, 495, 573)),
        ],
        figure_width=width, figure_height=height,
    )


def make_classification(status, figure=None, target_curve_type="capacitance_vs_vds",
                        reason="test reason", page=3):
    return ClassificationResult(
        target_curve_type=target_curve_type, status=status, figure=figure,
        score=10.0 if status == ClassificationStatus.MATCHED else 3.0,
        runner_up_score=1.0, margin=9.0, page=page, reason=reason, all_scores=[],
    )


def ok_stage5_result(curve_type="capacitance_vs_vds", device="DEV1", source_image="figures/fig_p3_000.png"):
    return build_result(
        device=device, curve_type=curve_type, source_image=source_image,
        status="ok", review_reason=None, duplicates_removed=0,
        calibration={"x_slope": 10.0, "x_intercept": 100.0, "y_slope": -50.0,
                    "y_intercept": 500.0, "x_log": False, "y_log": True},
        curves=[{"curve_name": "Ciss", "confidence": 0.9, "points": [{"x": 1.0, "y": 100.0}]},
                {"curve_name": "Coss", "confidence": 0.9, "points": [{"x": 1.0, "y": 50.0}]},
                {"curve_name": "Crss", "confidence": 0.9, "points": [{"x": 1.0, "y": 10.0}]}],
        units="pF",
    )


def build_adapter(monkeypatch, curve_type="capacitance_vs_vds", figures_by_page=None,
                  classify_result=None, claim_tracker=None, images_root="D:/fake_images",
                  extraction_result=None, mock_cv2_imread=True):
    """A LiveStages instance with every external dependency mocked.

    ``mock_cv2_imread=True`` (default) patches ``cv2.imread`` itself — as
    used by live_stages.py's own ``cv2.imread(...)`` call, i.e. the real
    ``cv2`` module's ``imread`` attribute, NOT a fake stand-in module — to
    return a small real array unconditionally, so classical-path tests
    never need a real image file on disk (``run_classical_pipeline`` is
    already mocked separately; this just satisfies the adapter's own
    pre-read). The ONE test that wants a genuine missing-file failure
    (``test_missing_image_file_for_classical_path_raises_not_crashes_silently``)
    passes ``mock_cv2_imread=False`` and supplies a real (but empty)
    ``tmp_path`` as ``images_root`` instead, so ``cv2.imread`` genuinely
    returns ``None`` for a file that doesn't exist.

    The classical path is now a per-curve-type registry lookup
    (``spec.classical_pipeline``), not a single ``live_stages_mod``-level
    name — so mocking it means replacing WHATEVER registry entry is
    currently active for ``curve_type`` (real or already faked by the
    caller, e.g. via its own ``monkeypatch.setitem`` before calling this)
    with an identical copy carrying ``classical_pipeline=run_classical_mock``,
    via ``dataclasses.replace`` (preserves every other field). Only done
    when that entry's ``method == "classical"`` — model/none entries are
    left untouched, exactly as before.

    Symmetric handling for ``model_pipeline`` (owner-approved routing
    addition, 2026-07-22 follow-up session): when the currently-active
    entry's ``method == "model"`` AND it already carries a real
    ``model_pipeline`` override (currently only if_vs_vsd's real registry
    entry, via ``model_if_vsd.run_model_pipeline``), that override is ALSO
    replaced with a mock via the same ``dataclasses.replace`` technique —
    so a test can exercise if_vs_vsd's dispatch without hitting the real
    (GPU-only) function. Plain model entries whose ``model_pipeline`` is
    ``None`` (capacitance_vs_vds, id_vs_vgs) are left completely
    untouched — they keep routing to the generic, separately-mocked
    ``run_pipeline`` exactly as before this addition.
    """
    figures_by_page = figures_by_page if figures_by_page is not None else {3: [make_figure()]}
    load_figures_mock = MagicMock(return_value=figures_by_page)
    monkeypatch.setattr(live_stages_mod, "load_figures_by_page", load_figures_mock)

    if classify_result is None:
        classify_result = (
            make_classification(ClassificationStatus.MATCHED, figure=make_figure(),
                                target_curve_type=curve_type),
            {make_figure().figure_id},
        )
    classify_device_mock = MagicMock(return_value=classify_result)
    monkeypatch.setattr(live_stages_mod, "classify_device", classify_device_mock)

    run_classical_mock = MagicMock(return_value=extraction_result or ok_stage5_result(curve_type))
    import dataclasses

    import src.extraction.extraction_registry as registry_mod
    current_spec = registry_mod._REGISTRY.get(curve_type)
    if current_spec is not None and current_spec.method == "classical":
        monkeypatch.setitem(
            registry_mod._REGISTRY, curve_type,
            dataclasses.replace(current_spec, classical_pipeline=run_classical_mock),
        )
    run_model_mock = MagicMock(return_value=extraction_result or ok_stage5_result(curve_type))
    monkeypatch.setattr(live_stages_mod, "run_pipeline", run_model_mock)
    load_model_mock = MagicMock(return_value=object())
    monkeypatch.setattr(live_stages_mod, "load_model", load_model_mock)

    model_pipeline_mock = MagicMock(return_value=extraction_result or ok_stage5_result(curve_type))
    # Re-fetch: the classical branch above may have just replaced the
    # registry entry for this curve_type (dataclasses.replace returns a
    # NEW object) -- current_spec must reflect that before this check.
    current_spec = registry_mod._REGISTRY.get(curve_type)
    if (current_spec is not None and current_spec.method == "model"
            and current_spec.model_pipeline is not None):
        monkeypatch.setitem(
            registry_mod._REGISTRY, curve_type,
            dataclasses.replace(current_spec, model_pipeline=model_pipeline_mock),
        )

    if mock_cv2_imread:
        import numpy as np
        fake_image = np.zeros((10, 10, 3), dtype=np.uint8)
        monkeypatch.setattr(live_stages_mod.cv2, "imread", MagicMock(return_value=fake_image))

    adapter = LiveStages(
        curve_type, stage3_root="D:/fake_stage3", images_root=images_root,
        claim_tracker=claim_tracker,
    )
    return adapter, {
        "load_figures": load_figures_mock, "classify_device": classify_device_mock,
        "run_classical_pipeline": run_classical_mock, "run_pipeline": run_model_mock,
        "load_model": load_model_mock, "model_pipeline": model_pipeline_mock,
    }


# ------------------------------------------------------- A. basic protocol

class TestBasicProtocol:
    def test_adapter_exposes_run_classification_and_run_extraction(self, monkeypatch):
        adapter, _ = build_adapter(monkeypatch)
        assert callable(adapter.run_classification)
        assert callable(adapter.run_extraction)

    def test_run_classification_returns_object_with_status_not_raw_tuple(self, monkeypatch):
        # classify_device itself returns (ClassificationResult, new_claimed) — a
        # tuple. The adapter must unwrap it: process_device does
        # getattr(classification, "status", None), which a raw tuple doesn't have.
        adapter, _ = build_adapter(monkeypatch)
        result = adapter.run_classification("DEV1")
        assert not isinstance(result, tuple)
        assert hasattr(result, "status")
        assert result.status == ClassificationStatus.MATCHED

    def test_adapter_does_not_mutate_input_figures_by_page(self, monkeypatch):
        figures_by_page = {3: [make_figure()]}
        original_snapshot = {3: list(figures_by_page[3])}
        adapter, _ = build_adapter(monkeypatch, figures_by_page=figures_by_page)
        adapter.run_classification("DEV1")
        assert figures_by_page == original_snapshot

    def test_adapter_does_not_mutate_figure_ocr_lines_in_place(self, monkeypatch):
        figure = make_figure()
        original_ocr_lines = list(figure.ocr_lines)
        adapter, mocks = build_adapter(
            monkeypatch, figures_by_page={3: [figure]},
            classify_result=(make_classification(ClassificationStatus.MATCHED, figure=figure), {figure.figure_id}),
        )
        classification = adapter.run_classification("DEV1")
        adapter.run_extraction("DEV1", classification)
        assert figure.ocr_lines == original_ocr_lines


# ---------------------------------------------- B. classification: matched

class TestClassificationMatched:
    def test_matched_result_has_correct_figure_attached(self, monkeypatch):
        figure = make_figure(figure_id="figures/fig_p3_000.png")
        adapter, _ = build_adapter(
            monkeypatch, figures_by_page={3: [figure]},
            classify_result=(make_classification(ClassificationStatus.MATCHED, figure=figure), {figure.figure_id}),
        )
        result = adapter.run_classification("DEV1")
        assert result.status == ClassificationStatus.MATCHED
        assert result.figure.figure_id == "figures/fig_p3_000.png"

    def test_run_classification_loads_figures_via_stage3_loader(self, monkeypatch):
        adapter, mocks = build_adapter(monkeypatch)
        adapter.run_classification("DEV1")
        mocks["load_figures"].assert_called_once_with("DEV1", "D:/fake_stage3")

    def test_claimed_figure_unavailable_to_second_curve_type(self, monkeypatch):
        # B.6: mutual exclusion across curve types via the SHARED ClaimTracker.
        figure = make_figure(figure_id="figures/fig_p3_000.png")
        tracker = ClaimTracker()

        adapter_a, mocks_a = build_adapter(
            monkeypatch, curve_type="id_vs_vgs", figures_by_page={3: [figure]},
            classify_result=(make_classification(ClassificationStatus.MATCHED, figure=figure,
                                                 target_curve_type="id_vs_vgs"), {figure.figure_id}),
            claim_tracker=tracker,
        )
        adapter_a.run_classification("DEV1")
        # classify_device for curve type A must have been called with an EMPTY
        # claimed set (nothing claimed yet).
        _, kwargs = mocks_a["classify_device"].call_args
        first_call_claimed = mocks_a["classify_device"].call_args[0][2] if len(
            mocks_a["classify_device"].call_args[0]) > 2 else kwargs.get("claimed")
        assert not first_call_claimed

        adapter_b, mocks_b = build_adapter(
            monkeypatch, curve_type="capacitance_vs_vds", figures_by_page={3: [figure]},
            classify_result=(make_classification(ClassificationStatus.NO_MATCH), set()),
            claim_tracker=tracker,
        )
        adapter_b.run_classification("DEV1")
        second_call_claimed = mocks_b["classify_device"].call_args[0][2] if len(
            mocks_b["classify_device"].call_args[0]) > 2 else mocks_b["classify_device"].call_args[1].get("claimed")
        assert figure.figure_id in second_call_claimed


# -------------------------------------------- C. classification: quarantined/no_match

class TestClassificationQuarantinedNoMatch:
    def test_quarantined_reason_preserved(self, monkeypatch):
        quarantined = make_classification(
            ClassificationStatus.QUARANTINED, figure=make_figure(),
            reason="scored 3.00 but threshold (5.0) not met — ambiguous, needs human review",
        )
        adapter, _ = build_adapter(monkeypatch, classify_result=(quarantined, set()))
        result = adapter.run_classification("DEV1")
        assert result.status == ClassificationStatus.QUARANTINED
        assert "ambiguous" in result.reason

    def test_no_match_when_device_has_no_figures(self, monkeypatch):
        no_match = make_classification(ClassificationStatus.NO_MATCH, reason="device has no figures on any page")
        adapter, _ = build_adapter(monkeypatch, figures_by_page={}, classify_result=(no_match, set()))
        result = adapter.run_classification("DEV1")
        assert result.status == ClassificationStatus.NO_MATCH

    def test_no_match_when_all_figures_already_claimed(self, monkeypatch):
        no_match = make_classification(ClassificationStatus.NO_MATCH, reason="no unclaimed figures available")
        figure = make_figure()
        tracker = ClaimTracker()
        tracker.update("DEV1", {figure.figure_id})
        adapter, mocks = build_adapter(
            monkeypatch, figures_by_page={3: [figure]},
            classify_result=(no_match, {figure.figure_id}), claim_tracker=tracker,
        )
        result = adapter.run_classification("DEV1")
        assert result.status == ClassificationStatus.NO_MATCH


# --------------------------------------------------- D. unregistered curve types

class TestUnregisteredCurveTypes:
    @pytest.mark.parametrize("curve_type", ["if_vs_vsd", "zth_vs_time", "vgsth_vs_tj", "vgs_vs_qg"])
    def test_run_classification_raises_keyerror_not_swallowed(self, monkeypatch, curve_type):
        # These four have NO src.classification.curve_registry entry today —
        # classify_device raises KeyError from get_spec(); the adapter must
        # NOT catch it itself (orchestrator.process_device's own try/except
        # is what turns this into failed_classification).
        adapter, mocks = build_adapter(monkeypatch, curve_type=curve_type)
        mocks["classify_device"].side_effect = KeyError(
            f"Unknown curve_type '{curve_type}'. Registered types: "
            "['capacitance_vs_vds', 'id_vs_vgs', 'rdson_vs_tj']"
        )
        with pytest.raises(KeyError):
            adapter.run_classification("DEV1")

    def test_newly_registered_curve_type_works_without_adapter_code_changes(self, monkeypatch):
        # D.11: registry-driven — patch a fake curve type into the REAL
        # classification registry and confirm the SAME LiveStages code path
        # (no special-casing) handles it once it exists.
        import src.classification.curve_registry as curve_registry_mod

        fake_spec = MagicMock()
        monkeypatch.setitem(curve_registry_mod._REGISTRY, "brand_new_curve_type", fake_spec)
        adapter, mocks = build_adapter(monkeypatch, curve_type="brand_new_curve_type")
        result = adapter.run_classification("DEV1")
        assert result.status == ClassificationStatus.MATCHED

    def test_vgs_vs_qg_extraction_returns_distinct_no_extractor_status(self, monkeypatch):
        # D.12: even if classification for vgs_vs_qg hypothetically matched,
        # extraction must recognize "no extractor slot" as a CLEAN, DISTINCT
        # outcome from a real crash — not a bare KeyError, not a silent pass.
        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure,
                                      target_curve_type="vgs_vs_qg")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="vgs_vs_qg", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        with pytest.raises(NoExtractorAvailable):
            adapter.run_extraction("DEV1", matched)
        mocks["run_classical_pipeline"].assert_not_called()
        mocks["run_pipeline"].assert_not_called()


# --------------------------------------------------- E. extraction routing

class TestExtractionRouting:
    def test_rdson_routes_to_classical_never_touches_model(self, monkeypatch):
        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure, target_curve_type="rdson_vs_tj")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="rdson_vs_tj", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        adapter.run_extraction("DEV1", matched)
        mocks["run_classical_pipeline"].assert_called_once()
        mocks["run_pipeline"].assert_not_called()
        mocks["load_model"].assert_not_called()

    def test_model_routed_curve_types_call_run_pipeline(self, monkeypatch):
        for curve_type in ("capacitance_vs_vds", "id_vs_vgs"):
            figure = make_figure()
            matched = make_classification(ClassificationStatus.MATCHED, figure=figure, target_curve_type=curve_type)
            adapter, mocks = build_adapter(
                monkeypatch, curve_type=curve_type, figures_by_page={3: [figure]},
                classify_result=(matched, {figure.figure_id}),
            )
            adapter.run_extraction("DEV1", matched)
            mocks["run_pipeline"].assert_called_once()
            mocks["run_classical_pipeline"].assert_not_called()

    def test_routing_is_data_driven_not_hardcoded_if_elif(self, monkeypatch):
        # E.16: prove it by inventing two brand-new curve types with opposite
        # methods and confirming both route correctly with ZERO adapter code
        # referencing their names.
        import src.extraction.extraction_registry as registry_mod

        monkeypatch.setitem(registry_mod._REGISTRY, "fake_classical_type", ExtractionSpec(
            curve_type="fake_classical_type", method="classical", checkpoint=None,
            config=None, score_thr=0.5, expected_curve_count=1))
        monkeypatch.setitem(registry_mod._REGISTRY, "fake_model_type", ExtractionSpec(
            curve_type="fake_model_type", method="model", checkpoint="ckpt.pth",
            config="cfg.py", score_thr=0.5, expected_curve_count=1))

        for curve_type, expect_classical in (("fake_classical_type", True), ("fake_model_type", False)):
            figure = make_figure()
            matched = make_classification(ClassificationStatus.MATCHED, figure=figure, target_curve_type=curve_type)
            adapter, mocks = build_adapter(
                monkeypatch, curve_type=curve_type, figures_by_page={3: [figure]},
                classify_result=(matched, {figure.figure_id}),
            )
            adapter.run_extraction("DEV1", matched)
            assert mocks["run_classical_pipeline"].called == expect_classical
            assert mocks["run_pipeline"].called == (not expect_classical)

    def test_unroutable_method_value_raises_clear_error_not_silent_fallback(self, monkeypatch):
        import src.extraction.extraction_registry as registry_mod

        monkeypatch.setitem(registry_mod._REGISTRY, "weird_type", ExtractionSpec(
            curve_type="weird_type", method="quantum_tunneling", checkpoint=None,
            config=None, score_thr=0.5, expected_curve_count=1))
        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure, target_curve_type="weird_type")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="weird_type", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        with pytest.raises(ValueError):
            adapter.run_extraction("DEV1", matched)
        mocks["run_classical_pipeline"].assert_not_called()
        mocks["run_pipeline"].assert_not_called()


# --------------------- E'. classical-dispatch routing fix (owner-approved, 2026-07-22)
#
# Was: live_stages.py hardcoded a SINGLE top-level
# `from src.extraction.classical import run_classical_pipeline` and called
# it for every method="classical" curve type regardless of which one —
# meaning vgsth_vs_tj (registered as "classical" in the prior session)
# would have silently called RDSON's wrapper if ever exercised end-to-end.
# test_a_future_classical_entry_routes_the_same_way_as_rdson (this file,
# written before classical_vgsth.py existed) actually encoded that bug as
# an intentional assumption — retired below, replaced by
# test_vgsth_and_rdson_route_to_independent_classical_functions, which
# inverts it.
#
# Now: ExtractionSpec carries the actual function object
# (spec.classical_pipeline); live_stages.py's classical branch calls it
# directly, no if/elif, no hardcoded import. These tests inject FAKE
# ExtractionSpec entries (classical_pipeline=a MagicMock) via
# monkeypatch.setitem — the same technique test_routing_is_data_driven_
# not_hardcoded_if_elif already uses — since the real rdson_vs_tj/
# vgsth_vs_tj entries point at HEAVY real functions (their own detection/
# naming/pipeline machinery, already covered by test_classical.py/
# test_classical_vgsth.py) that don't need re-exercising here; this file's
# job is only the adapter's OWN dispatch/routing logic.

def _classical_spec(curve_type, mock_fn, **overrides):
    fields = dict(curve_type=curve_type, method="classical", checkpoint=None,
                 config=None, score_thr=0.5, expected_curve_count=None,
                 classical_pipeline=mock_fn)
    fields.update(overrides)
    return ExtractionSpec(**fields)


class TestClassicalDispatchRoutingFix:
    def test_rdson_extraction_calls_its_own_classical_pipeline_by_identity(self, monkeypatch):
        # build_adapter wires classical_pipeline onto rdson_vs_tj's REAL
        # registry entry (via dataclasses.replace, preserving every other
        # field) -- fetching the spec fresh and checking IDENTITY against
        # the mock that then actually gets called is the genuine proof,
        # not just "a mock was called".
        from src.extraction.extraction_registry import get_extraction_spec

        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure,
                                      target_curve_type="rdson_vs_tj")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="rdson_vs_tj", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        assert get_extraction_spec("rdson_vs_tj").classical_pipeline is mocks["run_classical_pipeline"]
        adapter.run_extraction("DEV1", matched)
        mocks["run_classical_pipeline"].assert_called_once()

    def test_vgsth_and_rdson_route_to_independent_classical_functions(self, monkeypatch):
        # E.15 replacement for test_a_future_classical_entry_routes_the_
        # same_way_as_rdson: inverts the old (now-incorrect) assumption.
        # Two SEPARATE build_adapter calls (one per curve type) each wire
        # their OWN registry entry's classical_pipeline independently --
        # both processed here, proving genuine independence, not just
        # "each works alone in isolation".
        rdson_figure = make_figure()
        rdson_matched = make_classification(ClassificationStatus.MATCHED, figure=rdson_figure,
                                            target_curve_type="rdson_vs_tj")
        rdson_adapter, rdson_mocks = build_adapter(
            monkeypatch, curve_type="rdson_vs_tj", figures_by_page={3: [rdson_figure]},
            classify_result=(rdson_matched, {rdson_figure.figure_id}),
        )
        rdson_adapter.run_extraction("DEV1", rdson_matched)
        assert rdson_mocks["run_classical_pipeline"].call_count == 1

        vgsth_figure = make_figure()
        vgsth_matched = make_classification(ClassificationStatus.MATCHED, figure=vgsth_figure,
                                            target_curve_type="vgsth_vs_tj")
        vgsth_adapter, vgsth_mocks = build_adapter(
            monkeypatch, curve_type="vgsth_vs_tj", figures_by_page={3: [vgsth_figure]},
            classify_result=(vgsth_matched, {vgsth_figure.figure_id}),
        )
        vgsth_adapter.run_extraction("DEV2", vgsth_matched)

        assert vgsth_mocks["run_classical_pipeline"].call_count == 1
        assert rdson_mocks["run_classical_pipeline"].call_count == 1, \
            "rdson's mock must be untouched by vgsth's run"
        assert rdson_mocks["run_classical_pipeline"] is not vgsth_mocks["run_classical_pipeline"]

    def test_no_hardcoded_classical_module_reference_remains_in_dispatch(self):
        # Confirms the old if/elif/hardcoded-import is actually gone, not
        # just unused — the name must not exist in live_stages.py's own
        # module namespace at all.
        assert not hasattr(live_stages_mod, "run_classical_pipeline")

    def test_model_dispatch_unaffected_even_though_classical_pipeline_is_none(self, monkeypatch):
        from src.extraction.extraction_registry import get_extraction_spec

        assert get_extraction_spec("capacitance_vs_vds").classical_pipeline is None
        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure,
                                      target_curve_type="capacitance_vs_vds")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="capacitance_vs_vds", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        result = adapter.run_extraction("DEV1", matched)
        mocks["run_pipeline"].assert_called_once()
        mocks["run_classical_pipeline"].assert_not_called()
        assert result["status"] == "ok"

    def test_classical_method_with_none_pipeline_raises_clear_specific_error(self, monkeypatch):
        # A registry mistake (method="classical" but classical_pipeline
        # left None) must raise a CLEAR, SPECIFIC error -- never silently
        # fall through to the wrong function. build_adapter runs FIRST
        # (wires the other mocks); the broken spec is injected AFTER, so
        # it's the one active when run_extraction is actually called.
        import src.extraction.extraction_registry as registry_mod

        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure,
                                      target_curve_type="rdson_vs_tj")
        adapter, _ = build_adapter(
            monkeypatch, curve_type="rdson_vs_tj", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        monkeypatch.setitem(registry_mod._REGISTRY, "rdson_vs_tj",
                            _classical_spec("rdson_vs_tj", None))
        with pytest.raises((ValueError, TypeError)) as exc_info:
            adapter.run_extraction("DEV1", matched)
        assert "classical_pipeline" in str(exc_info.value)

    def test_classical_pipeline_called_with_expected_arguments(self, monkeypatch):
        # No argument drift from the extra indirection -- exactly the same
        # kwargs the old hardcoded call site already used.
        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure,
                                      target_curve_type="rdson_vs_tj")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="rdson_vs_tj", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        adapter.run_extraction("DEV1", matched)
        mock_fn = mocks["run_classical_pipeline"]
        assert mock_fn.call_count == 1
        _, kwargs = mock_fn.call_args
        assert kwargs["device"] == "DEV1"
        assert kwargs["curve_type"] == "rdson_vs_tj"
        assert kwargs["source_image"] == figure.image_path
        assert "image" in kwargs
        assert "ocr_lines" in kwargs

    def test_classical_modules_have_no_extraction_registry_reference(self):
        # Explicit static-import check (not assumed), per instruction.
        # Full transitive-closure investigation (documented in this
        # session's report): traced EVERY import of classical.py and
        # classical_vgsth.py down to leaves (curve_detection, naming/
        # rdson_vs_tj, naming/vgsth_vs_tj, naming/__init__,
        # extraction.pipeline, schema, skeletonize, extraction.dedup,
        # extraction.inference, calibration.ticks, training.eval_lineformer,
        # training.predict_to_cvat, dataset_tools.collect_images) --
        # NONE import extraction_registry, live_stages, or
        # orchestrator.pipeline at any depth. `grep -rln extraction_registry
        # src/` finds only live_stages.py itself as a current importer. No
        # cycle exists or would be created by extraction_registry.py
        # importing classical.py/classical_vgsth.py at module level. This
        # test is the runtime companion check: neither module has a name
        # bound that was imported FROM extraction_registry.
        import src.extraction.classical as classical_mod
        import src.extraction.classical_vgsth as classical_vgsth_mod

        for mod in (classical_mod, classical_vgsth_mod):
            for name, value in vars(mod).items():
                module_of_value = getattr(value, "__module__", None)
                assert module_of_value != "src.extraction.extraction_registry", (
                    f"{mod.__name__}.{name} was imported from extraction_registry"
                )


# --------------------- E''. model-dispatch routing addition (owner-approved,
# 2026-07-22 follow-up session)
#
# if_vs_vsd is registered as method="model" but, unlike capacitance_vs_vds/
# id_vs_vgs, needs its own expected-vs-detected safety net
# (model_if_vsd.run_model_pipeline) instead of the generic run_pipeline
# call -- the model-dispatch analogue of the classical_pipeline routing
# fix above. ExtractionSpec.model_pipeline carries the real override
# function object; live_stages.py's model branch calls it directly when
# present, falling through to run_pipeline exactly as before when it's
# None (every other model-routed curve type today).

class TestModelDispatchRoutingAddition:
    def test_if_vs_vsd_routes_through_its_own_model_pipeline_not_run_pipeline(self, monkeypatch):
        from src.extraction.extraction_registry import get_extraction_spec

        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure,
                                      target_curve_type="if_vs_vsd")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="if_vs_vsd", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        assert get_extraction_spec("if_vs_vsd").model_pipeline is mocks["model_pipeline"]
        result = adapter.run_extraction("DEV1", matched)
        mocks["model_pipeline"].assert_called_once()
        mocks["run_pipeline"].assert_not_called()
        assert result["status"] == "ok"

    def test_model_load_still_happens_before_dispatch_to_model_pipeline(self, monkeypatch):
        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure,
                                      target_curve_type="if_vs_vsd")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="if_vs_vsd", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        adapter.run_extraction("DEV1", matched)
        mocks["load_model"].assert_called_once()

    def test_plain_model_entries_unaffected_still_use_generic_run_pipeline(self, monkeypatch):
        # capacitance_vs_vds/id_vs_vgs's model_pipeline is None -- must
        # keep routing to run_pipeline exactly as before this addition.
        from src.extraction.extraction_registry import get_extraction_spec

        for curve_type in ("capacitance_vs_vds", "id_vs_vgs"):
            assert get_extraction_spec(curve_type).model_pipeline is None
            figure = make_figure()
            matched = make_classification(ClassificationStatus.MATCHED, figure=figure,
                                          target_curve_type=curve_type)
            adapter, mocks = build_adapter(
                monkeypatch, curve_type=curve_type, figures_by_page={3: [figure]},
                classify_result=(matched, {figure.figure_id}),
            )
            adapter.run_extraction("DEV1", matched)
            mocks["run_pipeline"].assert_called_once()
            mocks["model_pipeline"].assert_not_called()

    def test_model_pipeline_called_with_expected_arguments(self, monkeypatch):
        # Same kwargs run_pipeline itself would receive -- no argument
        # drift from the extra indirection.
        figure = make_figure(ocr_lines=[OcrLine(text="VSD (V)", bbox=(10, 20, 30, 40))])
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure,
                                      target_curve_type="if_vs_vsd")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="if_vs_vsd", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        adapter.run_extraction("DEV1", matched)
        mock_fn = mocks["model_pipeline"]
        assert mock_fn.call_count == 1
        _, kwargs = mock_fn.call_args
        assert kwargs["device"] == "DEV1"
        assert kwargs["curve_type"] == "if_vs_vsd"
        assert kwargs["image_path"] == "D:/fake_images/figures/fig_p3_000.png"
        assert kwargs["ocr_lines"][0]["text"] == "VSD (V)"
        assert kwargs["img_w"] == figure.figure_width
        assert kwargs["img_h"] == figure.figure_height
        assert "model" in kwargs
        assert "score_thr" in kwargs
        assert "expected_curve_count" in kwargs

    def test_routing_is_data_driven_fake_model_pipeline_entry(self, monkeypatch):
        # Proof the model_pipeline dispatch decision is a genuine data
        # lookup: a brand-new fake curve type with its own model_pipeline
        # override routes to it, with ZERO adapter code referencing its
        # name -- mirrors test_routing_is_data_driven_not_hardcoded_if_elif
        # above, one level deeper (which function the "model" branch itself
        # calls, not just classical-vs-model).
        import src.extraction.extraction_registry as registry_mod
        from src.extraction.extraction_registry import ExtractionSpec

        monkeypatch.setitem(registry_mod._REGISTRY, "fake_model_override_type", ExtractionSpec(
            curve_type="fake_model_override_type", method="model", checkpoint="ckpt.pth",
            config="cfg.py", score_thr=0.5, expected_curve_count=1,
            model_pipeline=MagicMock(),
        ))
        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure,
                                      target_curve_type="fake_model_override_type")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="fake_model_override_type", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        adapter.run_extraction("DEV1", matched)
        mocks["model_pipeline"].assert_called_once()
        mocks["run_pipeline"].assert_not_called()

    def test_no_hardcoded_if_vs_vsd_reference_in_dispatch(self):
        # Confirms the routing is genuinely data-driven, not a disguised
        # if/elif keyed on the curve_type string.
        import inspect

        source = inspect.getsource(live_stages_mod.LiveStages.run_extraction)
        assert "if_vs_vsd" not in source
        assert "model_if_vsd" not in source


# --------------------------------------------- F. model-based extraction specifics

class TestModelExtractionSpecifics:
    def test_correct_checkpoint_loaded_per_curve_type(self, monkeypatch):
        figure = make_figure()
        for curve_type in ("capacitance_vs_vds", "id_vs_vgs"):
            matched = make_classification(ClassificationStatus.MATCHED, figure=figure, target_curve_type=curve_type)
            adapter, mocks = build_adapter(
                monkeypatch, curve_type=curve_type, figures_by_page={3: [figure]},
                classify_result=(matched, {figure.figure_id}),
            )
            adapter.run_extraction("DEV1", matched)
            from src.extraction.extraction_registry import get_extraction_spec
            expected_checkpoint = get_extraction_spec(curve_type).checkpoint
            call_args = mocks["load_model"].call_args
            assert expected_checkpoint in call_args.args or expected_checkpoint in call_args.kwargs.values()

    def test_model_load_failure_surfaces_not_swallowed(self, monkeypatch):
        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure,
                                      target_curve_type="capacitance_vs_vds")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="capacitance_vs_vds", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        mocks["load_model"].side_effect = FileNotFoundError("checkpoint not found: /bad/path.pth")
        with pytest.raises(FileNotFoundError):
            adapter.run_extraction("DEV1", matched)

    def test_score_thr_and_expected_curve_count_passed_from_registry(self, monkeypatch):
        from src.extraction.extraction_registry import get_extraction_spec

        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure,
                                      target_curve_type="capacitance_vs_vds")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="capacitance_vs_vds", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        adapter.run_extraction("DEV1", matched)
        spec = get_extraction_spec("capacitance_vs_vds")
        _, kwargs = mocks["run_pipeline"].call_args
        assert kwargs.get("score_thr") == spec.score_thr
        assert kwargs.get("expected_curve_count") == spec.expected_curve_count

    def test_ocr_lines_passed_to_model_path_are_correctly_shaped_dicts(self, monkeypatch):
        figure = make_figure(ocr_lines=[OcrLine(text="VDS (V)", bbox=(10, 20, 30, 40))])
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure,
                                      target_curve_type="capacitance_vs_vds")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="capacitance_vs_vds", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        adapter.run_extraction("DEV1", matched)
        _, kwargs = mocks["run_pipeline"].call_args
        ocr_lines_arg = kwargs.get("ocr_lines") or mocks["run_pipeline"].call_args.args[3]
        assert isinstance(ocr_lines_arg, list)
        assert ocr_lines_arg[0]["text"] == "VDS (V)"
        assert ocr_lines_arg[0]["bounding_box"] == {"x1": 10, "y1": 20, "x2": 30, "y2": 40}


# ------------------------------------------------- G. classical extraction specifics

class TestClassicalExtractionSpecifics:
    def test_classical_path_delegates_entirely_to_run_classical_pipeline(self, monkeypatch):
        # G.21: the adapter must NOT reimplement color->mono fallback — just
        # call run_classical_pipeline once and return its result verbatim.
        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure, target_curve_type="rdson_vs_tj")
        expected = ok_stage5_result("rdson_vs_tj")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="rdson_vs_tj", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}), extraction_result=expected,
        )
        result = adapter.run_extraction("DEV1", matched)
        mocks["run_classical_pipeline"].assert_called_once()
        assert result == expected

    def test_ocr_lines_passed_to_classical_path_are_correctly_shaped_dicts(self, monkeypatch):
        # Symmetric with the model-path check above: classical.py's own
        # OcrLine is ALSO Dict[str, Any] with a "bounding_box" key (confirmed
        # by reading detect_rdson_units/run_classical_pipeline's body —
        # line["bounding_box"], dict access) — the dataclass->dict conversion
        # is not skipped just because this is the classical route.
        figure = make_figure(ocr_lines=[OcrLine(text="RDS(on) [mΩ]", bbox=(2, 100, 26, 190))])
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure, target_curve_type="rdson_vs_tj")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="rdson_vs_tj", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        adapter.run_extraction("DEV1", matched)
        _, kwargs = mocks["run_classical_pipeline"].call_args
        ocr_lines_arg = kwargs.get("ocr_lines") or mocks["run_classical_pipeline"].call_args.args[4]
        converted = list(ocr_lines_arg)
        assert converted[0]["text"] == "RDS(on) [mΩ]"
        assert converted[0]["bounding_box"] == {"x1": 2, "y1": 100, "x2": 26, "y2": 190}

    def test_missing_image_file_for_classical_path_raises_not_crashes_silently(self, monkeypatch, tmp_path):
        # Deliberately real (unmocked) cv2.imread against a real, empty
        # tmp_path — the file genuinely doesn't exist, so cv2.imread must
        # genuinely return None and the adapter must raise, not crash
        # silently or hand None to run_classical_pipeline unchecked.
        figure = make_figure(figure_id="figures/does_not_exist.png")
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure, target_curve_type="rdson_vs_tj")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="rdson_vs_tj", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}), images_root=str(tmp_path),
            mock_cv2_imread=False,
        )
        with pytest.raises(Exception):
            adapter.run_extraction("DEV1", matched)


# --------------------------------------------------- ocr_lines conversion (cross-route)

class TestOcrLineConversion:
    """The dataclass -> dict conversion (scoring.OcrLine -> {"text",
    "bounding_box"}) happens identically before EITHER extraction path is
    called — it is not split between classical and model (both consumers do
    dict access on ocr_lines, confirmed by reading their real body code).
    The SAME source figure run through a classical-routed curve type and a
    model-routed curve type must produce byte-identical converted output.
    """

    def test_same_ocr_lines_convert_identically_regardless_of_route(self, monkeypatch):
        raw_ocr_lines = [
            OcrLine(text="VDS (V)", bbox=(10.0, 20.0, 30.0, 40.0)),
            OcrLine(text="0.5", bbox=(5.0, 6.0, 7.0, 8.0)),
        ]
        expected = [
            {"text": "VDS (V)", "bounding_box": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0}},
            {"text": "0.5", "bounding_box": {"x1": 5.0, "y1": 6.0, "x2": 7.0, "y2": 8.0}},
        ]

        figure_classical = make_figure(ocr_lines=list(raw_ocr_lines))
        matched_classical = make_classification(
            ClassificationStatus.MATCHED, figure=figure_classical, target_curve_type="rdson_vs_tj")
        adapter_classical, mocks_classical = build_adapter(
            monkeypatch, curve_type="rdson_vs_tj", figures_by_page={3: [figure_classical]},
            classify_result=(matched_classical, {figure_classical.figure_id}),
        )
        adapter_classical.run_extraction("DEV1", matched_classical)
        classical_call = mocks_classical["run_classical_pipeline"].call_args
        classical_ocr_lines = list(
            classical_call.kwargs.get("ocr_lines") or classical_call.args[4])

        figure_model = make_figure(ocr_lines=list(raw_ocr_lines))
        matched_model = make_classification(
            ClassificationStatus.MATCHED, figure=figure_model, target_curve_type="capacitance_vs_vds")
        adapter_model, mocks_model = build_adapter(
            monkeypatch, curve_type="capacitance_vs_vds", figures_by_page={3: [figure_model]},
            classify_result=(matched_model, {figure_model.figure_id}),
        )
        adapter_model.run_extraction("DEV1", matched_model)
        model_call = mocks_model["run_pipeline"].call_args
        model_ocr_lines = list(model_call.kwargs.get("ocr_lines") or model_call.args[3])

        assert classical_ocr_lines == expected
        assert model_ocr_lines == expected
        assert classical_ocr_lines == model_ocr_lines


# --------------------------------------------------------- H. error isolation

class TestErrorIsolationBatchSafety:
    def test_one_devices_classification_exception_does_not_stop_batch(self, monkeypatch):
        from src.orchestrator.pipeline import run_batch

        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure)
        adapter, mocks = build_adapter(
            monkeypatch, figures_by_page={3: [figure]}, classify_result=(matched, {figure.figure_id}),
        )

        def flaky_classify(*args, **kwargs):
            device = args[0] if args else kwargs.get("device")
            raise RuntimeError("boom")

        # DEV_BAD raises; DEV_GOOD must still be processed by run_batch.
        original = mocks["classify_device"]

        def side_effect(figures_by_page, target_curve_type, claimed=None):
            return (matched, {figure.figure_id})

        call_log = []

        class SelectiveStages:
            def run_classification(self, device):
                call_log.append(device)
                if device == "DEV_BAD":
                    raise RuntimeError("boom")
                return adapter.run_classification(device)

            def run_extraction(self, device, classification):
                return adapter.run_extraction(device, classification)

        summary = run_batch(["DEV_BAD", "DEV_GOOD"], "capacitance_vs_vds", SelectiveStages(), {},
                            out_dir=Path("__test_scratch_ignore__"))
        assert "DEV_GOOD" in call_log
        assert summary["counts"]["failed_classification"] == 1

    def test_one_devices_extraction_exception_does_not_stop_batch(self, monkeypatch):
        from src.orchestrator.pipeline import run_batch

        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure)
        adapter, mocks = build_adapter(
            monkeypatch, figures_by_page={3: [figure]}, classify_result=(matched, {figure.figure_id}),
        )
        mocks["run_pipeline"].side_effect = None

        call_log = []

        class SelectiveStages:
            def run_classification(self, device):
                return adapter.run_classification(device)

            def run_extraction(self, device, classification):
                call_log.append(device)
                if device == "DEV_BAD":
                    raise RuntimeError("extraction boom")
                return adapter.run_extraction(device, classification)

        summary = run_batch(["DEV_BAD", "DEV_GOOD"], "capacitance_vs_vds", SelectiveStages(), {},
                            out_dir=Path("__test_scratch_ignore__"))
        assert "DEV_GOOD" in call_log
        assert summary["counts"]["failed_extraction"] == 1

    def test_malformed_ocr_input_for_one_device_does_not_crash_others(self, monkeypatch):
        good_figure = make_figure()

        def load_figures_side_effect(device, stage3_root):
            if device == "DEV_MALFORMED":
                # Malformed: bbox is a string instead of a 4-tuple.
                return {3: [make_figure(ocr_lines=[OcrLine(text="bad", bbox="not-a-tuple")])]}
            return {3: [good_figure]}

        monkeypatch.setattr(live_stages_mod, "load_figures_by_page",
                            MagicMock(side_effect=load_figures_side_effect))
        monkeypatch.setattr(live_stages_mod, "classify_device", MagicMock(
            return_value=(make_classification(ClassificationStatus.MATCHED, figure=good_figure), set())))
        monkeypatch.setattr(live_stages_mod, "run_pipeline", MagicMock(return_value=ok_stage5_result()))
        monkeypatch.setattr(live_stages_mod, "load_model", MagicMock(return_value=object()))

        adapter = LiveStages("capacitance_vs_vds", stage3_root="D:/fake", images_root="D:/fake_images")
        # DEV_GOOD must still classify successfully even if DEV_MALFORMED was
        # processed first and its malformed OCR data caused an error. Only
        # run_classification is exercised below (never run_extraction), so
        # no classical_pipeline mocking is needed here at all.
        try:
            adapter.run_classification("DEV_MALFORMED")
        except Exception:
            pass
        result = adapter.run_classification("DEV_GOOD")
        assert result.status == ClassificationStatus.MATCHED


# ----------------------------------------------- I. end-to-end per-device flow

class TestEndToEndPerDeviceFlow:
    def test_quarantined_classification_never_calls_extraction(self, monkeypatch):
        quarantined = make_classification(ClassificationStatus.QUARANTINED, figure=make_figure())
        adapter, mocks = build_adapter(monkeypatch, classify_result=(quarantined, set()))
        with pytest.raises(Exception):
            adapter.run_extraction("DEV1", quarantined)
        mocks["run_classical_pipeline"].assert_not_called()
        mocks["run_pipeline"].assert_not_called()

    def test_no_match_classification_never_calls_extraction(self, monkeypatch):
        no_match = make_classification(ClassificationStatus.NO_MATCH)
        adapter, mocks = build_adapter(monkeypatch, classify_result=(no_match, set()))
        with pytest.raises(Exception):
            adapter.run_extraction("DEV1", no_match)
        mocks["run_classical_pipeline"].assert_not_called()
        mocks["run_pipeline"].assert_not_called()

    def test_matched_extraction_called_with_correct_figure_image_path(self, monkeypatch):
        figure = make_figure(figure_id="figures/fig_p9_042.png")
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure, target_curve_type="rdson_vs_tj")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="rdson_vs_tj", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        adapter.run_extraction("DEV1", matched)
        call_args = mocks["run_classical_pipeline"].call_args
        assert "figures/fig_p9_042.png" in str(call_args)

    def test_full_flow_output_matches_stage5_schema_exactly(self, monkeypatch):
        from src.extraction.schema import validate_result

        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure, target_curve_type="rdson_vs_tj")
        expected = ok_stage5_result("rdson_vs_tj")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="rdson_vs_tj", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}), extraction_result=expected,
        )
        result = adapter.run_extraction("DEV1", matched)
        validate_result(result)  # raises if anything drifted from Stage 5's contract
        assert result == expected

    def test_running_same_device_twice_produces_identical_output(self, monkeypatch):
        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure, target_curve_type="rdson_vs_tj")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="rdson_vs_tj", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        c1 = adapter.run_classification("DEV1")
        r1 = adapter.run_extraction("DEV1", c1)
        c2 = adapter.run_classification("DEV1")
        r2 = adapter.run_extraction("DEV1", c2)
        assert r1 == r2
        assert c1.status == c2.status
        assert c1.figure.figure_id == c2.figure.figure_id


# --------------------------------------------------------- J. data source correctness

class TestDataSourceCorrectness:
    def test_stage3_root_defaults_to_env_var_when_not_passed(self, monkeypatch):
        monkeypatch.setenv("LINEFORMER_STAGE3_ROOT", "D:/env_configured_root")
        monkeypatch.setattr(live_stages_mod, "load_figures_by_page",
                            MagicMock(return_value={3: [make_figure()]}))
        monkeypatch.setattr(live_stages_mod, "classify_device", MagicMock(
            return_value=(make_classification(ClassificationStatus.MATCHED, figure=make_figure()), set())))
        adapter = LiveStages("capacitance_vs_vds", images_root="D:/fake_images")  # stage3_root omitted
        adapter.run_classification("DEV1")
        live_stages_mod.load_figures_by_page.assert_called_once_with("DEV1", "D:/env_configured_root")

    def test_missing_stage3_root_and_env_var_raises_clear_error(self, monkeypatch):
        monkeypatch.delenv("LINEFORMER_STAGE3_ROOT", raising=False)
        with pytest.raises((RuntimeError, ValueError)):
            LiveStages("capacitance_vs_vds", images_root="D:/fake_images")

    def test_adapter_fully_testable_with_mocked_dependencies_no_real_io(self, monkeypatch):
        # Capstone: every external dependency mocked, zero real file/network
        # access, full classify+extract flow still works end to end.
        figure = make_figure()
        matched = make_classification(ClassificationStatus.MATCHED, figure=figure, target_curve_type="rdson_vs_tj")
        adapter, mocks = build_adapter(
            monkeypatch, curve_type="rdson_vs_tj", figures_by_page={3: [figure]},
            classify_result=(matched, {figure.figure_id}),
        )
        classification = adapter.run_classification("DEV1")
        result = adapter.run_extraction("DEV1", classification)
        assert classification.status == ClassificationStatus.MATCHED
        assert result["status"] == "ok"
        mocks["load_figures"].assert_called()
        mocks["classify_device"].assert_called()
        mocks["run_classical_pipeline"].assert_called()


# --------------------------------------------------------------- ClaimTracker

class TestClaimTracker:
    def test_get_returns_empty_set_for_unseen_device(self):
        tracker = ClaimTracker()
        assert tracker.get("DEV1") == set()

    def test_update_then_get_returns_the_claimed_set(self):
        tracker = ClaimTracker()
        tracker.update("DEV1", {"fig1.png"})
        assert tracker.get("DEV1") == {"fig1.png"}

    def test_claims_are_isolated_per_device(self):
        tracker = ClaimTracker()
        tracker.update("DEV1", {"fig1.png"})
        tracker.update("DEV2", {"fig2.png"})
        assert tracker.get("DEV1") == {"fig1.png"}
        assert tracker.get("DEV2") == {"fig2.png"}

    def test_update_accumulates_does_not_overwrite(self):
        tracker = ClaimTracker()
        tracker.update("DEV1", {"fig1.png"})
        tracker.update("DEV1", {"fig2.png"})
        assert tracker.get("DEV1") == {"fig1.png", "fig2.png"}


# --------------------------------------------------------- K. device discovery
#
# discover_devices() lists the devices available under stage3_root for the
# CLI to feed into run_batch(). Unlike precomputed mode's CLI, which excludes
# a fixed set of known non-device stems (a name blocklist), a blocklist is
# NOT safe against the real Stage-3 output root: data/ accumulates non-device
# folders constantly (training image batches, COCO splits, overlays, raw
# downloads, ...), so a fixed list would silently miss new ones as they
# appear. Discovery is self-verifying instead: a subfolder counts as a
# device only if it directly contains full_extraction.json. No real Stage-3
# output exists on this machine yet (that OCR run hasn't happened), so this
# is tested against constructed tmp_path fixtures only.

def _make_device_dir(root, name, with_extraction=True):
    device_dir = root / name
    device_dir.mkdir(parents=True, exist_ok=True)
    if with_extraction:
        (device_dir / "full_extraction.json").write_text("{}", encoding="utf-8")
    return device_dir


class TestDeviceDiscovery:
    def test_finds_folder_with_full_extraction_json(self, tmp_path):
        _make_device_dir(tmp_path, "DEV1")
        adapter = LiveStages("capacitance_vs_vds", images_root="D:/fake_images",
                             stage3_root=tmp_path)
        assert adapter.discover_devices() == ["DEV1"]

    def test_skips_folder_without_full_extraction_json(self, tmp_path):
        _make_device_dir(tmp_path, "DEV1")
        _make_device_dir(tmp_path, "not_a_device", with_extraction=False)
        adapter = LiveStages("capacitance_vs_vds", images_root="D:/fake_images",
                             stage3_root=tmp_path)
        assert adapter.discover_devices() == ["DEV1"]

    def test_realistic_data_dir_shape(self, tmp_path):
        # Real-world mix: device folders alongside non-device artifact
        # folders that a hardcoded blocklist would eventually miss as new
        # ones appear.
        for real in ("BSP324H6327XTSA1", "AUIRF1010EZS", "R6010YND3TL1"):
            _make_device_dir(tmp_path, real)
        for decoy in ("training_batch_003", "coco_splits", "overlays",
                      "raw_downloads", "quarantined_gallery_assets"):
            _make_device_dir(tmp_path, decoy, with_extraction=False)
        adapter = LiveStages("capacitance_vs_vds", images_root="D:/fake_images",
                             stage3_root=tmp_path)
        assert adapter.discover_devices() == [
            "AUIRF1010EZS", "BSP324H6327XTSA1", "R6010YND3TL1",
        ]

    def test_returns_sorted_list(self, tmp_path):
        for name in ("ZDEV", "ADEV", "MDEV"):
            _make_device_dir(tmp_path, name)
        adapter = LiveStages("capacitance_vs_vds", images_root="D:/fake_images",
                             stage3_root=tmp_path)
        assert adapter.discover_devices() == ["ADEV", "MDEV", "ZDEV"]

    def test_empty_root_returns_empty_list(self, tmp_path):
        adapter = LiveStages("capacitance_vs_vds", images_root="D:/fake_images",
                             stage3_root=tmp_path)
        assert adapter.discover_devices() == []

    def test_all_decoys_returns_empty_list(self, tmp_path):
        _make_device_dir(tmp_path, "coco_splits", with_extraction=False)
        _make_device_dir(tmp_path, "overlays", with_extraction=False)
        adapter = LiveStages("capacitance_vs_vds", images_root="D:/fake_images",
                             stage3_root=tmp_path)
        assert adapter.discover_devices() == []

    def test_ignores_loose_files_at_root(self, tmp_path):
        _make_device_dir(tmp_path, "DEV1")
        (tmp_path / "full_extraction.json").write_text("{}", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("stray file", encoding="utf-8")
        adapter = LiveStages("capacitance_vs_vds", images_root="D:/fake_images",
                             stage3_root=tmp_path)
        assert adapter.discover_devices() == ["DEV1"]

    def test_ignores_nested_json_two_levels_down(self, tmp_path):
        # Only direct children of stage3_root are candidate device folders;
        # a full_extraction.json buried inside a subfolder's subfolder must
        # not promote that grandparent folder to "device".
        nested = tmp_path / "some_batch" / "DEV1"
        nested.mkdir(parents=True)
        (nested / "full_extraction.json").write_text("{}", encoding="utf-8")
        adapter = LiveStages("capacitance_vs_vds", images_root="D:/fake_images",
                             stage3_root=tmp_path)
        assert adapter.discover_devices() == []

    def test_full_extraction_json_must_be_a_file(self, tmp_path):
        # A directory that happens to be named full_extraction.json is not
        # the OCR output file and must not count.
        device_dir = tmp_path / "WEIRD"
        device_dir.mkdir()
        (device_dir / "full_extraction.json").mkdir()
        adapter = LiveStages("capacitance_vs_vds", images_root="D:/fake_images",
                             stage3_root=tmp_path)
        assert adapter.discover_devices() == []

    def test_device_folder_may_hold_other_files_too(self, tmp_path):
        device_dir = _make_device_dir(tmp_path, "DEV1")
        (device_dir / "figures").mkdir()
        (device_dir / "raw_ocr.json").write_text("{}", encoding="utf-8")
        adapter = LiveStages("capacitance_vs_vds", images_root="D:/fake_images",
                             stage3_root=tmp_path)
        assert adapter.discover_devices() == ["DEV1"]

    def test_nonexistent_root_raises_not_silently_empty(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        adapter = LiveStages("capacitance_vs_vds", images_root="D:/fake_images",
                             stage3_root=missing)
        with pytest.raises(FileNotFoundError):
            adapter.discover_devices()

    def test_reflects_filesystem_state_at_call_time(self, tmp_path):
        # discover_devices() is not cached — re-running after a new device
        # folder lands must pick it up (Stage-3 output grows over time).
        adapter = LiveStages("capacitance_vs_vds", images_root="D:/fake_images",
                             stage3_root=tmp_path)
        assert adapter.discover_devices() == []
        _make_device_dir(tmp_path, "DEV1")
        assert adapter.discover_devices() == ["DEV1"]

    def test_accepts_string_stage3_root_not_just_path(self, tmp_path):
        # __init__ does not normalize stage3_root into a Path (it's stored
        # and passed straight through to load_figures_by_page elsewhere,
        # same Union[str, PathLike] contract) — discover_devices() must
        # handle a plain string root itself, not assume Path already.
        _make_device_dir(tmp_path, "DEV1")
        adapter = LiveStages("capacitance_vs_vds", images_root="D:/fake_images",
                             stage3_root=str(tmp_path))
        assert adapter.discover_devices() == ["DEV1"]
