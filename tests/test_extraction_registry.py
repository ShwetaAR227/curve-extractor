"""Tests for src.extraction.extraction_registry — written FIRST (CLAUDE.md
§2, red phase). Module does not exist yet.

Data-only routing table (same pattern as
``src.classification.curve_registry``): one ``ExtractionSpec`` per curve
type says whether it routes to the classical (OpenCV) path or the
model-based (LineFormer) path, plus the parameters each path needs. No
if/elif chain anywhere — a lookup, exactly like ``get_spec``.

Real current-state gap this file tests around (per task instructions):
``capacitance_vs_vds``, ``id_vs_vgs``, ``rdson_vs_tj``, ``vgsth_vs_tj``
(2026-07-22), and now ``if_vs_vsd`` (2026-07-22 follow-up, this session)
get real entries here. ``zth_vs_time`` is still the one true gap (raises
KeyError, same as classification's gap).

``vgs_vs_qg`` is a deliberate exception (owner-flagged design point, see
this task's write-up): it gets an EXPLICIT sentinel entry
(``method="none"``) rather than being absent — "we know this curve type
has no extractor slot yet" is a different, calmer fact than "this curve
type was never registered at all". ``get_extraction_spec`` raises the
existing ``KeyError`` for the three truly-absent types, but a distinctly
named ``NoExtractorAvailable`` (raised by the ADAPTER when it sees
``method="none"``, not by this module) is exercised in test_live_stages.py,
not here — this file only proves the registry DATA itself is shaped that
way.
"""
import importlib

import pytest

from src.extraction.extraction_registry import ExtractionSpec, get_extraction_spec


# --------------------------------------------------------- routing decisions

def test_rdson_routes_to_classical():
    spec = get_extraction_spec("rdson_vs_tj")
    assert spec.method == "classical"


def test_capacitance_routes_to_model():
    spec = get_extraction_spec("capacitance_vs_vds")
    assert spec.method == "model"


def test_id_vs_vgs_routes_to_model():
    spec = get_extraction_spec("id_vs_vgs")
    assert spec.method == "model"


# ------------------------------------------------------- the real gap today

def test_zth_vs_time_not_yet_registered_raises_keyerror():
    with pytest.raises(KeyError):
        get_extraction_spec("zth_vs_time")


def test_unregistered_curve_type_error_message_lists_registered_types():
    # Same "immediately actionable" style as curve_registry.get_spec.
    with pytest.raises(KeyError, match="rdson_vs_tj"):
        get_extraction_spec("totally_unknown_curve_type")


# ---------------------------------------------------- vgs_vs_qg's sentinel

def test_vgs_vs_qg_has_an_explicit_none_method_entry_not_absent():
    # Deliberately registered (does NOT raise), method says "no extractor".
    spec = get_extraction_spec("vgs_vs_qg")
    assert spec.method == "none"


def test_vgs_vs_qg_sentinel_has_no_checkpoint_or_config():
    spec = get_extraction_spec("vgs_vs_qg")
    assert spec.checkpoint is None
    assert spec.config is None


# --------------------------------------------------- per-curve-type params

def test_capacitance_expects_exactly_three_curves():
    spec = get_extraction_spec("capacitance_vs_vds")
    assert spec.expected_curve_count == 3


def test_rdson_allows_one_or_two_curves():
    spec = get_extraction_spec("rdson_vs_tj")
    # rdson's classical path internally accepts 1 (IR template) or 2
    # (Infineon typ/max) — the registry must reflect both, not just one.
    assert 1 in spec.expected_curve_count
    assert 2 in spec.expected_curve_count


def test_classical_entries_carry_no_checkpoint_or_config():
    spec = get_extraction_spec("rdson_vs_tj")
    assert spec.checkpoint is None
    assert spec.config is None


# ------------------------------------------------------ vgsth_vs_tj (classical)

def test_vgsth_vs_tj_routes_to_classical():
    spec = get_extraction_spec("vgsth_vs_tj")
    assert spec.method == "classical"


def test_vgsth_vs_tj_classical_entry_carries_no_checkpoint_or_config():
    spec = get_extraction_spec("vgsth_vs_tj")
    assert spec.checkpoint is None
    assert spec.config is None


def test_vgsth_vs_tj_curve_count_is_not_a_fixed_set():
    # Unlike rdson_vs_tj's fixed (1, 2), vgsth's curve count is determined
    # dynamically per-chart by count_expected_curves(ocr_lines) inside
    # classical_vgsth.py itself -- there's no fixed set to register here
    # truthfully. Also: this field is never read on the "classical" dispatch
    # path at all (only run_pipeline's "model" branch consumes it) -- see
    # src/orchestrator/live_stages.py's run_extraction.
    spec = get_extraction_spec("vgsth_vs_tj")
    assert spec.expected_curve_count is None


def test_model_entries_carry_a_checkpoint_and_config():
    spec = get_extraction_spec("capacitance_vs_vds")
    assert spec.checkpoint
    assert spec.config


def test_checkpoints_are_distinct_per_model_curve_type():
    # F.18: id_vs_vgs must get ITS OWN (Run 3) checkpoint, never capacitance's.
    cap = get_extraction_spec("capacitance_vs_vds")
    idv = get_extraction_spec("id_vs_vgs")
    assert cap.checkpoint != idv.checkpoint


def test_model_entries_have_a_positive_score_threshold():
    spec = get_extraction_spec("capacitance_vs_vds")
    assert 0.0 < spec.score_thr <= 1.0


# ------------------------------------------------- data-driven, not if/elif

def test_registry_is_a_live_data_lookup_not_hardcoded_dispatch(monkeypatch):
    # Proof the routing decision is a genuine dict lookup: patching in a new
    # entry at runtime makes get_extraction_spec see it immediately, with NO
    # code path added anywhere — the same style already used for
    # src.extraction.pipeline.PLAUSIBILITY_SPECS /
    # src.extraction.naming._NAMING_REGISTRY in this codebase.
    import src.extraction.extraction_registry as registry_mod

    fake_spec = ExtractionSpec(
        curve_type="test_type", method="classical", checkpoint=None,
        config=None, score_thr=0.5, expected_curve_count=1,
    )
    monkeypatch.setitem(registry_mod._REGISTRY, "test_type", fake_spec)
    assert get_extraction_spec("test_type") is fake_spec


def test_registered_curve_types_are_a_strict_subset_of_in_scope_types():
    # No entry for anything outside CLAUDE.md §1's 7 in-scope curve types
    # (e.g. no accidental "breakdown_voltage" entry sneaking in).
    import src.extraction.extraction_registry as registry_mod

    in_scope = {
        "capacitance_vs_vds", "rdson_vs_tj", "if_vs_vsd", "id_vs_vgs",
        "vgs_vs_qg", "vgsth_vs_tj", "zth_vs_time",
    }
    assert set(registry_mod._REGISTRY) <= in_scope


# ============================================================
# classical_pipeline field (owner-approved routing fix, 2026-07-22)
#
# ExtractionSpec gains a new field: classical_pipeline, holding the ACTUAL
# function object (e.g. classical.run_classical_pipeline), not a string
# path — this is what lets live_stages.py's classical dispatch become a
# genuine data lookup (spec.classical_pipeline(...)) instead of a single
# hardcoded import shared by every "classical" curve type regardless of
# which one it actually is.
# ============================================================

def test_extraction_spec_accepts_classical_pipeline_field_defaults_to_none():
    # Omitting it entirely must not break existing construction calls
    # elsewhere in this file/codebase (e.g. test_registry_is_a_live_data_
    # lookup_not_hardcoded_dispatch above, which predates this field).
    spec = ExtractionSpec(
        curve_type="some_type", method="none", checkpoint=None,
        config=None, score_thr=0.0, expected_curve_count=0,
    )
    assert spec.classical_pipeline is None


def test_rdson_classical_pipeline_is_the_real_function_object():
    import src.extraction.classical as classical_mod

    spec = get_extraction_spec("rdson_vs_tj")
    assert spec.classical_pipeline is classical_mod.run_classical_pipeline


def test_vgsth_classical_pipeline_is_the_real_function_object():
    import src.extraction.classical_vgsth as classical_vgsth_mod

    spec = get_extraction_spec("vgsth_vs_tj")
    assert spec.classical_pipeline is classical_vgsth_mod.run_classical_pipeline


def test_capacitance_classical_pipeline_is_none():
    # Model-based entry — nothing classical to point at.
    assert get_extraction_spec("capacitance_vs_vds").classical_pipeline is None


def test_id_vs_vgs_classical_pipeline_is_none():
    assert get_extraction_spec("id_vs_vgs").classical_pipeline is None


def test_extraction_registry_module_imports_cleanly_no_circular_dependency():
    # Explicit, not assumed: re-triggers the module's own import machinery
    # (which will import classical.py and classical_vgsth.py at module
    # level once this field is implemented) and confirms it completes
    # without an ImportError. See test_classical_modules_have_no_
    # extraction_registry_reference in test_live_stages.py, and this
    # session's report, for the full transitive-closure investigation
    # this is a runtime companion to (grep found ONLY live_stages.py
    # currently imports extraction_registry anywhere in src/; neither
    # classical.py nor classical_vgsth.py nor anything in their transitive
    # import closure — curve_detection, naming/*, extraction.pipeline,
    # schema, skeletonize, dedup, inference, calibration.ticks,
    # training.eval_lineformer, training.predict_to_cvat,
    # dataset_tools.collect_images — imports extraction_registry,
    # live_stages, or orchestrator.pipeline at any depth).
    import src.extraction.extraction_registry as registry_mod

    importlib.reload(registry_mod)


# ============================================================
# model_pipeline field + if_vs_vsd entry (owner-approved routing addition,
# 2026-07-22 follow-up session)
#
# model_pipeline is the "model"-dispatch analogue of classical_pipeline:
# holds the ACTUAL function object (e.g. model_if_vsd.run_model_pipeline)
# for a "model"-method entry that needs its own override wrapper instead
# of the generic run_pipeline call — None for capacitance_vs_vds/
# id_vs_vgs, which have no such override.
# ============================================================

def test_extraction_spec_accepts_model_pipeline_field_defaults_to_none():
    spec = ExtractionSpec(
        curve_type="some_type", method="none", checkpoint=None,
        config=None, score_thr=0.0, expected_curve_count=0,
    )
    assert spec.model_pipeline is None


def test_capacitance_model_pipeline_is_none():
    # Plain model-routed entry -- no override, uses the generic run_pipeline.
    assert get_extraction_spec("capacitance_vs_vds").model_pipeline is None


def test_id_vs_vgs_model_pipeline_is_none():
    assert get_extraction_spec("id_vs_vgs").model_pipeline is None


def test_rdson_model_pipeline_is_none():
    # Classical entries have nothing to do with model_pipeline either.
    assert get_extraction_spec("rdson_vs_tj").model_pipeline is None


def test_vgsth_model_pipeline_is_none():
    assert get_extraction_spec("vgsth_vs_tj").model_pipeline is None


def test_if_vs_vsd_is_now_registered_routes_to_model():
    spec = get_extraction_spec("if_vs_vsd")
    assert spec.method == "model"


def test_if_vs_vsd_has_its_own_checkpoint_and_config():
    spec = get_extraction_spec("if_vs_vsd")
    assert spec.checkpoint == "body_diode_run1/best_segm_mAP_50_iter_2200.pth"
    assert spec.config == "src/training/configs/lineformer_body_diode_run1.py"


def test_if_vs_vsd_checkpoint_distinct_from_other_model_entries():
    if_vsd = get_extraction_spec("if_vs_vsd")
    cap = get_extraction_spec("capacitance_vs_vds")
    idv = get_extraction_spec("id_vs_vgs")
    assert if_vsd.checkpoint not in (cap.checkpoint, idv.checkpoint)


def test_if_vs_vsd_curve_count_is_not_a_fixed_set():
    # Real charts show 2 curves (common) or 4 (rare, compound percentile
    # label) -- but if_vs_vsd's model_pipeline override
    # (model_if_vsd.run_model_pipeline) never reads this field; its real
    # expected count comes dynamically per-chart from
    # naming.if_vs_vsd.count_expected_curves(ocr_lines) instead. No fixed
    # value would be truthful here -- same relationship as vgsth_vs_tj's
    # own expected_curve_count=None (classical entry, dynamic per-chart
    # count via its own naming module).
    spec = get_extraction_spec("if_vs_vsd")
    assert spec.expected_curve_count is None


def test_if_vs_vsd_has_a_positive_score_threshold():
    spec = get_extraction_spec("if_vs_vsd")
    assert 0.0 < spec.score_thr <= 1.0


def test_if_vs_vsd_classical_pipeline_is_none():
    # Model-based entry -- nothing classical to point at.
    assert get_extraction_spec("if_vs_vsd").classical_pipeline is None


def test_if_vs_vsd_model_pipeline_is_the_real_function_object():
    import src.extraction.model_if_vsd as model_if_vsd_mod

    spec = get_extraction_spec("if_vs_vsd")
    assert spec.model_pipeline is model_if_vsd_mod.run_model_pipeline


def test_if_vs_vsd_model_pipeline_distinct_from_other_curve_types_none():
    # Only if_vs_vsd carries a real override; every other entry's
    # model_pipeline stays None (no accidental cross-wiring).
    assert get_extraction_spec("if_vs_vsd").model_pipeline is not None
    for curve_type in ("capacitance_vs_vds", "id_vs_vgs", "rdson_vs_tj", "vgsth_vs_tj"):
        assert get_extraction_spec(curve_type).model_pipeline is None
