"""Tests for src.extraction.extraction_registry — written FIRST (CLAUDE.md
§2, red phase). Module does not exist yet.

Data-only routing table (same pattern as
``src.classification.curve_registry``): one ``ExtractionSpec`` per curve
type says whether it routes to the classical (OpenCV) path or the
model-based (LineFormer) path, plus the parameters each path needs. No
if/elif chain anywhere — a lookup, exactly like ``get_spec``.

Real current-state gap this file tests around (per task instructions):
only ``capacitance_vs_vds``, ``id_vs_vgs``, ``rdson_vs_tj`` get real
entries here. ``if_vs_vsd``, ``zth_vs_time``, ``vgsth_vs_tj`` have none at
all (raise KeyError, same as classification's gap).

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

def test_if_vs_vsd_not_yet_registered_raises_keyerror():
    with pytest.raises(KeyError):
        get_extraction_spec("if_vs_vsd")


def test_zth_vs_time_not_yet_registered_raises_keyerror():
    with pytest.raises(KeyError):
        get_extraction_spec("zth_vs_time")


def test_vgsth_vs_tj_not_yet_registered_raises_keyerror():
    with pytest.raises(KeyError):
        get_extraction_spec("vgsth_vs_tj")


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
