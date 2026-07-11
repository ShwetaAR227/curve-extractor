"""Tests for src.classification.curve_registry — written FIRST (CLAUDE.md §2).

Covers the data-only curve-type registry: lookup, unknown-type handling,
and shape of the two currently-registered specs.
"""
import pytest

from src.classification.curve_registry import (
    CurveTypeSpec,
    get_spec,
    list_registered_types,
)


def test_list_registered_types_contains_both_in_scope_types():
    types = list_registered_types()
    assert "capacitance_vs_vds" in types
    assert "id_vs_vgs" in types


def test_list_registered_types_returns_sorted_list():
    types = list_registered_types()
    assert types == sorted(types)


def test_get_spec_returns_curve_type_spec_instance():
    spec = get_spec("capacitance_vs_vds")
    assert isinstance(spec, CurveTypeSpec)


def test_get_spec_name_matches_lookup_key():
    spec = get_spec("id_vs_vgs")
    assert spec.name == "id_vs_vgs"


def test_get_spec_unknown_type_raises_key_error():
    with pytest.raises(KeyError):
        get_spec("not_a_real_curve_type")


def test_get_spec_unknown_type_error_message_lists_registered_types():
    with pytest.raises(KeyError) as exc_info:
        get_spec("bogus")
    message = str(exc_info.value)
    assert "capacitance_vs_vds" in message
    assert "id_vs_vgs" in message


def test_capacitance_spec_has_ciss_coss_crss_signal():
    spec = get_spec("capacitance_vs_vds")
    combined = " ".join(spec.caption_keywords).lower() + " ".join(
        p for p, _ in spec.positive_phrases
    ).lower()
    assert "ciss" in combined
    assert "coss" in combined
    assert "crss" in combined


def test_capacitance_spec_axis_keywords_have_x_and_y():
    spec = get_spec("capacitance_vs_vds")
    assert "x" in spec.axis_keywords
    assert "y" in spec.axis_keywords
    assert any("vds" in kw.lower() or "drain-to-source voltage" in kw.lower()
               for kw in spec.axis_keywords["x"])
    assert any("capacitance" in kw.lower() for kw in spec.axis_keywords["y"])


def test_id_vs_vgs_spec_axis_keywords_have_x_and_y():
    spec = get_spec("id_vs_vgs")
    assert any("vgs" in kw.lower() for kw in spec.axis_keywords["x"])
    assert any("id" in kw.lower() or "drain" in kw.lower()
               for kw in spec.axis_keywords["y"])


def test_id_vs_vgs_spec_caption_keywords_mention_transfer_characteristics():
    spec = get_spec("id_vs_vgs")
    combined = " ".join(spec.caption_keywords).lower()
    assert "transfer characteristic" in combined


def test_id_vs_vgs_spec_has_negative_phrase_for_capacitance():
    spec = get_spec("id_vs_vgs")
    negatives = [p.lower() for p, _ in spec.negative_phrases]
    assert any("capacitance" in p for p in negatives)


def test_capacitance_spec_has_negative_phrase_for_transfer_characteristics():
    spec = get_spec("capacitance_vs_vds")
    negatives = [p.lower() for p, _ in spec.negative_phrases]
    assert any("transfer characteristic" in p for p in negatives)


def test_positive_phrase_weights_are_positive_numbers():
    for curve_type in list_registered_types():
        spec = get_spec(curve_type)
        for phrase, weight in spec.positive_phrases:
            assert isinstance(phrase, str) and phrase
            assert weight > 0


def test_negative_phrase_weights_are_positive_numbers_applied_as_penalty():
    # Stored as positive magnitudes; scoring.py is responsible for subtracting.
    for curve_type in list_registered_types():
        spec = get_spec(curve_type)
        for phrase, weight in spec.negative_phrases:
            assert isinstance(phrase, str) and phrase
            assert weight > 0


def test_curve_type_spec_is_immutable():
    spec = get_spec("capacitance_vs_vds")
    with pytest.raises((AttributeError, TypeError)):
        spec.name = "hacked"
