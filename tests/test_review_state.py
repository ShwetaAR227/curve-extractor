"""Tests for src.review.review_state — written FIRST (CLAUDE.md §2).

Approve/reject decisions live in their own JSON file keyed by
device::curve_type — NEVER written back into Stage 5's output files
(stage outputs stay immutable). Loading is graceful: a missing or
malformed state file starts fresh instead of crashing, but never
silently overwrites recoverable data on save (atomic writes).
"""
import json

import pytest

from src.review.review_state import (
    decision_key,
    load_state,
    save_state,
    set_decision,
    validate_state,
)


def test_decision_key_combines_device_and_curve_type():
    assert decision_key("BSF050N03LQ3G", "capacitance_vs_vds") == \
        "BSF050N03LQ3G::capacitance_vs_vds"


def test_decision_key_rejects_empty_parts():
    with pytest.raises(ValueError):
        decision_key("", "capacitance_vs_vds")
    with pytest.raises(ValueError):
        decision_key("DEV1", "")


def test_set_decision_records_approve():
    state = {}
    state = set_decision(state, "DEV1", "capacitance_vs_vds", "approve")
    assert state["DEV1::capacitance_vs_vds"]["decision"] == "approve"


def test_set_decision_records_reject_and_overwrites_previous():
    state = set_decision({}, "DEV1", "capacitance_vs_vds", "approve")
    state = set_decision(state, "DEV1", "capacitance_vs_vds", "reject")
    assert state["DEV1::capacitance_vs_vds"]["decision"] == "reject"


def test_set_decision_rejects_invalid_decision_value():
    with pytest.raises(ValueError, match="decision"):
        set_decision({}, "DEV1", "capacitance_vs_vds", "maybe")


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "review_state.json"
    state = set_decision({}, "DEV1", "capacitance_vs_vds", "approve")
    state = set_decision(state, "DEV2", "capacitance_vs_vds", "reject")
    save_state(state, path)
    loaded = load_state(path)
    assert loaded == state


def test_load_state_missing_file_returns_empty(tmp_path):
    assert load_state(tmp_path / "does_not_exist.json") == {}


def test_load_state_malformed_json_returns_empty_without_crash(tmp_path):
    path = tmp_path / "review_state.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_state(path) == {}


def test_load_state_wrong_shape_returns_empty(tmp_path):
    path = tmp_path / "review_state.json"
    path.write_text(json.dumps(["a", "list"]), encoding="utf-8")
    assert load_state(path) == {}


def test_save_state_is_atomic_no_tmp_left(tmp_path):
    path = tmp_path / "review_state.json"
    save_state(set_decision({}, "D", "c", "approve"), path)
    assert list(tmp_path.rglob("*.tmp")) == []


def test_save_state_validates_before_write(tmp_path):
    path = tmp_path / "review_state.json"
    with pytest.raises(ValueError):
        save_state({"DEV1::ct": {"decision": "bogus"}}, path)
    assert not path.exists()


def test_validate_state_accepts_valid_state():
    validate_state({"DEV1::ct": {"decision": "approve"}})  # no raise


def test_validate_state_rejects_non_dict_entries():
    with pytest.raises(ValueError):
        validate_state({"DEV1::ct": "approve"})
