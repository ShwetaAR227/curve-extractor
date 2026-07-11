"""Tests for src.orchestrator.pipeline + queue — written FIRST (CLAUDE.md §2).

Stage 7 is pure orchestration: it calls Stage 4/5/6 through a small adapter
protocol (injected here as fakes) and never reimplements their logic. Every
device lands in exactly one of the six statuses; "finalized" strictly
requires an explicit human approval (no auto-pass) plus final validation.
"""
import json

import pytest

from src.orchestrator.pipeline import (
    STATUSES,
    DeviceResult,
    process_device,
    run_batch,
)
from src.orchestrator.queue import QUEUE_STATUSES, build_queue, write_queue
from src.review.review_state import set_decision


# ------------------------------------------------------------------ fakes

class FakeClassification:
    def __init__(self, status="matched"):
        self.status = status


def ok_stage5(device="DEV1", **overrides):
    result = {
        "device": device,
        "curve_type": "capacitance_vs_vds",
        "source_image": f"{device}__fig.png",
        "status": "ok",
        "review_reason": None,
        "duplicates_removed": 0,
        "calibration": {"x_slope": 10.0, "x_intercept": 100.0, "y_slope": -50.0,
                        "y_intercept": 500.0, "x_log": False, "y_log": True},
        "curves": [
            {"curve_name": "Ciss", "confidence": 0.9, "points": [{"x": 1.0, "y": 100.0}]},
            {"curve_name": "Coss", "confidence": 0.9, "points": [{"x": 1.0, "y": 50.0}]},
            {"curve_name": "Crss", "confidence": 0.9, "points": [{"x": 1.0, "y": 10.0}]},
        ],
        "units": "pF",
    }
    result.update(overrides)
    return result


class FakeStages:
    """Injectable stage adapter: maps device -> canned behavior."""

    def __init__(self, classification=None, stage5=None,
                 classify_raises=None, extract_raises=None):
        self._classification = classification or {}
        self._stage5 = stage5 or {}
        self._classify_raises = classify_raises or set()
        self._extract_raises = extract_raises or set()

    def run_classification(self, device):
        if device in self._classify_raises:
            raise RuntimeError("classifier exploded")
        return self._classification.get(device, FakeClassification("matched"))

    def run_extraction(self, device, classification):
        if device in self._extract_raises:
            raise RuntimeError("extraction exploded")
        return self._stage5[device]


def approved(device="DEV1"):
    return set_decision({}, device, "capacitance_vs_vds", "approve")


def rejected(device="DEV1"):
    return set_decision({}, device, "capacitance_vs_vds", "reject")


# ------------------------------------------------- status assignment (6 statuses)

def test_approved_valid_result_finalized():
    stages = FakeStages(stage5={"DEV1": ok_stage5()})
    result = process_device("DEV1", "capacitance_vs_vds", stages, approved())
    assert result.status == "finalized"
    assert result.final_record is not None


def test_no_decision_is_pending_review_never_auto_finalized():
    # THE no-auto-pass rule: perfectly clean ok result, no review entry.
    stages = FakeStages(stage5={"DEV1": ok_stage5()})
    result = process_device("DEV1", "capacitance_vs_vds", stages, {})
    assert result.status == "pending_review"
    assert result.final_record is None


def test_rejected_never_finalized_regardless_of_clean_data():
    stages = FakeStages(stage5={"DEV1": ok_stage5()})
    result = process_device("DEV1", "capacitance_vs_vds", stages, rejected())
    assert result.status == "rejected"
    assert result.final_record is None


def test_stage4_no_match_is_failed_classification():
    stages = FakeStages(classification={"DEV1": FakeClassification("no_match")},
                        stage5={"DEV1": ok_stage5()})
    result = process_device("DEV1", "capacitance_vs_vds", stages, approved())
    assert result.status == "failed_classification"


def test_stage4_quarantined_is_failed_classification():
    stages = FakeStages(classification={"DEV1": FakeClassification("quarantined")},
                        stage5={"DEV1": ok_stage5()})
    result = process_device("DEV1", "capacitance_vs_vds", stages, {})
    assert result.status == "failed_classification"


def test_stage5_exception_is_failed_extraction_not_a_crash():
    stages = FakeStages(extract_raises={"DEV1"})
    result = process_device("DEV1", "capacitance_vs_vds", stages, approved())
    assert result.status == "failed_extraction"
    assert "exploded" in result.reason


def test_stage4_exception_is_failed_classification_not_a_crash():
    stages = FakeStages(classify_raises={"DEV1"}, stage5={"DEV1": ok_stage5()})
    result = process_device("DEV1", "capacitance_vs_vds", stages, {})
    assert result.status == "failed_classification"


def test_stage5_needs_review_status_maps_to_needs_review():
    stage5 = ok_stage5(status="needs_review", review_reason="units_undetected",
                       units=None)
    stages = FakeStages(stage5={"DEV1": stage5})
    result = process_device("DEV1", "capacitance_vs_vds", stages, {})
    assert result.status == "needs_review"
    assert "units_undetected" in result.reason


def test_approved_but_invalid_downgrades_to_needs_review():
    # Approval is not a validation bypass: approved + missing units ->
    # needs_review with the validation reason, never silently shipped.
    stage5 = ok_stage5(status="needs_review", review_reason="implausible_calibration: x",
                       units=None)
    stages = FakeStages(stage5={"DEV1": stage5})
    result = process_device("DEV1", "capacitance_vs_vds", stages, approved())
    assert result.status == "needs_review"
    assert result.final_record is None


def test_stage5_needs_review_plus_reject_is_rejected():
    stage5 = ok_stage5(status="needs_review", review_reason="x", units=None)
    stages = FakeStages(stage5={"DEV1": stage5})
    result = process_device("DEV1", "capacitance_vs_vds", stages, rejected())
    assert result.status == "rejected"


def test_every_result_lands_in_exactly_one_known_status():
    cases = [
        (FakeStages(stage5={"D": ok_stage5(device="D")}), approved("D")),
        (FakeStages(stage5={"D": ok_stage5(device="D")}), {}),
        (FakeStages(stage5={"D": ok_stage5(device="D")}), rejected("D")),
        (FakeStages(classification={"D": FakeClassification("no_match")}), {}),
        (FakeStages(extract_raises={"D"}), {}),
        (FakeStages(stage5={"D": ok_stage5(device="D", status="needs_review",
                                           review_reason="r", units=None)}), {}),
    ]
    for stages, state in cases:
        result = process_device("D", "capacitance_vs_vds", stages, state)
        assert result.status in STATUSES


def test_auto_approve_mode_finalizes_valid_ok_without_decision():
    # require_manual_approval=False is the future flip; default stays True.
    stages = FakeStages(stage5={"DEV1": ok_stage5()})
    result = process_device("DEV1", "capacitance_vs_vds", stages, {},
                            require_manual_approval=False)
    assert result.status == "finalized"


def test_auto_approve_mode_still_respects_explicit_reject():
    stages = FakeStages(stage5={"DEV1": ok_stage5()})
    result = process_device("DEV1", "capacitance_vs_vds", stages, rejected(),
                            require_manual_approval=False)
    assert result.status == "rejected"


# ------------------------------------------------------------- final record

def test_final_record_has_provenance():
    stages = FakeStages(stage5={"DEV1": ok_stage5()})
    result = process_device("DEV1", "capacitance_vs_vds", stages, approved())
    record = result.final_record
    assert record["device"] == "DEV1"
    assert record["curve_type"] == "capacitance_vs_vds"
    assert record["provenance"]["review_decision"] == "approve"
    assert record["provenance"]["decided_at"]
    assert record["provenance"]["finalized_at"]
    assert record["units"] == "pF"
    assert len(record["curves"]) == 3


# ----------------------------------------------------------------- queue

def test_build_queue_includes_only_actionable_statuses():
    results = [
        DeviceResult("A", "ct", "finalized", "ok", None, None),
        DeviceResult("B", "ct", "pending_review", "awaiting decision", None, None),
        DeviceResult("C", "ct", "needs_review", "units_undetected", None, None),
        DeviceResult("D", "ct", "failed_classification", "no_match", None, None),
        DeviceResult("E", "ct", "failed_extraction", "boom", None, None),
        DeviceResult("F", "ct", "rejected", "reviewer rejected", None, None),
    ]
    queue = build_queue(results)
    devices = {entry["device"] for entry in queue}
    assert devices == {"B", "C", "D", "E"}
    assert set(QUEUE_STATUSES) == {"pending_review", "needs_review",
                                   "failed_classification", "failed_extraction"}


def test_queue_entries_carry_reasons():
    results = [DeviceResult("C", "ct", "needs_review", "units_undetected", None, None)]
    queue = build_queue(results)
    assert queue[0]["reason"] == "units_undetected"


def test_write_queue_rerun_overwrites_no_duplicates(tmp_path):
    path = tmp_path / "queue.json"
    results = [DeviceResult("B", "ct", "pending_review", "awaiting", None, None)]
    write_queue(build_queue(results), path)
    write_queue(build_queue(results), path)  # re-run
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert len(loaded) == 1


# ----------------------------------------------------------------- run_batch

def make_mixed_batch():
    stage5 = {
        "FIN": ok_stage5(device="FIN"),
        "PEND": ok_stage5(device="PEND"),
        "REJ": ok_stage5(device="REJ"),
        "NR": ok_stage5(device="NR", status="needs_review",
                        review_reason="units_undetected", units=None),
    }
    classification = {"FC": FakeClassification("no_match")}
    stages = FakeStages(classification=classification, stage5=stage5,
                        extract_raises={"FE"})
    state = set_decision({}, "FIN", "capacitance_vs_vds", "approve")
    state = set_decision(state, "REJ", "capacitance_vs_vds", "reject")
    return ["FIN", "PEND", "REJ", "NR", "FC", "FE"], stages, state


def test_run_batch_counts_add_up(tmp_path):
    devices, stages, state = make_mixed_batch()
    summary = run_batch(devices, "capacitance_vs_vds", stages, state, tmp_path)
    assert summary["counts"] == {
        "finalized": 1, "pending_review": 1, "rejected": 1,
        "needs_review": 1, "failed_classification": 1, "failed_extraction": 1,
    }
    assert sum(summary["counts"].values()) == len(devices)


def test_run_batch_writes_finalized_records_and_queue(tmp_path):
    devices, stages, state = make_mixed_batch()
    summary = run_batch(devices, "capacitance_vs_vds", stages, state, tmp_path)
    final_files = list((tmp_path / "final").rglob("*.json"))
    assert len(final_files) == 1
    record = json.loads(final_files[0].read_text(encoding="utf-8"))
    assert record["device"] == "FIN"
    queue = json.loads((tmp_path / "followup_queue.json").read_text(encoding="utf-8"))
    assert {e["device"] for e in queue} == {"PEND", "NR", "FC", "FE"}


def test_run_batch_one_bad_device_does_not_stop_the_rest(tmp_path):
    devices, stages, state = make_mixed_batch()
    # FE raises inside extraction; every other device still gets processed.
    summary = run_batch(devices, "capacitance_vs_vds", stages, state, tmp_path)
    assert summary["counts"]["failed_extraction"] == 1
    assert summary["counts"]["finalized"] == 1
