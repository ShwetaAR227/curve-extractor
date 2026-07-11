"""Tests for src.extraction.inference — written FIRST (CLAUDE.md §2).

Wraps the SAME mmdet calling convention already used by
src.training.eval_lineformer/predict_to_cvat (reused, not duplicated: mask
decoding via eval_lineformer._pred_to_bool_masks, score filtering via
predict_to_cvat.filter_by_score). Heavy imports (mmdet, torch) stay lazy
inside load_model/run_inference so this module needs no GPU to unit test
(CLAUDE.md §2) — the pure combining logic is factored into
detections_from_raw, and load_model/run_inference are tested by injecting a
fake mmdet.apis into sys.modules.
"""
import sys
import types

import numpy as np
import pytest

from src.extraction.inference import (
    DEFAULT_SCORE_THR,
    Detection,
    detections_from_raw,
    load_model,
    run_inference,
)


def make_bbox(score):
    return np.array([0.0, 0.0, 10.0, 10.0, score], dtype=float)


def test_detections_from_raw_keeps_scores_above_threshold():
    bbox_result = [[make_bbox(0.9), make_bbox(0.3)]]
    segm_result = [[np.ones((5, 5), dtype=bool), np.zeros((5, 5), dtype=bool)]]
    detections = detections_from_raw(bbox_result, segm_result, score_thr=0.5)
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.9)


def test_detections_from_raw_default_threshold_is_point_five():
    assert DEFAULT_SCORE_THR == 0.5


def test_detections_from_raw_keeps_boundary_score():
    bbox_result = [[make_bbox(0.5)]]
    segm_result = [[np.ones((3, 3), dtype=bool)]]
    detections = detections_from_raw(bbox_result, segm_result, score_thr=0.5)
    assert len(detections) == 1


def test_detections_from_raw_empty_result_returns_empty_list():
    assert detections_from_raw([[]], [[]]) == []


def test_detections_from_raw_returns_boolean_masks():
    bbox_result = [[make_bbox(0.8)]]
    segm_result = [[np.ones((4, 4), dtype=np.uint8)]]
    detections = detections_from_raw(bbox_result, segm_result, score_thr=0.5)
    assert detections[0].mask.dtype == bool


def test_detection_is_a_simple_dataclass():
    mask = np.ones((2, 2), dtype=bool)
    d = Detection(score=0.7, mask=mask)
    assert d.score == 0.7
    assert d.mask is mask


def test_load_model_calls_mmdet_init_detector(monkeypatch):
    calls = {}

    def fake_init_detector(config, checkpoint, device="cuda:0"):
        calls["args"] = (config, checkpoint, device)
        return "fake_model"

    fake_apis = types.SimpleNamespace(init_detector=fake_init_detector)
    fake_mmdet = types.ModuleType("mmdet")
    fake_mmdet.apis = fake_apis
    monkeypatch.setitem(sys.modules, "mmdet", fake_mmdet)
    monkeypatch.setitem(sys.modules, "mmdet.apis", fake_apis)

    model = load_model("ckpt.pth", "config.py", device="cpu")
    assert model == "fake_model"
    assert calls["args"] == ("config.py", "ckpt.pth", "cpu")


def test_run_inference_calls_inference_detector_and_filters(monkeypatch):
    bbox_result = [[make_bbox(0.9), make_bbox(0.1)]]
    segm_result = [[np.ones((3, 3), dtype=bool), np.zeros((3, 3), dtype=bool)]]

    def fake_inference_detector(model, image_path):
        assert model == "fake_model"
        return bbox_result, segm_result

    fake_apis = types.SimpleNamespace(inference_detector=fake_inference_detector)
    fake_mmdet = types.ModuleType("mmdet")
    fake_mmdet.apis = fake_apis
    monkeypatch.setitem(sys.modules, "mmdet", fake_mmdet)
    monkeypatch.setitem(sys.modules, "mmdet.apis", fake_apis)

    detections = run_inference("fake_model", "some_image.png", score_thr=0.5)
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.9)
