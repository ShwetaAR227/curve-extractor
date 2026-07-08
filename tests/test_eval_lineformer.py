"""Tests for src.training.eval_lineformer metric logic — written FIRST (CLAUDE.md §2).

Only the pure metric/report functions are unit-tested here (no GPU, no
network, no torch/mmdet imports). Model inference is integration-tested
against the real checkpoints on the GPU box.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.training.eval_lineformer import (
    DEFAULT_IOU_THR,
    DEFAULT_SCORE_THR,
    build_report,
    compute_recall,
    mask_iou,
    recall_by_curve,
)


def box_mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


# ------------------------------------------------------------------- mask_iou

class TestMaskIou:
    def test_identical_masks_iou_1(self):
        a = box_mask(20, 20, 5, 15, 5, 15)
        assert mask_iou(a, a) == pytest.approx(1.0)

    def test_disjoint_masks_iou_0(self):
        a = box_mask(20, 20, 0, 5, 0, 5)
        b = box_mask(20, 20, 10, 15, 10, 15)
        assert mask_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap_known_value(self):
        # a: 10x10=100 px, b: 10x10=100 px, overlap 5x10=50 px -> IoU 50/150
        a = box_mask(30, 30, 0, 10, 0, 10)
        b = box_mask(30, 30, 5, 15, 0, 10)
        assert mask_iou(a, b) == pytest.approx(50 / 150)

    def test_both_empty_masks_iou_0(self):
        a = np.zeros((10, 10), dtype=bool)
        assert mask_iou(a, a) == pytest.approx(0.0)


# -------------------------------------------------------------- compute_recall

class TestComputeRecall:
    def test_single_gt_matched(self):
        gt = [box_mask(20, 20, 0, 10, 0, 10)]
        preds = [box_mask(20, 20, 0, 10, 0, 10)]
        matched = compute_recall(gt, preds, scores=[0.9])
        assert matched == [True]

    def test_pred_below_score_threshold_ignored(self):
        gt = [box_mask(20, 20, 0, 10, 0, 10)]
        preds = [box_mask(20, 20, 0, 10, 0, 10)]
        matched = compute_recall(gt, preds, scores=[0.3])  # < 0.5 default
        assert matched == [False]

    def test_pred_below_iou_threshold_unmatched(self):
        gt = [box_mask(30, 30, 0, 10, 0, 10)]
        preds = [box_mask(30, 30, 8, 18, 0, 10)]  # IoU = 20/180 ≈ 0.11
        matched = compute_recall(gt, preds, scores=[0.9])
        assert matched == [False]

    def test_any_match_semantics_multiple_preds(self):
        gt = [box_mask(30, 30, 0, 10, 0, 10)]
        preds = [box_mask(30, 30, 20, 30, 20, 30),   # disjoint
                 box_mask(30, 30, 0, 10, 0, 10)]      # exact
        matched = compute_recall(gt, preds, scores=[0.9, 0.6])
        assert matched == [True]

    def test_zero_predictions_no_crash(self):
        gt = [box_mask(20, 20, 0, 10, 0, 10)]
        matched = compute_recall(gt, [], scores=[])
        assert matched == [False]

    def test_zero_gt_no_crash(self):
        preds = [box_mask(20, 20, 0, 10, 0, 10)]
        matched = compute_recall([], preds, scores=[0.9])
        assert matched == []

    def test_custom_thresholds_respected(self):
        gt = [box_mask(30, 30, 0, 10, 0, 10)]
        preds = [box_mask(30, 30, 2, 12, 0, 10)]  # IoU = 80/120 ≈ 0.667
        assert compute_recall(gt, preds, scores=[0.9], iou_thr=0.75) == [False]
        assert compute_recall(gt, preds, scores=[0.9], iou_thr=0.5) == [True]
        assert compute_recall(gt, preds, scores=[0.4], score_thr=0.3) == [True]


# ------------------------------------------------------------- recall_by_curve

class TestRecallByCurve:
    def test_grouping_counts(self):
        entries = [("Ciss", True), ("Ciss", True), ("Coss", False),
                   ("Coss", True), ("Crss", False)]
        by_curve = recall_by_curve(entries)
        assert by_curve["Ciss"] == {"matched": 2, "total": 2, "recall": 1.0}
        assert by_curve["Coss"] == {"matched": 1, "total": 2, "recall": 0.5}
        assert by_curve["Crss"] == {"matched": 0, "total": 1, "recall": 0.0}

    def test_empty_entries(self):
        assert recall_by_curve([]) == {}


# ---------------------------------------------------------------- build_report

class TestBuildReport:
    def _report(self):
        return build_report(
            checkpoint="ckpt.pth", config="cfg.py", dataset_hash="ab" * 32,
            map50=0.42, map75=0.21,
            match_entries=[("Ciss", True), ("Coss", False)],
            n_test_images=24,
        )

    def test_required_keys_present(self):
        report = self._report()
        for key in ("checkpoint", "config", "dataset_hash", "map50", "map75",
                    "recall_overall", "recall_by_curve", "n_test_images",
                    "n_gt_instances", "timestamp", "recall_score_thr",
                    "recall_iou_thr"):
            assert key in report, f"missing {key}"

    def test_recall_computed_from_entries(self):
        report = self._report()
        assert report["recall_overall"] == pytest.approx(0.5)
        assert report["n_gt_instances"] == 2
        assert report["recall_by_curve"]["Ciss"]["recall"] == 1.0

    def test_threshold_labels_recorded(self):
        report = self._report()
        assert report["recall_score_thr"] == DEFAULT_SCORE_THR == 0.5
        assert report["recall_iou_thr"] == DEFAULT_IOU_THR == 0.5

    def test_report_json_serializable(self):
        json.dumps(self._report())

    def test_zero_gt_dataset_recall_none(self):
        report = build_report(
            checkpoint="c", config="f", dataset_hash="x", map50=0.0,
            map75=0.0, match_entries=[], n_test_images=0)
        assert report["recall_overall"] is None
        assert report["n_gt_instances"] == 0
