"""Tests for src.training.checkpoint_retention — written FIRST (CLAUDE.md §2).

Pure planning logic (no filesystem I/O): given a checkpoint directory
listing, decide which files survive ("latest" periodic + "best") and which
get deleted. Actual deletion is a thin, untested I/O wrapper.
"""
import pytest

from src.training.checkpoint_retention import plan_retention


class TestPlanRetention:
    def test_keeps_latest_and_best_deletes_rest(self):
        files = ["iter_200.pth", "iter_400.pth", "iter_600.pth", "latest.pth",
                 "best_segm_mAP_50_iter_400.pth"]
        plan = plan_retention(files)
        assert set(plan["keep"]) == {"iter_600.pth", "latest.pth",
                                     "best_segm_mAP_50_iter_400.pth"}
        assert set(plan["delete"]) == {"iter_200.pth", "iter_400.pth"}

    def test_no_best_file_yet(self):
        files = ["iter_200.pth", "iter_400.pth", "latest.pth"]
        plan = plan_retention(files)
        assert set(plan["keep"]) == {"iter_400.pth", "latest.pth"}
        assert plan["delete"] == ["iter_200.pth"]

    def test_best_coincides_with_latest_iteration(self):
        files = ["iter_200.pth", "iter_400.pth", "latest.pth",
                 "best_segm_mAP_50_iter_400.pth"]
        plan = plan_retention(files)
        assert set(plan["keep"]) == {"iter_400.pth", "latest.pth",
                                     "best_segm_mAP_50_iter_400.pth"}
        assert plan["delete"] == ["iter_200.pth"]

    def test_multiple_stale_best_files_keeps_highest_iteration_only(self):
        # Should not normally happen (mmcv replaces the best file in place),
        # but the planner must be defensive rather than crash or keep both.
        files = ["best_segm_mAP_50_iter_200.pth",
                 "best_segm_mAP_50_iter_600.pth", "latest.pth"]
        plan = plan_retention(files)
        assert plan["keep"] == ["latest.pth", "best_segm_mAP_50_iter_600.pth"] \
            or set(plan["keep"]) == {"latest.pth", "best_segm_mAP_50_iter_600.pth"}
        assert plan["delete"] == ["best_segm_mAP_50_iter_200.pth"]

    def test_no_latest_marker_present(self):
        files = ["iter_200.pth", "iter_400.pth"]
        plan = plan_retention(files)
        assert plan["keep"] == ["iter_400.pth"]
        assert plan["delete"] == ["iter_200.pth"]

    def test_empty_directory(self):
        assert plan_retention([]) == {"keep": [], "delete": []}

    def test_only_best_file_present(self):
        files = ["best_segm_mAP_50_iter_400.pth"]
        plan = plan_retention(files)
        assert plan["keep"] == ["best_segm_mAP_50_iter_400.pth"]
        assert plan["delete"] == []

    def test_non_checkpoint_files_ignored(self):
        files = ["iter_200.pth", "iter_400.pth", "latest.pth",
                 "run_manifest.json", "resolved_config.json"]
        plan = plan_retention(files)
        assert "run_manifest.json" not in plan["keep"]
        assert "run_manifest.json" not in plan["delete"]

    def test_unrecognized_best_metric_name_still_handled(self):
        files = ["best_bbox_mAP_iter_300.pth", "iter_300.pth", "latest.pth"]
        plan = plan_retention(files)
        assert "best_bbox_mAP_iter_300.pth" in plan["keep"]

    def test_delete_list_has_no_duplicates_with_keep(self):
        files = ["iter_200.pth", "iter_400.pth", "iter_600.pth", "latest.pth",
                 "best_segm_mAP_50_iter_600.pth"]
        plan = plan_retention(files)
        assert not set(plan["keep"]) & set(plan["delete"])
