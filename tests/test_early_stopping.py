"""Tests for src.training.early_stopping — written FIRST (CLAUDE.md §2).

Pure decision logic, no GPU/torch/mmdet dependency: patience-based early
stopping on a val-metric history, and train/val divergence detection.
"""
import pytest

from src.training.early_stopping import (
    DEFAULT_DIVERGENCE_WINDOW,
    detect_divergence,
    format_status_line,
    should_stop,
)


class TestShouldStop:
    def test_no_history_never_stops(self):
        stop, info = should_stop([], patience_iters=600, eval_interval=200)
        assert stop is False
        assert info["best_iteration"] is None

    def test_still_improving_does_not_stop(self):
        history = [(200, 0.10), (400, 0.15), (600, 0.20)]
        stop, info = should_stop(history, patience_iters=600, eval_interval=200)
        assert stop is False
        assert info["best_iteration"] == 600
        assert info["iters_since_best"] == 0

    def test_stops_exactly_at_patience_boundary(self):
        # best at 200; patience 600 -> stop once we're 600 iters past it (800).
        history = [(200, 0.20), (400, 0.18), (600, 0.19), (800, 0.15)]
        stop, info = should_stop(history, patience_iters=600, eval_interval=200)
        assert stop is True
        assert info["best_iteration"] == 200
        assert info["iters_since_best"] == 600

    def test_does_not_stop_just_under_patience(self):
        history = [(200, 0.20), (400, 0.18), (600, 0.19)]
        stop, info = should_stop(history, patience_iters=600, eval_interval=200)
        assert stop is False
        assert info["iters_since_best"] == 400

    def test_new_best_resets_patience_counter(self):
        history = [(200, 0.10), (400, 0.30), (600, 0.25), (800, 0.20)]
        stop, info = should_stop(history, patience_iters=600, eval_interval=200)
        assert info["best_iteration"] == 400
        assert info["iters_since_best"] == 400
        assert stop is False

    def test_tie_keeps_earlier_best_iteration(self):
        # A later equal (not strictly greater) value must not shift "best".
        history = [(200, 0.20), (400, 0.20), (600, 0.20)]
        stop, info = should_stop(history, patience_iters=400, eval_interval=200)
        assert info["best_iteration"] == 200
        assert info["iters_since_best"] == 400
        assert stop is True

    def test_single_point_never_stops(self):
        stop, info = should_stop([(200, 0.5)], patience_iters=600, eval_interval=200)
        assert stop is False
        assert info["best_iteration"] == 200
        assert info["iters_since_best"] == 0


class TestDetectDivergence:
    def test_insufficient_history_no_flag(self):
        assert detect_divergence([1.0, 0.9], [0.1, 0.2]) is False

    def test_loss_down_map_flat_flags_divergence(self):
        losses = [1.0, 0.8, 0.6]
        maps = [0.20, 0.21, 0.19]
        assert detect_divergence(losses, maps, window=DEFAULT_DIVERGENCE_WINDOW) is True

    def test_loss_down_map_up_no_flag(self):
        losses = [1.0, 0.8, 0.6]
        maps = [0.10, 0.20, 0.30]
        assert detect_divergence(losses, maps, window=3) is False

    def test_loss_flat_map_down_no_flag(self):
        # Divergence specifically requires loss STILL decreasing.
        losses = [0.5, 0.5, 0.5]
        maps = [0.30, 0.20, 0.10]
        assert detect_divergence(losses, maps, window=3) is False

    def test_loss_increasing_no_flag(self):
        losses = [0.5, 0.6, 0.7]
        maps = [0.30, 0.20, 0.10]
        assert detect_divergence(losses, maps, window=3) is False

    def test_custom_window_uses_only_recent_points(self):
        # Older points look diverging, but the recent window does not.
        losses = [1.0, 0.5, 0.4, 0.3]
        maps = [0.10, 0.30, 0.31, 0.32]
        assert detect_divergence(losses, maps, window=2) is False

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            detect_divergence([1.0, 0.9, 0.8], [0.1, 0.2], window=2)


class TestFormatStatusLine:
    def test_contains_all_fields(self):
        line = format_status_line(800, 27.5, "segm_mAP_50", 0.50, 0.50, 800, 0)
        assert "iter=800" in line
        assert "train_loss=27.5000" in line
        assert "segm_mAP_50=0.5000" in line
        assert "best=0.5000@iter800" in line
        assert "iters_since_best=0" in line

    def test_single_line_no_newline(self):
        line = format_status_line(200, 38.4, "segm_mAP_50", 0.19, 0.19, 200, 0)
        assert "\n" not in line

    def test_none_best_values_render_as_na(self):
        line = format_status_line(200, 38.4, "segm_mAP_50", 0.19, None, None, None)
        assert "best=n/a" in line
        assert "iters_since_best=n/a" in line

    def test_different_metric_key_used_verbatim(self):
        line = format_status_line(100, 1.0, "bbox_mAP", 0.3, 0.3, 100, 0)
        assert "bbox_mAP=0.3000" in line
