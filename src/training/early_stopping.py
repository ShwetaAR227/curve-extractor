"""Pure early-stopping and train/val-divergence decision logic (task T7 Run A).

No GPU/torch/mmdet dependency — this is the tested decision logic behind the
custom mmcv hook in :mod:`src.training.hooks.early_stopping_hook`, which only
supplies the wiring (reading the runner's loss/metric state, calling these
functions, and stopping the runner).
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Overfitting-detection window: how many recent eval points to compare when
# deciding whether train loss is still falling while val mAP has stalled.
DEFAULT_DIVERGENCE_WINDOW = 3


def should_stop(
    history: Sequence[Tuple[int, float]],
    patience_iters: int,
    eval_interval: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Decide whether to stop given a (iteration, val_metric) history.

    Stops once the number of iterations since the best-so-far value reaches
    ``patience_iters`` (ties keep the EARLIER best iteration — a later equal
    value is not an improvement). ``eval_interval`` is accepted for callers'
    convenience/logging symmetry; the decision only depends on iteration
    deltas already present in ``history``.
    """
    if not history:
        return False, {"best_iteration": None, "best_value": None,
                       "iters_since_best": None}

    best_iteration, best_value = history[0]
    for iteration, value in history[1:]:
        if value > best_value:
            best_iteration, best_value = iteration, value

    latest_iteration = history[-1][0]
    iters_since_best = latest_iteration - best_iteration
    return iters_since_best >= patience_iters, {
        "best_iteration": best_iteration,
        "best_value": best_value,
        "iters_since_best": iters_since_best,
    }


def detect_divergence(
    train_losses: Sequence[float],
    val_maps: Sequence[float],
    window: int = DEFAULT_DIVERGENCE_WINDOW,
) -> bool:
    """Flag the earliest overfitting signal: train loss still falling while
    val mAP has flattened or dropped, over the last ``window`` eval points.

    Compares only the first and last value of the trailing window (a simple,
    deterministic trend check — not a regression fit). Returns False when
    there isn't yet ``window`` points of history.
    """
    if len(train_losses) != len(val_maps):
        raise ValueError(
            f"train_losses ({len(train_losses)}) and val_maps "
            f"({len(val_maps)}) must be the same length"
        )
    if len(train_losses) < window:
        return False

    loss_window = train_losses[-window:]
    map_window = val_maps[-window:]
    loss_decreasing = loss_window[-1] < loss_window[0]
    map_not_improving = map_window[-1] <= map_window[0]
    return loss_decreasing and map_not_improving


def format_status_line(
    iteration: int,
    train_loss: float,
    metric_key: str,
    metric_value: float,
    best_value: Optional[float],
    best_iteration: Optional[int],
    iters_since_best: Optional[int],
) -> str:
    """One-line, `cat`-able training status (task T7 Run A2).

    Written to ``status.txt`` in the work dir at every eval interval so
    progress can be checked without reading the full raw log. ``best_*``/
    ``iters_since_best`` may be ``None`` on the very first eval point (no
    history yet) — rendered as ``n/a``.
    """
    best_str = (f"{best_value:.4f}@iter{best_iteration}"
               if best_value is not None else "n/a")
    since_str = str(iters_since_best) if iters_since_best is not None else "n/a"
    return (f"iter={iteration} train_loss={train_loss:.4f} "
           f"{metric_key}={metric_value:.4f} best={best_str} "
           f"iters_since_best={since_str}")
