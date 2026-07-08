"""Custom mmcv Hook: early stopping + train/val divergence flagging.

mmcv 1.x / mmdet 2.x's old runner API (``IterBasedRunner``) has no built-in
early stopping — that only exists in the newer `mmengine` runner. This hook
supplies the wiring only; all decision logic lives in the unit-tested
:mod:`src.training.early_stopping` (``should_stop``, ``detect_divergence``).

Registered via config (``custom_hooks``), not instantiated directly:

    custom_hooks = [dict(type="EarlyStoppingDivergenceHook",
                         eval_interval=200, patience_iters=600,
                         metric_key="segm_mAP_50", divergence_window=3)]

Stopping mechanism: sets ``runner._max_iters = runner.iter + 1`` so the
IterBasedRunner's own loop condition ends the run after the current
iteration finishes — this lets ``after_run`` hooks (final checkpoint save,
logger flush) still execute normally, unlike raising an exception mid-loop.
"""
from typing import List, Tuple

from mmcv.runner import HOOKS, Hook

from src.common.log import get_logger
from src.training.early_stopping import detect_divergence, should_stop

logger = get_logger(__name__)


@HOOKS.register_module()
class EarlyStoppingDivergenceHook(Hook):
    """Stops training when val metric hasn't improved for ``patience_iters``;
    logs a divergence warning when train loss keeps falling while the metric
    has flattened or dropped."""

    def __init__(
        self,
        eval_interval: int,
        patience_iters: int,
        metric_key: str = "segm_mAP_50",
        divergence_window: int = 3,
    ) -> None:
        self.eval_interval = eval_interval
        self.patience_iters = patience_iters
        self.metric_key = metric_key
        self.divergence_window = divergence_window
        self._loss_buffer: List[float] = []
        self.val_history: List[Tuple[int, float]] = []
        self.train_loss_history: List[float] = []

    def after_train_iter(self, runner) -> None:
        loss = runner.outputs.get("loss")
        if loss is not None:
            self._loss_buffer.append(float(loss))

        if not self.every_n_iters(runner, self.eval_interval):
            return

        metric = runner.log_buffer.output.get(self.metric_key)
        if metric is None:
            logger.warning(
                "EarlyStoppingDivergenceHook: %r not found in log_buffer at "
                "iter %d (EvalHook may not have run this iteration) — "
                "skipping this check", self.metric_key, runner.iter,
            )
            return

        mean_train_loss = (sum(self._loss_buffer) / len(self._loss_buffer)
                           if self._loss_buffer else float("nan"))
        self._loss_buffer = []
        self.val_history.append((runner.iter, float(metric)))
        self.train_loss_history.append(mean_train_loss)

        if detect_divergence(self.train_loss_history, self.val_history_values(),
                             window=self.divergence_window):
            logger.warning(
                "DIVERGENCE FLAG at iter %d: train loss still falling "
                "(recent: %s) while val %s has flattened/dropped "
                "(recent: %s) — earliest overfitting signal",
                runner.iter, self.train_loss_history[-self.divergence_window:],
                self.metric_key,
                self.val_history_values()[-self.divergence_window:],
            )

        stop, info = should_stop(self.val_history, self.patience_iters,
                                 self.eval_interval)
        logger.info(
            "iter %d | train_loss=%.4f | val %s=%.4f | best=%.4f@iter%s | "
            "iters_since_best=%s",
            runner.iter, mean_train_loss, self.metric_key, metric,
            info["best_value"], info["best_iteration"], info["iters_since_best"],
        )
        if stop:
            logger.warning(
                "EARLY STOPPING at iter %d: no improvement in %s for %d "
                "iters (best %.4f at iter %d)", runner.iter, self.metric_key,
                info["iters_since_best"], info["best_value"], info["best_iteration"],
            )
            runner._max_iters = runner.iter + 1

    def val_history_values(self) -> List[float]:
        return [v for _, v in self.val_history]
