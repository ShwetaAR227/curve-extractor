"""T7/T10 — "Run A2": identical to Run A except the DATA and the schedule
numbers that scale with dataset size. Same init weights (official pretrained,
not Run A's own checkpoint — keeps both runs comparable against the same
starting point), same LR, same fp32/no-autocast, same augmentation, same
batch size, same checkpoint-retention policy.

Inherits Run A's config wholesale via `_base_` and overrides only the
iteration/interval/patience numbers below.

Why the numbers changed (owner asked for reasoning, not a blind copy):
train.json grew from 116 to 456 images after the T9 semiauto-batch merge —
a 456/116 ≈ 3.93x increase. Keeping Run A's raw iteration counts (max_iters
2000, eval every 200, patience 600) unchanged would mean this much bigger,
more diverse dataset is seen for only ~4.4 epochs total (2000/456), evaluated
every ~0.44 epochs, with early-stopping patience of just ~1.3 epochs — a much
thinner exposure than Run A's ~17.2 total epochs / ~1.72 epochs per eval /
~5.17 epochs patience (200/116, 600/116). Scaling every number by the same
~3.93x ratio preserves Run A's RELATIVE training schedule (same ~17.5 total
epochs, same 10 eval checkpoints across the run, same 3-eval-interval
patience in epoch-equivalent terms) while covering the larger dataset:
  max_iters:     2000  -> 8000   (8000/456 ≈ 17.5 epochs, vs 2000/116 ≈ 17.2)
  eval_interval:  200  ->  800   (800/456 ≈ 1.75 epochs/eval, vs 200/116 ≈ 1.72)
  patience:       600  -> 2400   (3 eval intervals either way)
"""
import os
import os.path as osp

# NOTE: this config is also loaded via mmcv.Config.fromfile, which copies
# .py configs into a tempfile.TemporaryDirectory() before exec'ing them —
# __file__-relative tricks here would resolve to that shallow temp path, not
# this file's real location (the exact landmine already hit and documented
# in lineformer_run_a.py). Use the same env-var mechanism instead.
REPO_ROOT = os.environ.get("LINEFORMER_REPO_ROOT")
if not REPO_ROOT:
    raise RuntimeError(
        "LINEFORMER_REPO_ROOT is not set — this config must be loaded via "
        "train_lineformer.py, which sets it before calling mmcv.Config.fromfile."
    )

_base_ = [osp.join(REPO_ROOT, "src", "training", "configs",
                   "lineformer_run_a.py")]

max_iters = 8000
runner = dict(type="IterBasedRunner", max_iters=max_iters)
evaluation = dict(interval=800, metric=["segm"], save_best="segm_mAP_50")
checkpoint_config = dict(
    interval=800, by_epoch=False, save_last=True, max_keep_ckpts=1)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(
        type="EarlyStoppingDivergenceHook",
        eval_interval=800,
        patience_iters=2400,      # 3 eval intervals, same as Run A
        metric_key="segm_mAP_50",
        divergence_window=3,
        priority=80,              # see the matching comment in run_a.py
    ),
]
