"""zth_vs_time — "Run 1": fine-tune the official pretrained LineFormer
checkpoint on the first multi-curve zth_vs_time annotation batch
(`data/coco/split_zth_multicurve_batch1/`, 41 train / 10 val images, 5-7
duty-cycle curves traced per chart — see PROGRESS.md 2026-07-16).

Inherits Run A wholesale via `_base_` (official pretrained init, fp32/no-
autocast, LR=5e-6, no horizontal/vertical flip, multi-scale resize +
brightness/contrast jitter, batch size 1, "best + latest only" checkpoint
retention) — only the DATA and the schedule numbers below are overridden,
exactly as Run A2 did for its own dataset. `classes = ("line",)` is
inherited unchanged and needs no override: this dataset's CVAT label is
also "line", and LineFormer trains one class ("is this pixel part of A
curve") regardless of how many curve instances (1 for rdson, 3 for
capacitance, 5-7 here) appear per image — curve IDENTITY (which duty
factor) is Stage 5 naming work, entirely downstream of this training run.
`num_queries=100` (inherited from the base config) comfortably covers this
dataset's max 7 instances/image, so it is not touched either.

Owner-specified schedule for this run (given directly, 2026-07-16 —
not derived via Run A2's dataset-size-ratio scaling, since the owner gave
the two numbers explicitly):
  max_iters = 8000       (same ceiling as Run A2)
  patience  = 2000       (owner-specified value for THIS run)
eval_interval is kept at 800 — 10% of max_iters, the same ratio Run A
(200/2000) and Run A2 (800/8000) both used — so patience=2000 works out to
~2.5 eval intervals, not the clean "3 eval intervals" of Run A/A2; this is
a deliberately looser patience than those runs, consistent with the
owner's explicit number rather than re-derived from it.

Flag for the owner (not something I'm authorized to second-guess here):
train has only 41 images, so 8000 iters is ~195 epochs — a far higher
epoch count than Run A (~17.2) or Run A2 (~17.5) saw on their much larger
splits. That may be intentional for a small first batch, or may want
revisiting once this run's curve is visible.
"""
import os
import os.path as osp

# NOTE: this config is loaded via mmcv.Config.fromfile, which copies .py
# configs into a tempfile.TemporaryDirectory() before exec'ing them —
# __file__-relative tricks here would resolve to that shallow temp path, not
# this file's real location (see the identical note in lineformer_run_a.py /
# lineformer_run_a2.py). Use the same env-var mechanism instead.
REPO_ROOT = os.environ.get("LINEFORMER_REPO_ROOT")
if not REPO_ROOT:
    raise RuntimeError(
        "LINEFORMER_REPO_ROOT is not set — this config must be loaded via "
        "train_lineformer.py, which sets it before calling mmcv.Config.fromfile."
    )

_base_ = [osp.join(REPO_ROOT, "src", "training", "configs",
                   "lineformer_run_a.py")]

# --------------------------------------------------------------------------
# Data: zth_vs_time multi-curve batch 1 (own split dir, own image folder —
# NOT the shared data/images/ used by capacitance/id_vs_vgs/rdson).
# --------------------------------------------------------------------------
_IMAGES_DIR = osp.join(REPO_ROOT, "data", "zth_vs_time_images")
_TRAIN_ANN = osp.join(REPO_ROOT, "data", "coco", "split_zth_multicurve_batch1", "train.json")
_VAL_ANN = osp.join(REPO_ROOT, "data", "coco", "split_zth_multicurve_batch1", "val.json")

data = dict(
    train=dict(ann_file=_TRAIN_ANN, img_prefix=_IMAGES_DIR),
    val=dict(ann_file=_VAL_ANN, img_prefix=_IMAGES_DIR),
    test=dict(ann_file=_VAL_ANN, img_prefix=_IMAGES_DIR),  # no separate held-out test split for this first batch
)

# --------------------------------------------------------------------------
# Schedule: 8000-iteration ceiling, patience 2000 (both owner-specified for
# this run). eval_interval=800 kept at the same 10%-of-max_iters ratio as
# Run A/A2 for a consistent checkpoint cadence.
# --------------------------------------------------------------------------
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
        patience_iters=2000,      # owner-specified for this run (see module docstring)
        metric_key="segm_mAP_50",
        divergence_window=3,
        priority=80,              # see the matching comment in run_a.py
    ),
]
