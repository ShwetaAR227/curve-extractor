"""zth_vs_time, multi-curve batch 1 — separate dataset from the single-curve
zth_vs_time split, same curve_type and image folder.

Identical to lineformer_zth_vs_time.py in every hyperparameter (official-
pretrained init, LR 5e-6, fp32, no flip, 8000-iter ceiling, eval/checkpoint
interval 200). Two things differ:
  - Data: data/coco/split_zth_multicurve_batch1/{train,val}.json (41 train /
    10 val images, 247/56 annotations — multi-curve-per-image batch, simple
    random per-image split, seed 42; see that folder's split_manifest.json).
    img_prefix stays data/images_zth_vs_time/, the same image folder used by
    the single-curve split — these are a subset of the same figure PNGs,
    just annotated with more than one curve per image.
  - patience_iters: 2000 from the start (owner instruction — same value
    already adopted for zth_vs_time's later patience-raised runs), not the
    base config's 600.
"""
# NOTE: mmcv.Config.fromfile copies .py configs into a tempfile.Temporary
# Directory() before exec'ing them (see lineformer_run_a.py's docstring for
# the full explanation) — a `__file__`-relative _base_ path would resolve to
# that shallow /tmp path, not this file's real location. Hardcoded absolute
# path instead, same fix used elsewhere in this project for the same issue.
import os
import os.path as osp

_base_ = ["/home/ec2-user/my-datasheet/src/training/configs/lineformer_zth_vs_time.py"]

# See lineformer_run_a.py for why REPO_ROOT must come from the environment
# rather than __file__ (mmcv.Config.fromfile copies configs to a temp dir).
REPO_ROOT = os.environ.get("LINEFORMER_REPO_ROOT")
if not REPO_ROOT:
    raise RuntimeError(
        "LINEFORMER_REPO_ROOT is not set — this config must be loaded via "
        "train_lineformer.py, which sets it before calling mmcv.Config.fromfile."
    )

_IMAGES_DIR = osp.join(REPO_ROOT, "data", "images_zth_vs_time")
_TRAIN_ANN = osp.join(REPO_ROOT, "data", "coco", "split_zth_multicurve_batch1", "train.json")
_VAL_ANN = osp.join(REPO_ROOT, "data", "coco", "split_zth_multicurve_batch1", "val.json")

data = dict(
    train=dict(ann_file=_TRAIN_ANN, img_prefix=_IMAGES_DIR),
    val=dict(ann_file=_VAL_ANN, img_prefix=_IMAGES_DIR),
    test=dict(ann_file=_VAL_ANN, img_prefix=_IMAGES_DIR),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(
        type="EarlyStoppingDivergenceHook",
        eval_interval=200,
        patience_iters=2000,
        metric_key="segm_mAP_50",
        divergence_window=3,
        priority=80,
    ),
]
