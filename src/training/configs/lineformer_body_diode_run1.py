"""First training run — body_diode / if_vs_vsd curve type.

Same approach as zth_multicurve_run1 / zth_vs_time / Run A: official-
pretrained init, LR x0.05, fp32, no flip, multi-scale + brightness/contrast
jitter, best+latest checkpoint retention, patience-based early stopping. See
lineformer_run_a.py's docstring for the full rationale behind each of those
choices — nothing here is a new decision. Structured as a full standalone
config (not chained onto lineformer_zth_vs_time.py the way
lineformer_zth_multicurve_run1.py is) because this is a genuinely different
curve_type/dataset, not another annotation batch of the same one — same
relationship id_vs_vgs.py and zth_vs_time.py already have to each other.

Differs from the base swin-t config (and matches zth_multicurve_run1's
values, per owner instruction to use that run's settings as the starting
point for this comparably small dataset):
  - Data: `data/coco/split_body_diode_batch1/{train,val}.json` (170 train /
    22 val images, 383/44 annotations — group-aware family split, see that
    folder's split_manifest.json) + `data/images_body_diode/` (single
    category "line", curve_name attribute holds the temperature label, e.g.
    "25C", "150C" — same convention as every other curve_type).
  - Schedule: max_iters=8000 ceiling, patience_iters=2000 from the start
    (not the base config's 600) — same as zth_multicurve_run1, whose
    41-train-image dataset is the closest precedent for a small batch here;
    170 train images is larger than that, so patience=2000 is a generous
    starting point, not a tight one.

NOT YET LAUNCHED as of 2026-07-18: data/images_body_diode/ is an empty
staging directory (see PROGRESS.md) — the actual source PNGs referenced by
split_body_diode_batch1/{train,val}.json have not been transferred to this
box yet. This config is prepared and ready; training starts once the images
land.
"""
import os
import os.path as osp

# See lineformer_run_a.py for why REPO_ROOT must come from the environment
# rather than __file__ (mmcv.Config.fromfile copies configs to a temp dir).
REPO_ROOT = os.environ.get("LINEFORMER_REPO_ROOT")
if not REPO_ROOT:
    raise RuntimeError(
        "LINEFORMER_REPO_ROOT is not set — this config must be loaded via "
        "train_lineformer.py, which sets it before calling mmcv.Config.fromfile."
    )

_base_ = [osp.join(REPO_ROOT, "third_party", "lineformer",
                   "lineformer_swin_t_config.py")]

# --------------------------------------------------------------------------
# Data: body_diode (if_vs_vsd) split.
# --------------------------------------------------------------------------
classes = ("line",)
_IMAGES_DIR = osp.join(REPO_ROOT, "data", "images_body_diode")
_TRAIN_ANN = osp.join(REPO_ROOT, "data", "coco", "split_body_diode_batch1", "train.json")
_VAL_ANN = osp.join(REPO_ROOT, "data", "coco", "split_body_diode_batch1", "val.json")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

_multi_scale = [(448, 448), (512, 512), (576, 576)]

train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, with_seg=False),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.7, 1.3),
        saturation_range=(1.0, 1.0),
        hue_delta=0),
    dict(type="RandomFlip", flip_ratio=0.0),
    dict(type="RandomShift", shift_ratio=0.3, max_shift_px=512 // 10),
    dict(
        type="RandomCrop",
        crop_size=(int(0.85 * 512), int(0.85 * 512)),
        crop_type="absolute",
        recompute_bbox=False,
        allow_negative_crop=True),
    dict(
        type="Resize",
        img_scale=_multi_scale,
        ratio_range=None,
        multiscale_mode="value",
        keep_ratio=True),
    dict(type="Pad", size=(512, 512),
        pad_val=dict(img=(255, 255, 255), masks=0, seg=255)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle", img_to_float=True),
    dict(type="Collect", keys=["img", "gt_labels", "gt_bboxes", "gt_masks"]),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type="CocoDataset", classes=classes,
        ann_file=_TRAIN_ANN, img_prefix=_IMAGES_DIR, pipeline=train_pipeline),
    val=dict(
        type="CocoDataset", classes=classes,
        ann_file=_VAL_ANN, img_prefix=_IMAGES_DIR),
    test=dict(
        type="CocoDataset", classes=classes,
        ann_file=_VAL_ANN, img_prefix=_IMAGES_DIR),
)

# --------------------------------------------------------------------------
# Init weights: OFFICIAL pretrained only (same as Run A).
# --------------------------------------------------------------------------
load_from = osp.join(REPO_ROOT, "data", "weights",
                     "lineformer_pretrained_official_iter3000.pth")
resume_from = None

# --------------------------------------------------------------------------
# Optimizer: same AdamW/paramwise_cfg as base, LR scaled x0.05 (same as Run A).
# --------------------------------------------------------------------------
_BASE_LR = 1e-4
optimizer = dict(lr=_BASE_LR * 0.05)  # = 5e-6

# --------------------------------------------------------------------------
# Schedule: 8000-iteration ceiling, same as zth_multicurve_run1.
# --------------------------------------------------------------------------
max_iters = 8000
runner = dict(type="IterBasedRunner", max_iters=max_iters)

# fp32 only — same as Run A, no fp16 key set anywhere.

# --------------------------------------------------------------------------
# Eval + checkpointing: interval 200; "best + latest only" retention.
# --------------------------------------------------------------------------
evaluation = dict(interval=200, metric=["segm"], save_best="segm_mAP_50")
checkpoint_config = dict(
    interval=200, by_epoch=False, save_last=True, max_keep_ckpts=1)

log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ])

# --------------------------------------------------------------------------
# Early stopping + train/val divergence flagging (patience=2000, same as
# zth_multicurve_run1 — owner instruction to use that run's settings as the
# starting point for this comparably small dataset).
# --------------------------------------------------------------------------
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
