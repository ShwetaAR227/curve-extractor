"""First training run — zth_vs_time (thermal_impedance) curve type.

Same approach as Run A / id_vs_vgs: official-pretrained init, LR x0.05, fp32,
no flip, multi-scale + brightness/contrast jitter, best+latest checkpoint
retention, patience=600 early stopping. See lineformer_run_a.py's docstring
for the full rationale behind each of those choices — nothing here is a new
decision.

Only two things differ from Run A, both owner-specified for this run:
- Data: `data/coco/split_zth_vs_time/{train,val}.json` + `data/images_zth_vs_time/`
  (owner-supplied; verified via validate_coco, zero missing images, zero
  overlap between train/val/test, single curve_name "single_pulse" per image
  — 221 train / 46 val / 48 test, all fully annotated, no thin-val-set
  caveat this time unlike id_vs_vgs).
- Schedule: max_iters=8000 from the start (id_vs_vgs needed a follow-up
  config to raise this after the first 2000-iter run; here it's requested
  up front). patience_iters stays 600, unscaled, per owner instruction —
  same as id_vs_vgs's 8000-iter run, not proportionally scaled the way T10
  (Run A2) scaled patience for its larger dataset.
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
# Data: zth_vs_time split (owner-supplied CVAT export -> COCO -> split).
# --------------------------------------------------------------------------
classes = ("line",)
_IMAGES_DIR = osp.join(REPO_ROOT, "data", "images_zth_vs_time")
_TRAIN_ANN = osp.join(REPO_ROOT, "data", "coco", "split_zth_vs_time", "train.json")
_VAL_ANN = osp.join(REPO_ROOT, "data", "coco", "split_zth_vs_time", "val.json")

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
# Schedule: 8000-iteration ceiling (owner-specified for this run, from the
# start — unlike id_vs_vgs which started at 2000 and got a follow-up config).
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
# Early stopping + train/val divergence flagging (patience=600, same as
# Run A and id_vs_vgs's 8000-iter run — unscaled, owner instruction).
# --------------------------------------------------------------------------
custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(
        type="EarlyStoppingDivergenceHook",
        eval_interval=200,
        patience_iters=600,
        metric_key="segm_mAP_50",
        divergence_window=3,
        priority=80,
    ),
]
