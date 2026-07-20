"""First training run — id_vs_vgs (transfer_char) curve type.

Same approach as Run A (capacitance_vs_vds, T7 Part B): only the data paths
differ. Every hyperparameter, augmentation choice, schedule, and
early-stopping rule below is copied unchanged from
`lineformer_run_a.py` per owner instruction ("same approach as Run A") —
see that file's docstring for the full rationale behind each choice
(official-pretrained init, LR x0.05, no flip, multi-scale + brightness/
contrast jitter, best+latest checkpoint retention, patience=600 early
stopping). Nothing here is a new decision; it is Run A's config replayed
against a different dataset.

Data: `data/coco/split_id_vs_vgs/{train,val}.json` + `data/images_id_vs_vgs/`
(owner-supplied, verified via `validate_coco` and a full file-existence
check against `data/images_id_vs_vgs/` before this config was used — see
PROGRESS.md / conversation record). curve_name values here are per-junction-
temperature series (`TJ_-40C`/`TJ_25C`/`TJ_150C`/`TJ_175C`), not
Ciss/Coss/Crss — irrelevant to training, which only sees the single "line"
category; curve_name is metadata carried through to Stage 5, unused here.

Known caveat, flagged not fixed: the val split is thin (29 images, only 7
with real annotations — 22 are unannotated/empty). Early-stopping/best-
checkpoint decisions in this run are based on a noisier val mAP@50 signal
than Run A's 24-image (all-annotated) val set.
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
# Data: id_vs_vgs split (owner-supplied CVAT export → COCO → split).
# --------------------------------------------------------------------------
classes = ("line",)
_IMAGES_DIR = osp.join(REPO_ROOT, "data", "images_id_vs_vgs")
_TRAIN_ANN = osp.join(REPO_ROOT, "data", "coco", "split_id_vs_vgs", "train.json")
_VAL_ANN = osp.join(REPO_ROOT, "data", "coco", "split_id_vs_vgs", "val.json")

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
# Schedule: 2000-iteration ceiling (same as Run A).
# --------------------------------------------------------------------------
max_iters = 2000
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
# Early stopping + train/val divergence flagging (same as Run A, patience=600).
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
