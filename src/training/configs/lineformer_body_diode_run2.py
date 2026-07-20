"""Second training run — body_diode / if_vs_vsd curve type, expanded dataset.

Same approach as body_diode_run1 (itself following zth_multicurve_run1 /
zth_vs_time / Run A): official-pretrained init, LR x0.05, fp32, no flip,
multi-scale + brightness/contrast jitter, best+latest checkpoint retention,
patience-based early stopping. See lineformer_run_a.py's docstring for the
full rationale behind each of those choices — nothing here is a new
decision. Structured as a full standalone config (not chained onto
lineformer_body_diode_run1.py), same reasoning as run1's own docstring:
each run gets its own complete config so schedule/data changes are explicit
and diffable, not layered through inheritance.

Differs from run1 only in data (owner instruction: "same hyperparameters as
run1 as a starting point"):
  - Data: `data/coco/split_body_diode_batch2/{train,val}.json` (287 train /
    38 val images, 633/91 annotations — group-aware family split over the
    combined 325-image pool [192 original + 133 newly-corrected], see that
    folder's split_manifest.json) + `data/images_body_diode_batch2_combined/`
    (new, symlink-only directory — the 325 images span two existing
    folders, `images_body_diode/` [192] and `images_body_diode_batch2/`
    [133], and mmdet's `img_prefix` only accepts one directory; this folder
    is 325 symlinks back to the two originals, nothing copied or moved,
    neither original folder touched).
  - Schedule: UNCHANGED from run1 (max_iters=8000, eval/checkpoint interval
    200, patience_iters=2000) per instruction to use run1's hyperparameters
    as the starting point.

FLAGGED, not applied (owner asked to be told, not to have this decided
silently): train set grew 170->287 images (x1.688). This project has an
established precedent for this exact situation — T10/Run A2 (see
PROGRESS.md) scaled every schedule number by the train-size ratio when
Run A's dataset grew ~4x, reasoning that leaving iteration counts fixed
silently shrinks the number of epochs actually seen. Applying the same
method here would give max_iters~=13500-14000, eval_interval~=350,
patience_iters~=3500 (preserves run1's ~47 total epochs / 10-eval-patience
shape almost exactly: 14000/287~=48.8 epochs, 3500/287~=12.2 epochs
patience, vs run1's 8000/170~=47.1 / 2000/170~=11.8). Not applied here
because the instruction was explicit about keeping run1's numbers as the
starting point; also worth noting run1 itself early-stopped at iter 4199,
well short of its 8000 ceiling, so under-training is a real but unproven
risk here, not a confirmed one. A follow-up config with the scaled numbers
is a reasonable next step if this run's val curve looks like it's still
improving when patience runs out.
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
# Data: body_diode (if_vs_vsd) batch2 split (192 + 133 = 325 images).
# --------------------------------------------------------------------------
classes = ("line",)
_IMAGES_DIR = osp.join(REPO_ROOT, "data", "images_body_diode_batch2_combined")
_TRAIN_ANN = osp.join(REPO_ROOT, "data", "coco", "split_body_diode_batch2", "train.json")
_VAL_ANN = osp.join(REPO_ROOT, "data", "coco", "split_body_diode_batch2", "val.json")

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
# Init weights: OFFICIAL pretrained only (same as Run A / run1).
# --------------------------------------------------------------------------
load_from = osp.join(REPO_ROOT, "data", "weights",
                     "lineformer_pretrained_official_iter3000.pth")
resume_from = None

# --------------------------------------------------------------------------
# Optimizer: same AdamW/paramwise_cfg as base, LR scaled x0.05 (same as run1).
# --------------------------------------------------------------------------
_BASE_LR = 1e-4
optimizer = dict(lr=_BASE_LR * 0.05)  # = 5e-6

# --------------------------------------------------------------------------
# Schedule: UNCHANGED from run1 (see module docstring for the scaling
# question this leaves open).
# --------------------------------------------------------------------------
max_iters = 8000
runner = dict(type="IterBasedRunner", max_iters=max_iters)

# fp32 only — same as Run A/run1, no fp16 key set anywhere.

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
# run1 — unscaled, per instruction; see module docstring).
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
