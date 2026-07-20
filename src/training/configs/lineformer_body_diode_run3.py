"""Third training run — body_diode / if_vs_vsd curve type, scaled schedule.

Same approach as body_diode_run2 (itself following run1 / zth_multicurve_run1
/ zth_vs_time / Run A): official-pretrained init, LR x0.05, fp32, no flip,
multi-scale + brightness/contrast jitter, best+latest checkpoint retention,
patience-based early stopping. See lineformer_run_a.py's docstring for the
full rationale behind each of those choices — nothing here is a new
decision. Structured as a full standalone config (not chained onto
lineformer_body_diode_run2.py), same reasoning as run1/run2's own
docstrings: each run gets its own complete config so schedule/data changes
are explicit and diffable, not layered through inheritance.

Identical to run2 in data and every other hyperparameter (same
`split_body_diode_batch2/` 287-train/38-val split, same
`images_body_diode_batch2_combined/` symlink image dir, same LR 5e-6, fp32,
no flip, official-pretrained init). Differs ONLY in schedule — this is the
scaled follow-up flagged in run2's own docstring and confirmed necessary by
run2's actual result:

  - `max_iters`: 8000 -> **14000**
  - `evaluation`/`checkpoint_config` interval: 200 -> **350**
  - `EarlyStoppingDivergenceHook.patience_iters`: 2000 -> **3500**

Rationale (T10/Run A2 precedent, see PROGRESS.md — scaling schedule numbers
by the train-size ratio rather than leaving iteration counts fixed when the
dataset grows): train set grew 170 (run1) -> 287 (run2/run3), a x1.688
ratio. 14000/287 ~= 48.8 epochs total and 3500/287 ~= 12.2 epochs patience,
closely matching run1's own 8000/170 ~= 47.1 total / 2000/170 ~= 11.8
patience (in epoch-equivalent terms) — this run gets the same relative
schedule shape run1 got, scaled for its larger dataset, rather than run2's
literal-copy schedule which left real headroom unused.

Confirmed necessary, not just theoretical: run2 (unscaled, max_iters=8000)
ran the full ceiling WITHOUT early stopping (`iters_since_best` was only
1600/2000 patience at iter 8000) and was still climbing/fluctuating near
its peak (0.65-0.68 segm_mAP_50) in the final 2000 iters — the iteration
budget cut it off, not convergence. Best checkpoint was
`best_segm_mAP_50_iter_6400.pth` (segm_mAP_50 0.6947).
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
# Data: body_diode (if_vs_vsd) batch2 split (192 + 133 = 325 images) —
# identical to run2.
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
# Init weights: OFFICIAL pretrained only (same as Run A / run1 / run2).
# --------------------------------------------------------------------------
load_from = osp.join(REPO_ROOT, "data", "weights",
                     "lineformer_pretrained_official_iter3000.pth")
resume_from = None

# --------------------------------------------------------------------------
# Optimizer: same AdamW/paramwise_cfg as base, LR scaled x0.05 (same as
# run1/run2).
# --------------------------------------------------------------------------
_BASE_LR = 1e-4
optimizer = dict(lr=_BASE_LR * 0.05)  # = 5e-6

# --------------------------------------------------------------------------
# Schedule: SCALED for the larger dataset (see module docstring).
# --------------------------------------------------------------------------
max_iters = 14000
runner = dict(type="IterBasedRunner", max_iters=max_iters)

# fp32 only — same as every prior run, no fp16 key set anywhere.

# --------------------------------------------------------------------------
# Eval + checkpointing: interval 350 (scaled from run2's 200); "best +
# latest only" retention.
# --------------------------------------------------------------------------
evaluation = dict(interval=350, metric=["segm"], save_best="segm_mAP_50")
checkpoint_config = dict(
    interval=350, by_epoch=False, save_last=True, max_keep_ckpts=1)

log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ])

# --------------------------------------------------------------------------
# Early stopping + train/val divergence flagging (patience=3500, scaled
# from run2's 2000 — see module docstring).
# --------------------------------------------------------------------------
custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(
        type="EarlyStoppingDivergenceHook",
        eval_interval=350,
        patience_iters=3500,
        metric_key="segm_mAP_50",
        divergence_window=3,
        priority=80,
    ),
]
