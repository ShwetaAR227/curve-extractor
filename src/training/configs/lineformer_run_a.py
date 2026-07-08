"""T7 Part B — "Run A": fine-tune the official pretrained LineFormer checkpoint
on our real (non-synthetic) 115-image training split.

Inherits everything from the upstream `lineformer_swin_t_config.py` (same
architecture as Baseline 1) via mmcv's `_base_` mechanism; only the deltas
below are owner-specified overrides. Every value NOT listed here is
inherited unchanged from the base config — the full resolved config (base +
overrides) is dumped to `run_manifest.json` by `train_lineformer.py` before
training starts, so nothing is left un-recorded.

Owner decisions baked in here (2026-07-08):
- Init weights: the OFFICIAL pretrained checkpoint, never the legacy one
  (Baseline 2 showed it is worthless — see PROGRESS.md T7a).
- LR = base LR (1e-4) x 0.05 = 5e-6 (tightened from the originally discussed
  x0.1, given this architecture's prior overfitting history at 10k iters).
- Horizontal AND vertical flip are removed entirely from the train pipeline
  (not just flip_ratio=0) — curve direction/physics correctness requirement,
  not a style preference. The base pipeline defaults flip_ratio=[0.3, 0.3]
  on BOTH directions, which would silently corrupt curve semantics if left in.
- Multi-scale resize and brightness/contrast jitter are ADDED (base pipeline
  had neither): "multi-scale" via a small range of scales around the base
  512x512; brightness/contrast via PhotoMetricDistortion with
  saturation/hue ranges pinned to no-op (owner asked for brightness/contrast
  only, not saturation/hue).
- Checkpoint retention: `max_keep_ckpts=1` (periodic "latest") + native
  `save_best='segm_mAP_50'` (mmcv's EvalHook replaces the previous best file
  in place) gives "best + latest only" without custom deletion code; the
  post-run audit in `checkpoint_retention.py` (tested) verifies this held.
- Early stopping (patience 600 iters = 3 eval intervals) and train/val
  divergence flagging are NOT native to mmcv 1.x/mmdet 2.x's old runner API
  (that only exists in the newer `mmengine` runner) — implemented as a
  custom hook, `EarlyStoppingDivergenceHook`, registered via `custom_hooks`
  below; its decision logic lives in `early_stopping.py` (unit tested).
"""
import os
import os.path as osp

# NOTE: this config is loaded by mmcv.Config.fromfile, which copies .py
# configs into a tempfile.TemporaryDirectory() before exec'ing them — so
# `__file__`-relative path tricks resolve to a shallow /tmp path here, not
# this file's real location. REPO_ROOT must come from the environment,
# set by the training script (train_lineformer.py) from ITS OWN __file__,
# which mmdet imports directly rather than through mmcv's temp-copy path.
REPO_ROOT = os.environ.get("LINEFORMER_REPO_ROOT")
if not REPO_ROOT:
    raise RuntimeError(
        "LINEFORMER_REPO_ROOT is not set — this config must be loaded via "
        "train_lineformer.py, which sets it before calling mmcv.Config.fromfile."
    )

_base_ = [osp.join(REPO_ROOT, "third_party", "lineformer",
                   "lineformer_swin_t_config.py")]

# --------------------------------------------------------------------------
# Data: our frozen split, not LineFormer's original PMC/AdobeSynth/LineEX mix.
# --------------------------------------------------------------------------
classes = ("line",)
_IMAGES_DIR = osp.join(REPO_ROOT, "data", "images")
_TRAIN_ANN = osp.join(REPO_ROOT, "data", "coco", "split", "train.json")
_VAL_ANN = osp.join(REPO_ROOT, "data", "coco", "split", "val.json")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# "Multi-scale resize": base config used a single fixed 512x512 scale
# (img_scale=[image_size], multiscale_mode='value' with one option is not
# actually multi-scale). Owner asked for multi-scale resize; scale band
# chosen as +/-12.5%/+12.5% around the base 512 — an assumption since no
# exact bounds were specified.
_multi_scale = [(448, 448), (512, 512), (576, 576)]

train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, with_seg=False),
    # "Brightness/contrast jitter" only: saturation_range=(1.0, 1.0) and
    # hue_delta=0 make PhotoMetricDistortion's saturation/hue steps no-ops,
    # since owner asked for brightness/contrast specifically, not color jitter.
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.7, 1.3),
        saturation_range=(1.0, 1.0),
        hue_delta=0),
    # RandomFlip disabled via flip_ratio=0.0 (owner requirement: curve
    # direction/physics correctness — a flipped C/I-vs-V curve is physically
    # wrong, not just a different-looking one). The base pipeline's active
    # train_pipeline had flip_ratio=[0.3, 0.3] on BOTH horizontal and
    # vertical — confirmed here, not assumed.
    #
    # Integration-test finding: initially this transform was removed
    # entirely (belt-and-suspenders against any "default-on" risk), but
    # mmdet's Collect/DefaultFormatBundle require results['flip'] to already
    # be set — populated by RandomFlip itself, even at flip_ratio=0 — so
    # removing the transform breaks the pipeline (KeyError: 'flip').
    # flip_ratio=0.0 is mmdet's own established disable idiom (used by this
    # very base config's test_pipeline) and is fully deterministic:
    # `random.random() < 0.0` is never true, so there is no hidden
    # "default-on" path — confirmed by reading mmdet's RandomFlip source,
    # not assumed.
    dict(type="RandomFlip", flip_ratio=0.0),
    #
    # NOTE — deviation from the base config, found during integration testing:
    # LineFormer's own install.sh does `pip install -e mmdetection` from a
    # *vendored, patched* mmdetection copy in third_party/lineformer/mmdetection
    # (same version string 2.28.2, but its RandomCrop adds a `crop_ratio`
    # probability gate that PyPI mmdet==2.28.2 — our T6 owner-approved pin —
    # does not have). Rather than silently swap to the vendored fork
    # mid-task, `crop_ratio` is dropped here: the crop always applies
    # (crop_type='absolute' with no probability gate) instead of applying
    # ~30% of the time. Flagged for owner awareness in SETUP.md; revisit if
    # this run's augmentation strength looks off.
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

# test_pipeline inherited unchanged from the base config: it already has
# flip=False / RandomFlip(flip_ratio=0.0) — no flip risk there either.

data = dict(
    samples_per_gpu=1,     # owner-specified batch size
    workers_per_gpu=2,     # not specified by owner; small dataset, conservative default
    train=dict(
        _delete_=True,  # base's data.train is a list (RepeatDataset entries);
        # ours is a single dataset dict — types differ, so mmcv requires
        # _delete_=True to replace rather than attempt a dict/list merge.
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
# Init weights: OFFICIAL pretrained only (never the legacy checkpoint).
# --------------------------------------------------------------------------
load_from = osp.join(REPO_ROOT, "data", "weights",
                     "lineformer_pretrained_official_iter3000.pth")
resume_from = None

# --------------------------------------------------------------------------
# Optimizer: same AdamW/paramwise_cfg as base, LR scaled x0.05.
# --------------------------------------------------------------------------
_BASE_LR = 1e-4
optimizer = dict(lr=_BASE_LR * 0.05)  # = 5e-6

# lr_config (policy/warmup/step) inherited unchanged. NOTE: base step=5000 is
# never reached within max_iters=2000 below, so in practice this run trains
# at a constant LR after a 10-iter linear warmup — recorded here so it's not
# a silent surprise when reading the resolved config.

# --------------------------------------------------------------------------
# Schedule: 2000-iteration ceiling, not a target (early stopping may cut it
# short — see custom_hooks below).
# --------------------------------------------------------------------------
max_iters = 2000
runner = dict(type="IterBasedRunner", max_iters=max_iters)

# fp32 only — no fp16 key is set anywhere in this config or the base config
# it inherits from (confirmed by grep during config review); do not add one.

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
# Early stopping + train/val divergence flagging (custom hook; see
# src/training/hooks/early_stopping_hook.py and its unit-tested logic in
# src/training/early_stopping.py).
# --------------------------------------------------------------------------
custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(
        type="EarlyStoppingDivergenceHook",
        eval_interval=200,
        patience_iters=600,       # 3 consecutive eval intervals, tightened from 1000
        metric_key="segm_mAP_50",
        divergence_window=3,
        # Priority 80: strictly between EvalHook (LOW=70, must run first so
        # the metric is written) and TextLoggerHook (VERY_LOW=90, which
        # calls log_buffer.clear_output() — confirmed by reading mmcv's
        # LoggerHook source — so we must read the metric BEFORE it does).
        # VERY_LOW alone is not enough: same-priority hooks run in
        # registration order, and TextLoggerHook (from log_config) is
        # registered before custom_hooks, so it would clear the buffer
        # first — verified by an actual smoke-test failure before this fix.
        priority=80,
    ),
]
