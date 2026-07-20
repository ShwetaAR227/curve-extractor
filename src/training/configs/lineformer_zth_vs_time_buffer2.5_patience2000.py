"""zth_vs_time, third attempt — buffer_px corrected + patience baked in from
the start.

Identical to lineformer_zth_vs_time.py in every hyperparameter (official-
pretrained init, LR 5e-6, fp32, no flip, 8000-iter ceiling, eval/checkpoint
interval 200). Two things differ, both already reflected in the underlying
data/config rather than needing overrides here:
  - data/coco/split_zth_vs_time/{train,val}.json were re-buffered in place
    at buffer_px=2.5 (down from 4.5 — see split_manifest.json's
    "buffer_rebuffer_2026_07_15" entry for the full measurement + visual-
    comparison rationale). The base config's ann_file paths are unchanged,
    so no override needed for that part.

The one thing this file DOES override: patience_iters, set to 2000 from the
start (not as a follow-up config the way id_vs_vgs and zth_vs_time's first
patience increase were done). Lesson learned from id_vs_vgs: a smaller
buffer can make the metric start at/near 0.0 and take longer to climb out
of that trough, so a short patience risks stopping before the model finds
its footing on the tighter masks. Building the longer patience in up front
this time instead of discovering it's needed after an early-stopped run.
"""
# NOTE: mmcv.Config.fromfile copies .py configs into a tempfile.Temporary
# Directory() before exec'ing them (see lineformer_run_a.py's docstring for
# the full explanation) — a `__file__`-relative _base_ path would resolve to
# that shallow /tmp path, not this file's real location. Hardcoded absolute
# path instead, same fix used elsewhere in this project for the same issue.
_base_ = ["/home/ec2-user/my-datasheet/src/training/configs/lineformer_zth_vs_time.py"]

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
