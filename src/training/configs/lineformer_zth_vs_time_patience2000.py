"""zth_vs_time, second attempt — identical to lineformer_zth_vs_time.py in
every respect (official-pretrained init, LR 5e-6, fp32, no flip, eval/
checkpoint interval 200, 8000-iteration ceiling — already the case in the
base config) except patience_iters, raised from 600 to 2000 per owner
instruction. The first run's early stopping (best iter 1000, stopped ~1600)
cut training short well before the 8000 ceiling; this gives it more room
to keep improving before patience triggers.
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
