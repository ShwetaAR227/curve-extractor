"""id_vs_vgs, 2px-buffer data, patience raised to 1200 (owner decision,
2026-07-13) after Run 4 (id_vs_vgs_run4_2px_8000iter) early-stopped at iter
800 with segm_mAP_50 stuck at exactly 0.0000 the whole way (patience 600).

Investigation (see conversation record / PROGRESS.md) found no evidence of a
data bug: train/val JSON structurally valid, correct counts, correct
provenance hash; post-pipeline mask survival for the 2px data (8.2% of
instances vanish under the training crop) is only modestly worse than the
4.5px data that trained fine (6.2%); loss curve well-behaved (no NaN/
explosion). Most likely explanation: the thinner GT masks need tighter
prediction precision to cross the IoU>=0.5 threshold, so the first nonzero
val score plausibly arrives later than it did for the 4.5px run (which got
lucky with a nonzero eval right at iter 200, resetting its patience clock
early enough to survive to iter 4800). Raising patience gives this run the
same kind of runway without changing anything else.

Identical to lineformer_id_vs_vgs_8000iter.py in every other respect
(official-pretrained init, LR 5e-6, fp32, no flip, max_iters 8000,
eval/checkpoint interval 200) -- only patience_iters changes, 600 -> 1200.
Data paths are unchanged (data/coco/split_id_vs_vgs/{train,val}.json); those
files are now the 2px-buffer versions on disk, so no config-side data change
is needed for the buffer switch itself.
"""
# NOTE: mmcv.Config.fromfile copies .py configs into a tempfile.Temporary
# Directory() before exec'ing them (see lineformer_run_a.py's docstring for
# the full explanation) — a `__file__`-relative _base_ path would resolve to
# that shallow /tmp path, not this file's real location. Hardcoded absolute
# path instead, same fix used elsewhere in this project for the same issue.
_base_ = ["/home/ec2-user/my-datasheet/src/training/configs/lineformer_id_vs_vgs_8000iter.py"]

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(
        type="EarlyStoppingDivergenceHook",
        eval_interval=200,
        patience_iters=1200,      # was 600; doubled per owner decision, 2026-07-13
        metric_key="segm_mAP_50",
        divergence_window=3,
        priority=80,
    ),
]
