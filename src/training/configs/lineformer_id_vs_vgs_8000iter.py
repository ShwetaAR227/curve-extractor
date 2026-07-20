"""id_vs_vgs, second attempt — identical to lineformer_id_vs_vgs.py in every
respect (official-pretrained init, LR 5e-6, fp32, no flip, eval/checkpoint
interval 200, early-stopping patience 600) except the iteration ceiling,
raised from 2000 to 8000 per owner instruction. Patience is deliberately
left UNCHANGED at 600 (not scaled the way Run A2's schedule was, per T10) —
an explicit owner decision, not an oversight.
"""
# NOTE: mmcv.Config.fromfile copies .py configs into a tempfile.Temporary
# Directory() before exec'ing them (see lineformer_run_a.py's docstring for
# the full explanation) — a `__file__`-relative _base_ path would resolve to
# that shallow /tmp path, not this file's real location. Hardcoded absolute
# path instead, same fix used elsewhere in this project for the same issue.
_base_ = ["/home/ec2-user/my-datasheet/src/training/configs/lineformer_id_vs_vgs.py"]

max_iters = 8000
runner = dict(type="IterBasedRunner", max_iters=max_iters)
