"""Pure checkpoint-retention planning logic (task T7 Run A).

Owner policy: keep only the best-on-val-mAP50 checkpoint and the latest
periodic checkpoint; delete everything else automatically. This module only
plans (given a directory listing, decide keep vs delete) — the thin,
untested I/O wrapper that lists a real directory and deletes files lives in
the training script, exercised only in the real/integration run.
"""
import re
from typing import Dict, List, Sequence

LATEST_NAME = "latest.pth"
_ITER_RE = re.compile(r"^iter_(\d+)\.pth$")
_BEST_RE = re.compile(r"^best_.+_iter_(\d+)\.pth$")


def plan_retention(files: Sequence[str]) -> Dict[str, List[str]]:
    """Decide which checkpoint files to keep vs. delete.

    Keeps: ``latest.pth`` if present, the numerically highest ``iter_N.pth``,
    and the ``best_*_iter_N.pth`` with the highest ``N`` (defensive against
    stale best files coexisting, though mmcv normally keeps only one).
    Non-checkpoint files (manifests, configs, etc.) are ignored — never
    listed in either ``keep`` or ``delete``.
    """
    keep: List[str] = []
    iter_files = []
    best_files = []

    for name in files:
        if name == LATEST_NAME:
            keep.append(name)
            continue
        iter_match = _ITER_RE.match(name)
        if iter_match:
            iter_files.append((int(iter_match.group(1)), name))
            continue
        best_match = _BEST_RE.match(name)
        if best_match:
            best_files.append((int(best_match.group(1)), name))

    if iter_files:
        keep.append(max(iter_files, key=lambda t: t[0])[1])
    if best_files:
        keep.append(max(best_files, key=lambda t: t[0])[1])

    keep_set = set(keep)
    delete = [name for _, name in iter_files if name not in keep_set]
    delete += [name for _, name in best_files if name not in keep_set]
    return {"keep": keep, "delete": delete}
