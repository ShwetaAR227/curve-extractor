"""LineFormer training orchestration — task T7 Part B, "Run A".

Fine-tunes the official pretrained checkpoint on the frozen real-data train
split (`data/coco/split/train.json`), validating against `val.json`. All
owner-mandated hyperparameters live in the config file
(`src/training/configs/lineformer_run_a.py`) as explicit, commented
overrides on top of the upstream base config — this script's job is to:

1. resolve the full config (base + overrides) and dump it, in full, to
   `run_manifest.json` BEFORE training starts (CLAUDE.md §7 — nothing about
   the run's configuration is left un-recorded);
2. run mmdet's standard training entry point with our custom early-stopping
   hook already registered (import side effect);
3. measure wall-clock time and peak GPU memory;
4. after training, audit checkpoint retention (tested logic in
   `checkpoint_retention.py`) and report what the run ultimately produced.

Heavy imports (torch/mmcv/mmdet) happen inside `main()`/`run_training()`,
not at module scope, so this file can still be imported for its (thin, GPU-
free) helper functions without requiring the training env to be installed.

CLI (run on the GPU box inside the `lineformer` conda env):
    python -m src.training.train_lineformer \
        --config src/training/configs/lineformer_run_a.py \
        --work-dir /mnt/data/my-datasheet/checkpoints/run_a --seed 42
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from src.common.log import get_logger
from src.dataset_tools.collect_images import _sha256
from src.training.checkpoint_retention import plan_retention

logger = get_logger(__name__)

MANIFEST_NAME = "run_manifest.json"


def build_manifest(
    cfg: Any,
    config_path: str,
    seed: int,
    lineformer_commit: Optional[str],
) -> Dict[str, Any]:
    """Assemble the pre-training manifest: full resolved config plus the
    quick-reference fields the owner asked to have on record explicitly."""
    train_ann = cfg.data["train"]["ann_file"]
    val_ann = cfg.data["val"]["ann_file"]
    manifest = {
        "timestamp_start": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "seed": seed,
        "lineformer_commit": lineformer_commit,
        "init_checkpoint": cfg.load_from,
        "init_checkpoint_sha256": _sha256(cfg.load_from) if cfg.load_from and
                                  Path(cfg.load_from).is_file() else None,
        "train_ann": train_ann,
        "train_ann_sha256": _sha256(train_ann) if Path(train_ann).is_file() else None,
        "val_ann": val_ann,
        "val_ann_sha256": _sha256(val_ann) if Path(val_ann).is_file() else None,
        # Quick-reference fields (owner explicitly wants these on record,
        # in addition to the full resolved config below):
        "quick_reference": {
            # compat_cfg() (applied before this is called) moves
            # samples_per_gpu/workers_per_gpu under data.train_dataloader.
            "batch_size": cfg.data["train_dataloader"]["samples_per_gpu"],
            "max_iters": cfg.runner["max_iters"],
            "learning_rate": cfg.optimizer["lr"],
            "optimizer_type": cfg.optimizer["type"],
            "weight_decay": cfg.optimizer.get("weight_decay"),
            "lr_schedule_policy": cfg.lr_config.get("policy"),
            "lr_schedule_step": cfg.lr_config.get("step"),
            "warmup": cfg.lr_config.get("warmup"),
            "warmup_iters": cfg.lr_config.get("warmup_iters"),
            "warmup_ratio": cfg.lr_config.get("warmup_ratio"),
            "eval_interval": cfg.evaluation["interval"],
            "checkpoint_interval": cfg.checkpoint_config["interval"],
            "save_best_metric": cfg.evaluation.get("save_best"),
            "max_keep_ckpts": cfg.checkpoint_config.get("max_keep_ckpts"),
            "num_queries": cfg.model["panoptic_head"]["num_queries"],
            "loss_cls_weight": cfg.model["panoptic_head"]["loss_cls"]["loss_weight"],
            "loss_mask_weight": cfg.model["panoptic_head"]["loss_mask"]["loss_weight"],
            "loss_dice_weight": cfg.model["panoptic_head"]["loss_dice"]["loss_weight"],
            "backbone_lr_mult": cfg.optimizer["paramwise_cfg"]["custom_keys"]
                                ["backbone"]["lr_mult"],
        },
        # Full resolved config (base + all our overrides merged) — nothing
        # about this run's configuration is left un-recorded.
        "resolved_config": cfg._cfg_dict.to_dict() if hasattr(cfg, "_cfg_dict")
                           else dict(cfg),
    }
    return manifest


def audit_checkpoints(work_dir: str) -> Dict[str, Any]:
    """Post-run retention audit: report what mmcv's own max_keep_ckpts +
    save_best left behind, flag (but do not silently fix) any drift from
    the "best + latest only" policy so it's visible in the run report."""
    work_path = Path(work_dir)
    files = sorted(p.name for p in work_path.glob("*.pth"))
    plan = plan_retention(files)
    unexpected = [f for f in files if f not in plan["keep"]]
    if unexpected:
        logger.warning(
            "Checkpoint retention drift: mmcv left %d file(s) beyond "
            "best+latest: %s — not deleted automatically, review manually",
            len(unexpected), unexpected,
        )
    return {"present": files, "expected_keep": plan["keep"],
            "unexpected": unexpected}


def run_training(config_path: str, work_dir: str, seed: int = 42) -> Dict[str, Any]:
    """Resolve config, dump the pre-training manifest, train, report."""
    import mmcv
    import torch
    from mmcv.runner import init_dist, set_random_seed
    from mmdet.apis import train_detector
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.utils import compat_cfg

    # Import side effect: registers EarlyStoppingDivergenceHook with mmcv's
    # HOOKS registry so the config's custom_hooks entry can find it by name.
    import src.training.hooks.early_stopping_hook  # noqa: F401

    # mmcv.Config.fromfile copies .py configs into a temp dir before exec'ing
    # them, so __file__-relative tricks inside the config would resolve to
    # that shallow temp path. This script's own __file__ IS reliable (mmdet
    # imports it directly), so REPO_ROOT is computed here and handed to the
    # config via the environment — see the matching guard in
    # configs/lineformer_run_a.py.
    repo_root = Path(__file__).resolve().parents[2]
    os.environ["LINEFORMER_REPO_ROOT"] = str(repo_root)

    cfg = mmcv.Config.fromfile(config_path)
    cfg = compat_cfg(cfg)
    cfg.work_dir = work_dir
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    set_random_seed(seed, deterministic=False)
    cfg.seed = seed

    # status_path depends on work_dir, which is only known at runtime (not
    # inside the static config file) — inject it into the already-configured
    # EarlyStoppingDivergenceHook entry so `cat status.txt` works for any run.
    status_path = str(Path(work_dir) / "status.txt")
    for hook_cfg in cfg.get("custom_hooks", []):
        if hook_cfg.get("type") == "EarlyStoppingDivergenceHook":
            hook_cfg["status_path"] = status_path

    commit = None
    commit_file = repo_root / "envs" / "lineformer.commit"
    if commit_file.is_file():
        commit = commit_file.read_text().strip()

    manifest = build_manifest(cfg, config_path, seed, commit)
    manifest_path = Path(work_dir) / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str),
                             encoding="utf-8")
    logger.info("Pre-training manifest written: %s", manifest_path)

    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"),
                           test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    datasets = [build_dataset(cfg.data.train)]
    model.CLASSES = datasets[0].CLASSES

    # Normally set by mmdet's own tools/train.py CLI (single-GPU here since
    # we call train_detector directly rather than through that script).
    cfg.gpu_ids = [0]
    cfg.device = "cuda"

    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    train_detector(model, datasets, cfg, distributed=False, validate=True,
                   meta={"seed": seed})
    wall_clock_s = time.monotonic() - start
    peak_mem_mib = torch.cuda.max_memory_allocated() / 1024**2

    audit = audit_checkpoints(work_dir)
    result = {
        "wall_clock_seconds": wall_clock_s,
        "peak_gpu_memory_mib": peak_mem_mib,
        "checkpoint_audit": audit,
        "work_dir": work_dir,
        "timestamp_end": datetime.now(timezone.utc).isoformat(),
    }
    result_path = Path(work_dir) / "run_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Training finished: %.1fs wall-clock, %.0f MiB peak GPU memory",
                wall_clock_s, peak_mem_mib)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(description="Train LineFormer (Run A).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        result = run_training(args.config, args.work_dir, seed=args.seed)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("Training failed: %s", exc)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
