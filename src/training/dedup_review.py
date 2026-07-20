"""Standalone dedup review tool for zth_multicurve_run1 (2026-07-17 investigation).

**Standalone only — deliberately NOT called by predict_to_cvat.py.** The
owner asked to review this dedup step's before/after numbers once more
before it goes live in the actual CVAT pre-annotation push; wiring it into
predict_to_cvat.py is a separate, later step.

Reuses existing tested code exclusively — no new inference or dedup logic:
  - src.extraction.inference.{load_model,run_inference} for score-filtered
    (>=score_thr) Detections.
  - src.extraction.dedup.dedup_detections(..., use_flat_curve_heuristic) for
    the actual de-duplication. Default here is ``use_flat_curve_heuristic=
    False`` (IoU-only) — the investigation on split_zth_multicurve_batch1
    (2026-07-17) found the flat-curve heuristic (tuned for
    capacitance_vs_vds's flat plateaus) over-merges zth_multicurve's near-
    parallel duty-cycle curve families; IoU-only at the existing 0.5
    threshold scored 78.4% exact curve-count match with only 2 regressions,
    vs. 66.7%/4 regressions with the heuristic on. See PROGRESS.md for the
    full comparison.

Only the summarization logic (build_report_row/summarize_dedup_report) is
pure/GPU-free and unit-tested (CLAUDE.md §2); the inference-driving CLI path
is exercised manually against real images per the same convention as
eval_lineformer.py/predict_to_cvat.py.

CLI (run on the GPU box inside the `lineformer` conda env):
    python -m src.training.dedup_review \
        --checkpoint <path> --config <path> --images-dir <dir> \
        --ann <coco.json> [--ann <coco2.json> ...] \
        --out <report.csv> [--score-thr 0.5] [--use-flat-curve-heuristic]
"""
import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.common.log import get_logger
from src.extraction.dedup import dedup_detections
from src.extraction.inference import DEFAULT_SCORE_THR, load_model, run_inference

logger = get_logger(__name__)


def build_report_row(
    image: str, split: str, gt_count: int, raw_count: int,
    dedup_count: int, n_removed: int,
) -> Dict[str, Any]:
    """One image's before/after-dedup curve-count comparison row."""
    return {
        "image": image,
        "split": split,
        "gt_count": gt_count,
        "raw_count": raw_count,
        "dedup_count": dedup_count,
        "n_removed": n_removed,
        "raw_match": raw_count == gt_count,
        "dedup_match": dedup_count == gt_count,
    }


def summarize_dedup_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate before/after stats + regression flags from report rows.

    A "regression" is an image that matched exactly before dedup but no
    longer does after — dedup removed a real, distinct curve. Separately,
    ``under_detected_worsened`` flags images that were already under-
    detected before dedup and lost even more detections (dedup can only
    remove, never add, so this can't happen to an exact or over-detected
    image without first becoming a regression).
    """
    n = len(rows)
    exact_before = sum(1 for r in rows if r["raw_match"])
    exact_after = sum(1 for r in rows if r["dedup_match"])
    over_after = sum(1 for r in rows if r["dedup_count"] > r["gt_count"])
    under_after = sum(1 for r in rows if r["dedup_count"] < r["gt_count"])
    regressions = [r["image"] for r in rows
                   if r["raw_match"] and not r["dedup_match"]]
    under_detected_worsened = [
        r["image"] for r in rows
        if r["raw_count"] < r["gt_count"] and r["dedup_count"] < r["raw_count"]
    ]
    total_removed = sum(r["n_removed"] for r in rows)

    return {
        "n_images": n,
        "exact_before": exact_before,
        "exact_after": exact_after,
        "exact_after_pct": (100.0 * exact_after / n) if n else 0.0,
        "over_after": over_after,
        "under_after": under_after,
        "regressions": regressions,
        "under_detected_worsened": under_detected_worsened,
        "total_removed": total_removed,
    }


def run_dedup_review(
    checkpoint: Union[str, Path],
    config: Union[str, Path],
    images_dir: Union[str, Path],
    ann_paths: Sequence[Union[str, Path]],
    out_path: Union[str, Path],
    score_thr: float = DEFAULT_SCORE_THR,
    use_flat_curve_heuristic: bool = False,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Run inference + dedup on every image across ``ann_paths``, report before/after.

    ``ann_paths`` are COCO json files (e.g. train.json + val.json); each
    file's basename-minus-extension is recorded as the row's "split".
    """
    from pycocotools.coco import COCO

    images_dir = Path(images_dir)
    model = load_model(str(checkpoint), str(config), device=device)

    rows: List[Dict[str, Any]] = []
    for ann_path in ann_paths:
        ann_path = Path(ann_path)
        split_name = ann_path.stem
        coco_gt = COCO(str(ann_path))
        for img_id in sorted(coco_gt.getImgIds()):
            info = coco_gt.loadImgs(img_id)[0]
            fname = info["file_name"]
            img_path = images_dir / fname
            if not img_path.is_file():
                logger.warning("Missing image, skipped: %s", img_path)
                continue

            gt_count = len(coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id)))
            raw_detections = run_inference(model, str(img_path), score_thr=score_thr)
            deduped, n_removed = dedup_detections(
                raw_detections, use_flat_curve_heuristic=use_flat_curve_heuristic)

            row = build_report_row(fname, split_name, gt_count,
                                   len(raw_detections), len(deduped), n_removed)
            rows.append(row)
            logger.info(
                "[%s] %s: GT=%d raw=%d dedup=%d (removed %d) "
                "raw_match=%s dedup_match=%s",
                split_name, fname, gt_count, row["raw_count"], row["dedup_count"],
                n_removed, row["raw_match"], row["dedup_match"],
            )

    summary = summarize_dedup_report(rows)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "gt_count",
                                               "raw_count", "dedup_count",
                                               "n_removed", "raw_match",
                                               "dedup_match"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), out_path)

    summary["out_path"] = str(out_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(
        description="Standalone before/after dedup review (not wired into "
                     "predict_to_cvat.py).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--ann", action="append", required=True,
                        help="COCO json (repeatable, e.g. --ann train.json --ann val.json)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--score-thr", type=float, default=DEFAULT_SCORE_THR)
    parser.add_argument(
        "--use-flat-curve-heuristic", action="store_true",
        help="Also apply the same-band/x-span heuristic (default off — "
             "IoU-only was found better for zth_multicurve, see module docstring).")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    summary = run_dedup_review(
        args.checkpoint, args.config, args.images_dir, args.ann, args.out,
        score_thr=args.score_thr,
        use_flat_curve_heuristic=args.use_flat_curve_heuristic,
        device=args.device,
    )
    print(f"Images: {summary['n_images']}")
    print(f"Exact match BEFORE dedup: {summary['exact_before']}")
    print(f"Exact match AFTER dedup : {summary['exact_after']} "
          f"({summary['exact_after_pct']:.1f}%)")
    print(f"Total duplicates removed: {summary['total_removed']}")
    print(f"Regressions (dedup broke a prior exact match): "
          f"{len(summary['regressions'])} {summary['regressions']}")
    print(f"Under-detected images made worse by dedup: "
          f"{len(summary['under_detected_worsened'])} "
          f"{summary['under_detected_worsened']}")
    print(f"Report: {summary['out_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
