"""LineFormer checkpoint evaluation against the frozen test split (task T7).

Metrics:
- **mAP@50 / mAP@75** — standard COCO *mask* AP via pycocotools' COCOeval
  (never hand-rolled). COCO AP sweeps confidence thresholds natively; these
  numbers are NOT at a fixed confidence.
- **Per-curve recall** — at a FIXED confidence threshold of 0.5 (the legacy
  inference default): a ground-truth instance counts as matched when ANY
  prediction with score >= 0.5 overlaps it with mask IoU >= 0.5. Reported
  overall and broken down by ``curve_name`` (carried in annotation
  ``attributes`` since T2/T3).

The two families of numbers are labeled distinctly in the report so they are
never confused: ``map50``/``map75`` (threshold-sweeping) vs
``recall_*`` (fixed ``recall_score_thr``/``recall_iou_thr``).

Pure metric functions in this module import only numpy so they are
unit-testable without GPU/torch (CLAUDE.md §2); torch/mmdet/pycocotools are
imported lazily inside the inference path only.

CLI (run on the GPU box inside the ``lineformer`` conda env):
    python -m src.training.eval_lineformer \
        --checkpoint <path> --config <path> \
        --test-coco data/coco/split/test.json --images data/images/ \
        --out <report.json>
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.common.log import get_logger
from src.dataset_tools.collect_images import _sha256

logger = get_logger(__name__)

# Fixed thresholds for the recall metric (legacy inference default is 0.5).
DEFAULT_SCORE_THR = 0.5
DEFAULT_IOU_THR = 0.5

MatchEntry = Tuple[str, bool]  # (curve_name, matched)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU of two boolean masks; 0.0 when the union is empty."""
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(mask_a, mask_b).sum() / union)


def compute_recall(
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    scores: Sequence[float],
    score_thr: float = DEFAULT_SCORE_THR,
    iou_thr: float = DEFAULT_IOU_THR,
) -> List[bool]:
    """Per-GT matched flags: ANY prediction with ``score >= score_thr`` and
    mask IoU >= ``iou_thr`` matches the ground-truth instance (no one-to-one
    assignment — this is a pure did-we-find-it recall)."""
    kept = [m for m, s in zip(pred_masks, scores) if s >= score_thr]
    return [any(mask_iou(gt, pred) >= iou_thr for pred in kept)
            for gt in gt_masks]


def recall_by_curve(entries: Sequence[MatchEntry]) -> Dict[str, Dict[str, Any]]:
    """Group (curve_name, matched) entries into per-curve recall stats."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for curve_name, matched in entries:
        stats = grouped.setdefault(curve_name, {"matched": 0, "total": 0})
        stats["total"] += 1
        stats["matched"] += int(matched)
    for stats in grouped.values():
        stats["recall"] = stats["matched"] / stats["total"]
    return grouped


def build_report(
    checkpoint: str,
    config: str,
    dataset_hash: str,
    map50: float,
    map75: float,
    match_entries: Sequence[MatchEntry],
    n_test_images: int,
    score_thr: float = DEFAULT_SCORE_THR,
    iou_thr: float = DEFAULT_IOU_THR,
) -> Dict[str, Any]:
    """Assemble the JSON-serializable evaluation report."""
    n_gt = len(match_entries)
    n_matched = sum(int(m) for _, m in match_entries)
    return {
        "checkpoint": checkpoint,
        "config": config,
        "dataset_hash": dataset_hash,
        # COCO-native mask AP (sweeps confidence thresholds internally):
        "map50": map50,
        "map75": map75,
        # Fixed-threshold recall (score >= recall_score_thr, IoU >= recall_iou_thr):
        "recall_overall": (n_matched / n_gt) if n_gt else None,
        "recall_by_curve": recall_by_curve(match_entries),
        "recall_score_thr": score_thr,
        "recall_iou_thr": iou_thr,
        "n_test_images": n_test_images,
        "n_gt_instances": n_gt,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------
# Inference path — GPU box only; heavy imports stay inside these functions.
# --------------------------------------------------------------------------

def _pred_to_bool_masks(segm_result: Any) -> List[np.ndarray]:
    """Normalize mmdet 2.x per-class segm results (ndarray or RLE) to bool
    numpy masks, flattened across classes (single-class model in practice)."""
    import pycocotools.mask as mask_util

    masks: List[np.ndarray] = []
    for class_masks in segm_result:
        for m in class_masks:
            if isinstance(m, dict):  # RLE
                masks.append(mask_util.decode(m).astype(bool))
            else:
                masks.append(np.asarray(m, dtype=bool))
    return masks


def run_eval(
    checkpoint: Union[str, Path],
    config: Union[str, Path],
    test_coco: Union[str, Path],
    images_dir: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    score_thr: float = DEFAULT_SCORE_THR,
    iou_thr: float = DEFAULT_IOU_THR,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Evaluate a checkpoint on the frozen test COCO. Returns the report dict."""
    import os

    import pycocotools.mask as mask_util
    from mmdet.apis import inference_detector, init_detector
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Configs using our _base_ chain (e.g. lineformer_run_a.py) need this env
    # var — see the matching comment in train_lineformer.py. Harmless no-op
    # for configs that don't reference it (e.g. LineFormer's own upstream
    # config, used directly for the baseline evals).
    os.environ["LINEFORMER_REPO_ROOT"] = str(Path(__file__).resolve().parents[2])

    test_coco = Path(test_coco)
    images_dir = Path(images_dir)
    coco_gt = COCO(str(test_coco))
    img_ids = sorted(coco_gt.getImgIds())
    logger.info("Evaluating %s on %s (%d images)", checkpoint, test_coco,
                len(img_ids))

    model = init_detector(str(config), str(checkpoint), device=device)

    detections: List[Dict[str, Any]] = []  # COCO-format results for COCOeval
    match_entries: List[MatchEntry] = []
    for img_id in img_ids:
        info = coco_gt.loadImgs(img_id)[0]
        img_path = images_dir / info["file_name"]
        if not img_path.is_file():
            raise FileNotFoundError(f"Test image missing: {img_path}")
        result = inference_detector(model, str(img_path))
        bbox_result, segm_result = result
        scores = [float(b[4]) for class_boxes in bbox_result
                  for b in class_boxes]
        pred_masks = _pred_to_bool_masks(segm_result)

        # COCO-format detections (all scores; COCOeval sweeps internally).
        for score, mask in zip(scores, pred_masks):
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            detections.append({"image_id": img_id, "category_id": 1,
                               "segmentation": rle, "score": score})

        # Fixed-threshold recall per GT instance.
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        gt_masks = [coco_gt.annToMask(a).astype(bool) for a in anns]
        matched = compute_recall(gt_masks, pred_masks, scores,
                                 score_thr=score_thr, iou_thr=iou_thr)
        for ann, m in zip(anns, matched):
            curve = (ann.get("attributes") or {}).get("curve_name", "?")
            match_entries.append((curve, m))
        logger.info("%s: %d preds (%d >= %.2f), %d/%d GT matched",
                    info["file_name"], len(pred_masks),
                    sum(s >= score_thr for s in scores), score_thr,
                    sum(matched), len(matched))

    if detections:
        coco_dt = coco_gt.loadRes(detections)
        ev = COCOeval(coco_gt, coco_dt, iouType="segm")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        # stats[1] = AP@.50, stats[2] = AP@.75 (COCO summarize layout)
        map50, map75 = float(ev.stats[1]), float(ev.stats[2])
    else:
        logger.warning("Zero predictions across the whole test set")
        map50 = map75 = 0.0

    report = build_report(
        checkpoint=str(checkpoint), config=str(config),
        dataset_hash=_sha256(test_coco), map50=map50, map75=map75,
        match_entries=match_entries, n_test_images=len(img_ids),
        score_thr=score_thr, iou_thr=iou_thr,
    )
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Report written: %s", out_path)
    return report


def format_summary(report: Dict[str, Any]) -> str:
    """Human-readable console summary of an eval report."""
    lines = [
        f"Checkpoint : {report['checkpoint']}",
        f"Config     : {report['config']}",
        f"Test set   : {report['n_test_images']} images, "
        f"{report['n_gt_instances']} GT instances "
        f"(sha256 {report['dataset_hash'][:12]}…)",
        "-- COCO mask AP (confidence swept internally by COCOeval) --",
        f"mAP@50     : {report['map50']:.4f}",
        f"mAP@75     : {report['map75']:.4f}",
        f"-- Recall at FIXED score >= {report['recall_score_thr']}, "
        f"IoU >= {report['recall_iou_thr']} --",
        f"overall    : "
        + (f"{report['recall_overall']:.4f}" if report["recall_overall"]
           is not None else "n/a (no GT)"),
    ]
    for curve, stats in sorted(report["recall_by_curve"].items()):
        lines.append(f"  {curve:<10}: {stats['recall']:.4f} "
                     f"({stats['matched']}/{stats['total']})")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(
        description="Evaluate a LineFormer checkpoint on the frozen test split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--test-coco", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", default=None, help="Report JSON path")
    parser.add_argument("--score-thr", type=float, default=DEFAULT_SCORE_THR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        report = run_eval(args.checkpoint, args.config, args.test_coco,
                          args.images, out_path=args.out,
                          score_thr=args.score_thr, device=args.device)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1
    print(format_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
