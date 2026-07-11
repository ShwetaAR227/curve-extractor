"""Model-inference wrapper (Stage 5), reusing the existing mmdet calling
convention rather than rewriting it.

Mask decoding is imported directly from
:func:`src.training.eval_lineformer._pred_to_bool_masks` and score
filtering from :func:`src.training.predict_to_cvat.filter_by_score` — the
exact same helpers ``run_eval``/``run_predict_to_cvat`` already use, so
this module adds no second implementation of either.

GPU-only; ``mmdet``/``torch`` imports stay lazy inside ``load_model``/
``run_inference`` (same convention as those two callers), so the pure
combining logic (:func:`detections_from_raw`) is unit-testable without a
GPU (CLAUDE.md §2) and ``load_model``/``run_inference`` are tested by
injecting a fake ``mmdet.apis`` module.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np

from src.common.log import get_logger
from src.training.eval_lineformer import _pred_to_bool_masks
from src.training.predict_to_cvat import DEFAULT_SCORE_THR, filter_by_score

logger = get_logger(__name__)


@dataclass
class Detection:
    """One kept model detection: confidence score + boolean HxW mask."""

    score: float
    mask: np.ndarray


def detections_from_raw(
    bbox_result: Any, segm_result: Any, score_thr: float = DEFAULT_SCORE_THR
) -> List[Detection]:
    """Combine raw mmdet ``(bbox_result, segm_result)`` into kept Detections.

    Args:
        bbox_result: Per-class list of ``[x1, y1, x2, y2, score]`` arrays.
        segm_result: Per-class list of masks (ndarray or RLE dict).
        score_thr: Minimum confidence to keep (default matches the eval
            baseline's fixed recall threshold).

    Returns:
        Detections with score >= ``score_thr``, order preserved.
    """
    scores = [float(b[4]) for class_boxes in bbox_result for b in class_boxes]
    masks = _pred_to_bool_masks(segm_result)
    kept = filter_by_score(list(zip(scores, masks)), score_thr=score_thr)
    return [Detection(score=score, mask=mask) for score, mask in kept]


def load_model(checkpoint: str, config: str, device: str = "cuda:0") -> Any:
    """Load a LineFormer/mmdet model (GPU-only; heavy import stays lazy)."""
    import os

    from mmdet.apis import init_detector

    os.environ.setdefault(
        "LINEFORMER_REPO_ROOT", str(Path(__file__).resolve().parents[2])
    )
    logger.info("load_model: checkpoint=%s config=%s device=%s", checkpoint, config, device)
    return init_detector(config, checkpoint, device=device)


def run_inference(
    model: Any, image_path: str, score_thr: float = DEFAULT_SCORE_THR
) -> List[Detection]:
    """Run ``model`` on one image, keeping detections with score >= ``score_thr``."""
    from mmdet.apis import inference_detector

    bbox_result, segm_result = inference_detector(model, str(image_path))
    detections = detections_from_raw(bbox_result, segm_result, score_thr)
    logger.info(
        "run_inference(%s): %d kept >= %.2f", image_path, len(detections), score_thr
    )
    return detections
