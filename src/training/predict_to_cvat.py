"""Convert LineFormer predictions into CVAT-importable pre-annotations (T8b).

Owner approved (2026-07-08, T8a): predictions from the Run A best checkpoint
are tight and high-confidence enough to pre-annotate from, so this tool turns
model output into a CVAT-for-images-1.1 XML file a human can import into a
CVAT task and correct, instead of tracing every curve from scratch.

Predicted masks are filled blobs (the model was trained on 4.5px-buffered
polygon masks — see the T4 buffer-radius decision), so each kept mask is
emitted as a CVAT **polygon** (its contour), not a skeletonized polyline.
This matches the "polygon for thick bands" half of our existing CVAT
convention already read by `cvat_to_coco.parse_cvat_xml`, is lossless/robust
(no skeleton-branch artifacts at curve endpoints or crossings), and is just
as easy for a human to drag-correct in CVAT as a polyline.

Every polygon's `curve_name` attribute is written as the literal placeholder
``CURVE_NAME_PLACEHOLDER`` ("TODO") — the model has no notion of Ciss vs Coss
vs Crss (single-class "line"), so the annotator must fill in the real name
for every curve before the corrected file is ever run back through
`cvat_to_coco.py` (which hard-errors on an *empty* curve_name, but does not
yet validate against a fixed vocabulary — leaving "TODO" unedited would slip
through that check; flagged here, not fixed, since cvat_to_coco.py is a
frozen pipeline stage — see CLAUDE.md §4).

KNOWN LIMITATION — documented, not auto-filtered (owner decision, 2026-07-08,
T8a): the model occasionally emits a near-duplicate second mask on flat,
low-texture curves (seen on Ciss plateaus in the T8a check). No NMS/dedup is
applied here — annotators should expect to occasionally delete one duplicate
polygon per batch during correction.

CLI (run on the GPU box inside the `lineformer` conda env):
    python -m src.training.predict_to_cvat \
        --checkpoint <path> --config <path> --images-dir <dir> \
        --out <preannotations.xml> [--score-thr 0.5] [--limit N]
"""
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

from src.common.log import get_logger

logger = get_logger(__name__)

DEFAULT_SCORE_THR = 0.5
CURVE_NAME_PLACEHOLDER = "TODO"
# A polygon needs >=3 vertices to be valid geometry.
MIN_POLYGON_POINTS = 3
# cv2.approxPolyDP epsilon as a fraction of the contour perimeter — keeps
# polygons small/editable without materially changing their shape.
APPROX_EPSILON_RATIO = 0.003

T = TypeVar("T")
Point = Tuple[float, float]


def mask_to_polygon(mask: np.ndarray) -> Optional[List[Point]]:
    """Largest-contour polygon of a boolean mask, as (x, y) float points.

    Returns ``None`` for an empty mask or one whose largest contour has
    fewer than ``MIN_POLYGON_POINTS`` vertices after simplification (nothing
    worth annotating). Uses only the LARGEST external contour — a defensive
    choice against noisy/disconnected mask predictions producing more than
    one component for what is meant to be a single curve instance.
    """
    import cv2

    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    epsilon = APPROX_EPSILON_RATIO * cv2.arcLength(largest, True)
    simplified = cv2.approxPolyDP(largest, epsilon, True)
    points = [(float(pt[0][0]), float(pt[0][1])) for pt in simplified]
    if len(points) < MIN_POLYGON_POINTS:
        return None
    return points


def filter_by_score(
    predictions: Sequence[Tuple[float, T]], score_thr: float = DEFAULT_SCORE_THR
) -> List[Tuple[float, T]]:
    """Keep (score, item) pairs with score >= score_thr, order preserved."""
    return [p for p in predictions if p[0] >= score_thr]


def build_cvat_xml(images: Sequence[Dict[str, Any]]) -> str:
    """Build a CVAT-for-images-1.1 XML string from per-image polygon lists.

    ``images``: ``[{"name", "width", "height", "polygons": [[(x,y), ...], ...]}, ...]``.
    Images with zero polygons are still included (so the annotator sees every
    image in the task, just with nothing pre-filled).
    """
    parts = ['<?xml version="1.0" encoding="utf-8"?>',
            "<annotations>", "<version>1.1</version>"]
    for img_id, img in enumerate(images):
        parts.append(
            f'<image id="{img_id}" name="{_xml_escape(img["name"])}" '
            f'width="{img["width"]}" height="{img["height"]}">'
        )
        for polygon in img["polygons"]:
            points_str = ";".join(f"{x:.2f},{y:.2f}" for x, y in polygon)
            parts.append(
                f'<polygon label="line" occluded="0" points="{points_str}" z_order="0">'
                f'<attribute name="curve_name">{CURVE_NAME_PLACEHOLDER}</attribute>'
                "</polygon>"
            )
        parts.append("</image>")
    parts.append("</annotations>")
    return "".join(parts)


def _xml_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# --------------------------------------------------------------------------
# Inference path — GPU box only; heavy imports stay inside this function.
# --------------------------------------------------------------------------

def run_predict_to_cvat(
    checkpoint: str,
    config: str,
    images_dir: str,
    out_path: str,
    score_thr: float = DEFAULT_SCORE_THR,
    limit: Optional[int] = None,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Run inference over every image in ``images_dir`` and write a CVAT
    pre-annotation XML. Returns a small summary dict for the CLI/report."""
    import os

    import cv2
    from mmdet.apis import inference_detector, init_detector

    os.environ.setdefault("LINEFORMER_REPO_ROOT",
                          str(Path(__file__).resolve().parents[2]))
    model = init_detector(config, checkpoint, device=device)

    images_path = Path(images_dir)
    file_names = sorted(p.name for p in images_path.glob("*.png"))
    if limit is not None:
        file_names = file_names[:limit]

    images: List[Dict[str, Any]] = []
    counts: List[int] = []
    for name in file_names:
        img_path = images_path / name
        h, w = cv2.imread(str(img_path)).shape[:2]
        bbox_result, segm_result = inference_detector(model, str(img_path))
        scores = [float(b[4]) for cb in bbox_result for b in cb]
        masks = [m for cm in segm_result for m in cm]
        kept = filter_by_score(list(zip(scores, masks)), score_thr=score_thr)

        polygons = []
        for score, mask in kept:
            poly = mask_to_polygon(np.asarray(mask, dtype=bool))
            if poly is not None:
                polygons.append(poly)
        images.append({"name": name, "width": w, "height": h,
                       "polygons": polygons})
        counts.append(len(polygons))
        logger.info("%s: %d predictions >= %.2f -> %d polygons written",
                    name, len(kept), score_thr, len(polygons))

    xml_text = build_cvat_xml(images)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml_text, encoding="utf-8")

    summary = {
        "n_images": len(images),
        "n_polygons_total": sum(counts),
        "polygons_per_image": dict(zip(file_names, counts)),
        "out_path": str(out),
    }
    logger.info("Pre-annotations written: %s (%d images, %d polygons)",
                out, summary["n_images"], summary["n_polygons_total"])
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(
        description="Convert LineFormer predictions to a CVAT pre-annotation XML.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--score-thr", type=float, default=DEFAULT_SCORE_THR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        summary = run_predict_to_cvat(
            args.checkpoint, args.config, args.images_dir, args.out,
            score_thr=args.score_thr, limit=args.limit, device=args.device)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("Pre-annotation generation failed: %s", exc)
        return 1
    print(f"Images processed : {summary['n_images']}")
    print(f"Polygons written : {summary['n_polygons_total']}")
    print(f"Output           : {summary['out_path']}")
    print("\nKNOWN LIMITATION: duplicate masks can occur on flat/low-texture "
          "curves (e.g. Ciss plateaus) — not filtered here; expect to "
          "occasionally delete a duplicate polygon during correction.")
    print(f"Every polygon's curve_name is the placeholder "
          f"'{CURVE_NAME_PLACEHOLDER}' — fill in the real name "
          f"(Ciss/Coss/Crss) for every curve before exporting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
