"""Overlay visual check of buffered masks (task T4).

Renders sampled COCO annotations over their source figure PNGs so the owner
can judge by eye whether the polyline buffer radius (baked into the COCO at
conversion time; canonical 4.5 px per T4 review) matches how curves render. Output:
one overlay PNG per sampled image plus a contact-sheet ``overlay_check.html``.

Source images live outside the repo. Three directory layouts are resolved,
in order, for a COCO ``file_name`` like ``DEVICE__fig_p8_021.png``:
  1. ``<dir>/DEVICE__fig_p8_021.png``            (exact)
  2. ``<dir>/DEVICE_fig_p8_021.png``             (single-underscore exports)
  3. ``<dir>/DEVICE/figures/fig_p8_021.png``     (pipeline cache layout)

CLI:
    python -m src.dataset_tools.overlay_check <coco.json> <output_dir>
        --images-dir <dir> [--images-dir <dir2> ...] [--n 6] [--seed 42]
"""
import argparse
import html
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from src.common.log import get_logger

logger = get_logger(__name__)

DEFAULT_SAMPLE_SIZE = 6
DEFAULT_SEED = 42
FILL_OPACITY = 0.4
# Distinct BGR colors (tab10-like), cycled per annotation within an image.
PALETTE = [
    (180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214),
    (189, 103, 148), (75, 86, 140), (194, 119, 227), (127, 127, 127),
    (34, 189, 188), (207, 190, 23),
]
HTML_NAME = "overlay_check.html"


def resolve_image(file_name: str, images_dirs: Sequence[Path]) -> Optional[Path]:
    """Locate a COCO file_name on disk across known directory layouts."""
    device, sep, fig = file_name.partition("__")
    for d in images_dirs:
        candidates = [d / file_name, d / file_name.replace("__", "_")]
        if sep:
            candidates.append(d / device / "figures" / fig)
        for c in candidates:
            if c.is_file():
                return c
    return None


def sample_images(
    images: List[Dict[str, Any]], n: int, seed: int
) -> List[Dict[str, Any]]:
    """Deterministically sample up to ``n`` images, maximizing variety.

    The merged COCO carries no source-task field, so device-name prefix
    (first 4 characters) plus image size is used as a variety proxy: samples
    are drawn round-robin across those groups.
    """
    rng = random.Random(seed)
    groups: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    for img in images:
        key = (img["file_name"][:4], img["width"], img["height"])
        groups.setdefault(key, []).append(img)
    for members in groups.values():
        rng.shuffle(members)
    keys = sorted(groups)  # sort for determinism, then shuffle
    rng.shuffle(keys)

    sampled: List[Dict[str, Any]] = []
    while len(sampled) < n and any(groups[k] for k in keys):
        for key in keys:
            if groups[key] and len(sampled) < n:
                sampled.append(groups[key].pop())
    return sampled


def draw_overlay(
    image_path: Path, annotations: List[Dict[str, Any]], out_path: Path
) -> None:
    """Draw each annotation's polygons filled at ``FILL_OPACITY`` plus a
    curve_name tag near the curve start, and write the result as PNG."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    fills = img.copy()
    for i, ann in enumerate(annotations):
        color = PALETTE[i % len(PALETTE)]
        for ring in ann["segmentation"]:
            pts = np.asarray(ring, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(fills, [pts], color)
    blended = cv2.addWeighted(fills, FILL_OPACITY, img, 1 - FILL_OPACITY, 0)

    h, w = blended.shape[:2]
    for i, ann in enumerate(annotations):
        color = PALETTE[i % len(PALETTE)]
        ring = ann["segmentation"][0]
        x = int(min(max(ring[0] + 4, 2), w - 60))
        y = int(min(max(ring[1] - 6, 12), h - 4))
        name = ann.get("attributes", {}).get("curve_name", "?")
        # Dark halo behind the label keeps it readable on any background.
        cv2.putText(blended, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(blended, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), blended)


def write_contact_sheet(
    entries: List[Dict[str, Any]], missing: List[str], output_dir: Path,
    coco_name: str,
) -> Path:
    """Write ``overlay_check.html`` referencing the overlay PNGs."""
    rows = []
    for e in entries:
        curves = ", ".join(html.escape(c) for c in e["curve_names"])
        rows.append(
            f'<figure><img src="{html.escape(e["png"])}" alt="">'
            f'<figcaption><b>{html.escape(e["file_name"])}</b> '
            f'({e["width"]}×{e["height"]}, {e["n_curves"]} curves)<br>'
            f'{curves}</figcaption></figure>'
        )
    missing_html = ""
    if missing:
        items = "".join(f"<li>{html.escape(m)}</li>" for m in missing)
        missing_html = (f"<h2>Sampled but not found in supplied image dirs "
                        f"({len(missing)})</h2><ul>{items}</ul>")
    page = (
        "<!doctype html><meta charset='utf-8'>"
        "<title>T4 overlay check</title>"
        "<style>body{font-family:sans-serif;margin:2em}"
        "figure{display:inline-block;margin:1em;vertical-align:top;max-width:640px}"
        "img{max-width:100%;border:1px solid #999}"
        "figcaption{font-size:0.85em;margin-top:0.4em}</style>"
        f"<h1>Overlay check — {html.escape(coco_name)}</h1>"
        "<p>Filled masks at 40% opacity, one color per annotation. "
        "Judge: does the mask thickness match the rendered curve stroke?</p>"
        + "".join(rows) + missing_html
    )
    out = output_dir / HTML_NAME
    out.write_text(page, encoding="utf-8")
    return out


def run_overlay_check(
    coco_path: Union[str, Path],
    images_dirs: Sequence[Union[str, Path]],
    output_dir: Union[str, Path],
    n: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Sample images, render overlays, write the contact sheet.

    Sampled images missing from every supplied directory are reported (and
    replaced by further samples so the reviewer still gets ``n`` overlays
    when possible). Returns a summary dict.
    """
    coco_path = Path(coco_path)
    output_dir = Path(output_dir)
    dirs = [Path(d) for d in images_dirs]
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    annotated = [img for img in coco["images"] if anns_by_image.get(img["id"])]
    logger.info("Loaded %s: %d annotated images; sampling %d (seed=%d)",
                coco_path, len(annotated), n, seed)

    # Oversample so missing files can be backfilled; the first n found win.
    candidates = sample_images(annotated, min(len(annotated), n * 4), seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    missing: List[str] = []
    for img in candidates:
        if len(entries) >= n:
            break
        src = resolve_image(img["file_name"], dirs)
        if src is None:
            missing.append(img["file_name"])
            logger.warning("Sampled image not found in supplied dirs: %s",
                           img["file_name"])
            continue
        anns = anns_by_image[img["id"]]
        png_name = f"overlay_{img['file_name'].replace('/', '_')}"
        draw_overlay(src, anns, output_dir / png_name)
        entries.append({
            "file_name": img["file_name"], "png": png_name,
            "width": img["width"], "height": img["height"],
            "n_curves": len(anns),
            "curve_names": [a.get("attributes", {}).get("curve_name", "?")
                            for a in anns],
        })
        logger.info("Overlay written: %s (%d curves, source %s)",
                    png_name, len(anns), src)

    sheet = write_contact_sheet(entries, missing, output_dir, coco_path.name)
    logger.info("Contact sheet: %s | %d overlays, %d sampled-but-missing",
                sheet, len(entries), len(missing))
    return {"overlays": [e["png"] for e in entries], "missing": missing,
            "html": str(sheet), "output_dir": str(output_dir)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(
        description="Render sampled COCO annotations over source figures for "
                    "visual review of the polyline buffer thickness.")
    parser.add_argument("coco_json", help="COCO file produced by cvat_to_coco")
    parser.add_argument("output_dir", help="Directory for overlay PNGs + HTML")
    parser.add_argument("--images-dir", action="append", required=True,
                        dest="images_dirs",
                        help="Directory of source images (repeatable)")
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help=f"Sample size (default {DEFAULT_SAMPLE_SIZE})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Sampling seed (default {DEFAULT_SEED})")
    args = parser.parse_args(argv)
    try:
        result = run_overlay_check(args.coco_json, args.images_dirs,
                                   args.output_dir, n=args.n, seed=args.seed)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("Overlay check failed: %s", exc)
        return 1
    print(f"Overlays written to: {result['output_dir']}")
    print(f"Contact sheet:       {result['html']}")
    if result["missing"]:
        print("Sampled images NOT found (skipped):")
        for name in result["missing"]:
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
