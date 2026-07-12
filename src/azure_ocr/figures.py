"""Stage-3 figure extraction: Doc Intel bounding regions -> cropped PNGs.

Primary path crops each figure reported by Azure Document Intelligence from
the pre-rendered page PNG (polygon coordinates are in inches; multiplied by
the render DPI). Fallback when Doc Intel reports no figures: OpenCV contour
detection. Both were legacy-verified behaviors; skips are always logged with
the reason (figures are never silently dropped).
"""
import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Union

import cv2
import numpy as np

from .config import DEFAULT_DPI, FIGURE_PADDING_PX

logger = logging.getLogger(__name__)

# --- package-outline flagging ------------------------------------------------

_PACKAGE_OUTLINE_KEYWORDS = [
    "dimensions", "outline", "inches", "package outline",
    "mechanical", "footprint", "land pattern",
]

# If any of these appear the figure is almost certainly a graph — override.
_GRAPH_OVERRIDE_KEYWORDS = [
    "parameter:", "= f(", "vs.", "typ.", "max.",
    "vgs", "vds", "rdson", "rds(on)", "id =", "tj",
    "gate charge", "capacitance", "impedance",
]

# "mm" must be standalone (not "[mm]") — Azure misreads mOhm/mΩ as "[mm]".
_STANDALONE_MM_RE = re.compile(r"(?<!\[)\bmm\b")

# --- contour-fallback gates (legacy-verified values) --------------------------

_MIN_AREA_RATIO = 0.02
_MAX_AREA_RATIO = 0.40
_MIN_ASPECT = 0.3
_MAX_ASPECT = 4.0
_MIN_WIDTH_PX = 200
_MIN_HEIGHT_PX = 150


def extract_figures(
    doc_result: Any,
    page_images_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> List[dict]:
    """Extract figure crops for a document and save them under
    ``<output_dir>/figures/``.

    Args:
        doc_result: Object exposing ``.figures`` (Doc Intel figure dicts)
            and ``.pages`` properties.
        page_images_dir: Directory of pre-rendered ``page_NNN.png`` files.
        output_dir: Run root; crops go to ``<output_dir>/figures/``.

    Returns:
        One dict per extracted figure: index, page, image_path (relative to
        ``output_dir``), caption, bounding_box (pixels), detection_method,
        is_package_outline (initially False).
    """
    page_images_dir = Path(page_images_dir)
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    api_figures = getattr(doc_result, "figures", None) or []
    if api_figures:
        logger.info("Extracting %d figure(s) via Doc Intel bounding regions", len(api_figures))
        return _extract_doc_intel_figures(api_figures, page_images_dir, figures_dir)

    logger.info("Doc Intel reported no figures — using contour fallback")
    return _extract_contour_figures(page_images_dir, figures_dir)


def flag_package_outlines(figures: List[dict]) -> List[dict]:
    """Set ``is_package_outline`` on every figure from its ``ocr_text``.

    Keyword-based: mechanical-drawing keywords set the flag; standalone
    "mm" counts but bracketed "[mm]" does not (known Azure misread of mΩ);
    graph-characteristic keywords override to False. Modifies in place.
    """
    for fig in figures:
        combined = " ".join(fig.get("ocr_text") or []).lower()
        is_outline = any(kw in combined for kw in _PACKAGE_OUTLINE_KEYWORDS)
        if not is_outline and _STANDALONE_MM_RE.search(combined):
            is_outline = True
        if is_outline and any(kw in combined for kw in _GRAPH_OVERRIDE_KEYWORDS):
            is_outline = False
        fig["is_package_outline"] = is_outline
    return figures


def _extract_doc_intel_figures(
    api_figures: List[dict],
    page_images_dir: Path,
    figures_dir: Path,
) -> List[dict]:
    """Crop figures at their Doc Intel polygons (inches -> pixels at DPI)."""
    results: List[dict] = []
    for idx, figure in enumerate(api_figures):
        regions = figure.get("boundingRegions") or []
        if not regions:
            logger.warning("Figure %d: no boundingRegions — skipped", idx)
            continue
        region = regions[0]
        page_num = region.get("pageNumber", 1)
        polygon = region.get("polygon") or []
        if len(polygon) < 8:
            logger.warning(
                "Figure %d (page %d): malformed polygon (%d values) — skipped",
                idx, page_num, len(polygon),
            )
            continue

        page_path = page_images_dir / f"page_{page_num:03d}.png"
        img = cv2.imread(str(page_path)) if page_path.exists() else None
        if img is None:
            logger.warning(
                "Figure %d: page image %s missing/unreadable — skipped", idx, page_path.name
            )
            continue

        img_h, img_w = img.shape[:2]
        xs = [polygon[i] * DEFAULT_DPI for i in range(0, len(polygon), 2)]
        ys = [polygon[i] * DEFAULT_DPI for i in range(1, len(polygon), 2)]
        x1 = int(max(0, min(xs) - FIGURE_PADDING_PX))
        y1 = int(max(0, min(ys) - FIGURE_PADDING_PX))
        x2 = int(min(img_w, max(xs) + FIGURE_PADDING_PX))
        y2 = int(min(img_h, max(ys) + FIGURE_PADDING_PX))

        name = f"fig_p{page_num}_{idx:03d}.png"
        if not cv2.imwrite(str(figures_dir / name), img[y1:y2, x1:x2]):
            logger.warning("Figure %d: failed to write crop %s — skipped", idx, name)
            continue

        results.append({
            "index": idx,
            "page": page_num,
            "image_path": f"figures/{name}",
            "caption": (figure.get("caption") or {}).get("content"),
            "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "detection_method": "doc_intel",
            "is_package_outline": False,
        })

    logger.info("Extracted %d/%d figure(s) via Doc Intel", len(results), len(api_figures))
    return results


def _extract_contour_figures(page_images_dir: Path, figures_dir: Path) -> List[dict]:
    """Detect figure-like regions with OpenCV contours (no captions)."""
    results: List[dict] = []
    global_idx = 0

    page_images = sorted(page_images_dir.glob("page_*.png"))
    if not page_images:
        logger.warning("No page images in %s for contour fallback", page_images_dir)
        return results

    for page_path in page_images:
        try:
            page_num = int(page_path.stem.split("_")[1])
        except (IndexError, ValueError):
            logger.warning("Cannot parse page number from %s — skipped", page_path.name)
            continue
        img = cv2.imread(str(page_path))
        if img is None:
            logger.warning("Unreadable page image %s — skipped", page_path.name)
            continue

        img_h, img_w = img.shape[:2]
        page_area = img_h * img_w
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area_ratio = (w * h) / page_area if page_area else 0.0
            aspect = w / h if h else 0.0
            if not (_MIN_AREA_RATIO <= area_ratio <= _MAX_AREA_RATIO):
                continue
            if not (_MIN_ASPECT <= aspect <= _MAX_ASPECT):
                continue
            if w < _MIN_WIDTH_PX or h < _MIN_HEIGHT_PX:
                continue

            x1 = max(0, x - FIGURE_PADDING_PX)
            y1 = max(0, y - FIGURE_PADDING_PX)
            x2 = min(img_w, x + w + FIGURE_PADDING_PX)
            y2 = min(img_h, y + h + FIGURE_PADDING_PX)
            name = f"fig_p{page_num}_{global_idx:03d}.png"
            if not cv2.imwrite(str(figures_dir / name), img[y1:y2, x1:x2]):
                logger.warning("Failed to write contour crop %s — skipped", name)
                continue

            results.append({
                "index": global_idx,
                "page": page_num,
                "image_path": f"figures/{name}",
                "caption": None,
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "detection_method": "contour_fallback",
                "is_package_outline": False,
            })
            global_idx += 1

    logger.info("Extracted %d figure(s) via contour fallback", len(results))
    return results
