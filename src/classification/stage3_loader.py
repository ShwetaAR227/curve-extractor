"""Stage-3 output loader (CLAUDE.md §1, stage 4 input).

Parses a device's real Stage-3 ``full_extraction.json`` into the exact
input :func:`src.classification.classify.classify_device` needs — a
``Dict[int, List[FigureCandidate]]`` keyed by page number. This is the
piece that never existed as a tested module (every prior real-data check
in this project — T13, T24, T27 — parsed ``full_extraction.json`` via
throwaway, uncommitted scripts; see ``src/classification/README.md``:
"not yet connected to real stage-3 data").

Layout assumed (matches the real corpus, e.g.
``D:\\Extractor\\data\\OCR1-OCR13\\<device>\\...``, and every ad-hoc script
that has read it so far):
    <stage3_root>/<device>/full_extraction.json
    <stage3_root>/<device>/<image_path>   (image_path taken from the JSON,
                                            e.g. "figures/fig_p8_019.png")

No new scoring/classification logic lives here — this module only builds
the data :mod:`src.classification.scoring`/:mod:`src.classification.classify`
already know how to consume.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import cv2

from src.common.log import get_logger
from src.classification.scoring import FigureCandidate, OcrLine

logger = get_logger(__name__)

FULL_EXTRACTION_FILENAME = "full_extraction.json"


def _convert_ocr_line(raw: Dict[str, Any]) -> OcrLine:
    """Convert one Stage-3 OCR line dict into a :class:`OcrLine`.

    Stage-3's shape is ``{"text": str, "bounding_box": {"x1","y1","x2","y2"}}``;
    ``OcrLine.bbox`` wants a plain ``(x1, y1, x2, y2)`` tuple.
    """
    bbox = raw.get("bounding_box")
    if bbox is None:
        return OcrLine(text=raw.get("text", ""), bbox=None)
    return OcrLine(
        text=raw.get("text", ""),
        bbox=(float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])),
    )


def _read_image_dimensions(image_full_path: Path) -> tuple:
    """Return ``(width, height)`` of the referenced figure PNG, or
    ``(None, None)`` if it's missing/unreadable — a known, non-fatal Stage
    1-3 gap (documented in T25/T27: a figure listed in the JSON manifest
    whose PNG was never rendered), not a crash."""
    if not image_full_path.is_file():
        logger.warning(
            "stage3_loader: referenced image not found (Stage 1-3 render "
            "gap, not fatal): %s", image_full_path,
        )
        return None, None
    image = cv2.imread(str(image_full_path))
    if image is None:
        logger.warning(
            "stage3_loader: referenced image unreadable (corrupt/unsupported, "
            "not fatal): %s", image_full_path,
        )
        return None, None
    height, width = image.shape[:2]
    return width, height


def load_figures_by_page(
    device: str, stage3_root: Union[str, Path]
) -> Dict[int, List[FigureCandidate]]:
    """Load a device's Stage-3 figures, grouped by page number.

    Args:
        device: Device identifier (the folder name under ``stage3_root``).
        stage3_root: Root directory containing one subfolder per device.

    Returns:
        ``{page_number: [FigureCandidate, ...]}``. Empty dict if the device
        has no figures at all (a valid state, not an error). Each
        candidate's ``figure_width``/``figure_height`` are ``None`` if its
        referenced image is missing/unreadable — that one figure degrades
        gracefully (scoring treats unknown zone as neutral) rather than
        failing the whole device.

    Raises:
        FileNotFoundError: ``<stage3_root>/<device>`` or its
            ``full_extraction.json`` doesn't exist — a real data gap, not
            silently swallowed.
        ValueError: ``full_extraction.json`` is not valid JSON, or is
            missing the expected ``"figures"`` key.
    """
    device_dir = Path(stage3_root) / device
    if not device_dir.is_dir():
        raise FileNotFoundError(f"Stage-3 device directory not found: {device_dir}")

    json_path = device_dir / FULL_EXTRACTION_FILENAME
    if not json_path.is_file():
        raise FileNotFoundError(f"Stage-3 output not found: {json_path}")

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed Stage-3 JSON at {json_path}: {exc}") from exc

    if "figures" not in payload:
        raise ValueError(f"Stage-3 JSON at {json_path} is missing the 'figures' key")

    figures_by_page: Dict[int, List[FigureCandidate]] = {}
    for raw_figure in payload["figures"]:
        image_path = raw_figure["image_path"]
        width, height = _read_image_dimensions(device_dir / image_path)
        candidate = FigureCandidate(
            figure_id=image_path,
            page=raw_figure["page"],
            figure_index=raw_figure.get("index", 0),
            image_path=image_path,
            caption=raw_figure.get("caption"),
            ocr_lines=[_convert_ocr_line(line) for line in raw_figure.get("ocr_lines", [])],
            figure_width=width,
            figure_height=height,
        )
        figures_by_page.setdefault(candidate.page, []).append(candidate)

    logger.info(
        "load_figures_by_page(%s): %d page(s), %d figure(s) total from %s",
        device, len(figures_by_page), sum(len(v) for v in figures_by_page.values()), json_path,
    )
    return figures_by_page
