"""Stage-3 OCR attachment: run OCR over each figure crop, attach text + boxes.

The OCR callable is INJECTED (``ocr_fn(image_bytes) -> analyzeResult dict or
None``) so unit tests never touch Azure; production passes the real Azure
Read API client wrapper. Legacy-verified behaviors kept: idempotent re-runs
(figures with ``ocr_text`` already set are skipped — no re-spent quota),
call budget, per-call rate limit, and per-figure failure isolation (any OCR
failure yields empty text, logged, batch continues).
"""
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .config import DEFAULT_MAX_OCR_CALLS, OCR_RATE_LIMIT_S
from .figures import flag_package_outlines

logger = logging.getLogger(__name__)

OcrFn = Callable[[bytes], Optional[dict]]


def ocr_figures(
    figures: List[dict],
    base_dir: Union[str, Path],
    ocr_fn: OcrFn,
    max_calls: int = DEFAULT_MAX_OCR_CALLS,
) -> List[dict]:
    """Populate ``ocr_text``/``ocr_lines`` on each figure via ``ocr_fn``.

    Args:
        figures: Figure dicts from :func:`~.figures.extract_figures`; each
            needs ``image_path`` relative to ``base_dir``.
        base_dir: Root that ``image_path`` values are relative to.
        ocr_fn: ``(image_bytes) -> analyzeResult dict | None``. Injected so
            tests can mock; production wires the Azure Read API client.
        max_calls: OCR call budget; figures beyond it are left un-OCR'd
            (``ocr_text`` stays absent) with a warning.

    Returns:
        The same list, modified in place. ``is_package_outline`` is
        re-evaluated on every figure once OCR text is available.
    """
    base_dir = Path(base_dir)
    calls_made = 0

    for fig in figures:
        if fig.get("ocr_text") is not None:
            logger.debug("Figure %s already OCR'd — skipped", fig.get("index"))
            continue
        if calls_made >= max_calls:
            remaining = sum(1 for f in figures if f.get("ocr_text") is None)
            logger.warning(
                "OCR budget of %d calls exhausted — %d figure(s) left without OCR",
                max_calls, remaining,
            )
            break

        image_path = base_dir / fig.get("image_path", "")
        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            logger.warning(
                "Figure %s: cannot read %s (%s) — empty OCR", fig.get("index"), image_path, exc
            )
            fig["ocr_text"] = []
            fig["ocr_lines"] = []
            continue

        result: Optional[dict] = None
        try:
            result = ocr_fn(image_bytes)
        except Exception as exc:  # failure isolation: one figure never kills the batch
            logger.warning("Figure %s: OCR call raised %s — empty OCR", fig.get("index"), exc)
        calls_made += 1

        if result is None:
            fig["ocr_text"] = []
            fig["ocr_lines"] = []
        else:
            fig["ocr_text"], fig["ocr_lines"] = parse_ocr_result(result)
            logger.debug(
                "Figure %s: %d OCR line(s)", fig.get("index"), len(fig["ocr_text"])
            )

        if calls_made < max_calls:
            time.sleep(OCR_RATE_LIMIT_S)

    logger.info("OCR complete: %d call(s) across %d figure(s)", calls_made, len(figures))
    return flag_package_outlines(figures)


def parse_ocr_result(analyze_result: Dict[str, Any]) -> Tuple[List[str], List[dict]]:
    """Parse an Azure Read API ``analyzeResult`` into (ocr_text, ocr_lines).

    ``ocr_text`` is a flat list of line strings; ``ocr_lines`` pairs each
    text with a min/max ``bounding_box`` derived from the line polygon.
    Blank lines are dropped; malformed polygons yield an all-zero box.
    """
    ocr_text: List[str] = []
    ocr_lines: List[dict] = []
    for page_result in analyze_result.get("readResults") or []:
        for line in page_result.get("lines") or []:
            text = (line.get("text") or "").strip()
            if not text:
                continue
            ocr_text.append(text)
            ocr_lines.append({
                "text": text,
                "bounding_box": _bbox_from_polygon(line.get("boundingBox") or []),
            })
    return ocr_text, ocr_lines


def _bbox_from_polygon(polygon: List[float]) -> Dict[str, int]:
    if len(polygon) < 8:
        return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
    xs = [polygon[i] for i in range(0, len(polygon), 2)]
    ys = [polygon[i] for i in range(1, len(polygon), 2)]
    return {"x1": int(min(xs)), "y1": int(min(ys)), "x2": int(max(xs)), "y2": int(max(ys))}
