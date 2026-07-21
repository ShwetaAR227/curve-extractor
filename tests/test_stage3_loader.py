"""Tests for src.classification.stage3_loader — written FIRST (CLAUDE.md §2,
red phase). Module does not exist yet.

``load_figures_by_page(device, stage3_root) -> Dict[int, List[FigureCandidate]]``
parses a device's real Stage-3 ``full_extraction.json`` (the same format used
throughout this project's ad-hoc real-data scripts, e.g. T13/T24/T27) into the
exact input ``classify_device`` needs — the piece that's never existed as a
tested module (src/classification/README.md: "not yet connected to real
stage-3 data").

Design decisions this test file bakes in (flagged for owner review, not yet
approved beyond the interface signature):
- ``full_extraction.json`` lives at ``<stage3_root>/<device>/full_extraction.json``,
  matching the real corpus layout (``D:\\Extractor\\data\\OCR1-OCR13\\<device>\\...``)
  used by every prior real-data check in this project.
- Each figure's own image lives at ``<stage3_root>/<device>/<image_path>``
  (image_path taken verbatim from the JSON, e.g. ``figures/fig_p8_019.png``),
  same tree — no separate images_root needed for this loader.
- ``FigureCandidate.figure_width``/``figure_height`` come from actually
  reading the referenced PNG (its real pixel dimensions), NOT from the
  JSON's own per-figure ``bounding_box`` (a different, page-relative crop
  region) — OCR line bboxes are recorded in the rendered PNG's own pixel
  space, so the zone-scoring heuristic (`_classify_zone`) needs the PNG's
  real dimensions to be meaningful.
- ``figure_id`` is the ``image_path`` string as-is (unique within one
  device's figures, which is all the ``claimed`` mechanism needs since
  ``ClaimTracker`` already partitions claims by device).
- A missing/unreadable referenced image degrades that ONE figure to
  ``figure_width=None``/``figure_height=None`` (score_figure already treats
  unknown zone as neutral, partial-credit scoring) rather than failing the
  whole device — consistent with this project's quarantine-not-crash
  philosophy, but NOT the same as a missing/malformed ``full_extraction.json``
  itself, which is a real error and must raise.

No GPU, no network — every fixture writes real files under ``tmp_path`` and
optionally a tiny real PNG via cv2 (already a project dependency).
"""
import json

import cv2
import numpy as np
import pytest

from src.classification.stage3_loader import load_figures_by_page
from src.classification.scoring import FigureCandidate


def _write_full_extraction(root, device, figures):
    device_dir = root / device
    device_dir.mkdir(parents=True, exist_ok=True)
    (device_dir / "full_extraction.json").write_text(
        json.dumps({"figures": figures}), encoding="utf-8"
    )
    return device_dir


def _write_png(path, width=800, height=650, color=255):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _ocr_line(text, x1, y1, x2, y2):
    return {"text": text, "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def _figure(index, page, image_path, caption=None, ocr_lines=None):
    return {
        "index": index,
        "page": page,
        "image_path": image_path,
        "caption": caption,
        "bounding_box": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
        "detection_method": "azure_ocr",
        "is_package_outline": False,
        "ocr_text": "",
        "ocr_lines": ocr_lines or [],
    }


# ---------------------------------------------------------------- grouping

def test_figures_grouped_by_page_number(tmp_path):
    device_dir = _write_full_extraction(tmp_path, "DEV1", [
        _figure(0, page=3, image_path="figures/fig_p3_000.png"),
        _figure(1, page=5, image_path="figures/fig_p5_001.png"),
        _figure(2, page=5, image_path="figures/fig_p5_002.png"),
    ])
    _write_png(device_dir / "figures" / "fig_p3_000.png")
    _write_png(device_dir / "figures" / "fig_p5_001.png")
    _write_png(device_dir / "figures" / "fig_p5_002.png")

    result = load_figures_by_page("DEV1", tmp_path)
    assert set(result.keys()) == {3, 5}
    assert len(result[3]) == 1
    assert len(result[5]) == 2


def test_all_returned_items_are_figure_candidates(tmp_path):
    device_dir = _write_full_extraction(tmp_path, "DEV1", [
        _figure(0, page=1, image_path="figures/fig_p1_000.png"),
    ])
    _write_png(device_dir / "figures" / "fig_p1_000.png")
    result = load_figures_by_page("DEV1", tmp_path)
    assert isinstance(result[1][0], FigureCandidate)


# ------------------------------------------------------------- field mapping

def test_caption_and_image_path_populated_verbatim(tmp_path):
    device_dir = _write_full_extraction(tmp_path, "DEV1", [
        _figure(0, page=3, image_path="figures/fig_p3_000.png",
                caption="Fig 3. Typical Transfer Characteristics"),
    ])
    _write_png(device_dir / "figures" / "fig_p3_000.png")
    fig = load_figures_by_page("DEV1", tmp_path)[3][0]
    assert fig.caption == "Fig 3. Typical Transfer Characteristics"
    assert fig.image_path == "figures/fig_p3_000.png"
    assert fig.page == 3


def test_ocr_lines_converted_from_bounding_box_dict_to_bbox_tuple(tmp_path):
    # Stage-3's raw shape is {"text", "bounding_box": {"x1","y1","x2","y2"}};
    # scoring.OcrLine wants a (x1, y1, x2, y2) tuple in .bbox.
    device_dir = _write_full_extraction(tmp_path, "DEV1", [
        _figure(0, page=3, image_path="figures/fig_p3_000.png", ocr_lines=[
            _ocr_line("VGS, Gate-to-Source Voltage (V)", 177, 547, 495, 573),
        ]),
    ])
    _write_png(device_dir / "figures" / "fig_p3_000.png")
    fig = load_figures_by_page("DEV1", tmp_path)[3][0]
    assert len(fig.ocr_lines) == 1
    line = fig.ocr_lines[0]
    assert line.text == "VGS, Gate-to-Source Voltage (V)"
    assert line.bbox == (177.0, 547.0, 495.0, 573.0)


def test_figure_width_height_read_from_real_image_dimensions(tmp_path):
    device_dir = _write_full_extraction(tmp_path, "DEV1", [
        _figure(0, page=1, image_path="figures/fig_p1_000.png"),
    ])
    _write_png(device_dir / "figures" / "fig_p1_000.png", width=731, height=905)
    fig = load_figures_by_page("DEV1", tmp_path)[1][0]
    assert fig.figure_width == 731
    assert fig.figure_height == 905


def test_figure_id_unique_within_device(tmp_path):
    device_dir = _write_full_extraction(tmp_path, "DEV1", [
        _figure(0, page=1, image_path="figures/fig_p1_000.png"),
        _figure(1, page=2, image_path="figures/fig_p2_001.png"),
    ])
    _write_png(device_dir / "figures" / "fig_p1_000.png")
    _write_png(device_dir / "figures" / "fig_p2_001.png")
    result = load_figures_by_page("DEV1", tmp_path)
    ids = {result[1][0].figure_id, result[2][0].figure_id}
    assert len(ids) == 2


# --------------------------------------------------------------- edge cases

def test_device_with_empty_figures_list_returns_empty_dict(tmp_path):
    _write_full_extraction(tmp_path, "DEV1", [])
    assert load_figures_by_page("DEV1", tmp_path) == {}


def test_missing_referenced_image_degrades_to_none_dimensions_not_a_crash(tmp_path):
    # The JSON references a figure whose PNG was never rendered (the exact
    # real Stage 1-3 gap documented in T25/T27) — must not crash the device.
    _write_full_extraction(tmp_path, "DEV1", [
        _figure(0, page=1, image_path="figures/never_rendered.png"),
    ])
    fig = load_figures_by_page("DEV1", tmp_path)[1][0]
    assert fig.figure_width is None
    assert fig.figure_height is None


def test_missing_full_extraction_json_raises_clear_error(tmp_path):
    (tmp_path / "DEV_NO_JSON").mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        load_figures_by_page("DEV_NO_JSON", tmp_path)


def test_malformed_json_raises_clear_error(tmp_path):
    device_dir = tmp_path / "DEV_BAD_JSON"
    device_dir.mkdir()
    (device_dir / "full_extraction.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_figures_by_page("DEV_BAD_JSON", tmp_path)


def test_device_directory_itself_missing_raises_clear_error(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        load_figures_by_page("NEVER_SEEN_DEVICE", tmp_path)
