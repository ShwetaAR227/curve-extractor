"""Tests for src/azure_ocr/figures.py — figure extraction + outline flagging.

Page images are tiny synthetic PNGs built with numpy/cv2; the Doc Intel
result is a plain namespace — no Azure, no network.
"""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.azure_ocr.config import DEFAULT_DPI, FIGURE_PADDING_PX
from src.azure_ocr.figures import extract_figures, flag_package_outlines


def make_page(dir_, page_num, w=800, h=1000):
    """White page with a dark rectangle 'chart' at a known location."""
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 200), (500, 600), (0, 0, 0), 2)
    path = dir_ / f"page_{page_num:03d}.png"
    cv2.imwrite(str(path), img)
    return path


def doc_result(figures, pages=None):
    return SimpleNamespace(figures=figures, pages=pages or [{"pageNumber": 1}])


def api_figure(page=1, poly_inches=(0.5, 1.0, 2.5, 1.0, 2.5, 3.0, 0.5, 3.0), caption="Fig 1"):
    return {
        "boundingRegions": [{"pageNumber": page, "polygon": list(poly_inches)}],
        "caption": {"content": caption},
    }


class TestDocIntelPath:
    def test_crop_saved_and_metadata_correct(self, tmp_path):
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        make_page(pages_dir, 1)
        figs = extract_figures(doc_result([api_figure()]), pages_dir, tmp_path)
        assert len(figs) == 1
        fig = figs[0]
        assert fig["page"] == 1
        assert fig["detection_method"] == "doc_intel"
        assert fig["caption"] == "Fig 1"
        assert (tmp_path / fig["image_path"]).exists()

    def test_polygon_inches_to_pixels(self, tmp_path):
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        make_page(pages_dir, 1, w=2000, h=2000)
        figs = extract_figures(doc_result([api_figure()]), pages_dir, tmp_path)
        bb = figs[0]["bounding_box"]
        # polygon x range 0.5-2.5 inches -> 100-500 px at 200 DPI, +/- padding
        assert bb["x1"] == int(0.5 * DEFAULT_DPI - FIGURE_PADDING_PX)
        assert bb["x2"] == int(2.5 * DEFAULT_DPI + FIGURE_PADDING_PX)

    def test_padding_clamped_at_page_edge(self, tmp_path):
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        make_page(pages_dir, 1, w=400, h=400)
        # Polygon reaching the exact page edge
        fig = api_figure(poly_inches=(0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0))
        figs = extract_figures(doc_result([fig]), pages_dir, tmp_path)
        bb = figs[0]["bounding_box"]
        assert bb["x1"] == 0 and bb["y1"] == 0
        assert bb["x2"] <= 400 and bb["y2"] <= 400

    def test_missing_bounding_regions_skipped_not_crashed(self, tmp_path, caplog):
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        make_page(pages_dir, 1)
        bad = {"boundingRegions": [], "caption": {}}
        figs = extract_figures(doc_result([bad, api_figure()]), pages_dir, tmp_path)
        assert len(figs) == 1  # bad one skipped, good one survives

    def test_malformed_polygon_skipped(self, tmp_path):
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        make_page(pages_dir, 1)
        bad = {"boundingRegions": [{"pageNumber": 1, "polygon": [1.0, 2.0]}]}
        figs = extract_figures(doc_result([bad]), pages_dir, tmp_path)
        assert figs == []

    def test_missing_page_image_skipped(self, tmp_path):
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()  # no page written
        figs = extract_figures(doc_result([api_figure()]), pages_dir, tmp_path)
        assert figs == []


class TestContourFallback:
    def test_no_api_figures_falls_back_to_contours(self, tmp_path):
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        make_page(pages_dir, 1)  # contains one rectangle "chart"
        figs = extract_figures(doc_result([]), pages_dir, tmp_path)
        assert len(figs) == 1
        assert figs[0]["detection_method"] == "contour_fallback"
        assert (tmp_path / figs[0]["image_path"]).exists()

    def test_blank_page_yields_nothing(self, tmp_path):
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        blank = np.full((1000, 800, 3), 255, np.uint8)
        cv2.imwrite(str(pages_dir / "page_001.png"), blank)
        assert extract_figures(doc_result([]), pages_dir, tmp_path) == []

    def test_no_page_images_at_all(self, tmp_path):
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        assert extract_figures(doc_result([]), pages_dir, tmp_path) == []


class TestPackageOutlineFlag:
    def make(self, lines):
        return [{"ocr_text": lines, "ocr_lines": []}]

    def test_outline_keyword_flags(self):
        figs = flag_package_outlines(self.make(["Package Outline Dimensions"]))
        assert figs[0]["is_package_outline"] is True

    def test_standalone_mm_flags(self):
        figs = flag_package_outlines(self.make(["All dims in mm unless noted"]))
        assert figs[0]["is_package_outline"] is True

    def test_bracketed_mm_does_not_flag(self):
        # Azure misreads mOhm as [mm] — must not mark a chart as a drawing
        figs = flag_package_outlines(self.make(["RDS(on) [mm]"]))
        assert figs[0]["is_package_outline"] is False

    def test_graph_keywords_override(self):
        figs = flag_package_outlines(
            self.make(["outline of gate charge behaviour", "vgs = 10 V"])
        )
        assert figs[0]["is_package_outline"] is False

    def test_no_ocr_text_not_flagged(self):
        figs = flag_package_outlines([{"ocr_text": None}])
        assert figs[0]["is_package_outline"] is False
