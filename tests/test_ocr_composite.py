"""Tests for src/azure_ocr/composite.py — composite-figure splitting.

Includes the known legacy bug as an explicit regression case: a caption line
spanning the full width UNDER both charts defeated legacy pixel-gap detection
(which required near-100% white columns over the FULL image height), so
two-chart composites shipped merged. Our detector measures whiteness over the
central band only, so spanning caption text can't mask the gap.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.azure_ocr.composite import find_composite_candidates, split_composite_figures


# ---------------------------------------------------------------------------
# Synthetic composite builders
# ---------------------------------------------------------------------------

def draw_chart(img, x1, y1, x2, y2):
    """Dark-bordered 'chart' with some interior gridlines."""
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
    for gx in range(x1 + 40, x2 - 10, 60):
        cv2.line(img, (gx, y1), (gx, y2), (150, 150, 150), 1)


def make_two_up(dir_: Path, name="fig_p1_000.png", w=900, h=400,
                spanning_caption=False):
    """Two charts side by side with a clean white gap in the middle."""
    img = np.full((h, w, 3), 255, np.uint8)
    draw_chart(img, 30, 30, 400, 330)     # left chart
    draw_chart(img, 500, 30, 870, 330)    # right chart
    if spanning_caption:
        # Caption text row spanning the FULL width under both charts —
        # the exact condition that broke legacy full-height gap detection.
        cv2.putText(img, "Figure 7. Typical characteristics at Tj = 25 C and 175 C",
                    (40, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    figures_dir = dir_ / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(figures_dir / name), img)
    return f"figures/{name}", w, h


def ocr_line(text, x1, y1, x2, y2):
    return {"text": text, "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def two_up_figure(image_path, w, h, extra_lines=()):
    """Figure dict for the two-up image with x-axis titles under each chart."""
    return {
        "index": 0,
        "page": 1,
        "image_path": image_path,
        "caption": "Typical characteristics",
        "bounding_box": {"x1": 0, "y1": 0, "x2": w, "y2": h},
        "detection_method": "doc_intel",
        "is_package_outline": False,
        "ocr_text": ["VDS, DRAIN-TO-SOURCE VOLTAGE (V)"],
        "ocr_lines": [
            ocr_line("VDS, DRAIN-TO-SOURCE VOLTAGE (V)", 120, 340, 340, 355),
            ocr_line("VGS, GATE-TO-SOURCE VOLTAGE (V)", 590, 340, 800, 355),
            *extra_lines,
        ],
    }


def single_chart_figure(dir_, name="fig_p1_001.png", w=400, h=380):
    img = np.full((h, w, 3), 255, np.uint8)
    draw_chart(img, 30, 30, 370, 330)
    figures_dir = dir_ / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(figures_dir / name), img)
    return {
        "index": 1,
        "page": 1,
        "image_path": f"figures/{name}",
        "caption": "Fig 3",
        "bounding_box": {"x1": 0, "y1": 0, "x2": w, "y2": h},
        "detection_method": "doc_intel",
        "is_package_outline": False,
        "ocr_text": ["ID (A)"],
        "ocr_lines": [ocr_line("VDS, DRAIN-TO-SOURCE VOLTAGE (V)", 100, 345, 300, 360)],
    }


# ---------------------------------------------------------------------------
# Candidate detection
# ---------------------------------------------------------------------------

class TestCandidates:
    def test_wide_aspect_is_candidate(self, tmp_path):
        path, w, h = make_two_up(tmp_path)
        fig = two_up_figure(path, w, h)
        assert find_composite_candidates([fig, single_chart_figure(tmp_path)]) == [fig]

    def test_two_subcaptions_is_candidate_signal(self, tmp_path):
        # LEGACY GAP FIXED: squarish 2-up (aspect < 2.2) with two "Figure N"
        # captions was missed by legacy; caption count is now a signal.
        fig = {
            "index": 0, "image_path": "figures/x.png", "is_package_outline": False,
            "caption": "", "detection_method": "doc_intel",
            "bounding_box": {"x1": 0, "y1": 0, "x2": 800, "y2": 400},
            "ocr_text": [], "ocr_lines": [
                ocr_line("Figure 1. Output characteristics", 100, 380, 300, 395),
                ocr_line("Figure 2. Transfer characteristics", 500, 380, 700, 395),
            ],
        }
        assert fig in find_composite_candidates([fig])

    def test_package_outline_never_candidate(self):
        fig = {
            "index": 0, "is_package_outline": True, "caption": "",
            "detection_method": "doc_intel",
            "bounding_box": {"x1": 0, "y1": 0, "x2": 2000, "y2": 400},
            "ocr_lines": [ocr_line("x", 0, 0, 1, 1)],
        }
        assert find_composite_candidates([fig]) == []

    def test_already_split_never_candidate(self):
        fig = {
            "index": 0, "is_package_outline": False, "caption": "",
            "detection_method": "composite_split",
            "bounding_box": {"x1": 0, "y1": 0, "x2": 2000, "y2": 400},
            "ocr_lines": [ocr_line("x", 0, 0, 1, 1)],
        }
        assert find_composite_candidates([fig]) == []

    def test_no_ocr_lines_never_candidate(self):
        fig = {
            "index": 0, "is_package_outline": False, "caption": "",
            "detection_method": "doc_intel",
            "bounding_box": {"x1": 0, "y1": 0, "x2": 2000, "y2": 400},
            "ocr_lines": [],
        }
        assert find_composite_candidates([fig]) == []


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

class TestSplit:
    def test_two_up_splits_into_two(self, tmp_path):
        path, w, h = make_two_up(tmp_path)
        figs = split_composite_figures([two_up_figure(path, w, h)], tmp_path)
        assert len(figs) == 2
        assert all(f["detection_method"] == "composite_split" for f in figs)
        for f in figs:
            assert (tmp_path / f["image_path"]).exists()

    def test_spanning_caption_regression(self, tmp_path):
        """THE known legacy bug: caption under both charts -> merged output."""
        path, w, h = make_two_up(tmp_path, spanning_caption=True)
        figs = split_composite_figures([two_up_figure(path, w, h)], tmp_path)
        assert len(figs) == 2, "spanning caption must not defeat the split"

    def test_ocr_lines_reassigned_with_local_coords(self, tmp_path):
        path, w, h = make_two_up(tmp_path)
        figs = split_composite_figures([two_up_figure(path, w, h)], tmp_path)
        left, right = sorted(figs, key=lambda f: f["bounding_box"]["x1"])
        assert left["ocr_text"] == ["VDS, DRAIN-TO-SOURCE VOLTAGE (V)"]
        assert right["ocr_text"] == ["VGS, GATE-TO-SOURCE VOLTAGE (V)"]
        # right line's x must be translated into the right crop's local frame
        rx1 = right["ocr_lines"][0]["bounding_box"]["x1"]
        assert 0 <= rx1 < right["bounding_box"]["x2"] - right["bounding_box"]["x1"]

    def test_non_candidate_passes_through_unchanged(self, tmp_path):
        fig = single_chart_figure(tmp_path)
        before = dict(fig)
        figs = split_composite_figures([fig], tmp_path)
        assert figs == [before | {"index": 0}]

    def test_unsplittable_candidate_flagged_not_silent(self, tmp_path):
        """LEGACY BUG NOT PORTED: a candidate that can't be split is flagged
        composite_suspect=True instead of silently passing as normal."""
        # Solid image with NO white gap anywhere — unsplittable
        img = np.full((400, 900, 3), 255, np.uint8)
        draw_chart(img, 10, 10, 890, 390)
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir(parents=True)
        cv2.imwrite(str(figures_dir / "solid.png"), img)
        fig = two_up_figure("figures/solid.png", 900, 400)
        # Make OCR give no usable split signal either
        fig["ocr_lines"] = [ocr_line("some text", 400, 200, 500, 215)]
        figs = split_composite_figures([fig], tmp_path)
        assert len(figs) == 1
        assert figs[0]["composite_suspect"] is True

    def test_indices_resequenced(self, tmp_path):
        path, w, h = make_two_up(tmp_path)
        single = single_chart_figure(tmp_path)
        figs = split_composite_figures([two_up_figure(path, w, h), single], tmp_path)
        assert [f["index"] for f in figs] == list(range(len(figs)))

    def test_missing_image_kept_not_crashed(self, tmp_path):
        fig = two_up_figure("figures/ghost.png", 900, 400)
        figs = split_composite_figures([fig], tmp_path)
        assert len(figs) == 1  # kept as-is

    def test_sub_captions_assigned(self, tmp_path):
        path, w, h = make_two_up(tmp_path)
        fig = two_up_figure(path, w, h, extra_lines=(
            ocr_line("Figure 1. Output", 150, 360, 320, 375),
            ocr_line("Figure 2. Transfer", 600, 360, 790, 375),
        ))
        figs = split_composite_figures([fig], tmp_path)
        left, right = sorted(figs, key=lambda f: f["bounding_box"]["x1"])
        assert "Figure 1" in left["caption"]
        assert "Figure 2" in right["caption"]

    def test_empty_list_passthrough(self, tmp_path):
        assert split_composite_figures([], tmp_path) == []
