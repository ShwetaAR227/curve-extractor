"""Tests for src.training.predict_to_cvat — written FIRST (CLAUDE.md §2).

Pure logic only (mask -> polygon geometry, score filtering, CVAT XML
writing): no GPU/torch/mmdet dependency. Model inference is
integration-tested on the GPU box, same precedent as eval_lineformer.py /
train_lineformer.py.
"""
import numpy as np
import pytest

from src.cvat_to_coco import parse_cvat_xml
from src.training.predict_to_cvat import (
    CURVE_NAME_PLACEHOLDER,
    MIN_POLYGON_POINTS,
    build_cvat_xml,
    filter_by_score,
    mask_to_polygon,
)


def box_mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


# ------------------------------------------------------------- mask_to_polygon

class TestMaskToPolygon:
    def test_square_mask_returns_four_ish_points_right_area(self):
        mask = box_mask(50, 50, 10, 30, 10, 30)  # 20x20 = 400 px
        pts = mask_to_polygon(mask)
        assert pts is not None
        assert len(pts) >= 3
        # shoelace area should be close to the true 400 px square
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        area = 0.5 * abs(sum(xs[i] * ys[(i + 1) % len(xs)] -
                             xs[(i + 1) % len(xs)] * ys[i]
                             for i in range(len(xs))))
        assert area == pytest.approx(400, rel=0.15)

    def test_points_are_xy_not_rowcol(self):
        # Tall, narrow mask: rows 0..40 (height), cols 0..5 (width).
        mask = box_mask(50, 50, 0, 40, 0, 5)
        pts = mask_to_polygon(mask)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        assert max(xs) - min(xs) < max(ys) - min(ys)  # narrow in x, tall in y

    def test_points_are_floats(self):
        mask = box_mask(30, 30, 5, 15, 5, 15)
        pts = mask_to_polygon(mask)
        assert all(isinstance(p[0], float) and isinstance(p[1], float)
                  for p in pts)

    def test_empty_mask_returns_none(self):
        mask = np.zeros((30, 30), dtype=bool)
        assert mask_to_polygon(mask) is None

    def test_tiny_mask_below_min_points_returns_none(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[10, 10] = True  # single pixel
        assert mask_to_polygon(mask) is None

    def test_two_components_keeps_only_largest(self):
        mask = box_mask(60, 60, 0, 5, 0, 5)      # small: 25 px
        mask |= box_mask(60, 60, 20, 50, 20, 50)  # large: 900 px
        pts = mask_to_polygon(mask)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # Should bound the LARGE component (20..50), not the small one.
        assert min(xs) >= 15 and max(xs) <= 55
        assert min(ys) >= 15 and max(ys) <= 55

    def test_contour_not_duplicated_closing_point(self):
        mask = box_mask(40, 40, 5, 25, 5, 25)
        pts = mask_to_polygon(mask)
        assert pts[0] != pts[-1]

    def test_returns_at_least_min_polygon_points(self):
        mask = box_mask(40, 40, 5, 25, 5, 25)
        pts = mask_to_polygon(mask)
        assert len(pts) >= MIN_POLYGON_POINTS


# --------------------------------------------------------------- filter_by_score

class TestFilterByScore:
    def test_keeps_only_at_or_above_threshold(self):
        preds = [(0.9, "a"), (0.5, "b"), (0.4, "c"), (0.6, "d")]
        kept = filter_by_score(preds, score_thr=0.5)
        assert [p[1] for p in kept] == ["a", "b", "d"]

    def test_boundary_value_included(self):
        preds = [(0.5, "x")]
        assert filter_by_score(preds, score_thr=0.5) == [(0.5, "x")]

    def test_empty_input_no_crash(self):
        assert filter_by_score([], score_thr=0.5) == []

    def test_all_below_threshold_empty_output(self):
        preds = [(0.1, "a"), (0.2, "b")]
        assert filter_by_score(preds, score_thr=0.5) == []

    def test_order_preserved(self):
        preds = [(0.9, "a"), (0.7, "b"), (0.55, "c")]
        kept = filter_by_score(preds, score_thr=0.5)
        assert [p[1] for p in kept] == ["a", "b", "c"]


# ---------------------------------------------------------------- build_cvat_xml

class TestBuildCvatXml:
    def _images(self):
        return [
            {"name": "dev1__fig1.png", "width": 100, "height": 80,
             "polygons": [[(1.0, 2.0), (50.0, 2.0), (50.0, 40.0), (1.0, 40.0)]]},
            {"name": "dev2__fig2.png", "width": 60, "height": 60,
             "polygons": []},
        ]

    def test_round_trips_through_existing_parser(self, tmp_path):
        xml_text = build_cvat_xml(self._images())
        xml_path = tmp_path / "preanno.xml"
        xml_path.write_text(xml_text, encoding="utf-8")
        parsed = parse_cvat_xml(xml_path)
        assert len(parsed) == 2
        dev1 = next(p for p in parsed if p["name"] == "dev1__fig1.png")
        assert dev1["width"] == 100 and dev1["height"] == 80
        assert len(dev1["shapes"]) == 1
        assert dev1["shapes"][0]["type"] == "polygon"
        assert dev1["shapes"][0]["points"] == [
            (1.0, 2.0), (50.0, 2.0), (50.0, 40.0), (1.0, 40.0)]

    def test_placeholder_curve_name_used(self, tmp_path):
        xml_text = build_cvat_xml(self._images())
        xml_path = tmp_path / "preanno.xml"
        xml_path.write_text(xml_text, encoding="utf-8")
        parsed = parse_cvat_xml(xml_path)
        dev1 = next(p for p in parsed if p["name"] == "dev1__fig1.png")
        assert dev1["shapes"][0]["curve_name"] == CURVE_NAME_PLACEHOLDER

    def test_zero_polygon_image_included_with_no_shapes(self, tmp_path):
        xml_text = build_cvat_xml(self._images())
        xml_path = tmp_path / "preanno.xml"
        xml_path.write_text(xml_text, encoding="utf-8")
        parsed = parse_cvat_xml(xml_path)
        dev2 = next(p for p in parsed if p["name"] == "dev2__fig2.png")
        assert dev2["shapes"] == []

    def test_multiple_polygons_one_image(self, tmp_path):
        images = [{"name": "d__f.png", "width": 50, "height": 50,
                   "polygons": [[(1.0, 1.0), (10.0, 1.0), (10.0, 10.0)],
                               [(20.0, 20.0), (30.0, 20.0), (30.0, 30.0)]]}]
        xml_text = build_cvat_xml(images)
        xml_path = tmp_path / "preanno.xml"
        xml_path.write_text(xml_text, encoding="utf-8")
        parsed = parse_cvat_xml(xml_path)
        assert len(parsed[0]["shapes"]) == 2

    def test_version_1_1_declared(self):
        xml_text = build_cvat_xml(self._images())
        assert "<version>1.1</version>" in xml_text

    def test_empty_image_list(self, tmp_path):
        xml_text = build_cvat_xml([])
        xml_path = tmp_path / "preanno.xml"
        xml_path.write_text(xml_text, encoding="utf-8")
        assert parse_cvat_xml(xml_path) == []
