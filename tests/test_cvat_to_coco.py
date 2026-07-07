"""Tests for src.cvat_to_coco — written FIRST per CLAUDE.md §2 (strict TDD).

Covers: parse_cvat_xml, buffer_polyline, polygon_area, bbox_from_segmentation,
convert, validate_coco, and the CLI entry point.
"""
import json
import math
from pathlib import Path

import pytest

from src.cvat_to_coco import (
    CATEGORY_NAME,
    DEFAULT_BUFFER_PX,
    bbox_from_segmentation,
    buffer_polyline,
    convert,
    main,
    parse_cvat_xml,
    polygon_area,
    validate_coco,
)

FIXTURES = Path(__file__).parent / "fixtures" / "cvat"
SAMPLE_XML = FIXTURES / "sample.xml"


# ---------------------------------------------------------------- parse_cvat_xml

class TestParseCvatXml:
    def test_returns_all_images(self):
        images = parse_cvat_xml(SAMPLE_XML)
        assert len(images) == 2

    def test_image_metadata(self):
        images = parse_cvat_xml(SAMPLE_XML)
        first = images[0]
        assert first["name"] == "fig_001.png"
        assert first["width"] == 800
        assert first["height"] == 600

    def test_polyline_points_parsed_exactly(self):
        images = parse_cvat_xml(SAMPLE_XML)
        shape = images[0]["shapes"][0]
        assert shape["type"] == "polyline"
        assert shape["points"] == [(10.0, 20.0), (100.5, 80.25), (200.0, 90.0)]

    def test_polygon_shape_parsed(self):
        images = parse_cvat_xml(SAMPLE_XML)
        shapes = images[1]["shapes"]
        assert len(shapes) == 1
        assert shapes[0]["type"] == "polygon"
        assert len(shapes[0]["points"]) == 4

    def test_curve_name_attribute_extracted(self):
        images = parse_cvat_xml(SAMPLE_XML)
        names = [s["curve_name"] for s in images[0]["shapes"]]
        assert names == ["Vgs=4.5V", "Vgs=10V"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_cvat_xml(tmp_path / "nope.xml")

    def test_malformed_xml_raises(self, tmp_path):
        bad = tmp_path / "bad.xml"
        bad.write_text("<annotations><image></annotations>")
        with pytest.raises(ValueError):
            parse_cvat_xml(bad)

    def test_empty_curve_name_raises(self, tmp_path):
        # Empty curve_name mirrors the stage-5 rule: empty curve keys are an error.
        xml = tmp_path / "empty_name.xml"
        xml.write_text(
            '<?xml version="1.0"?><annotations>'
            '<image id="0" name="f.png" width="100" height="100">'
            '<polyline label="line" points="1.0,1.0;50.0,50.0">'
            '<attribute name="curve_name"></attribute>'
            "</polyline></image></annotations>"
        )
        with pytest.raises(ValueError, match="curve_name"):
            parse_cvat_xml(xml)

    def test_missing_curve_name_attribute_raises(self, tmp_path):
        xml = tmp_path / "no_attr.xml"
        xml.write_text(
            '<?xml version="1.0"?><annotations>'
            '<image id="0" name="f.png" width="100" height="100">'
            '<polyline label="line" points="1.0,1.0;50.0,50.0"/>'
            "</image></annotations>"
        )
        with pytest.raises(ValueError, match="curve_name"):
            parse_cvat_xml(xml)

    def test_unsupported_shapes_skipped_not_fatal(self):
        # sample.xml contains a <box>; it must be skipped (with a logged warning),
        # never silently converted.
        images = parse_cvat_xml(SAMPLE_XML)
        types = {s["type"] for s in images[0]["shapes"]}
        assert types == {"polyline"}
        assert len(images[0]["shapes"]) == 2


# ------------------------------------------------------------------- geometry

class TestBufferPolyline:
    def test_output_is_flat_even_length_polygon(self):
        seg = buffer_polyline([(0.0, 0.0), (100.0, 0.0)], buffer_px=3.0)
        assert isinstance(seg, list)
        assert len(seg) >= 6
        assert len(seg) % 2 == 0
        assert all(isinstance(v, float) for v in seg)

    def test_area_matches_length_times_width_plus_caps(self):
        # Straight line of length L buffered by radius r (round caps):
        # area ≈ L*2r + pi*r^2.
        r = 3.0
        seg = buffer_polyline([(0.0, 0.0), (100.0, 0.0)], buffer_px=r)
        expected = 100.0 * 2 * r + math.pi * r * r
        assert polygon_area(seg) == pytest.approx(expected, rel=0.02)

    def test_buffer_contains_original_points(self):
        from shapely.geometry import Point, Polygon

        pts = [(10.0, 20.0), (100.5, 80.25), (200.0, 90.0)]
        seg = buffer_polyline(pts, buffer_px=2.0)
        poly = Polygon(list(zip(seg[0::2], seg[1::2])))
        for x, y in pts:
            assert poly.contains(Point(x, y))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            buffer_polyline([(5.0, 5.0)], buffer_px=3.0)

    def test_nonpositive_buffer_raises(self):
        with pytest.raises(ValueError):
            buffer_polyline([(0.0, 0.0), (10.0, 10.0)], buffer_px=0.0)
        with pytest.raises(ValueError):
            buffer_polyline([(0.0, 0.0), (10.0, 10.0)], buffer_px=-1.0)


class TestPolygonMath:
    def test_area_unit_square(self):
        assert polygon_area([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]) == pytest.approx(1.0)

    def test_area_orientation_independent(self):
        ccw = [0.0, 0.0, 4.0, 0.0, 4.0, 3.0, 0.0, 3.0]
        cw = [0.0, 0.0, 0.0, 3.0, 4.0, 3.0, 4.0, 0.0]
        assert polygon_area(ccw) == pytest.approx(12.0)
        assert polygon_area(cw) == pytest.approx(12.0)

    def test_bbox_from_segmentation(self):
        seg = [10.0, 5.0, 30.0, 5.0, 30.0, 25.0, 10.0, 25.0]
        assert bbox_from_segmentation(seg) == pytest.approx([10.0, 5.0, 20.0, 20.0])


# -------------------------------------------------------------------- convert

class TestConvert:
    def test_coco_structure(self):
        coco = convert(SAMPLE_XML)
        assert set(coco) >= {"images", "annotations", "categories"}
        assert len(coco["images"]) == 2
        assert len(coco["categories"]) == 1
        assert coco["categories"][0]["name"] == CATEGORY_NAME

    def test_annotation_counts_and_unique_ids(self):
        coco = convert(SAMPLE_XML)
        # 2 polylines + 1 polygon (box skipped) = 3 annotations
        assert len(coco["annotations"]) == 3
        ids = [a["id"] for a in coco["annotations"]]
        assert len(set(ids)) == len(ids)
        image_ids = {img["id"] for img in coco["images"]}
        assert all(a["image_id"] in image_ids for a in coco["annotations"])

    def test_polygon_passes_through_unbuffered(self):
        coco = convert(SAMPLE_XML)
        band = [a for a in coco["annotations"]
                if a["attributes"]["curve_name"] == "typical-band"][0]
        assert band["segmentation"] == [
            [100.0, 100.0, 200.0, 100.0, 200.0, 150.0, 100.0, 150.0]
        ]
        assert band["area"] == pytest.approx(100.0 * 50.0)

    def test_polyline_is_buffered(self):
        coco = convert(SAMPLE_XML, buffer_px=DEFAULT_BUFFER_PX)
        ann = [a for a in coco["annotations"]
               if a["attributes"]["curve_name"] == "Vgs=4.5V"][0]
        seg = ann["segmentation"][0]
        # Buffered outline has more vertices than the 3-point source polyline.
        assert len(seg) > 6
        assert ann["area"] > 0
        assert ann["iscrowd"] == 0

    def test_curve_name_preserved(self):
        coco = convert(SAMPLE_XML)
        names = {a["attributes"]["curve_name"] for a in coco["annotations"]}
        assert names == {"Vgs=4.5V", "Vgs=10V", "typical-band"}

    def test_writes_valid_json_file(self, tmp_path):
        out = tmp_path / "out.json"
        convert(SAMPLE_XML, output_path=out)
        loaded = json.loads(out.read_text())
        assert validate_coco(loaded) == []


# -------------------------------------------------------------- validate_coco

class TestValidateCoco:
    def _valid(self):
        return convert(SAMPLE_XML)

    def test_valid_output_has_no_errors(self):
        assert validate_coco(self._valid()) == []

    def test_duplicate_annotation_ids_flagged(self):
        coco = self._valid()
        coco["annotations"][1]["id"] = coco["annotations"][0]["id"]
        assert any("id" in e for e in validate_coco(coco))

    def test_bbox_outside_image_flagged(self):
        coco = self._valid()
        coco["annotations"][0]["bbox"] = [9000.0, 9000.0, 10.0, 10.0]
        assert validate_coco(coco) != []

    def test_empty_segmentation_flagged(self):
        coco = self._valid()
        coco["annotations"][0]["segmentation"] = [[]]
        assert validate_coco(coco) != []

    def test_unknown_image_id_flagged(self):
        coco = self._valid()
        coco["annotations"][0]["image_id"] = 999
        assert validate_coco(coco) != []

    def test_empty_curve_name_flagged(self):
        coco = self._valid()
        coco["annotations"][0]["attributes"]["curve_name"] = ""
        assert any("curve_name" in e for e in validate_coco(coco))


# ------------------------------------------------------------------------ CLI

class TestCli:
    def test_cli_converts_and_writes(self, tmp_path):
        out = tmp_path / "coco.json"
        rc = main([str(SAMPLE_XML), str(out), "--buffer-px", "4"])
        assert rc == 0
        coco = json.loads(out.read_text())
        assert validate_coco(coco) == []
        assert len(coco["annotations"]) == 3

    def test_cli_missing_input_returns_nonzero(self, tmp_path):
        rc = main([str(tmp_path / "missing.xml"), str(tmp_path / "o.json")])
        assert rc != 0
