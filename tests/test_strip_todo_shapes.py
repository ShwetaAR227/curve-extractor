"""Tests for src.dataset_tools.strip_todo_shapes — written FIRST (CLAUDE.md §2)."""
import xml.etree.ElementTree as ET

import pytest

from src.cvat_to_coco import parse_cvat_xml
from src.dataset_tools.strip_todo_shapes import strip_todo_xml


def make_cvat_xml(path, images):
    """images: list of (name, width, height, [(shape_type, points_str, curve_name), ...])"""
    parts = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "<version>1.1</version>"]
    for img_id, (name, w, h, shapes) in enumerate(images):
        parts.append(f'<image id="{img_id}" name="{name}" width="{w}" height="{h}">')
        for shape_type, points, curve_name in shapes:
            parts.append(
                f'<{shape_type} label="line" occluded="0" points="{points}" z_order="0">'
                f'<attribute name="curve_name">{curve_name}</attribute>'
                f"</{shape_type}>"
            )
        parts.append("</image>")
    parts.append("</annotations>")
    path.write_text("".join(parts), encoding="utf-8")
    return path


class TestStripTodoShapes:
    def test_removes_only_todo_shapes(self, tmp_path):
        src = make_cvat_xml(tmp_path / "src.xml", [
            ("dev1__fig.png", 100, 100, [
                ("polygon", "1,1;10,1;10,10", "TODO"),
                ("polygon", "20,20;30,20;30,30", "Ciss"),
            ]),
        ])
        out = tmp_path / "out.xml"
        report = strip_todo_xml(src, out)
        parsed = parse_cvat_xml(out)
        assert len(parsed[0]["shapes"]) == 1
        assert parsed[0]["shapes"][0]["curve_name"] == "Ciss"
        assert report["shapes_removed"] == 1

    def test_other_curve_names_untouched(self, tmp_path):
        src = make_cvat_xml(tmp_path / "src.xml", [
            ("dev1__fig.png", 100, 100, [
                ("polygon", "1,1;10,1;10,10", "Ciss"),
                ("polygon", "20,20;30,20;30,30", "Coss"),
                ("polygon", "40,40;50,40;50,50", "Crss"),
            ]),
        ])
        out = tmp_path / "out.xml"
        strip_todo_xml(src, out)
        parsed = parse_cvat_xml(out)
        names = {s["curve_name"] for s in parsed[0]["shapes"]}
        assert names == {"Ciss", "Coss", "Crss"}
        assert len(parsed[0]["shapes"]) == 3

    def test_image_with_zero_todo_completely_unmodified(self, tmp_path):
        src = make_cvat_xml(tmp_path / "src.xml", [
            ("clean__fig.png", 50, 50, [("polygon", "1,1;5,1;5,5", "Ciss")]),
        ])
        out = tmp_path / "out.xml"
        report = strip_todo_xml(src, out)
        parsed = parse_cvat_xml(out)
        assert len(parsed[0]["shapes"]) == 1
        assert parsed[0]["shapes"][0]["points"] == [(1.0, 1.0), (5.0, 1.0), (5.0, 5.0)]
        assert report["images_affected"] == []

    def test_image_where_all_shapes_are_todo_becomes_valid_empty_entry(self, tmp_path):
        src = make_cvat_xml(tmp_path / "src.xml", [
            ("alltodo__fig.png", 60, 60, [
                ("polygon", "1,1;5,1;5,5", "TODO"),
                ("polygon", "10,10;15,10;15,15", "TODO"),
            ]),
        ])
        out = tmp_path / "out.xml"
        report = strip_todo_xml(src, out)
        parsed = parse_cvat_xml(out)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "alltodo__fig.png"
        assert parsed[0]["shapes"] == []
        assert report["shapes_removed"] == 2

    def test_output_round_trips_through_parser(self, tmp_path):
        src = make_cvat_xml(tmp_path / "src.xml", [
            ("dev1__fig.png", 100, 100, [
                ("polyline", "1,1;10,1;10,10", "TODO"),
                ("polygon", "20,20;30,20;30,30", "Ciss"),
            ]),
        ])
        out = tmp_path / "out.xml"
        strip_todo_xml(src, out)
        # Would raise if the output weren't valid parseable CVAT XML.
        parsed = parse_cvat_xml(out)
        assert len(parsed) == 1

    def test_report_lists_affected_images_and_counts(self, tmp_path):
        src = make_cvat_xml(tmp_path / "src.xml", [
            ("a__fig.png", 50, 50, [("polygon", "1,1;5,1;5,5", "TODO")]),
            ("b__fig.png", 50, 50, [("polygon", "1,1;5,1;5,5", "Ciss")]),
            ("c__fig.png", 50, 50, [
                ("polygon", "1,1;5,1;5,5", "TODO"),
                ("polygon", "6,6;9,6;9,9", "TODO"),
            ]),
        ])
        out = tmp_path / "out.xml"
        report = strip_todo_xml(src, out)
        assert report["shapes_removed"] == 3
        assert sorted(report["images_affected"]) == ["a__fig.png", "c__fig.png"]

    def test_original_file_never_modified(self, tmp_path):
        src = make_cvat_xml(tmp_path / "src.xml", [
            ("dev1__fig.png", 100, 100, [("polygon", "1,1;10,1;10,10", "TODO")]),
        ])
        original_text = src.read_text(encoding="utf-8")
        out = tmp_path / "out.xml"
        strip_todo_xml(src, out)
        assert src.read_text(encoding="utf-8") == original_text

    def test_multiple_images_mixed_todo_and_clean(self, tmp_path):
        src = make_cvat_xml(tmp_path / "src.xml", [
            ("clean1__fig.png", 50, 50, [("polygon", "1,1;5,1;5,5", "Ciss")]),
            ("mixed__fig.png", 50, 50, [
                ("polygon", "1,1;5,1;5,5", "TODO"),
                ("polygon", "6,6;9,6;9,9", "Coss"),
            ]),
            ("clean2__fig.png", 50, 50, [("polygon", "1,1;5,1;5,5", "Crss")]),
        ])
        out = tmp_path / "out.xml"
        report = strip_todo_xml(src, out)
        parsed = {i["name"]: i for i in parse_cvat_xml(out)}
        assert len(parsed) == 3
        assert len(parsed["clean1__fig.png"]["shapes"]) == 1
        assert len(parsed["mixed__fig.png"]["shapes"]) == 1
        assert parsed["mixed__fig.png"]["shapes"][0]["curve_name"] == "Coss"
        assert len(parsed["clean2__fig.png"]["shapes"]) == 1
        assert report["images_affected"] == ["mixed__fig.png"]
