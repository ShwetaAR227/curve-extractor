"""Tests for src.dataset_tools.collect_images — written FIRST (CLAUDE.md §2).

All fixtures are built in tmp dirs; no real data involved.
"""
import json
from pathlib import Path

import pytest

from src.dataset_tools.collect_images import collect_images, main


def make_coco(path: Path, file_names: list) -> Path:
    coco = {
        "images": [{"id": i, "file_name": n, "width": 100, "height": 100}
                   for i, n in enumerate(file_names, start=1)],
        "annotations": [], "categories": [{"id": 1, "name": "line"}],
    }
    path.write_text(json.dumps(coco))
    return path


def put(root: Path, rel: str, content: bytes = b"png-bytes") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


class TestMatching:
    def test_exact_basename_match_copied_flat(self, tmp_path):
        coco = make_coco(tmp_path / "c.json", ["DEV1__fig_p8_021.png"])
        root = tmp_path / "root"
        put(root, "some/nested/dir/DEV1__fig_p8_021.png", b"A")
        out = tmp_path / "out"
        report = collect_images(coco, [root], out)
        assert report["missing"] == []
        assert (out / "DEV1__fig_p8_021.png").read_bytes() == b"A"

    def test_nested_device_figures_layout_match(self, tmp_path):
        # Legacy tree: <root>/.../<DEVICE>/figures/<fig>.png with bare fig name.
        coco = make_coco(tmp_path / "c.json", ["DEV1__fig_p8_021.png"])
        root = tmp_path / "root"
        put(root, "OCR1/DEV1/figures/fig_p8_021.png", b"B")
        out = tmp_path / "out"
        report = collect_images(coco, [root], out)
        assert report["missing"] == []
        # Copied flat under the COCO file_name, not the bare fig name.
        assert (out / "DEV1__fig_p8_021.png").read_bytes() == b"B"

    def test_bare_fig_name_outside_device_dir_not_matched(self, tmp_path):
        # A fig_*.png under the WRONG device (or no device dir) must not match.
        coco = make_coco(tmp_path / "c.json", ["DEV1__fig_p8_021.png"])
        root = tmp_path / "root"
        put(root, "OTHERDEV/figures/fig_p8_021.png", b"X")
        put(root, "loose/fig_p8_021.png", b"Y")
        report = collect_images(coco, [root], tmp_path / "out")
        assert report["missing"] == ["DEV1__fig_p8_021.png"]

    def test_distractor_variants_ignored(self, tmp_path):
        # Legacy figures/ dirs contain *_cv_overlay.png and validated_*.png.
        coco = make_coco(tmp_path / "c.json", ["DEV1__fig_p8_021.png"])
        root = tmp_path / "root"
        put(root, "DEV1/figures/fig_p8_021_cv_overlay.png", b"N")
        put(root, "DEV1/figures/validated_fig_p8_021.png", b"N")
        report = collect_images(coco, [root], tmp_path / "out")
        assert report["missing"] == ["DEV1__fig_p8_021.png"]

    def test_multiple_roots_searched(self, tmp_path):
        coco = make_coco(tmp_path / "c.json",
                         ["DEV1__fig_p1_001.png", "DEV2__fig_p2_002.png"])
        r1, r2 = tmp_path / "r1", tmp_path / "r2"
        put(r1, "DEV1/figures/fig_p1_001.png", b"1")
        put(r2, "DEV2/figures/fig_p2_002.png", b"2")
        out = tmp_path / "out"
        report = collect_images(coco, [r1, r2], out)
        assert report["missing"] == []
        assert len(report["copied"]) == 2

    def test_unrelated_files_not_copied(self, tmp_path):
        coco = make_coco(tmp_path / "c.json", ["DEV1__fig_p1_001.png"])
        root = tmp_path / "root"
        put(root, "DEV1/figures/fig_p1_001.png", b"1")
        put(root, "DEV1/figures/fig_p9_999.png", b"other")
        put(root, "DEV1/DEV1.pdf", b"pdf")
        out = tmp_path / "out"
        collect_images(coco, [root], out)
        assert sorted(p.name for p in out.iterdir()) == ["DEV1__fig_p1_001.png"]


class TestDuplicatesAndConflicts:
    def test_identical_duplicates_copied_once_no_conflict(self, tmp_path):
        coco = make_coco(tmp_path / "c.json", ["DEV1__fig_p1_001.png"])
        root = tmp_path / "root"
        put(root, "a/DEV1/figures/fig_p1_001.png", b"SAME")
        put(root, "b/DEV1__fig_p1_001.png", b"SAME")
        out = tmp_path / "out"
        report = collect_images(coco, [root], out)
        assert report["conflicts"] == {}
        assert report["missing"] == []
        assert (out / "DEV1__fig_p1_001.png").read_bytes() == b"SAME"

    def test_differing_content_is_conflict_and_not_copied(self, tmp_path):
        coco = make_coco(tmp_path / "c.json", ["DEV1__fig_p1_001.png"])
        root = tmp_path / "root"
        put(root, "a/DEV1/figures/fig_p1_001.png", b"AAA")
        put(root, "b/DEV1/figures/fig_p1_001.png", b"BBB")
        out = tmp_path / "out"
        report = collect_images(coco, [root], out)
        assert "DEV1__fig_p1_001.png" in report["conflicts"]
        assert len(report["conflicts"]["DEV1__fig_p1_001.png"]) == 2
        assert not (out / "DEV1__fig_p1_001.png").exists()

    def test_source_tree_never_modified(self, tmp_path):
        coco = make_coco(tmp_path / "c.json", ["DEV1__fig_p1_001.png"])
        root = tmp_path / "root"
        src = put(root, "DEV1/figures/fig_p1_001.png", b"KEEP")
        collect_images(coco, [root], tmp_path / "out")
        assert src.read_bytes() == b"KEEP"


class TestReportAndCli:
    def test_missing_reported(self, tmp_path):
        coco = make_coco(tmp_path / "c.json",
                         ["FOUND__fig_p1_001.png", "GONE__fig_p2_002.png"])
        root = tmp_path / "root"
        put(root, "FOUND/figures/fig_p1_001.png")
        report = collect_images(coco, [root], tmp_path / "out")
        assert report["missing"] == ["GONE__fig_p2_002.png"]

    def test_cli_exit_zero_when_all_found(self, tmp_path):
        coco = make_coco(tmp_path / "c.json", ["DEV1__fig_p1_001.png"])
        root = tmp_path / "root"
        put(root, "DEV1/figures/fig_p1_001.png")
        rc = main([str(coco), str(tmp_path / "out"), str(root)])
        assert rc == 0

    def test_cli_exit_nonzero_when_missing(self, tmp_path, capsys):
        coco = make_coco(tmp_path / "c.json", ["GONE__fig_p2_002.png"])
        root = tmp_path / "root"
        root.mkdir()
        rc = main([str(coco), str(tmp_path / "out"), str(root)])
        assert rc != 0
        assert "GONE__fig_p2_002.png" in capsys.readouterr().out

    def test_cli_exit_nonzero_on_conflict(self, tmp_path):
        # A conflicted file is not collected, so the "all 164 before
        # training" guarantee is not met — must fail.
        coco = make_coco(tmp_path / "c.json", ["DEV1__fig_p1_001.png"])
        root = tmp_path / "root"
        put(root, "a/DEV1/figures/fig_p1_001.png", b"AAA")
        put(root, "b/DEV1/figures/fig_p1_001.png", b"BBB")
        rc = main([str(coco), str(tmp_path / "out"), str(root)])
        assert rc != 0
