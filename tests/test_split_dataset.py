"""Tests for src.dataset_tools.split_dataset — written FIRST (CLAUDE.md §2).

Covers device extraction, family assignment, group-aware split invariants,
determinism, ratio tolerance, written-COCO validity, and manifest content.
"""
import json
from pathlib import Path

import pytest

from src.cvat_to_coco import validate_coco
from src.dataset_tools.split_dataset import (
    RATIOS,
    assign_family,
    extract_device,
    group_split,
    main,
    propose_split,
    write_split,
)


def make_coco(path: Path, file_names: list) -> Path:
    """Minimal valid COCO: one 3-vertex annotation per image."""
    images, annotations = [], []
    for i, name in enumerate(file_names, start=1):
        images.append({"id": i, "file_name": name, "width": 100, "height": 100})
        annotations.append({
            "id": i, "image_id": i, "category_id": 1,
            "segmentation": [[10.0, 10.0, 60.0, 10.0, 60.0, 40.0]],
            "area": 750.0, "bbox": [10.0, 10.0, 50.0, 30.0], "iscrowd": 0,
            "attributes": {"curve_name": f"c{i}"},
        })
    path.write_text(json.dumps({"images": images, "annotations": annotations,
                                "categories": [{"id": 1, "name": "line"}]}))
    return path


def family_names(prefix: str, n_devices: int, figs_per_device: int = 1) -> list:
    return [f"{prefix}{100 + d}XYZ__fig_p{f}_00{f}.png"
            for d in range(n_devices) for f in range(figs_per_device)]


# ---------------------------------------------------------------- extraction

class TestExtractDevice:
    def test_normal_name(self):
        assert extract_device("BSZ100N03MSGATMA1__fig_p8_020.png") == "BSZ100N03MSGATMA1"

    def test_digit_leading_device(self):
        assert extract_device("94-3316__fig_p4_007.png") == "94-3316"

    def test_no_separator_falls_back_to_stem(self):
        assert extract_device("oddname.png") == "oddname"


class TestAssignFamily:
    def test_prefix_letters_up_to_first_digit(self):
        assert assign_family("BSZ100N03MSGATMA1") == "BSZ"
        assert assign_family("BSC032N04LSATMA1") == "BSC"
        assert assign_family("AUIRF1010EZS") == "AUIRF"
        assert assign_family("AUIRL3705N") == "AUIRL"
        assert assign_family("IAUC120N04S6L005ATMA1") == "IAUC"
        assert assign_family("IPB012N04NF2SATMA1") == "IPB"

    def test_lowercase_uppercased(self):
        assert assign_family("bsz100n03") == "BSZ"

    def test_no_letter_prefix_is_own_family(self):
        assert assign_family("94-3316") == "94-3316"


# ---------------------------------------------------------------- group_split

class TestGroupSplit:
    def _families(self):
        # 100 images across families of varied size.
        fams = {}
        next_id = 1
        for name, size in [("A", 30), ("B", 20), ("C", 15), ("D", 10),
                           ("E", 10), ("F", 5), ("G", 5), ("H", 3),
                           ("I", 1), ("J", 1)]:
            fams[name] = list(range(next_id, next_id + size))
            next_id += size
        return fams

    def test_no_family_straddles_splits(self):
        fams = self._families()
        split = group_split(fams, seed=42)
        for members in fams.values():
            sides = {s for s, ids in split.items()
                     if any(i in ids for i in members)}
            assert len(sides) == 1

    def test_every_image_in_exactly_one_split(self):
        fams = self._families()
        split = group_split(fams, seed=42)
        all_ids = [i for ids in split.values() for i in ids]
        assert sorted(all_ids) == sorted(i for m in fams.values() for i in m)

    def test_all_splits_non_empty(self):
        split = group_split(self._families(), seed=42)
        assert all(split[s] for s in ("train", "val", "test"))

    def test_deterministic_for_same_seed(self):
        fams = self._families()
        a = group_split(fams, seed=7)
        b = group_split(fams, seed=7)
        assert {k: sorted(v) for k, v in a.items()} == \
               {k: sorted(v) for k, v in b.items()}

    def test_ratio_tolerance_when_feasible(self):
        # 20 families x 5 images: fine-grained enough to hit ±5%.
        fams = {f"F{i}": list(range(i * 5, i * 5 + 5)) for i in range(20)}
        split = group_split(fams, seed=42)
        total = 100
        for side, ratio in zip(("train", "val", "test"), RATIOS):
            assert abs(len(split[side]) - ratio * total) <= 5

    def test_fewer_than_three_families_raises(self):
        with pytest.raises(ValueError):
            group_split({"A": [1, 2], "B": [3]}, seed=42)


# ------------------------------------------------------- write_split and CLI

class TestWriteSplitAndCli:
    def _coco(self, tmp_path):
        names = (family_names("BSZ", 8) + family_names("BSC", 6)
                 + family_names("AUIRF", 4) + family_names("IAUC", 3)
                 + family_names("IPB", 2) + ["94-3316__fig_p4_007.png"])
        return make_coco(tmp_path / "src.json", names)

    def test_written_splits_are_valid_filtered_cocos(self, tmp_path):
        coco_path = self._coco(tmp_path)
        proposal = propose_split(coco_path, seed=42)
        out = tmp_path / "split"
        write_split(coco_path, proposal, out)
        source = json.loads(coco_path.read_text())
        seen_ids = []
        for side in ("train", "val", "test"):
            part = json.loads((out / f"{side}.json").read_text())
            assert validate_coco(part) == []
            seen_ids += [img["id"] for img in part["images"]]
            # ids preserved from source, annotations follow their images
            part_img_ids = {img["id"] for img in part["images"]}
            assert all(a["image_id"] in part_img_ids
                       for a in part["annotations"])
        assert sorted(seen_ids) == [img["id"] for img in source["images"]]

    def test_manifest_contents(self, tmp_path):
        coco_path = self._coco(tmp_path)
        proposal = propose_split(coco_path, seed=42)
        out = tmp_path / "split"
        write_split(coco_path, proposal, out)
        manifest = json.loads((out / "split_manifest.json").read_text())
        assert manifest["seed"] == 42
        assert set(manifest["families"].values()) <= {"train", "val", "test"}
        for side in ("train", "val", "test"):
            part = json.loads((out / f"{side}.json").read_text())
            assert manifest["counts"][side]["images"] == len(part["images"])
            assert manifest["counts"][side]["annotations"] == len(part["annotations"])
        assert len(manifest["source_coco_sha256"]) == 64

    def test_dry_run_writes_nothing(self, tmp_path, capsys):
        coco_path = self._coco(tmp_path)
        out = tmp_path / "split"
        rc = main([str(coco_path), str(out), "--seed", "42", "--dry-run"])
        assert rc == 0
        assert not out.exists()
        assert "train" in capsys.readouterr().out

    def test_cli_real_run_writes_files(self, tmp_path):
        coco_path = self._coco(tmp_path)
        out = tmp_path / "split"
        rc = main([str(coco_path), str(out), "--seed", "42"])
        assert rc == 0
        for f in ("train.json", "val.json", "test.json", "split_manifest.json"):
            assert (out / f).exists()
