"""Tests for src.dataset_tools.split_dataset — written FIRST (CLAUDE.md §2).

Covers device extraction, family assignment, group-aware split invariants,
determinism, ratio tolerance, written-COCO validity, and manifest content.
"""
import json
from pathlib import Path

import pytest

from src.dataset_tools.cvat_to_coco import validate_coco
from src.dataset_tools.split_dataset import (
    FAMILY_MERGE_MAP,
    PINNED_FAMILIES,
    RATIOS,
    allocate_new_batch,
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
        assert assign_family("AUIRF1010EZS") == "AUIRF"
        assert assign_family("AUIRL3705N") == "AUIRL"
        assert assign_family("IPB012N04NF2SATMA1") == "IPB"
        assert assign_family("BSP125H6327XTSA1") == "BSP"

    def test_lowercase_uppercased(self):
        assert assign_family("ipb012n04") == "IPB"

    def test_no_letter_prefix_is_own_family(self):
        assert assign_family("94-3316") == "94-3316"

    def test_owner_merge_map_is_explicit_dict(self):
        # Owner decision 2026-07-07 (T5): template-true families.
        assert FAMILY_MERGE_MAP == {
            "IAUA": "IAU", "IAUAN": "IAU", "IAUC": "IAU",
            "AUIRFS": "AUIRF", "AUIRFP": "AUIRF", "AUIRFZ": "AUIRF",
            "AUIRLU": "AUIRL",
            "BSC": "BSC-BSZ", "BSZ": "BSC-BSZ",
        }

    def test_merged_families_applied(self):
        assert assign_family("BSZ100N03MSGATMA1") == "BSC-BSZ"
        assert assign_family("BSC032N04LSATMA1") == "BSC-BSZ"
        assert assign_family("IAUC120N04S6L005ATMA1") == "IAU"
        assert assign_family("IAUA120N04S5N014AUMA1") == "IAU"
        assert assign_family("IAUAN04S7N004AUMA1") == "IAU"
        assert assign_family("AUIRFS4115-7P") == "AUIRF"
        assert assign_family("AUIRFP4568-E") == "AUIRF"
        assert assign_family("AUIRFZ34N") == "AUIRF"
        assert assign_family("AUIRLU3114Z") == "AUIRL"

    def test_bsc_bsz_pinned_to_train(self):
        assert PINNED_FAMILIES == {"BSC-BSZ": "train"}


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

    def test_pinned_family_respected(self):
        # Unpinned, D lands in val (greedy); pinning must override that.
        fams = self._families()
        unpinned = group_split(fams, seed=42)
        assert set(fams["D"]) <= set(unpinned["val"])
        for seed in (42, 7):  # pinning is not luck of the seed
            split = group_split(fams, seed=seed, pinned={"D": "test"})
            assert set(fams["D"]) <= set(split["test"])

    def test_eval_split_minimums_enforced(self):
        # Owner invariant: val and test each need >=2 families and >=15
        # images. A dataset that cannot satisfy it must hard-error.
        fams = {"A": list(range(50)), "B": [50, 51], "C": [52, 53]}
        with pytest.raises(ValueError, match="val|test"):
            group_split(fams, seed=42)


# ------------------------------------------------------- write_split and CLI

class TestWriteSplitAndCli:
    def _coco(self, tmp_path):
        # Large enough that the >=2-family / >=15-image eval minimums are
        # satisfiable (BSZ+BSC merge into one pinned train family).
        names = (family_names("BSZ", 15) + family_names("BSC", 15)
                 + family_names("AUIRF", 20) + family_names("IAUC", 18)
                 + family_names("IPB", 16) + family_names("BTS", 12)
                 + family_names("BUZ", 10) + family_names("BSP", 10)
                 + family_names("BSS", 8) + ["94-3316__fig_p4_007.png"])
        return make_coco(tmp_path / "src.json", names)

    def test_pinned_family_lands_in_train(self, tmp_path):
        proposal = propose_split(self._coco(tmp_path), seed=42)
        assert proposal["families"]["BSC-BSZ"] == "train"

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


# --------------------------------------------------------- allocate_new_batch
# T9: promoting the ad-hoc semiauto-batch split (commit 2c33a17) into a
# tested, reusable tool. Routes a NEW batch's families into train/val only —
# never references or writes test data.

class TestAllocateNewBatch:
    def _families(self):
        return {"A": [1, 2, 3, 4, 5], "B": [6, 7], "C": [8], "D": [9, 10, 11]}

    def test_val_family_count_takes_smallest_n(self):
        result = allocate_new_batch(self._families(), val_family_count=2)
        assert set(result["val"]) == {6, 7, 8}  # smallest: C(1), B(2)
        assert set(result["train"]) == {1, 2, 3, 4, 5, 9, 10, 11}

    def test_val_family_count_ties_broken_alphabetically(self):
        families = {"B": [1, 2], "A": [3, 4], "C": [5, 6, 7]}
        result = allocate_new_batch(families, val_family_count=1)
        assert result["val"] == [3, 4]  # 'A' ties with 'B' at size 2, A wins alphabetically

    def test_val_image_target_stops_at_closest_boundary(self):
        families = {"A": [1], "B": [2], "C": [3], "D": [4, 5, 6, 7, 8]}
        result = allocate_new_batch(families, val_image_target=2)
        assert set(result["val"]) == {1, 2}
        assert set(result["train"]) == {3, 4, 5, 6, 7, 8}

    def test_no_family_straddles_train_val(self):
        families = self._families()
        for kwargs in [{"val_family_count": 2}, {"val_image_target": 5}]:
            result = allocate_new_batch(families, **kwargs)
            val_set, train_set = set(result["val"]), set(result["train"])
            for fam_ids in families.values():
                sides = {"val" if i in val_set else "train" for i in fam_ids}
                assert len(sides) == 1, f"family straddled: {fam_ids}"

    def test_all_images_accounted_for_no_duplicates(self):
        families = self._families()
        result = allocate_new_batch(families, val_family_count=1)
        all_ids = [i for ids in families.values() for i in ids]
        combined = result["train"] + result["val"]
        assert sorted(combined) == sorted(all_ids)
        assert not (set(result["train"]) & set(result["val"]))

    def test_deterministic(self):
        families = self._families()
        a = allocate_new_batch(families, val_family_count=2)
        b = allocate_new_batch(families, val_family_count=2)
        assert a == b

    def test_neither_mode_given_raises(self):
        with pytest.raises(ValueError):
            allocate_new_batch(self._families())

    def test_both_modes_given_raises(self):
        with pytest.raises(ValueError):
            allocate_new_batch(self._families(), val_family_count=1, val_image_target=5)

    def test_val_family_count_exceeding_total_is_safe(self):
        families = self._families()
        result = allocate_new_batch(families, val_family_count=10)
        assert result["train"] == []
        assert sorted(result["val"]) == sorted(i for ids in families.values() for i in ids)


class TestAllocateNewBatchCli:
    def _existing_split(self, tmp_path):
        split_dir = tmp_path / "existing_split"
        split_dir.mkdir()
        make_coco(split_dir / "train.json", family_names("BSC", 3))
        make_coco(split_dir / "val.json", family_names("BSP", 2))
        make_coco(split_dir / "test.json", family_names("BSS", 2))
        return split_dir

    def test_test_json_byte_identical_before_and_after(self, tmp_path):
        split_dir = self._existing_split(tmp_path)
        before = (split_dir / "test.json").read_bytes()
        new_batch = make_coco(tmp_path / "new_batch.json",
                              family_names("IPD", 4) + family_names("IPP", 2))
        out = tmp_path / "out"
        rc = main(["allocate-new-batch", str(new_batch), str(split_dir),
                  "--val-families", "1", "--out", str(out)])
        assert rc == 0
        after = (out / "test.json").read_bytes()
        assert after == before
        # Original, untouched location is also unaffected.
        assert (split_dir / "test.json").read_bytes() == before

    def test_merge_appends_existing_and_new_without_duplication(self, tmp_path):
        split_dir = self._existing_split(tmp_path)
        existing_train_names = {
            i["file_name"] for i in
            json.loads((split_dir / "train.json").read_text())["images"]
        }
        new_batch = make_coco(tmp_path / "new_batch.json",
                              family_names("IPD", 4) + family_names("IPP", 2))
        out = tmp_path / "out"
        rc = main(["allocate-new-batch", str(new_batch), str(split_dir),
                  "--val-families", "1", "--out", str(out)])
        assert rc == 0
        train_out = json.loads((out / "train.json").read_text())
        names = [i["file_name"] for i in train_out["images"]]
        assert len(names) == len(set(names))  # no duplicates
        assert existing_train_names <= set(names)  # all originals preserved
        assert len(names) > len(existing_train_names)  # new ones were added
        ids = [i["id"] for i in train_out["images"]]
        assert len(ids) == len(set(ids))  # ids unique after renumbering
        assert validate_coco(train_out) == []
