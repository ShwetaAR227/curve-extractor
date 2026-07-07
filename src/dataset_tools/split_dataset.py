"""Group-aware train/val/test split of the curve dataset (task T5).

The 164 annotated figures contain near-duplicate template families (same
manufacturer plot style across device series, e.g. Infineon BSZ*/BSC*,
IR AUIRF*). A naive random split leaks near-twins across train/test and
inflates metrics — suspected contributor to the misleading legacy 73% mAP@50.
So the split is BY FAMILY: every image of a family lands on the same side.

Family heuristic: device-name letters up to the first digit, uppercased
(BSZ, BSC, AUIRF, IAUC, IPB, ...). Devices with no letter prefix become their
own single-device family (safe default).

Assignment: greedy balanced — families sorted by image count (descending,
seed breaks ties), each whole family goes to the split currently furthest
below its target image count. Hard invariants: no family straddles splits,
every image lands in exactly one split, all splits non-empty.

CLI:
    python -m src.dataset_tools.split_dataset <coco.json> <out_dir>
        [--seed 42] [--dry-run]
``--dry-run`` prints the proposal without writing anything.
"""
import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.common.log import get_logger
from src.cvat_to_coco import validate_coco
from src.dataset_tools.collect_images import _sha256

logger = get_logger(__name__)

RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)
SIDES = ("train", "val", "test")
MANIFEST_NAME = "split_manifest.json"
_LETTER_PREFIX = re.compile(r"^([A-Za-z]+)")


def extract_device(file_name: str) -> str:
    """Device name: the part before ``__`` (else the file stem)."""
    device, sep, _ = file_name.partition("__")
    return device if sep else Path(file_name).stem


def assign_family(device: str) -> str:
    """Family key for a device: leading letters uppercased, or — for names
    with no letter prefix — the device itself (its own group)."""
    match = _LETTER_PREFIX.match(device)
    return match.group(1).upper() if match else device


def group_split(
    families: Dict[str, List[int]],
    ratios: Tuple[float, float, float] = RATIOS,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Assign whole families to train/val/test, balancing image counts.

    Greedy: families in descending size order (ties broken by ``seed``) each
    go to the side furthest below its target image count. If greedy leaves a
    side empty, the smallest family from the side richest in families is
    moved there. Raises ``ValueError`` for fewer than 3 families.
    """
    if len(families) < len(SIDES):
        raise ValueError(
            f"Need at least {len(SIDES)} families to fill every split, "
            f"got {len(families)}"
        )
    total = sum(len(ids) for ids in families.values())
    targets = {side: ratio * total for side, ratio in zip(SIDES, ratios)}

    rng = random.Random(seed)
    tiebreak = {fam: rng.random() for fam in sorted(families)}
    order = sorted(families, key=lambda f: (-len(families[f]), tiebreak[f]))

    by_side: Dict[str, List[str]] = {side: [] for side in SIDES}
    counts = {side: 0 for side in SIDES}
    for fam in order:
        side = max(SIDES, key=lambda s: targets[s] - counts[s])
        by_side[side].append(fam)
        counts[side] += len(families[fam])

    for side in SIDES:  # enforce the non-empty invariant
        if not by_side[side]:
            donor = max(SIDES, key=lambda s: (len(by_side[s]), counts[s]))
            fam = min(by_side[donor], key=lambda f: (len(families[f]), f))
            by_side[donor].remove(fam)
            by_side[side].append(fam)
            counts[donor] -= len(families[fam])
            counts[side] += len(families[fam])
            logger.warning("Moved family %s to empty split %r", fam, side)

    return {side: [i for fam in by_side[side] for i in families[fam]]
            for side in SIDES}


def propose_split(
    coco_path: Union[str, Path],
    ratios: Tuple[float, float, float] = RATIOS,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build the full split proposal for a COCO file (no files written).

    Returns inventory (family -> devices/images/example), the per-side image
    id assignment, family->side mapping, counts, achieved ratios, seed, and
    the source file's sha256.
    """
    coco_path = Path(coco_path)
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    families: Dict[str, List[int]] = {}
    inventory: Dict[str, Dict[str, Any]] = {}
    for img in coco["images"]:
        device = extract_device(img["file_name"])
        fam = assign_family(device)
        families.setdefault(fam, []).append(img["id"])
        entry = inventory.setdefault(
            fam, {"devices": set(), "images": 0, "example": img["file_name"]})
        entry["devices"].add(device)
        entry["images"] += 1
    for entry in inventory.values():
        entry["devices"] = len(entry["devices"])

    assignment = group_split(families, ratios=ratios, seed=seed)
    families_by_split = {
        fam: side for side in SIDES for fam in
        {assign_family(extract_device(img["file_name"]))
         for img in coco["images"] if img["id"] in set(assignment[side])}
    }
    anns_per_image: Dict[int, int] = {}
    for ann in coco["annotations"]:
        anns_per_image[ann["image_id"]] = anns_per_image.get(ann["image_id"], 0) + 1
    total = len(coco["images"])
    counts = {}
    for side in SIDES:
        ids = assignment[side]
        counts[side] = {
            "images": len(ids),
            "annotations": sum(anns_per_image.get(i, 0) for i in ids),
        }
    proposal = {
        "seed": seed,
        "ratios": list(ratios),
        "inventory": inventory,
        "assignment": assignment,
        "families": families_by_split,
        "counts": counts,
        "achieved_ratios": {s: counts[s]["images"] / total for s in SIDES},
        "source_coco": str(coco_path),
        "source_coco_sha256": _sha256(coco_path),
    }
    logger.info("Split proposal for %s (seed=%d): %s", coco_path, seed,
                {s: counts[s] for s in SIDES})
    return proposal


def write_split(
    coco_path: Union[str, Path],
    proposal: Dict[str, Any],
    out_dir: Union[str, Path],
) -> None:
    """Write train/val/test.json (filtered COCO, source ids preserved) plus
    ``split_manifest.json``. Each part is validated before writing."""
    coco = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for side in SIDES:
        ids = set(proposal["assignment"][side])
        part = {
            "images": [i for i in coco["images"] if i["id"] in ids],
            "annotations": [a for a in coco["annotations"]
                            if a["image_id"] in ids],
            "categories": coco["categories"],
        }
        errors = validate_coco(part)
        if errors:
            raise ValueError(f"{side} split invalid:\n" + "\n".join(errors))
        (out_dir / f"{side}.json").write_text(json.dumps(part, indent=2),
                                              encoding="utf-8")
        logger.info("Wrote %s: %d images, %d annotations", side,
                    len(part["images"]), len(part["annotations"]))

    manifest = {k: proposal[k] for k in
                ("seed", "ratios", "families", "counts", "achieved_ratios",
                 "source_coco", "source_coco_sha256")}
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2),
                                         encoding="utf-8")
    logger.info("Wrote %s", out_dir / MANIFEST_NAME)


def format_proposal(proposal: Dict[str, Any]) -> str:
    """Human-readable proposal report (used by --dry-run)."""
    lines = [f"Split proposal — seed {proposal['seed']}, "
             f"ratios {proposal['ratios']}, "
             f"source {proposal['source_coco']} "
             f"(sha256 {proposal['source_coco_sha256'][:12]}…)", ""]
    inv = proposal["inventory"]
    lines.append(f"{'family':<22}{'side':<7}{'devices':>8}{'images':>8}  example")
    for fam in sorted(inv, key=lambda f: -inv[f]["images"]):
        e = inv[fam]
        lines.append(f"{fam:<22}{proposal['families'][fam]:<7}"
                     f"{e['devices']:>8}{e['images']:>8}  {e['example']}")
    lines.append("")
    for side in SIDES:
        c = proposal["counts"][side]
        lines.append(f"{side:<6}: {c['images']:>4} images, "
                     f"{c['annotations']:>4} annotations "
                     f"({proposal['achieved_ratios'][side]:.1%})")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(
        description="Group-aware (family-level) train/val/test split of a "
                    "COCO dataset.")
    parser.add_argument("coco_json", help="Source COCO file")
    parser.add_argument("out_dir", help="Directory for split files + manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the proposal; write nothing")
    args = parser.parse_args(argv)
    try:
        proposal = propose_split(args.coco_json, seed=args.seed)
        print(format_proposal(proposal))
        if not args.dry_run:
            write_split(args.coco_json, proposal, args.out_dir)
            print(f"\nSplit written to: {args.out_dir}")
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("Split failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
