"""Recover annotated source images from legacy data trees (task T4a).

Reads every ``file_name`` from a COCO json and recursively searches one or
more roots for the matching source PNGs, copying matches flat into an output
directory. The legacy tree is read-only reference material (CLAUDE.md §6) —
sources are never modified.

Matching, for a COCO name like ``DEVICE__fig_p8_021.png``:
  1. exact basename match anywhere under a root, or
  2. the legacy nested layout ``.../DEVICE/figures/fig_p8_021.png``
     (bare fig basename directly inside ``<DEVICE>/figures/``).
Both are exact-name matches — variants such as ``fig_p8_021_cv_overlay.png``
or ``validated_fig_p8_021.png`` never match.

When the same target is found in several places the contents are hashed:
identical copies are fine (one is copied); differing content is a conflict —
nothing is copied for that name and the owner gets the list.

The CLI exits non-zero unless EVERY file was found and copied (training needs
the complete set).

CLI:
    python -m src.dataset_tools.collect_images <coco.json> <output_dir> <root> [<root> ...]
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.common.log import get_logger

logger = get_logger(__name__)

FIGURES_DIR_NAME = "figures"
HASH_CHUNK_BYTES = 1 << 20


def _sha256(path: Path) -> str:
    """Content hash used to distinguish identical duplicates from conflicts."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_matches(
    file_names: Sequence[str], roots: Sequence[Path]
) -> Dict[str, List[Path]]:
    """Map each COCO file_name to every path under ``roots`` that matches it.

    Walks each root once. A file matches by exact basename, or — for names of
    the form ``DEVICE__fig.png`` — by bare ``fig.png`` basename located
    directly in ``<DEVICE>/figures/``.
    """
    matches: Dict[str, List[Path]] = {name: [] for name in file_names}
    # bare fig basename -> [(device, coco file_name), ...]
    nested: Dict[str, List[Tuple[str, str]]] = {}
    for name in file_names:
        device, sep, fig = name.partition("__")
        if sep:
            nested.setdefault(fig, []).append((device, name))

    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            for base in filenames:
                if base in matches:
                    matches[base].append(Path(dirpath) / base)
                elif base in nested:
                    parent = Path(dirpath)
                    if parent.name == FIGURES_DIR_NAME:
                        for device, name in nested[base]:
                            if parent.parent.name == device:
                                matches[name].append(parent / base)
    return matches


def collect_images(
    coco_path: Union[str, Path],
    roots: Sequence[Union[str, Path]],
    output_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Search ``roots`` for every COCO file_name and copy matches flat into
    ``output_dir``. Returns ``{"copied", "missing", "conflicts"}``.

    Sources are only ever read. Duplicate finds with identical content are
    copied once; differing content is reported in ``conflicts`` (mapping
    file_name -> list of source paths) and not copied at all.
    """
    coco_path = Path(coco_path)
    output_dir = Path(output_dir)
    root_paths = [Path(r) for r in roots]
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    file_names = [img["file_name"] for img in coco["images"]]
    logger.info("Collecting %d images from %s across %d root(s): %s",
                len(file_names), coco_path, len(root_paths),
                [str(r) for r in root_paths])

    matches = find_matches(file_names, root_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    missing: List[str] = []
    conflicts: Dict[str, List[str]] = {}
    for name in file_names:
        found = matches[name]
        if not found:
            missing.append(name)
            continue
        hashes = {_sha256(p) for p in found}
        if len(hashes) > 1:
            conflicts[name] = [str(p) for p in found]
            logger.warning("Conflict for %s: %d sources with differing "
                           "content, copying none: %s", name, len(found), found)
            continue
        shutil.copyfile(found[0], output_dir / name)
        copied.append(name)

    logger.info("Collected %d/%d images -> %s | missing: %d | conflicts: %d",
                len(copied), len(file_names), output_dir,
                len(missing), len(conflicts))
    for name in missing:
        logger.warning("No match anywhere for: %s", name)
    return {"copied": copied, "missing": missing, "conflicts": conflicts}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Exit 0 only if every COCO image was found and copied."""
    parser = argparse.ArgumentParser(
        description="Recursively collect the source images referenced by a "
                    "COCO json into one flat directory (read-only search).")
    parser.add_argument("coco_json", help="COCO file with the image list")
    parser.add_argument("output_dir", help="Flat destination directory")
    parser.add_argument("roots", nargs="+", help="Search root(s)")
    args = parser.parse_args(argv)
    try:
        report = collect_images(args.coco_json, args.roots, args.output_dir)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("Collection failed: %s", exc)
        return 1

    total = len(report["copied"]) + len(report["missing"]) + len(report["conflicts"])
    print(f"Copied {len(report['copied'])}/{total} images to {args.output_dir}")
    if report["conflicts"]:
        print(f"CONFLICTS ({len(report['conflicts'])}) — same name, differing "
              "content; copied none, owner must resolve:")
        for name, paths in report["conflicts"].items():
            print(f"  - {name}")
            for p in paths:
                print(f"      {p}")
    if report["missing"]:
        print(f"MISSING ({len(report['missing'])}) — no match under any root:")
        for name in report["missing"]:
            print(f"  - {name}")
    return 0 if not report["missing"] and not report["conflicts"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
