"""Remove leftover 'TODO' placeholder shapes from a corrected CVAT export.

After semi-automatic pre-annotation (T8b), an annotator may run out of time
before relabeling every model-generated shape, leaving some with the literal
placeholder ``curve_name == "TODO"`` (see `predict_to_cvat.py`). Rather than
hand-delete those in CVAT, this strips them directly from the XML: the whole
``<polyline>``/``<polygon>`` element is removed (not just the attribute),
every other shape — correct curve names, other images — is left byte-for-byte
untouched, and the original export file is never modified.

CLI:
    python -m src.dataset_tools.strip_todo_shapes <input.xml> <output.xml>
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.common.log import get_logger

logger = get_logger(__name__)

TODO_PLACEHOLDER = "TODO"
_SHAPE_TAGS = ("polyline", "polygon")


def strip_todo_xml(
    xml_path: Union[str, Path], out_path: Union[str, Path]
) -> Dict[str, Any]:
    """Remove every shape whose ``curve_name`` attribute is exactly "TODO".

    Writes the cleaned tree to ``out_path`` (the input file is only read,
    never modified) and returns a report:
    ``{"shapes_removed": int, "images_affected": [names, ...]}``.
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    shapes_removed = 0
    images_affected: List[str] = []
    for image_elem in root.iter("image"):
        removed_here = 0
        for shape_elem in list(image_elem):
            if shape_elem.tag not in _SHAPE_TAGS:
                continue
            curve_name = None
            for attr in shape_elem.findall("attribute"):
                if attr.get("name") == "curve_name":
                    curve_name = (attr.text or "").strip()
                    break
            if curve_name == TODO_PLACEHOLDER:
                image_elem.remove(shape_elem)
                removed_here += 1
        if removed_here:
            shapes_removed += removed_here
            images_affected.append(image_elem.get("name"))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

    logger.info(
        "Stripped %d TODO shape(s) from %d image(s) in %s -> %s",
        shapes_removed, len(images_affected), xml_path, out_path,
    )
    return {"shapes_removed": shapes_removed, "images_affected": images_affected}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(
        description="Remove leftover TODO-placeholder shapes from a CVAT export.")
    parser.add_argument("input_xml", help="Corrected CVAT export XML")
    parser.add_argument("output_xml", help="Cleaned XML to write (never the input)")
    args = parser.parse_args(argv)
    try:
        report = strip_todo_xml(args.input_xml, args.output_xml)
    except (FileNotFoundError, ET.ParseError) as exc:
        logger.error("Strip failed: %s", exc)
        return 1
    print(f"Shapes removed  : {report['shapes_removed']}")
    print(f"Images affected : {len(report['images_affected'])}")
    for name in report["images_affected"]:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
