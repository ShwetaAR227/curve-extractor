"""CVAT "for images 1.1" XML → COCO instance-segmentation converter (task T2).

Why this exists: CVAT's own COCO exporter silently drops polylines, so curve
annotations must be exported as "CVAT for images 1.1" XML and converted here.
Thin curves are annotated as polylines and buffered (via shapely) into polygon
masks; thick bands are annotated as polygons and pass through unchanged.

Every annotation carries its ``curve_name`` attribute; an empty or missing
``curve_name`` is a hard error (mirrors the stage-5 rule that empty curve keys
are a write-time error).

CLI:
    python -m src.dataset_tools.cvat_to_coco <cvat_export.xml> <output_coco.json> [--buffer-px N]
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from shapely.geometry import LineString

from src.common.log import get_logger

logger = get_logger(__name__)

# Buffer radius in pixels applied to each polyline (mask thickness ≈ 2×this).
# Owner decision after the T4 three-radius visual review (2026-07-07): 4.5 px.
# Annotation polylines jitter 1–2 px off the stroke center, and an
# under-covered stroke hurts training more than a slightly wide mask.
# Override via --buffer-px; changing the default requires owner approval.
DEFAULT_BUFFER_PX = 4.5
CATEGORY_NAME = "line"
CATEGORY_ID = 1
# COCO polygons need at least 3 vertices = 6 coordinates.
MIN_SEGMENTATION_COORDS = 6

Point = Tuple[float, float]
Shape = Dict[str, Any]
CvatImage = Dict[str, Any]

_SUPPORTED_SHAPE_TAGS = ("polyline", "polygon")


def _parse_points(raw: str) -> List[Point]:
    """Parse a CVAT points string ``"x1,y1;x2,y2;..."`` into (x, y) tuples."""
    try:
        return [
            (float(x), float(y))
            for x, y in (pair.split(",") for pair in raw.split(";"))
        ]
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Malformed points string: {raw!r}") from exc


def _extract_curve_name(elem: ET.Element, image_name: str) -> str:
    """Read the non-empty ``curve_name`` attribute of a shape element."""
    for attr in elem.findall("attribute"):
        if attr.get("name") == "curve_name":
            value = (attr.text or "").strip()
            if not value:
                raise ValueError(
                    f"Empty curve_name on {elem.tag} in image {image_name!r}"
                )
            return value
    raise ValueError(f"Missing curve_name attribute on {elem.tag} in image {image_name!r}")


def parse_cvat_xml(xml_path: Union[str, Path]) -> List[CvatImage]:
    """Parse a "CVAT for images 1.1" export into a list of image records.

    Each record: ``{"id", "name", "width", "height", "shapes"}`` where each
    shape is ``{"type": "polyline"|"polygon", "points": [(x, y), ...],
    "curve_name": str}``. Unsupported shape types (e.g. boxes) are skipped
    with a warning. Raises ``FileNotFoundError`` for a missing file,
    ``ValueError`` for malformed XML, bad points, or empty/missing curve_name.
    """
    xml_path = Path(xml_path)
    if not xml_path.is_file():
        raise FileNotFoundError(f"CVAT XML not found: {xml_path}")
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Malformed CVAT XML {xml_path}: {exc}") from exc

    images: List[CvatImage] = []
    skipped = 0
    for image_elem in root.iter("image"):
        name = image_elem.get("name", "")
        try:
            record: CvatImage = {
                "id": int(image_elem.get("id")),
                "name": name,
                "width": int(image_elem.get("width")),
                "height": int(image_elem.get("height")),
                "shapes": [],
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Image element missing id/width/height: {name!r}") from exc

        for child in image_elem:
            if child.tag == "attribute":
                continue
            if child.tag not in _SUPPORTED_SHAPE_TAGS:
                skipped += 1
                logger.warning(
                    "Skipping unsupported shape <%s> in image %s (only %s are converted)",
                    child.tag, name, "/".join(_SUPPORTED_SHAPE_TAGS),
                )
                continue
            record["shapes"].append({
                "type": child.tag,
                "points": _parse_points(child.get("points", "")),
                "curve_name": _extract_curve_name(child, name),
            })
        images.append(record)

    total_shapes = sum(len(img["shapes"]) for img in images)
    logger.info(
        "Parsed %s: %d images, %d shapes kept, %d unsupported shapes skipped",
        xml_path, len(images), total_shapes, skipped,
    )
    return images


def buffer_polyline(points: Sequence[Point], buffer_px: float) -> List[float]:
    """Buffer a polyline into a polygon outline, returned as a flat COCO
    segmentation list ``[x1, y1, x2, y2, ...]``.

    ``buffer_px`` is the buffer radius (mask thickness ≈ ``2 * buffer_px``).
    Raises ``ValueError`` for fewer than 2 points or a non-positive radius.
    """
    if len(points) < 2:
        raise ValueError(f"Polyline needs at least 2 points, got {len(points)}")
    if buffer_px <= 0:
        raise ValueError(f"buffer_px must be positive, got {buffer_px}")
    polygon = LineString(points).buffer(buffer_px)
    if polygon.is_empty:
        raise ValueError(f"Buffering produced an empty polygon for points {points!r}")
    # Exterior ring repeats the first vertex at the end; drop the duplicate.
    coords = list(polygon.exterior.coords)[:-1]
    return [float(v) for xy in coords for v in xy]


def polygon_area(segmentation: Sequence[float]) -> float:
    """Shoelace area of a flat ``[x1, y1, x2, y2, ...]`` polygon (always ≥ 0)."""
    xs = segmentation[0::2]
    ys = segmentation[1::2]
    n = len(xs)
    twice_area = sum(
        xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n)
    )
    return abs(twice_area) / 2.0


def bbox_from_segmentation(segmentation: Sequence[float]) -> List[float]:
    """COCO ``[x, y, width, height]`` bounding box of a flat polygon."""
    xs = segmentation[0::2]
    ys = segmentation[1::2]
    x_min, y_min = min(xs), min(ys)
    return [x_min, y_min, max(xs) - x_min, max(ys) - y_min]


def _build_coco(images: List[CvatImage], buffer_px: float) -> Dict[str, Any]:
    """Build a COCO dict from parsed CVAT image records (ids used as given)."""
    coco: Dict[str, Any] = {
        "images": [
            {"id": img["id"], "file_name": img["name"],
             "width": img["width"], "height": img["height"]}
            for img in images
        ],
        "annotations": [],
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME}],
    }

    ann_id = 1
    buffered = passed_through = 0
    for img in images:
        for shape in img["shapes"]:
            if shape["type"] == "polyline":
                seg = buffer_polyline(shape["points"], buffer_px)
                buffered += 1
            else:
                seg = [float(v) for xy in shape["points"] for v in xy]
                passed_through += 1
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img["id"],
                "category_id": CATEGORY_ID,
                "segmentation": [seg],
                "area": polygon_area(seg),
                "bbox": bbox_from_segmentation(seg),
                "iscrowd": 0,
                "attributes": {"curve_name": shape["curve_name"]},
            })
            ann_id += 1

    logger.info(
        "Built COCO: %d images, %d annotations (%d polylines buffered @ %.1fpx, "
        "%d polygons passed through)",
        len(coco["images"]), len(coco["annotations"]),
        buffered, buffer_px, passed_through,
    )
    return coco


def _validate_and_write(
    coco: Dict[str, Any], output_path: Optional[Union[str, Path]]
) -> Dict[str, Any]:
    """Validate a COCO dict, then optionally write it to JSON.

    Validation failures raise ``ValueError`` before anything touches disk
    (write-time error, never a silent bad file).
    """
    errors = validate_coco(coco)
    if errors:
        raise ValueError(
            "Conversion produced invalid COCO:\n" + "\n".join(errors)
        )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        logger.info("Wrote COCO file: %s", output_path)
    return coco


def convert(
    xml_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    buffer_px: float = DEFAULT_BUFFER_PX,
) -> Dict[str, Any]:
    """Convert one CVAT XML export to a COCO dict; optionally write it to JSON.

    Polylines are buffered by ``buffer_px``; polygons pass through unchanged.
    CVAT image ids are kept as-is, including unannotated images. For combining
    multiple exports use :func:`merge_convert`.
    """
    coco = _build_coco(parse_cvat_xml(xml_path), buffer_px)
    return _validate_and_write(coco, output_path)


def merge_convert(
    xml_paths: Sequence[Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None,
    buffer_px: float = DEFAULT_BUFFER_PX,
) -> Dict[str, Any]:
    """Merge multiple CVAT XML exports into one COCO dict (task T3).

    CVAT numbers image ids from 0 in every export, so images are renumbered
    sequentially (1..N) and annotation ids run 1..M across the whole merge.
    Unannotated images are dropped (CVAT exports every task frame regardless
    of annotation state). A duplicate ``file_name`` among the kept images —
    within one file or across files — raises ``ValueError``: this is the
    permanent guard against double-counting annotations from overlapping
    exports (e.g. a pilot job re-exported inside a later task).
    """
    if not xml_paths:
        raise ValueError("merge_convert needs at least one input XML")

    kept: List[CvatImage] = []
    dropped = 0
    for xml_path in xml_paths:
        images = parse_cvat_xml(xml_path)
        annotated = [img for img in images if img["shapes"]]
        dropped += len(images) - len(annotated)
        kept.extend(annotated)

    seen: Dict[str, int] = {}
    duplicates = []
    for img in kept:
        if img["name"] in seen:
            duplicates.append(img["name"])
        seen[img["name"]] = 1
    if duplicates:
        raise ValueError(
            "Duplicate file_name(s) across merge inputs — refusing to "
            f"double-count annotations: {sorted(set(duplicates))}"
        )

    for new_id, img in enumerate(kept, start=1):
        img["id"] = new_id
    logger.info(
        "Merging %d exports: %d annotated images kept, %d unannotated dropped",
        len(xml_paths), len(kept), dropped,
    )
    coco = _build_coco(kept, buffer_px)
    return _validate_and_write(coco, output_path)


def validate_coco(coco: Dict[str, Any]) -> List[str]:
    """Validate a COCO dict; return a list of error strings (empty = valid).

    Checks: required top-level keys, unique image/annotation ids, annotation
    references, segmentation shape, positive area, bbox within image bounds,
    and non-empty ``curve_name`` attributes.
    """
    errors: List[str] = []
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            errors.append(f"Missing top-level key: {key!r}")
    if errors:
        return errors

    image_ids = [img["id"] for img in coco["images"]]
    if len(set(image_ids)) != len(image_ids):
        errors.append("Duplicate image ids")
    images_by_id = {img["id"]: img for img in coco["images"]}
    category_ids = {cat["id"] for cat in coco["categories"]}

    seen_ann_ids = set()
    for ann in coco["annotations"]:
        label = f"annotation id={ann.get('id')}"
        if ann.get("id") in seen_ann_ids:
            errors.append(f"Duplicate annotation id: {ann.get('id')}")
        seen_ann_ids.add(ann.get("id"))

        if ann.get("image_id") not in images_by_id:
            errors.append(f"{label}: unknown image_id {ann.get('image_id')}")
            continue
        if ann.get("category_id") not in category_ids:
            errors.append(f"{label}: unknown category_id {ann.get('category_id')}")

        seg = ann.get("segmentation") or [[]]
        for ring in seg:
            if len(ring) < MIN_SEGMENTATION_COORDS or len(ring) % 2 != 0:
                errors.append(
                    f"{label}: segmentation ring must have an even count of "
                    f"≥{MIN_SEGMENTATION_COORDS} coords, got {len(ring)}"
                )
        if not ann.get("area") or ann["area"] <= 0:
            errors.append(f"{label}: non-positive area")

        img = images_by_id[ann["image_id"]]
        bbox = ann.get("bbox") or []
        if len(bbox) != 4:
            errors.append(f"{label}: bbox must be [x, y, w, h]")
        else:
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                errors.append(f"{label}: non-positive bbox size")
            # Buffered outlines may poke slightly past the frame; only a bbox
            # entirely outside the image is an error.
            if x >= img["width"] or y >= img["height"] or x + w <= 0 or y + h <= 0:
                errors.append(f"{label}: bbox {bbox} lies outside image {img['id']}")

        if not (ann.get("attributes") or {}).get("curve_name", "").strip():
            errors.append(f"{label}: empty or missing curve_name attribute")

    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(
        description="Convert CVAT-for-images-1.1 XML export(s) to COCO JSON, "
                    "buffering polylines into polygon masks. Multiple inputs "
                    "are merged (ids renumbered, unannotated images dropped, "
                    "duplicate file_names are a hard error).",
    )
    parser.add_argument("xml_paths", nargs="+", help="CVAT XML export file(s)")
    parser.add_argument("output_path", help="COCO JSON file to write")
    parser.add_argument(
        "--buffer-px", type=float, default=DEFAULT_BUFFER_PX,
        help=f"Polyline buffer radius in pixels (default {DEFAULT_BUFFER_PX})",
    )
    args = parser.parse_args(argv)
    try:
        if len(args.xml_paths) == 1:
            convert(args.xml_paths[0], args.output_path, buffer_px=args.buffer_px)
        else:
            merge_convert(args.xml_paths, args.output_path, buffer_px=args.buffer_px)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Conversion failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
