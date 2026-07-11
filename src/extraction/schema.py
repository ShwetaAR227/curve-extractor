"""Stage-5 curve-extraction output schema (CLAUDE.md §1).

Fresh design — deliberately NOT the legacy flat/keyed ``cv_curves.json``
fork (CLAUDE.md §6). ``curve_type`` is always present and non-empty
(``validate_result`` hard-rejects an empty one before anything reaches
disk), which directly prevents the legacy "" curve_type key-collision bug.

**One file per device per curve type** (:func:`result_path`), not a single
multi-curve-type keyed file — chosen specifically so no merge logic is ever
needed, which makes the legacy schema-fork/silent-overwrite bug class
structurally impossible here rather than merely guarded against.
:func:`write_result` still hard-fails if a write would land on a path
already holding a *different* curve_type's result (a path-construction bug
guard, defense in depth), and every write is validated before ever
touching disk, then written via tmp-file + ``os.replace`` so a crash mid-write
never leaves a partial/corrupt file.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

VALID_STATUSES = {"ok", "needs_review"}


def validate_result(result: Dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``result`` does not conform to the output schema.

    Checks: required keys present; ``curve_type`` non-empty; ``status`` is
    ``"ok"``/``"needs_review"`` with a consistent ``review_reason``
    (``None`` iff ``ok``); ``duplicates_removed`` is a non-negative int;
    each curve has a non-empty, unique ``curve_name``, a confidence in
    ``[0, 1]``, and only finite point coordinates.
    """
    required = {
        "device", "curve_type", "source_image", "status", "review_reason",
        "duplicates_removed", "calibration", "curves", "units",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"result missing required key(s): {sorted(missing)}")

    if not result["curve_type"]:
        raise ValueError("curve_type must be present and non-empty")

    units = result["units"]
    if units is not None and not isinstance(units, str):
        raise ValueError(f"units must be a string or None, got {units!r}")
    if result.get("review_reason") == "units_undetected" and units is not None:
        raise ValueError('units must be None when review_reason == "units_undetected" (never guessed)')

    if result["status"] not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}, got {result['status']!r}")

    if result["status"] == "ok" and result["review_reason"] is not None:
        raise ValueError('review_reason must be None when status == "ok"')
    if result["status"] == "needs_review" and not result["review_reason"]:
        raise ValueError('review_reason must be a non-empty string when status == "needs_review"')

    if not isinstance(result["duplicates_removed"], int) or result["duplicates_removed"] < 0:
        raise ValueError("duplicates_removed must be a non-negative int")

    if result["calibration"] is not None:
        _validate_calibration(result["calibration"])

    curves = result["curves"]
    names_seen = set()
    for curve in curves:
        name = curve.get("curve_name")
        if not name:
            raise ValueError("every curve needs a non-empty curve_name")
        if name in names_seen:
            raise ValueError(f"duplicate curve_name '{name}' in result")
        names_seen.add(name)

        confidence = curve.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise ValueError(f"curve '{name}' confidence must be in [0, 1], got {confidence!r}")

        for point in curve.get("points", []):
            x, y = point.get("x"), point.get("y")
            for label, value in (("x", x), ("y", y)):
                if not isinstance(value, (int, float)) or not _is_finite(value):
                    raise ValueError(f"curve '{name}' has a non-finite {label} value: {value!r}")


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))  # NaN != NaN


_CALIBRATION_NUMERIC_KEYS = ("x_slope", "x_intercept", "y_slope", "y_intercept")
_CALIBRATION_BOOL_KEYS = ("x_log", "y_log")


def _validate_calibration(calibration: Dict[str, Any]) -> None:
    required = set(_CALIBRATION_NUMERIC_KEYS) | set(_CALIBRATION_BOOL_KEYS)
    missing = required - set(calibration)
    if missing:
        raise ValueError(f"calibration missing required key(s): {sorted(missing)}")
    for key in _CALIBRATION_NUMERIC_KEYS:
        value = calibration[key]
        if not isinstance(value, (int, float)) or not _is_finite(value):
            raise ValueError(f"calibration['{key}'] must be a finite number, got {value!r}")
    for key in _CALIBRATION_BOOL_KEYS:
        if not isinstance(calibration[key], bool):
            raise ValueError(f"calibration['{key}'] must be a bool, got {calibration[key]!r}")


def build_result(
    device: str,
    curve_type: str,
    source_image: str,
    status: str,
    review_reason: Optional[str],
    duplicates_removed: int,
    calibration: Optional[Dict[str, Any]],
    curves: List[Dict[str, Any]],
    units: Optional[str],
) -> Dict[str, Any]:
    """Assemble and validate a Stage-5 result dict (fails fast, before any write).

    Args:
        units: Detected y-axis unit (e.g. ``"pF"``/``"nF"``/``"uF"``), or
            ``None`` if undetected/ambiguous — never guessed.
    """
    result = {
        "device": device,
        "curve_type": curve_type,
        "source_image": source_image,
        "status": status,
        "review_reason": review_reason,
        "duplicates_removed": duplicates_removed,
        "calibration": calibration,
        "curves": curves,
        "units": units,
    }
    validate_result(result)
    return result


def result_path(output_dir: Path, device: str, curve_type: str) -> Path:
    """Return the one file per device per curve_type this result belongs at.

    Raises:
        ValueError: If ``curve_type`` is empty — refuses to even construct a
            path for the legacy "" curve_type bug case.
    """
    if not curve_type:
        raise ValueError("curve_type must be non-empty to build a result path")
    return Path(output_dir) / device / f"{curve_type}.json"


def write_result(result: Dict[str, Any], path: Path) -> None:
    """Validate, then atomically write ``result`` to ``path``.

    Refuses (raises ``ValueError``, no write) if ``path`` already holds a
    result for a *different* curve_type — a defense-in-depth guard against
    a path-construction bug accidentally aiming two curve types at the same
    file; readable but unparseable existing files are treated as safe to
    overwrite (nothing valid to lose).
    """
    validate_result(result)

    path = Path(path)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_curve_type = existing.get("curve_type")
        except (json.JSONDecodeError, OSError):
            existing_curve_type = None
        if existing_curve_type and existing_curve_type != result["curve_type"]:
            raise ValueError(
                f"Refusing to overwrite {path}: existing curve_type "
                f"'{existing_curve_type}' != new curve_type '{result['curve_type']}'"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, path)
