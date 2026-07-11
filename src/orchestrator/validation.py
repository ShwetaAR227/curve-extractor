"""Stage-7 final validation (CLAUDE.md §1, T20).

The last, cheap safety gate before a record is marked "finalized". It
REUSES Stage 5's schema validator for everything that validator already
covers (required keys, non-empty curve_type, finite points, calibration
shape) and adds only the finalization-specific requirements schema
validation deliberately leaves optional at the Stage-5 level:

- every expected curve name for the curve type is present (per the naming
  registry — same source module as the naming function, so no drift), and
  no unexpected names;
- ``units`` is present (Stage 5 allows None; a finalized record must not
  ship without units — the BTS247Z nF-vs-pF lesson from T17);
- ``calibration`` is present.

Never raises: returns a reason string (for the needs_review downgrade) or
``None`` when the record is fit to finalize.
"""
from typing import Any, Dict, Optional

from src.common.log import get_logger
from src.extraction.naming import get_expected_names
from src.extraction.schema import validate_result

logger = get_logger(__name__)


def validate_final(result: Dict[str, Any]) -> Optional[str]:
    """Validate a Stage-5 result for finalization.

    Args:
        result: A Stage-5 result dict (see :mod:`src.extraction.schema`).

    Returns:
        ``None`` if the record may be finalized, otherwise a human-readable
        reason string. Never raises — any internal error becomes a reason.
    """
    try:
        validate_result(result)  # reuse Stage 5's validator, don't reimplement
    except ValueError as exc:
        return f"schema validation failed: {exc}"

    curve_type = result["curve_type"]
    try:
        expected = set(get_expected_names(curve_type))
    except KeyError as exc:
        return f"no expected-name registry entry: {exc}"

    actual = {curve.get("curve_name") for curve in result["curves"]}
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        return f"missing expected curve name(s): {sorted(missing)}"
    if unexpected:
        return f"unexpected curve name(s): {sorted(unexpected)}"

    if result.get("units") is None:
        return "units missing (undetected at Stage 5) — cannot finalize without units"
    if result.get("calibration") is None:
        return "calibration missing — cannot finalize without calibration"

    return None
