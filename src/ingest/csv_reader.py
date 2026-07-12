"""Stage-1 orchestrator: device-list CSV -> clean canonical device records.

Reads a vendor CSV (DigiKey-style exports, varying header conventions),
resolves headers via :mod:`column_mapping`, parses values via :mod:`parsers`,
normalizes part numbers via :mod:`model_name_utils`, and returns an
:class:`IngestResult`.

DELIBERATE CHANGE from legacy ``csv_reader.py``: rows are never silently
dropped — every skipped row is recorded as a :class:`SkippedRow` with a
reason and summarized in the log.
"""
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .column_mapping import map_columns
from .model_name_utils import normalize_model_name
from .parsers import (
    parse_capacitance,
    parse_charge,
    parse_current,
    parse_energy,
    parse_numeric,
    parse_power,
    parse_price,
    parse_rdson_mohm,
    parse_temperature_range,
    parse_thermal_resistance,
    parse_timing,
    parse_voltage,
)

logger = logging.getLogger(__name__)

#: Canonical field -> parser producing the field's fixed unit (see parsers.py).
FIELD_PARSERS: Dict[str, Callable] = {
    "Vdss": parse_voltage,
    "Vgs_th": parse_voltage,
    "Vgs_max": parse_voltage,
    "Vf": parse_voltage,
    "Vce_sat": parse_voltage,
    "Id": parse_current,
    "Id_pulse": parse_current,
    "Is": parse_current,
    "Rdson": parse_rdson_mohm,
    "Qg": parse_charge,
    "Qgs": parse_charge,
    "Qgd": parse_charge,
    "Qrr": parse_charge,
    "Ciss": parse_capacitance,
    "Coss": parse_capacitance,
    "Crss": parse_capacitance,
    "Pd": parse_power,
    "R_theta_JC": parse_thermal_resistance,
    "R_theta_JA": parse_thermal_resistance,
    "Cost": parse_price,
    "eon_ref": parse_energy,
    "eoff_ref": parse_energy,
    "td_on": parse_timing,
    "td_off": parse_timing,
    "tr": parse_timing,
    "tf": parse_timing,
    "trr": parse_timing,
    "stock_available": parse_numeric,
}

#: Fields kept as raw strings (no numeric parsing); "-" and "" become None.
STRING_FIELDS = frozenset({
    "ModelName", "manufacturer", "package", "mounting", "channel_polarity",
    "product_status", "technology", "pdf_url", "supplier_part_number",
    "description",
})


@dataclass(frozen=True)
class SkippedRow:
    """One skipped CSV row: 1-based row number (header = row 1) + reason."""

    row_num: int
    reason: str


@dataclass
class IngestResult:
    """Outcome of reading one CSV: clean records plus the skip report."""

    records: List[dict] = field(default_factory=list)
    skipped: List[SkippedRow] = field(default_factory=list)


def _clean_string(raw: str) -> Optional[str]:
    s = raw.strip()
    return s if s and s != "-" else None


def read_csv(
    file_path: Union[str, Path],
    device_type: str,
    limit: Optional[int] = None,
) -> IngestResult:
    """Read a device-list CSV into canonical device records.

    Each record is keyed by canonical field names (values in the fixed
    units documented in :mod:`parsers`), plus ``_raw_part_number`` holding
    the part number before DigiKey-suffix stripping. Unparseable optional
    values become None (row kept); a missing/empty ``ModelName`` skips the
    row with a recorded reason. The batch never crashes on a bad row.

    Args:
        file_path: CSV path (UTF-8, BOM tolerated).
        device_type: Registered device type, e.g. ``"Si-MOSFET"``.
        limit: Optional cap on the number of records returned.

    Returns:
        :class:`IngestResult` with ``records`` and the ``skipped`` report.

    Raises:
        KeyError: Unknown ``device_type``.
        ValueError: CSV has no recognizable ModelName / part-number column.
    """
    file_path = Path(file_path)
    logger.info("Reading %s as %s", file_path.name, device_type)

    result = IngestResult()

    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        col_mapping = map_columns(headers, device_type)

        if "ModelName" not in col_mapping:
            raise ValueError(
                f"No ModelName/part-number column found in {file_path.name}; "
                f"headers were: {headers}"
            )

        for row_num, row in enumerate(reader, start=2):
            if limit is not None and len(result.records) >= limit:
                break

            parsed: Dict[str, Optional[object]] = {}
            for field_name, col_idx in col_mapping.items():
                raw_val = (row.get(headers[col_idx]) or "").strip()
                if field_name in STRING_FIELDS:
                    parsed[field_name] = _clean_string(raw_val)
                elif field_name == "Tjmax":
                    parsed["Tjmin"], parsed["Tjmax"] = parse_temperature_range(raw_val)
                elif field_name in FIELD_PARSERS:
                    parsed[field_name] = FIELD_PARSERS[field_name](raw_val)
                else:
                    parsed[field_name] = _clean_string(raw_val)

            model_name = parsed.get("ModelName")
            if not model_name:
                reason = "missing required field 'ModelName'"
                logger.warning("Row %d skipped: %s", row_num, reason)
                result.skipped.append(SkippedRow(row_num=row_num, reason=reason))
                continue

            parsed["_raw_part_number"] = model_name
            parsed["ModelName"] = normalize_model_name(model_name)
            result.records.append(parsed)

    logger.info(
        "%s: %d records kept, %d rows skipped",
        file_path.name, len(result.records), len(result.skipped),
    )
    for skip in result.skipped:
        logger.info("  skipped row %d: %s", skip.row_num, skip.reason)
    return result
