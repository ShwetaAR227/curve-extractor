"""Stage-1 column registry: vendor CSV header -> canonical field name.

The mapping is DATA, not code (same registry pattern as stage 4's
:mod:`src.classification.curve_registry`): one dict per device type keyed by
canonical field name, value = ordered list of known header aliases
(first match wins). Header aliases were taken verbatim from real DigiKey
exports as documented in the legacy repo's mapping tables — clean-room
reimplementation, behavior verified against those tables.

Adding a column variant is a one-line registry edit, never a new code path.
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_MOSFET_COLUMNS: Dict[str, List[str]] = {
    "ModelName": ["Mfr Part #", "Part Number", "Manufacturer Part Number", "Mfr Part Number"],
    "manufacturer": ["Mfr", "Manufacturer", "Manufacture"],
    "Vdss": ["Drain to Source Voltage (Vdss)", "Voltage - Drain to Source", "Drain Source Voltage", "Vdss"],
    "Id": [
        "Current - Continuous Drain (Id) @ 25°C",
        "Drain Current", "Continuous Drain Current", "Id",
    ],
    "Id_pulse": ["Current - Pulsed Drain (Idm)", "Pulsed Drain Current"],
    "Rdson": [
        "Rds On (Max) @ Id, Vgs", "Rds(on)", "RDS(ON)",
        "Drain-Source On Resistance", "Drain Source On Resistance",
    ],
    "Vgs_th": ["Vgs(th) (Max) @ Id", "Gate Threshold Voltage", "Vgs_th"],
    "Qg": ["Gate Charge (Qg) (Max) @ Vgs", "Total Gate Charge", "Charge - Gate (Qg)", "Qg"],
    "Qgs": ["Gate Charge (Qgs)", "Gate-Source Charge", "Charge - Gate to Source"],
    "Qgd": ["Gate Charge (Qgd)", "Gate-Drain Charge", "Charge - Gate to Drain"],
    "Ciss": ["Input Capacitance (Ciss) (Max) @ Vds", "Input Capacitance", "Capacitance @ Vds - Ciss"],
    "Coss": ["Output Capacitance (Coss)", "Output Capacitance", "Capacitance @ Vds - Coss"],
    "Crss": ["Reverse Transfer Capacitance (Crss)", "Reverse Transfer Capacitance"],
    "Qrr": ["Reverse Recovery Charge (Qrr)", "Reverse Recovery Charge"],
    "Pd": ["Power Dissipation (Max)", "Power Dissipation", "Power - Max"],
    "R_theta_JC": ["Thermal Resistance Junction-Case", "Rth(j-c)", "Rthjc"],
    "R_theta_JA": ["Thermal Resistance Junction-Ambient"],
    "Tjmax": ["Operating Temperature", "Operating Temperature - Junction"],
    "package": ["Package / Case", "Package/Case", "Package"],
    "mounting": ["Mounting Type", "Mounting"],
    "channel_polarity": ["FET Type", "Channel Polarity"],
    "product_status": ["Product Status", "Part Status", "Status"],
    "technology": ["Technology"],
    "Cost": ["Price", "Unit Price", "Unit Price (USD)"],
    "pdf_url": ["Datasheet", "Datasheet Link", "Datasheet URL"],
    "Vf": ["Diode Forward Voltage", "Body Diode Forward Voltage"],
    "td_on": ["Turn-On Delay Time"],
    "td_off": ["Turn-Off Delay Time"],
    "tr": ["Rise Time"],
    "tf": ["Fall Time"],
    "trr": ["Reverse Recovery Time"],
    "Is": ["Diode Continuous Forward Current", "Current - Diode (Continuous)"],
    "Vgs_max": ["Vgs (Max)", "Gate-Source Voltage (Max)", "Drive Voltage (Max)"],
    "supplier_part_number": ["Digi-Key Part #", "Supplier Part Number", "DigiKey Part Number"],
    "description": ["Description", "Detailed Description"],
    "stock_available": ["Stock", "Quantity Available", "Digi-Key Stock"],
}

_IGBT_COLUMNS: Dict[str, List[str]] = {
    "ModelName": ["Mfr Part #", "Part Number", "Manufacturer Part Number"],
    "manufacturer": ["Mfr", "Manufacturer", "Manufacture"],
    "Vdss": ["Vces", "VCES", "Collector-Emitter Breakdown Voltage", "Voltage - Collector Emitter Breakdown"],
    "Id": ["Ic", "IC", "Collector Current", "Collector Current - Continuous", "Current - Collector (Ic)"],
    "Vce_sat": ["Vce(sat)", "VCE(sat)", "Collector-Emitter Saturation Voltage"],
    "Vgs_th": ["Vge(th)", "VGE(th)", "Gate Threshold Voltage"],
    "Qg": ["Qg", "QG", "Total Gate Charge", "Charge - Gate"],
    "Ciss": ["Ciss", "CISS", "Input Capacitance"],
    "R_theta_JC": ["Rth(j-c)", "Rthjc", "Thermal Resistance Junction-Case"],
    "eon_ref": ["Eon", "EON", "Turn-On Energy"],
    "eoff_ref": ["Eoff", "EOFF", "Turn-Off Energy"],
    "Tjmax": ["Tj(max)", "TJ(max)", "Operating Temperature - Junction", "Operating Temperature"],
    "package": ["Package", "Package / Case", "Case"],
    "mounting": ["Mounting", "Mounting Type"],
    "Cost": ["Unit Price", "Price", "Unit Price (USD)"],
    "product_status": ["Product Status", "Status", "Part Status"],
    "pdf_url": ["Datasheet", "Datasheet Link"],
    "stock_available": ["Stock", "Quantity Available", "Digi-Key Stock"],
}

_GAN_COLUMNS: Dict[str, List[str]] = {
    "ModelName": ["Mfr Part #", "Part Number", "Manufacturer Part Number"],
    "manufacturer": ["Mfr", "Manufacturer", "Manufacture"],
    "Vdss": ["Vds", "VDS", "Drain-Source Voltage", "Voltage - Drain to Source"],
    "Id": ["Id", "ID", "Drain Current", "Drain Current - Continuous", "Current - Continuous Drain"],
    "Rdson": ["Rds(on)", "RDS(ON)", "Drain-Source On Resistance", "Rdson", "RDSon"],
    "Vgs_th": ["Vgs(th)", "VGS(th)", "Gate Threshold Voltage"],
    "Qg": ["Qg", "QG", "Total Gate Charge", "Charge - Gate"],
    "Ciss": ["Ciss", "CISS", "Input Capacitance"],
    "Crss": ["Crss", "CRSS", "Reverse Transfer Capacitance"],
    "Coss": ["Coss", "COSS", "Output Capacitance"],
    "R_theta_JC": ["Rth(j-c)", "Rthjc", "Thermal Resistance", "Thermal Resistance Junction-Case"],
    "Qrr": ["Qrr", "QRR", "Reverse Recovery Charge"],
    "Tjmax": ["Tj(max)", "TJ(max)", "Operating Temperature", "Operating Temperature - Junction"],
    "package": ["Package", "Package / Case", "Case"],
    "mounting": ["Mounting", "Mounting Type"],
    "Cost": ["Unit Price", "Price", "Unit Price (USD)"],
    "product_status": ["Product Status", "Status", "Part Status"],
    "pdf_url": ["Datasheet", "Datasheet Link"],
    "stock_available": ["Stock", "Quantity Available", "Digi-Key Stock"],
}

#: Device type -> {canonical field -> ordered header aliases}.
#: SiC shares Si's table: DigiKey uses identical column names for both.
DEVICE_TYPE_COLUMNS: Dict[str, Dict[str, List[str]]] = {
    "Si-MOSFET": _MOSFET_COLUMNS,
    "SiC-MOSFET": dict(_MOSFET_COLUMNS),
    "GaN-HEMT": _GAN_COLUMNS,
    "IGBT": _IGBT_COLUMNS,
}


def list_device_types() -> List[str]:
    """Return every registered device type, sorted."""
    return sorted(DEVICE_TYPE_COLUMNS)


def _get_table(device_type: str) -> Dict[str, List[str]]:
    try:
        return DEVICE_TYPE_COLUMNS[device_type]
    except KeyError:
        raise KeyError(
            f"Unknown device_type '{device_type}'. Registered types: {list_device_types()}"
        ) from None


def find_column(headers: List[str], field_name: str, device_type: str) -> Optional[int]:
    """Find the index of the CSV header matching a canonical field name.

    Matching is case-insensitive and whitespace-insensitive; aliases are
    tried in registry order (first match wins).

    Args:
        headers: CSV header row.
        field_name: Canonical field name, e.g. ``"Rdson"``.
        device_type: A key of :data:`DEVICE_TYPE_COLUMNS`.

    Returns:
        Column index, or None if no alias matches.

    Raises:
        KeyError: If ``device_type`` is not registered.
    """
    aliases = _get_table(device_type).get(field_name, [field_name])
    headers_norm = [h.strip().lower() for h in headers]
    for alias in aliases:
        alias_norm = alias.strip().lower()
        for i, h in enumerate(headers_norm):
            if h == alias_norm:
                return i
    return None


def map_columns(headers: List[str], device_type: str) -> Dict[str, int]:
    """Map every discoverable canonical field to its column index.

    Headers that match no registered alias are ignored (logged at DEBUG),
    never an error — vendor exports carry many columns we don't consume.

    Args:
        headers: CSV header row.
        device_type: A key of :data:`DEVICE_TYPE_COLUMNS`.

    Returns:
        ``{canonical_field: column_index}`` for all fields found.

    Raises:
        KeyError: If ``device_type`` is not registered.
    """
    table = _get_table(device_type)
    mapping: Dict[str, int] = {}
    for field_name in table:
        idx = find_column(headers, field_name, device_type)
        if idx is not None:
            mapping[field_name] = idx

    mapped_indices = set(mapping.values())
    unmatched = [h for i, h in enumerate(headers) if i not in mapped_indices]
    if unmatched:
        logger.debug("Ignored %d unmatched headers: %s", len(unmatched), unmatched)
    logger.info("Mapped %d/%d headers for %s", len(mapping), len(headers), device_type)
    return mapping
