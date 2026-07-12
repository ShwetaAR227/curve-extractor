"""Stage-1 value parsers: raw CSV cell strings -> floats in canonical units.

Canonical units per field family (fixed, documented here once):
voltage V, current A, on-resistance mOhm, capacitance pF, charge nC,
power W, energy mJ, timing ns, thermal resistance C/W, price USD,
temperature C.

Behavior verified against the legacy parsers' documented real DigiKey cell
formats (e.g. ``"79mOhm @ 13.2A, 15V"``, ``"13.5A (Ta), 22A (Tc)"``).
Every parser returns ``None`` for empty/unparseable input — the caller
decides whether that skips the row.
"""
import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_EMPTY_VALUES = {"", "-", "N/A", "n/a", "null", "None"}

_NUM = r"([+-]?\d+\.?\d*)"


def _is_empty(raw) -> bool:
    return raw is None or str(raw).strip() in _EMPTY_VALUES


def parse_numeric(raw) -> Optional[float]:
    """Generic numeric parse: ranges (``~`` takes first value), ``@`` conditions,
    thousands commas, trailing unit letters. None if unparseable."""
    if _is_empty(raw):
        return None
    cleaned = re.sub(r"[VAmWnCpFKSΩ℧°/]+$", "", str(raw).strip())
    if "~" in cleaned:
        cleaned = cleaned.split("~")[0]
    if "@" in cleaned:
        cleaned = cleaned.split("@")[0]
    try:
        return float(cleaned.strip().replace(",", ""))
    except ValueError:
        return None


def parse_voltage(raw) -> Optional[float]:
    """``"650 V"``, ``"1.2kV"``, ``"500mV"``, ``"±20V"`` -> volts (absolute)."""
    if _is_empty(raw):
        return None
    s = str(raw).strip().lstrip("±").strip()
    m = re.match(_NUM + r"\s*[kK]V", s)
    if m:
        return float(m.group(1)) * 1000.0
    m = re.match(_NUM + r"\s*mV", s)
    if m:
        return float(m.group(1)) / 1000.0
    m = re.match(_NUM + r"\s*V", s)
    if m:
        return float(m.group(1))
    return parse_numeric(s)


def _parse_ta_preferring(raw, unit_re: str, milli_prefix: str) -> Optional[float]:
    """Shared logic for current/power cells like ``"13.5A (Ta), 22A (Tc)"``:
    prefer the (Ta) part, fall back to any part."""
    if _is_empty(raw):
        return None
    parts = [p.strip() for p in str(raw).strip().split(",")]
    pattern = _NUM + r"\s*(" + unit_re + r")"

    def _match(part):
        m = re.match(pattern, part)
        if not m:
            return None
        val = float(m.group(1))
        return val / 1000.0 if m.group(2) == milli_prefix else val

    for part in parts:
        if "(Ta)" in part or (len(parts) == 1 and "(Tc)" not in part):
            val = _match(part)
            if val is not None:
                return val
    for part in parts:
        val = _match(part)
        if val is not None:
            return val
    return None


def parse_current(raw) -> Optional[float]:
    """``"49A (Tc)"``, ``"200mA (Ta)"`` -> amps; prefers the (Ta) value."""
    return _parse_ta_preferring(raw, r"m?A", "mA")


def parse_power(raw) -> Optional[float]:
    """``"164W (Tc)"``, ``"400mW (Ta)"`` -> watts; prefers the (Ta) value."""
    return _parse_ta_preferring(raw, r"m?W", "mW")


def parse_rdson_mohm(raw) -> Optional[float]:
    """On-resistance -> milliohms.

    Handles uOhm/mOhm/Ohm with ASCII or Unicode omega plus the OCR
    artifacts ``W``/``Q`` standing in for omega; a bare number is assumed
    to already be mOhm (legacy-verified DigiKey convention).
    """
    if _is_empty(raw):
        return None
    s = str(raw).strip()
    omega = r"[OΩ℧WQ]"
    m = re.search(r"([\d.]+)\s*[uµ]" + omega, s, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 1000.0
    m = re.search(r"([\d.]+)\s*m" + omega, s, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Plain Ohm last — the mOhm/uOhm branches above already claimed their cases
    m = re.search(r"([\d.]+)\s*" + omega, s, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1000.0
    return parse_numeric(s)


def parse_capacitance(raw) -> Optional[float]:
    """``"1621 pF @ 400 V"``, ``"2.4nF"`` -> picofarads."""
    if _is_empty(raw):
        return None
    s = str(raw).strip()
    m = re.match(_NUM + r"\s*nF", s)
    if m:
        return float(m.group(1)) * 1000.0
    m = re.match(_NUM + r"\s*[uµ]F", s)
    if m:
        return float(m.group(1)) * 1e6
    m = re.match(_NUM + r"\s*pF", s)
    if m:
        return float(m.group(1))
    return None


def parse_charge(raw) -> Optional[float]:
    """``"59 nC @ 15 V"``, ``"1.2uC"``, ``"300pC"`` -> nanocoulombs."""
    if _is_empty(raw):
        return None
    s = str(raw).strip()
    m = re.match(_NUM + r"\s*nC", s)
    if m:
        return float(m.group(1))
    m = re.match(_NUM + r"\s*[uµ]C", s)
    if m:
        return float(m.group(1)) * 1000.0
    m = re.match(_NUM + r"\s*pC", s)
    if m:
        return float(m.group(1)) / 1000.0
    return None


def parse_energy(raw) -> Optional[float]:
    """``"1.2mJ @ 400V, 40A"``, ``"500uJ"`` -> millijoules."""
    if _is_empty(raw):
        return None
    s = str(raw).strip()
    m = re.match(_NUM + r"\s*mJ", s, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.match(_NUM + r"\s*[uµ]J", s, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 1000.0
    m = re.match(_NUM + r"\s*J", s)
    if m:
        return float(m.group(1)) * 1000.0
    return parse_numeric(s)


def parse_timing(raw) -> Optional[float]:
    """``"35ns @ 400V, 40A"``, ``"1.5us"`` -> nanoseconds."""
    if _is_empty(raw):
        return None
    s = str(raw).strip()
    m = re.match(_NUM + r"\s*[uµ]s", s, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1000.0
    m = re.match(_NUM + r"\s*ns", s, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.match(_NUM + r"\s*ps", s, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 1000.0
    return parse_numeric(s)


def parse_thermal_resistance(raw) -> Optional[float]:
    """``"0.65 °C/W"``, ``"1.1 K/W"`` -> C/W (K/W is numerically identical)."""
    if _is_empty(raw):
        return None
    m = re.match(_NUM + r"\s*[°ºK]?\s*[CK]?/?W", str(raw).strip())
    if m:
        return float(m.group(1))
    return parse_numeric(raw)


def parse_price(raw) -> Optional[float]:
    """``"$15.51"`` or ``"15.51"`` -> USD float."""
    if _is_empty(raw):
        return None
    s = str(raw).strip().lstrip("$").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_temperature_range(raw) -> Tuple[Optional[float], Optional[float]]:
    """``"-40 C ~ 175 C (TJ)"`` -> (-40.0, 175.0); a single value is Tjmax.

    Returns:
        ``(tjmin, tjmax)`` — either element may be None.
    """
    if _is_empty(raw):
        return None, None
    s = str(raw).strip()
    m = re.match(_NUM + r"\s*°?\s*C\s*~\s*" + _NUM + r"\s*°?\s*C", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(_NUM + r"\s*°?\s*C", s)
    if m:
        return None, float(m.group(1))
    return None, None
