"""Stage-1 unit normalization: ``(value, source_unit) -> value in target unit``.

Conversion factors are DATA: ``UNIT_CONVERSIONS[target][source] = multiplier``.

DELIBERATE CHANGE from legacy ``unit_conversion.py``: an unrecognized source
unit returns None with a WARNING instead of silently assuming 1:1 (legacy
bug class: silently guessed values poisoning downstream data).
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

#: target unit -> {source unit variant -> multiplier (target = source * m)}
UNIT_CONVERSIONS: Dict[str, Dict[str, float]] = {
    "mOhm": {
        "ohm": 1000.0, "Ω": 1000.0, "℧": 1000.0,
        "mohm": 1.0, "milliohm": 1.0, "mili ohm": 1.0, "miliohm": 1.0,
        "mΩ": 1.0, "m℧": 1.0,
        "uohm": 0.001, "microohm": 0.001, "µΩ": 0.001, "uΩ": 0.001, "µ℧": 0.001,
    },
    "Ohm": {
        "ohm": 1.0, "Ω": 1.0, "℧": 1.0,
        "mohm": 0.001, "mΩ": 0.001, "m℧": 0.001, "milliohm": 0.001,
    },
    "pF": {"pf": 1.0, "nf": 1000.0, "µf": 1e6, "uf": 1e6, "ff": 0.001},
    "nC": {"nc": 1.0, "pc": 0.001, "µc": 1000.0, "uc": 1000.0},
    "V": {"v": 1.0, "mv": 0.001, "kv": 1000.0},
    "A": {"a": 1.0, "ma": 0.001},
    "C/W": {"°c/w": 1.0, "ºc/w": 1.0, "c/w": 1.0, "k/w": 1.0},
    "mJ": {"mj": 1.0, "j": 1000.0, "µj": 0.001, "uj": 0.001},
    "USD": {"usd": 1.0, "$": 1.0, "": 1.0},
    "ns": {"ns": 1.0, "us": 1000.0, "µs": 1000.0, "ms": 1e6, "s": 1e9, "ps": 0.001},
    "W": {"w": 1.0, "mw": 0.001, "kw": 1000.0},
    "S": {"s": 1.0, "ms": 0.001},
}


def normalize_unit(value, source_unit: str, target_unit: str) -> Optional[float]:
    """Convert ``value`` from ``source_unit`` to ``target_unit``.

    Matching of the source unit is case-insensitive (tables are stored
    lowercase; Unicode symbols kept as-is).

    Args:
        value: Numeric value in ``source_unit``; None passes through as None.
        source_unit: Unit string as found in the data, e.g. ``"mOhm"``.
        target_unit: Canonical target, a key of :data:`UNIT_CONVERSIONS`.

    Returns:
        Converted float, or None when ``value`` is None/non-numeric or the
        source unit is unrecognized (flagged via WARNING, never guessed).

    Raises:
        KeyError: If ``target_unit`` has no conversion table (programmer
            error — targets are fixed at design time).
    """
    if target_unit not in UNIT_CONVERSIONS:
        raise KeyError(
            f"Unknown target unit '{target_unit}'. Known targets: {sorted(UNIT_CONVERSIONS)}"
        )
    if value is None or not isinstance(value, (int, float)):
        return None

    conversions = UNIT_CONVERSIONS[target_unit]
    source = (source_unit or "").strip()
    multiplier = conversions.get(source, conversions.get(source.lower()))

    if multiplier is None:
        logger.warning(
            "Unknown source unit '%s' for target '%s' — value dropped, not guessed",
            source_unit, target_unit,
        )
        return None
    return value * multiplier
