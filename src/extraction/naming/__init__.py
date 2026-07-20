"""Curve-naming registry (Stage 5), mirroring the Stage 4 curve_registry pattern.

Each curve type owns its own naming module/function (position-based for
capacitance_vs_vds; future types like id_vs_vgs will need a different rule,
e.g. temperature-based) — the pipeline looks the function up by curve_type,
it never hardcodes curve-type-specific naming logic itself.
"""
from typing import Callable, Dict, List, Sequence, Tuple

from src.extraction.naming.capacitance_vs_vds import (
    CURVE_ORDER_TOP_TO_BOTTOM as _CAPACITANCE_NAMES,
    name_curves as _capacitance_vs_vds,
)
from src.extraction.naming.rdson_vs_tj import (
    CURVE_NAMES as _RDSON_NAMES,
    name_curves as _rdson_vs_tj,
)

Point = Tuple[float, float]
NamingFn = Callable[[Sequence[Sequence[Point]]], List[str]]

_NAMING_REGISTRY: Dict[str, NamingFn] = {
    "capacitance_vs_vds": _capacitance_vs_vds,
    "rdson_vs_tj": _rdson_vs_tj,
}

# The canonical curve-name set each naming function can produce — sourced
# from the same per-curve-type module as the function itself, so the two
# can never drift apart. Used by Stage 7's final validation ("are all
# expected curves present?").
_EXPECTED_NAMES: Dict[str, List[str]] = {
    "capacitance_vs_vds": list(_CAPACITANCE_NAMES),
    "rdson_vs_tj": list(_RDSON_NAMES),
}


def get_expected_names(curve_type: str) -> List[str]:
    """Return the canonical curve names for ``curve_type``.

    Raises:
        KeyError: If ``curve_type`` has no registered name set. The message
            lists every registered type.
    """
    try:
        return list(_EXPECTED_NAMES[curve_type])
    except KeyError:
        raise KeyError(
            f"No expected curve names registered for curve_type '{curve_type}'. "
            f"Registered types: {sorted(_EXPECTED_NAMES)}"
        ) from None


def get_naming_fn(curve_type: str) -> NamingFn:
    """Look up the naming function registered for ``curve_type``.

    Raises:
        KeyError: If no naming function is registered for ``curve_type``.
            The message lists every registered type.
    """
    try:
        return _NAMING_REGISTRY[curve_type]
    except KeyError:
        raise KeyError(
            f"No naming function registered for curve_type '{curve_type}'. "
            f"Registered types: {sorted(_NAMING_REGISTRY)}"
        ) from None


def list_registered_naming_types() -> List[str]:
    """Return every curve_type with a registered naming function, sorted."""
    return sorted(_NAMING_REGISTRY)
