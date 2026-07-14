"""Stage-4 curve-type registry (CLAUDE.md §1).

Curve-type "fingerprints" are DATA, not code: one :class:`CurveTypeSpec` per
curve type, held in a single dict. Adding a new curve type is a registry
entry, never a new code path — :mod:`scoring` and :mod:`classify` are
identical for every entry.

Wording below (captions, axis-label text) was confirmed against real
``full_extraction.json`` OCR output from ``D:\\Extractor\\data`` before being
committed here, not guessed from the curve-type name alone.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class CurveTypeSpec:
    """Data-only fingerprint for one curve type. No behavior lives here."""

    name: str
    caption_keywords: List[str]
    axis_keywords: Dict[str, List[str]]  # keys: "x", "y"
    positive_phrases: List[Tuple[str, float]]
    negative_phrases: List[Tuple[str, float]]


_REGISTRY: Dict[str, CurveTypeSpec] = {
    "capacitance_vs_vds": CurveTypeSpec(
        name="capacitance_vs_vds",
        caption_keywords=[
            "capacitance vs. drain-to-source voltage",
            "capacitance vs drain-to-source voltage",
            "typical capacitance",
            "typ. capacitance",
        ],
        axis_keywords={
            "y": ["capacitance", "pf", "ciss", "coss", "crss"],
            "x": ["vds", "drain-to-source voltage", "drain-source voltage"],
        },
        positive_phrases=[
            ("ciss", 2.0),
            ("coss", 2.0),
            ("crss", 2.0),
            ("cds shorted", 1.5),
        ],
        negative_phrases=[
            ("transfer characteristic", 3.0),
            ("gate charge", 3.0),
        ],
    ),
    "rdson_vs_tj": CurveTypeSpec(
        name="rdson_vs_tj",
        # Wording confirmed against real OCR across 50 devices (T24 survey,
        # 2026-07-13; runtime-injected spec matched 27/50, then extended
        # from the 15-quarantine diagnosis). Two template families:
        # - IR/AUIRF: caption "Fig N. Normalized On-Resistance vs.
        #   Temperature", clean labels "RDS(on) , Drain-to-Source On
        #   Resistance" / "(Normalized)" / "TJ , Junction Temperature (℃)".
        # - Infineon "Diagram": the Stage-3 caption off-by-one bug usually
        #   shifts "Diagram N: Drain-source on-state resistance" onto the
        #   WRONG figure (page-header logo, output char, RDS(on)=f(ID)),
        #   leaving the true chart captionless with OCR-mangled labels:
        #   y "RDS(on) [m_2]"/"[22]"/"[mQ]"/"[mW]", x "Tj[C]"/"T [C]"/
        #   "T, [º℃]"/"Tj [ºC]" — hence the mangled-Tj x keywords below.
        caption_keywords=[
            "normalized on-resistance",
            "on-state resistance",
        ],
        axis_keywords={
            # "r ds(on)" (space inside RDS) is a real OCR mangling seen on
            # BSB012N03LX3G; it never co-occurs with "rds(on)" in one line,
            # so the two keywords can't double-count the same label.
            "y": ["rds(on)", "r ds(on)", "on resistance", "normalized"],
            # "tj["/"tj ["/"t, ["/"t [c" are the observed OCR manglings of
            # "Tj [°C]". Deliberately NOT "tc [" — case-temperature axes
            # (power dissipation, drain-current derating) must not match.
            "x": ["junction temperature", "tj [", "tj[", "t, [", "t [c"],
        },
        positive_phrases=[
            ("normalized", 1.5),
            ("junction temperature", 1.5),
        ],
        negative_phrases=[
            # Same-device distractors that share RDS(on)/on-resistance
            # wording, each observed winning or crowding the margin in the
            # T24 diagnosis:
            ("gate voltage", 3.0),           # "Typical On-Resistance vs. Gate Voltage"
            ("drain current", 3.0),          # "... vs. Drain Current" (IR caption)
            ("safe operating area", 3.0),    # SOA: "LIMITED BY RDS(on)"
            ("output characteristic", 3.0),  # shifted-caption output char
            ("parameter: v", 2.5),           # multi-Vgs-curve footers: "parameter: VGs"/"Vas"
            ("transfer characteristic", 3.0),
            ("capacitance", 3.0),
            ("gate charge", 3.0),
        ],
    ),
    "id_vs_vgs": CurveTypeSpec(
        name="id_vs_vgs",
        caption_keywords=[
            "transfer characteristic",
            "transfer characteristics",
        ],
        axis_keywords={
            "y": ["id, drain", "drain-to-source current", "drain current", "id,"],
            "x": ["vgs", "gate-to-source voltage", "gate-source voltage"],
        },
        positive_phrases=[
            ("vgs, gate", 1.5),
            ("id, drain", 1.5),
        ],
        negative_phrases=[
            ("capacitance", 3.0),
            ("gate charge", 3.0),
            ("threshold voltage", 2.0),
        ],
    ),
}


def get_spec(curve_type: str) -> CurveTypeSpec:
    """Look up the registered spec for ``curve_type``.

    Args:
        curve_type: Registry key, e.g. ``"id_vs_vgs"``.

    Raises:
        KeyError: If ``curve_type`` is not registered. The message lists
            every registered type so callers/logs are immediately actionable.
    """
    try:
        return _REGISTRY[curve_type]
    except KeyError:
        raise KeyError(
            f"Unknown curve_type '{curve_type}'. Registered types: {sorted(_REGISTRY)}"
        ) from None


def list_registered_types() -> List[str]:
    """Return every registered curve_type key, sorted."""
    return sorted(_REGISTRY)
