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
            # Same-device distractor found in a real Stage-4 spot-check
            # (RD3G08CBKHRBTL, 2026-07-23): "Fig.7 Normalized Breakdown
            # Voltage vs. Junction Temperature" was outscoring the SAME
            # device's real on-resistance chart (10.5 vs 9.0) purely
            # because both mention "normalized" and both have a junction-
            # temperature x-axis -- neither word is specific to on-
            # resistance. Confirmed safe against the real true chart's own
            # text, which never mentions "breakdown voltage" (see
            # tests/test_curve_registry_rdson.py's own
            # test_real_onresistance_chart_never_mentions_breakdown_voltage).
            ("breakdown voltage", 3.0),
            # Same-device distractor found in a real Stage-4 spot-check
            # (RS6G100BGTB1, 2026-07-24): "Fig.1 Power Dissipation
            # Derating Curve" was outscoring the SAME device's real
            # on-resistance chart (6.5 vs 4.0) purely because both are
            # plotted against junction temperature -- the same root
            # cause, and the same fix pattern, as the "breakdown
            # voltage" guard directly above. Confirmed safe against the
            # real true chart's own text, which never mentions "power
            # dissipation" (see tests/test_curve_registry_rdson.py's own
            # test_real_rs6g100b_onresistance_chart_never_mentions_power_dissipation).
            ("power dissipation", 3.0),
        ],
    ),
    "vgsth_vs_tj": CurveTypeSpec(
        name="vgsth_vs_tj",
        # Wording confirmed against the one real corpus example directly
        # available in this codebase — BSC010N04LSATMA1 page 9 (Infineon
        # "Diagram" template; the SAME device/page already used in
        # test_curve_registry_rdson.py's own end-to-end fixture, where
        # this exact figure appears as rdson_vs_tj's "must NOT match"
        # distractor — see tests/test_curve_registry_vgsth.py, which
        # imports that same fixture rather than re-typing it): y-axis
        # "VGS(th) [V]", x-axis "Tj [ºC]" (the SAME mangled-Tj pattern as
        # rdson_vs_tj — same chart family, same Stage-3 OCR pipeline; x
        # keywords below are reused verbatim from rdson_vs_tj's own T25
        # battle-tested list, not re-derived), body text carrying
        # current-value labels like "250 HA" (OCR-mangled "250 uA" — a
        # Stage-5 naming concern, not a classification signal).
        #
        # Captions are UNRELIABLE for this template (the SAME known
        # Stage-3 off-by-one bug that shifts rdson_vs_tj's own caption
        # onto the wrong figure ALSO affects this one — confirmed
        # directly on this exact page: the true vgsth figure here carries
        # a WRONG "Typ. capacitances" caption, shifted from elsewhere).
        # Because of that, deliberately NO "capacitance" negative phrase
        # here (unlike every other entry in this registry) — it would
        # fire on vgsth's own mis-shifted caption on this exact real
        # example and defeat the match; the axis keywords alone already
        # keep a genuine capacitance chart (y=Capacitance, x=VDS) at zero
        # score, so the negative phrase would be a redundant, actively
        # harmful safety net for this specific template family — same
        # caption-unreliability lesson rdson_vs_tj's own entry already
        # learned, applied here too.
        #
        # TRIED AND REVERTED (documented so it isn't re-added by accident):
        # "gate threshold voltage" (spelled out) — the standard
        # unabbreviated JEDEC-style term for the same "VGS(th)" concept —
        # was initially added as a plausible caption_keyword/positive_phrase
        # for an unconfirmed IR/AUIRF-style verbose template. Caught by
        # test_end_to_end_page9_lineup_matches_vgsth_true_chart: on the
        # SAME real page-9 lineup, THAT exact phrase is the caption
        # wrongly shifted onto rdson_vs_tj's own true chart (not vgsth's) —
        # so it scored rdson's chart at 7.5 against THIS spec, beating the
        # real vgsth chart's own 7.0. Removed; only the one corpus-
        # confirmed signal ("vgs(th)") remains. If a verbose caption
        # template is ever confirmed for real, re-add it then, verified
        # against that real example first.
        caption_keywords=[
            "vgs(th)",
        ],
        axis_keywords={
            "y": ["vgs(th)"],
            "x": ["junction temperature", "tj [", "tj[", "t, [", "t [c"],
        },
        positive_phrases=[
            ("vgs(th)", 2.0),
        ],
        negative_phrases=[
            ("transfer characteristic", 3.0),
            ("gate charge", 3.0),
            ("on-state resistance", 3.0),
            ("on-resistance", 3.0),
            ("drain current", 3.0),
            ("safe operating area", 3.0),
        ],
    ),
    "if_vs_vsd": CurveTypeSpec(
        name="if_vs_vsd",
        # Wording as specified by the owner (2026-07-22, real captions
        # reviewed outside this session) -- kept to exactly the confirmed
        # phrases, no invented/unverified additions (CLAUDE.md's "confirmed
        # against real OCR, not guessed" standard). Unlike rdson_vs_tj's
        # 50-device survey or even vgsth_vs_tj's one embedded real
        # fixture, no real OCR text is directly available in this session
        # to build an end-to-end fixture from -- FLAGGED for the owner to
        # sanity-check against an actual figure's OCR output once one is
        # available (see tests/test_curve_registry_if_vsd.py's own
        # docstring for the same caveat).
        caption_keywords=[
            "forward characteristics",
            "reverse diode",
            # Real second wording found on device RD3G08CBKHRBTL
            # (2026-07-24, grepped verbatim from its real
            # full_extraction.json): "Fig.20 Source Current vs. Source
            # Drain Voltage" -- the original two phrases above scored 0
            # on this real chart, so it was never picked at all. Caption-
            # keyword ONLY, deliberately NOT also added as a
            # positive_phrase: "source current" is a literal substring of
            # id_vs_vgs's own registered y-axis keyword "drain-to-source
            # current" ("source current" in "drain-to-source current" ==
            # True) -- a positive_phrase scans every OCR line (not just
            # the caption), so it would silently award if_vs_vsd points
            # on a genuine id_vs_vgs chart. Caption-only avoids this: real
            # id_vs_vgs captions say "Typical Transfer Characteristics",
            # never "source current" (checked directly, not assumed, in
            # test_source_current_phrase_not_used_as_positive_phrase_substring_risk).
            "source current",
        ],
        axis_keywords={
            "y": ["i_f", "i_sd", "if,", "isd,"],
            # "vsd" (source-to-drain) is the REVERSE letter order of
            # capacitance_vs_vds's own "vds" (drain-to-source) token --
            # genuinely different physical quantities on real body-diode
            # charts, not a typo; kept as two distinct literal strings on
            # purpose (checked directly, not assumed, in
            # test_x_axis_keyword_is_vsd_not_vds_no_accidental_reversal).
            # "vsd [" (2026-07-24, same real RD3G08CBKHRBTL chart): the
            # real x-axis text is "Source - Drain Voltage : VSD [V]" --
            # bare "VSD" with no underscore/comma, so neither existing
            # x-axis keyword matched it. Bracket-anchored (mirrors
            # rdson_vs_tj/vgsth_vs_tj's own "tj [" convention) rather than
            # a bare "vsd" substring, for the same defensive reasoning.
            "x": ["v_sd", "vsd,", "vsd ["],
        },
        positive_phrases=[
            ("forward characteristics", 2.0),
            ("reverse diode", 1.5),
        ],
        negative_phrases=[
            ("capacitance", 3.0),                  # capacitance_vs_vds
            ("gate charge", 3.0),                    # vgs_vs_qg
            ("transfer characteristic", 3.0),        # id_vs_vgs
            ("threshold voltage", 2.0),               # vgsth_vs_tj / id_vs_vgs
            ("vgs(th)", 2.0),                          # vgsth_vs_tj
            ("on-resistance", 3.0),                     # rdson_vs_tj / vgsth
            ("normalized on-resistance", 3.0),           # rdson_vs_tj
            ("safe operating area", 3.0),                 # SOA (out of scope, common distractor)
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
    "zth_vs_time": CurveTypeSpec(
        name="zth_vs_time",
        # Wording confirmed against real OCR from TWO real chart templates
        # this session (both grep'd directly from real full_extraction.json,
        # same standard as rdson_vs_tj's own survey, not guessed):
        # - ROHM "ratio" template (AG087FGD3HRBTL, page 4, fig index 7):
        #   caption "Fig.3 Normalized Transient Thermal Resistance vs.
        #   Pulse Width"; y-axis label is "Normalized Transient Resistance
        #   : r(t)" -- note it never says "thermal" (that word only
        #   appears in the CAPTION on this template), so the y-axis
        #   keyword below is "transient resistance", not "thermal
        #   resistance", to actually match the real label text; x-axis
        #   "Pulse Width : PW [s]".
        # - ROHM "direct units" template (SCT3030ARC15, page 5, fig index
        #   17): NO caption at all -- the real caption ("Fig.3 Typical
        #   Transient Thermal Impedance vs. Pulse Width") is shifted onto
        #   the neighboring Safe-Operating-Area figure, the SAME caption-
        #   misattribution pattern vgsth_vs_tj's own entry already hit for
        #   its own template (see that entry's comment) -- so this
        #   template's match must clear threshold from axis text alone,
        #   confirmed by tests/test_curve_registry_zth.py's own
        #   test_direct_units_chart_clears_match_threshold_despite_no_caption.
        #   y-axis "ZthJC [K/W]" / "Transient Thermal Impedance :"; x-axis
        #   "Pulse Width : PW [s]".
        # A THIRD template (Infineon "Diagram N: Max. transient thermal
        # impedance", y-axis "Z_thJC [K/W]", x-axis "t_p [s]") was reviewed
        # VISUALLY this session -- no OCR JSON available for those images,
        # so this wording is owner-confirmed real rather than independently
        # OCR-grepped (same provenance standard if_vs_vsd's own entry
        # already used). "tp [s]" is kept as its own x-axis keyword
        # specifically because this template's x-axis label is NOT the
        # ROHM templates' "Pulse Width : PW [s]" text.
        #
        # Cross-contamination checked explicitly (same diligence as the
        # vgs(th)-vs-id_vs_vgs / vsd-vs-vds checks already in this
        # registry) against every OTHER registered type, both directions
        # -- see tests/test_curve_registry_zth.py's own cross-match tests.
        caption_keywords=[
            "transient thermal impedance",
            "transient thermal resistance",
        ],
        axis_keywords={
            "y": ["zthjc", "thermal impedance", "transient resistance"],
            "x": ["pulse width", "tp [s]"],
        },
        positive_phrases=[
            ("thermal impedance", 2.0),
            ("thermal resistance", 2.0),
            ("pulse width", 1.5),
        ],
        negative_phrases=[
            ("capacitance", 3.0),
            ("transfer characteristic", 3.0),
            ("gate charge", 3.0),
            ("forward characteristics", 3.0),
            ("reverse diode", 3.0),
            ("on-state resistance", 3.0),
            ("on-resistance", 3.0),
            ("vgs(th)", 3.0),
            ("safe operating area", 3.0),
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
