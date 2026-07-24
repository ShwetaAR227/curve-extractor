"""Tests for the if_vs_vsd Stage-4 registry entry — written FIRST
(CLAUDE.md §2, red phase). RED PHASE ONLY — the registry entry does not
exist yet.

Wording is the owner-specified confirmed phrase set (2026-07-22, real
captions reviewed outside this session): caption keywords "forward
characteristics" + "reverse diode", axis tokens I_F/I_SD (y) and V_SD (x).

FLAGGED FOR OWNER REVIEW (per task instruction, honestly noted — same
spirit as vgsth_vs_tj's own thinner-grounding flag, but here there is
LESS: unlike rdson_vs_tj's 50-device survey or even vgsth_vs_tj's one real
embedded fixture, no real OCR text is directly available in this session
to build an end-to-end fixture from. The end-to-end "true chart" fixture
below is SYNTHETIC, built strictly from the confirmed phrase set above —
not invented wording beyond it — and should be sanity-checked against a
real if_vs_vsd figure's OCR output once one is available.

Cross-contamination checked explicitly (same diligence as vgsth_vs_tj's
own "vgs(th)" vs "vgs" check): if_vs_vsd's x-axis token "vsd" and
capacitance_vs_vds's x-axis token "vds" are different strings (reversed
letter order) with no substring relationship either direction, but this
is exactly a case a careless refactor could bridge, so it's tested for
directly, not assumed.
"""
from src.classification.classify import MATCH_THRESHOLD, ClassificationStatus, classify_page
from src.classification.curve_registry import CurveTypeSpec, get_spec, list_registered_types
from src.classification.scoring import score_figure

from tests.test_curve_registry_rdson import (
    BODY_BBOX,
    X_LABEL_BBOX,
    Y_LABEL_BBOX,
    fig,
    infineon_true_chart,
    ir_true_chart,
)


def if_vsd_true_chart(figure_id="if_vsd_true", page=6):
    """Synthetic (not real-OCR-embedded — see module docstring) if_vs_vsd
    figure, built strictly from the owner-confirmed phrase set: caption
    "forward characteristics"/"reverse diode", y-axis I_F/I_SD, x-axis
    V_SD, body text carrying a per-curve temperature label (the same kind
    of annotation id_vs_vgs's own real reference fixture carries, e.g.
    "TJ = 25°C")."""
    return fig(figure_id, "Fig 6. Typical Source-Drain Diode Forward "
               "Characteristics (Reverse Diode)", [
        ("IF, ISD, Source-Drain Current (A)", Y_LABEL_BBOX),
        ("TJ = 25°C", BODY_BBOX),
        ("VSD, Source-to-Drain Voltage (V)", X_LABEL_BBOX),
    ], page=page)


# ------------------------------------------------------------- registry entry

def test_if_vs_vsd_is_registered():
    assert "if_vs_vsd" in list_registered_types()


def test_if_vs_vsd_spec_name_matches_key_and_has_both_axes():
    spec = get_spec("if_vs_vsd")
    assert isinstance(spec, CurveTypeSpec)
    assert spec.name == "if_vs_vsd"
    assert set(spec.axis_keywords) == {"x", "y"}


def test_no_caption_keyword_is_substring_of_another():
    keywords = [k.lower() for k in get_spec("if_vs_vsd").caption_keywords]
    for i, a in enumerate(keywords):
        for j, b in enumerate(keywords):
            assert i == j or a not in b, f"{a!r} is a substring of {b!r}"


def test_x_axis_keyword_is_vsd_not_vds_no_accidental_reversal():
    # Genuine risk, not trivial: "vds" (capacitance_vs_vds's own x-axis
    # token) and "vsd" (this entry's) are the SAME three letters reversed
    # -- confirm the literal strings actually differ, not merely that a
    # human reading them thinks they differ.
    if_vsd_x = [k.lower() for k in get_spec("if_vs_vsd").axis_keywords["x"]]
    cap_x = [k.lower() for k in get_spec("capacitance_vs_vds").axis_keywords["x"]]
    assert not set(if_vsd_x) & set(cap_x)
    for tok in if_vsd_x:
        assert "vds" not in tok
    for tok in cap_x:
        assert "vsd" not in tok


# ------------------------------------------------- the true chart must match

def test_real_if_vsd_chart_clears_match_threshold():
    score = score_figure(if_vsd_true_chart(), get_spec("if_vs_vsd"))
    assert score.total_score >= MATCH_THRESHOLD


def test_end_to_end_page_lineup_matches_if_vsd_true_chart():
    figures = [
        fig("logo", "Infineon", [("Infineon", (67, 47, 293, 109))], page=6),
        fig("rdson_distractor", "Diagram 5: Normalized on-resistance", [
            ("RDS(on)", Y_LABEL_BBOX), ("Tj [ºC]", X_LABEL_BBOX),
        ], page=6),
        if_vsd_true_chart("if_vsd_true", page=6),
    ]
    result = classify_page(figures, "if_vs_vsd")
    assert result.status is ClassificationStatus.MATCHED
    assert result.figure.figure_id == "if_vsd_true"


# ------------------------------------------------ must not cross-match other types

def test_real_if_vsd_chart_does_not_match_capacitance():
    assert score_figure(if_vsd_true_chart(), get_spec("capacitance_vs_vds")).total_score \
        < MATCH_THRESHOLD


def test_real_if_vsd_chart_does_not_match_id_vs_vgs():
    assert score_figure(if_vsd_true_chart(), get_spec("id_vs_vgs")).total_score \
        < MATCH_THRESHOLD


def test_real_if_vsd_chart_does_not_match_rdson():
    assert score_figure(if_vsd_true_chart(), get_spec("rdson_vs_tj")).total_score \
        < MATCH_THRESHOLD


def test_real_if_vsd_chart_does_not_match_vgsth():
    assert score_figure(if_vsd_true_chart(), get_spec("vgsth_vs_tj")).total_score \
        < MATCH_THRESHOLD


def test_rdson_true_charts_do_not_match_if_vsd():
    spec = get_spec("if_vs_vsd")
    for chart in (ir_true_chart(), infineon_true_chart()):
        assert score_figure(chart, spec).total_score < MATCH_THRESHOLD


def test_capacitance_reference_figure_does_not_match_if_vsd():
    cap_chart = fig("cap", "Typical Capacitance vs. Drain-to-Source Voltage", [
        ("C, Capacitance (pF)", Y_LABEL_BBOX),
        ("Ciss", BODY_BBOX),
        ("VDS, Drain-to-Source Voltage (V)", X_LABEL_BBOX),
    ])
    assert score_figure(cap_chart, get_spec("if_vs_vsd")).total_score < MATCH_THRESHOLD


def test_id_vs_vgs_reference_figure_does_not_match_if_vsd():
    id_chart = fig("id_vgs", "Typical Transfer Characteristics", [
        ("ID, Drain-to-Source Current (A)", Y_LABEL_BBOX),
        ("VGS, Gate-to-Source Voltage (V)", X_LABEL_BBOX),
        ("TJ = 25℃", BODY_BBOX),
    ])
    assert score_figure(id_chart, get_spec("if_vs_vsd")).total_score < MATCH_THRESHOLD


def test_vgsth_reference_figure_does_not_match_if_vsd():
    vgsth_chart = fig("vgsth", "Diagram: Typ. gate threshold voltage", [
        ("VGS(th) [V]", Y_LABEL_BBOX),
        ("Tj [ºC]", X_LABEL_BBOX),
    ])
    assert score_figure(vgsth_chart, get_spec("if_vs_vsd")).total_score < MATCH_THRESHOLD


def test_gate_charge_reference_figure_does_not_match_if_vsd():
    qg_chart = fig("qg", "Typical Gate Charge Waveform", [
        ("VGS, Gate-to-Source Voltage (V)", Y_LABEL_BBOX),
        ("QG, Total Gate Charge (nC)", X_LABEL_BBOX),
    ])
    assert score_figure(qg_chart, get_spec("if_vs_vsd")).total_score < MATCH_THRESHOLD


# ============================================================
# Real-device wording gap (2026-07-24): device RD3G08CBKHRBTL's real
# body-diode chart is captioned "Fig.20 Source Current vs. Source Drain
# Voltage", x-axis "Source - Drain Voltage : VSD [V]" -- different wording
# than the original entry recognized ("forward characteristics"/"reverse
# diode"). Caption and x-axis text below are grepped VERBATIM from this
# device's real full_extraction.json (not invented) -- confirmed to score
# 0 against the original entry before this addition.
#
# This real figure has NO y-axis OCR line at all (a genuine, separate
# Stage-3 OCR gap for this exact figure -- the caption says "Source
# Current" but that text was never captured as its own OCR line) --
# faithfully reproduced here, not papered over with an invented y-axis
# line. So even correctly recognized, it lands QUARANTINED
# (incomplete_axes, flagged for a human to review) rather than
# auto-MATCHED -- confirmed directly, not assumed. The real, honest
# outcome of this fix is: invisible (NO_MATCH) -> flagged for review
# (QUARANTINED), not invisible -> auto-approved.
# ============================================================

def real_rd3g08cbkhrbtl_source_current_chart(figure_id="real_rd3g08_if_vsd", page=9):
    return fig(figure_id, "Fig.20 Source Current vs. Source Drain Voltage", [
        ("Source - Drain Voltage : VSD [V]", X_LABEL_BBOX),
    ], page=page)


def test_real_source_current_chart_now_recognized_not_invisible():
    figures = [real_rd3g08cbkhrbtl_source_current_chart()]
    result = classify_page(figures, "if_vs_vsd", claimed=set())
    assert result.score >= MATCH_THRESHOLD
    assert result.status == ClassificationStatus.QUARANTINED
    assert "incomplete_axes" in result.reason
    assert result.figure.figure_id == "real_rd3g08_if_vsd"


def test_real_source_current_chart_score_breakdown_is_exactly_two_signals():
    # Precise audit: the new caption keyword + the new x-axis keyword,
    # nothing else contributing (no accidental extra matches).
    result = score_figure(real_rd3g08cbkhrbtl_source_current_chart(), get_spec("if_vs_vsd"))
    assert result.total_score == 5.5
    signal_texts = {(s.source, s.text) for s in result.matched_signals}
    assert signal_texts == {
        ("caption_keyword", "source current"),
        ("axis_x", "vsd ["),
    }


def test_real_source_current_chart_does_not_score_high_against_id_vs_vgs():
    # "vgs" (unknown-zone partial credit) is the only signal id_vs_vgs
    # picks up here (from body text like "VGS= 0V" on the real chart,
    # not tested in this synthetic fixture) -- well below MATCH_THRESHOLD
    # either way.
    result = score_figure(real_rd3g08cbkhrbtl_source_current_chart(), get_spec("id_vs_vgs"))
    assert result.total_score < MATCH_THRESHOLD


def test_source_current_phrase_not_used_as_positive_phrase_substring_risk():
    # Explicit, documented reasoning (not just relying on the reader to
    # notice): "source current" is a literal substring of id_vs_vgs's OWN
    # registered y-axis keyword "drain-to-source current" --
    #   "source current" in "drain-to-source current" == True
    # Confirmed directly below. This is exactly why the new phrase is
    # added ONLY as a caption_keyword (checks figure.caption alone) and
    # NOT as a positive_phrase (which would scan every OCR line,
    # including a real id_vs_vgs chart's own y-axis text, and silently
    # award if_vs_vsd points on a genuine id_vs_vgs chart).
    assert "source current" in "drain-to-source current"
    spec = get_spec("if_vs_vsd")
    assert ("source current", 2.0) not in spec.positive_phrases
    assert not any(phrase == "source current" for phrase, _ in spec.positive_phrases)
    # And the real regression this reasoning protects against: a genuine
    # id_vs_vgs chart using that exact phrasing must still score well
    # below MATCH_THRESHOLD against if_vs_vsd (already covered by
    # test_id_vs_vgs_reference_figure_does_not_match_if_vsd above, which
    # uses this exact "Drain-to-Source Current" phrasing -- re-asserted
    # here for visibility next to the reasoning it's protecting).
    id_chart = fig("id_vgs2", "Typical Transfer Characteristics", [
        ("ID, Drain-to-Source Current (A)", Y_LABEL_BBOX),
        ("VGS, Gate-to-Source Voltage (V)", X_LABEL_BBOX),
    ])
    assert score_figure(id_chart, spec).total_score < MATCH_THRESHOLD


def test_x_axis_keyword_vsd_bracket_does_not_collide_with_capacitance_vds():
    # "vsd [" vs capacitance_vs_vds's "vds" -- same reversed-letter-order
    # safety already established for the original "v_sd"/"vsd," keywords,
    # re-checked for the new "vsd [" variant specifically.
    assert "vsd [" not in "vds, drain-to-source voltage (v)"
    cap_chart = fig("cap2", "Typical Capacitance vs. Drain-to-Source Voltage", [
        ("VDS, Drain-to-Source Voltage (V)", X_LABEL_BBOX),
    ])
    assert score_figure(cap_chart, get_spec("if_vs_vsd")).total_score < MATCH_THRESHOLD
