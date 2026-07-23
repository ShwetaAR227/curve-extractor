"""Tests for the vgsth_vs_tj Stage-4 registry entry — written FIRST
(CLAUDE.md §2).

Real corpus grounding is thinner here than rdson_vs_tj's own 50-device
survey — CLAUDE.md's "confirmed against real OCR, not guessed" standard is
met for ONE real example: BSC010N04LSATMA1, page 9 (Infineon "Diagram"
template), already embedded as a fixture in
tests/test_curve_registry_rdson.py's own
``test_infineon_page9_end_to_end_true_chart_beats_captioned_logo`` (used
there as rdson's "must NOT match" distractor) — imported from there rather
than re-typed, so this file and rdson's stay pinned to the exact same real
text. That figure: y-axis "VGS(th) [V]", x-axis "Tj [ºC]" (the SAME
mangled-Tj pattern as rdson_vs_tj — same chart family, same Stage-3 OCR
pipeline; x keywords below are reused verbatim from rdson_vs_tj's own
T25 battle-tested list, not re-derived), caption WRONGLY shifted to "Typ.
capacitances" (the same known Stage-3 off-by-one bug that shifts
rdson_vs_tj's own caption also hits this template).

FLAGGED for owner sanity-check (per task instruction — this entry has real
judgment calls, only ONE real example backs it, unlike rdson's 50-device
survey):
- Deliberately NO "capacitance" negative phrase (unlike every other entry
  in this registry) — see test_no_capacitance_negative_phrase_deliberately_absent
  below for why: it would fire on vgsth's own real mis-shifted caption on
  the one example available and defeat the match. Axis keywords alone
  already keep a genuine capacitance chart at zero score without it.
- "gate threshold voltage" (spelled out) was TRIED as a caption_keyword/
  positive_phrase for a hypothetical unconfirmed IR/AUIRF-style verbose
  template, then REVERTED —
  test_end_to_end_page9_lineup_matches_vgsth_true_chart caught it scoring
  rdson_vs_tj's own true chart HIGHER than the real vgsth chart, because
  on this exact real page that phrase is the caption wrongly shifted onto
  rdson's figure, not vgsth's. Only the one corpus-confirmed signal
  ("vgs(th)") remains — see curve_registry.py's own comment for the full
  account.
"""
from src.classification.classify import MATCH_MARGIN, MATCH_THRESHOLD, ClassificationStatus, classify_page
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


def vgsth_true_chart(figure_id="vgsth_true", page=9):
    """BSC010N04LSATMA1 page 9's real vgsth figure (real OCR text) — the
    exact fixture already embedded in test_curve_registry_rdson.py's own
    end-to-end page-9 test, used there as rdson's distractor. Caption is
    the REAL (wrongly-shifted) one, not invented."""
    return fig(figure_id, "Diagram 11: Typ. capacitances", [
        ("VGS(th) [V]", Y_LABEL_BBOX),
        ("250 HA", BODY_BBOX),
        ("Tj [ºC]", X_LABEL_BBOX),
    ], page=page)


# ------------------------------------------------------------- registry entry

def test_vgsth_vs_tj_is_registered():
    assert "vgsth_vs_tj" in list_registered_types()


def test_vgsth_spec_name_matches_key_and_has_both_axes():
    spec = get_spec("vgsth_vs_tj")
    assert isinstance(spec, CurveTypeSpec)
    assert spec.name == "vgsth_vs_tj"
    assert set(spec.axis_keywords) == {"x", "y"}


def test_no_caption_keyword_is_substring_of_another():
    # Same T14 id_vs_vgs lesson rdson_vs_tj's own test already checks.
    keywords = [k.lower() for k in get_spec("vgsth_vs_tj").caption_keywords]
    for i, a in enumerate(keywords):
        for j, b in enumerate(keywords):
            assert i == j or a not in b, f"{a!r} is a substring of {b!r}"


def test_x_axis_keywords_reused_verbatim_from_rdson():
    # Same physical quantity (junction temperature), same OCR pipeline,
    # same manglings -- deliberately not re-derived.
    assert get_spec("vgsth_vs_tj").axis_keywords["x"] == get_spec("rdson_vs_tj").axis_keywords["x"]


def test_no_capacitance_negative_phrase_deliberately_absent():
    # See module docstring: "capacitance" as a negative phrase would fire
    # on vgsth's own real mis-shifted caption ("Typ. capacitances") and
    # defeat the match -- confirmed by test_real_vgsth_chart_clears_match_threshold
    # below, which would otherwise regress if this phrase were added back.
    negatives = [p.lower() for p, _ in get_spec("vgsth_vs_tj").negative_phrases]
    assert not any("capacitance" in p for p in negatives)


# ------------------------------------------------- the true chart must match

def test_real_vgsth_chart_clears_match_threshold():
    score = score_figure(vgsth_true_chart(), get_spec("vgsth_vs_tj"))
    assert score.total_score >= MATCH_THRESHOLD


def test_real_vgsth_chart_scores_despite_wrongly_shifted_caption():
    # The caption is "Typ. capacitances" (wrong, shifted) -- the true
    # chart must still clear threshold from axis text + positive phrase
    # alone, not from any caption credit.
    result = score_figure(vgsth_true_chart(), get_spec("vgsth_vs_tj"))
    assert not any(s.source == "caption_keyword" for s in result.matched_signals)
    assert result.total_score >= MATCH_THRESHOLD


def test_end_to_end_page9_lineup_matches_vgsth_true_chart():
    # The real BSC010N04LSATMA1 page-9 three-figure lineup (shifted-caption
    # logo, rdson's true chart carrying vgsth's OWN rightful caption, and
    # the real vgsth figure) -- classify_page("vgsth_vs_tj") must pick the
    # real vgsth figure, not either distractor.
    figures = [
        fig("logo", "Diagram 9: Drain-source on-state resistance", [
            ("Infineon", (67, 47, 293, 109)),
        ], page=9),
        fig("rdson_true", "Diagram 10: Typ. gate threshold voltage", [
            ("RDS(on)[m2]", Y_LABEL_BBOX),
            ("max", BODY_BBOX), ("typ", (250, 300, 330, 320)),
            ("Tj [ºC]", X_LABEL_BBOX),
        ], page=9),
        vgsth_true_chart("vgsth_true", page=9),
    ]
    result = classify_page(figures, "vgsth_vs_tj")
    assert result.status is ClassificationStatus.MATCHED
    assert result.figure.figure_id == "vgsth_true"


# ------------------------------------------------ must not cross-match other types

def test_real_vgsth_chart_does_not_match_capacitance():
    assert score_figure(vgsth_true_chart(), get_spec("capacitance_vs_vds")).total_score \
        < MATCH_THRESHOLD


def test_real_vgsth_chart_does_not_match_id_vs_vgs():
    # "VGS(th)" contains "vgs" as a substring -- id_vs_vgs's own x-axis
    # keyword -- so this is a genuine cross-contamination risk, not a
    # trivial check. Confirmed safely below threshold (id_vs_vgs's own
    # "threshold voltage" negative phrase + the wrong-zone-only partial
    # credit keep it there).
    assert score_figure(vgsth_true_chart(), get_spec("id_vs_vgs")).total_score \
        < MATCH_THRESHOLD


def test_real_vgsth_chart_does_not_match_rdson():
    assert score_figure(vgsth_true_chart(), get_spec("rdson_vs_tj")).total_score \
        < MATCH_THRESHOLD


def test_rdson_true_charts_do_not_match_vgsth():
    spec = get_spec("vgsth_vs_tj")
    for chart in (ir_true_chart(), infineon_true_chart()):
        assert score_figure(chart, spec).total_score < MATCH_THRESHOLD


def test_capacitance_reference_figure_does_not_match_vgsth():
    cap_chart = fig("cap", "Typical Capacitance vs. Drain-to-Source Voltage", [
        ("C, Capacitance (pF)", Y_LABEL_BBOX),
        ("Ciss", BODY_BBOX),
        ("VDS, Drain-to-Source Voltage (V)", X_LABEL_BBOX),
    ])
    assert score_figure(cap_chart, get_spec("vgsth_vs_tj")).total_score < MATCH_THRESHOLD


def test_id_vs_vgs_reference_figure_does_not_match_vgsth():
    id_chart = fig("id_vgs", "Typical Transfer Characteristics", [
        ("ID, Drain-to-Source Current (A)", Y_LABEL_BBOX),
        ("VGS, Gate-to-Source Voltage (V)", X_LABEL_BBOX),
        ("TJ = 25℃", BODY_BBOX),
    ])
    assert score_figure(id_chart, get_spec("vgsth_vs_tj")).total_score < MATCH_THRESHOLD
