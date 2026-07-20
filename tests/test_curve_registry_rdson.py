"""Tests for the rdson_vs_tj Stage-4 registry entry — written FIRST (CLAUDE.md §2).

Every caption/OCR string here is REAL corpus text (D:/Extractor/data/
OCR1-OCR13, T24 survey + quarantine diagnosis, 2026-07-13), not invented:

- IR/AUIRF template (the 27 already-matching charts): caption
  "Fig 10. Normalized On-Resistance vs. Temperature", clean axis labels.
- Infineon "Diagram" template (the 15 quarantined devices): captions are
  SHIFTED onto the wrong figures by the known Stage-3 off-by-one bug, so
  the true chart is usually captionless with OCR-mangled labels —
  y "RDS(on) [m_2]" / "[22]" / "[mQ]" / "[mW]", x "Tj[C]" / "T [C]" /
  "T, [º℃]" — and the wrong figures (page-header logo, on-resistance
  vs. DRAIN CURRENT, output characteristics) carry the good captions.

The entry must therefore: score the true charts >= MATCH_THRESHOLD from
their mangled axis text alone, keep every known same-device distractor
strictly below the true chart, and stay out of capacitance_vs_vds's way.
"""
import pytest

from src.classification.classify import (
    MATCH_MARGIN, MATCH_THRESHOLD, ClassificationStatus, classify_page,
)
from src.classification.curve_registry import (
    CurveTypeSpec, get_spec, list_registered_types,
)
from src.classification.scoring import FigureCandidate, OcrLine, score_figure

# Figure geometry for zone classification: labels get bboxes shaped/placed
# like the real ones (y label: tall+narrow, near left edge; x label:
# wide+short, near bottom edge).
FIG_W, FIG_H = 640.0, 600.0
Y_LABEL_BBOX = (10, 180, 40, 430)    # tall & narrow, near left
X_LABEL_BBOX = (240, 565, 450, 590)  # wide & short, near bottom
BODY_BBOX = (250, 250, 330, 270)     # mid-figure (zone unknown)


def fig(figure_id, caption, lines, page=1, index=0):
    return FigureCandidate(
        figure_id=figure_id, page=page, figure_index=index,
        image_path=f"{figure_id}.png", caption=caption,
        ocr_lines=[OcrLine(text=t, bbox=b) for t, b in lines],
        figure_width=FIG_W, figure_height=FIG_H,
    )


def ir_true_chart(figure_id="ir_true", page=1):
    """AUIRF1010EZS Fig 10 — the reference already-matching IR chart."""
    return fig(figure_id, "Fig 10. Normalized On-Resistance vs. Temperature", [
        ("RDS(on) , Drain-to-Source On Resistance", Y_LABEL_BBOX),
        ("(Normalized)", (45, 220, 65, 330)),
        ("TJ , Junction Temperature (℃)", X_LABEL_BBOX),
        ("ID = 84A", (150, 60, 230, 80)),
        ("VGS = 10V", (150, 85, 230, 105)),
    ], page=page)


def infineon_true_chart(figure_id="inf_true", page=1,
                        y_text="RDS(on) [m_2]", x_text="Tj[C]"):
    """BSC009NE2LS5ATMA1 fig_p8_019 — captionless, mangled labels."""
    return fig(figure_id, "", [
        ("3.0", (60, 40, 90, 60)), ("2.5", (60, 120, 90, 140)),
        (y_text, Y_LABEL_BBOX),
        ("typ", BODY_BBOX),
        ("-60", (100, 540, 130, 560)), ("180", (560, 540, 590, 560)),
        (x_text, X_LABEL_BBOX),
    ], page=page)


# ------------------------------------------------------------- registry entry

def test_rdson_vs_tj_is_registered():
    assert "rdson_vs_tj" in list_registered_types()


def test_rdson_spec_name_matches_key_and_has_both_axes():
    spec = get_spec("rdson_vs_tj")
    assert isinstance(spec, CurveTypeSpec)
    assert spec.name == "rdson_vs_tj"
    assert set(spec.axis_keywords) == {"x", "y"}


def test_no_caption_keyword_is_substring_of_another():
    # The T14 id_vs_vgs lesson: substring-overlapping caption keywords
    # double-count the same phrase and inflate scores.
    keywords = [k.lower() for k in get_spec("rdson_vs_tj").caption_keywords]
    for i, a in enumerate(keywords):
        for j, b in enumerate(keywords):
            assert i == j or a not in b, f"{a!r} is a substring of {b!r}"


# ------------------------------------------- IR template (the 27 must keep working)

def test_ir_true_chart_clears_match_threshold():
    score = score_figure(ir_true_chart(), get_spec("rdson_vs_tj"))
    assert score.total_score >= MATCH_THRESHOLD


def test_ir_gate_voltage_distractor_scores_below_true_chart():
    # AUIRF7640S2TR Fig 3 "Typical On-Resistance vs. Gate Voltage" — shares
    # the RDS(on) y-label wording, must lose on the negative phrase.
    distractor = fig("ir_fig3", "Fig. 3 Typical On-Resistance vs. Gate Voltage", [
        ("RDS(on), Drain-to -Source On Resistance (ms2)", Y_LABEL_BBOX),
        ("VGS. Gate -to -Source Voltage (V)", X_LABEL_BBOX),
        ("TJ = 125℃", BODY_BBOX),
    ])
    spec = get_spec("rdson_vs_tj")
    assert score_figure(distractor, spec).total_score \
        < score_figure(ir_true_chart(), spec).total_score - MATCH_MARGIN


def test_ir_soa_distractor_scores_below_threshold():
    # AUIRF1010EZS Fig 8 SOA — mentions RDS(on) mid-plot only.
    soa = fig("ir_soa", "Fig 8. Maximum Safe Operating Area", [
        ("OPERATION IN THIS AREA LIMITED BY RDS (on);", BODY_BBOX),
        ("ID, Drain-to-Source Current (A)", Y_LABEL_BBOX),
        ("VDS, Drain-to-Source Voltage (V)", X_LABEL_BBOX),
    ])
    assert score_figure(soa, get_spec("rdson_vs_tj")).total_score < MATCH_THRESHOLD


def test_ir_page_still_picks_true_chart_with_margin():
    figures = [
        ir_true_chart("ir_true", page=4),
        fig("ir_fig4", "Fig. 4 Typical On-Resistance vs. Drain Current", [
            ("RDS(on), Drain-to -Source On Resistance ( m22)", Y_LABEL_BBOX),
            ("ID, Drain Current (A)", X_LABEL_BBOX),
            ("Vgs = 10V", BODY_BBOX),
        ], page=4),
    ]
    result = classify_page(figures, "rdson_vs_tj")
    assert result.status is ClassificationStatus.MATCHED
    assert result.figure.figure_id == "ir_true"


# ------------------- Infineon "Diagram" template (the 15 quarantined must now match)

@pytest.mark.parametrize("y_text,x_text", [
    ("RDS(on) [m_2]", "Tj[C]"),      # BSC009NE2LS5ATMA1 fig_p8_019
    ("RDS(on) [22]", "T [C]"),       # 2N7002H6327XTSA2 fig_p6_013
    ("RDS(on) [mQ]", "Tj[ºC]"),      # BSB053N03LPG fig_p6_012
    ("RDS(on) [mW]", "T, [º℃]"),     # BSB165N15NZ3GXUMA3 fig_p8_019
    ("RDS(on)[m2]", "Tj [ºC]"),      # BSC010N04LSATMA1 fig_p9_020
    ("R DS(on) [mQ]", "Tj[º℃]"),     # BSB012N03LX3G fig_p6_012 (space inside RDS)
])
def test_infineon_true_chart_mangled_labels_clear_threshold(y_text, x_text):
    chart = infineon_true_chart(y_text=y_text, x_text=x_text)
    score = score_figure(chart, get_spec("rdson_vs_tj"))
    assert score.total_score >= MATCH_THRESHOLD, \
        f"true chart with y={y_text!r} x={x_text!r} scored {score.total_score}"


def test_infineon_vs_drain_current_distractor_stays_below_threshold():
    # BSC009NE2LS5ATMA1 fig_p7_015 — RDS(on)=f(ID), previously the device's
    # best candidate at 3.5. Real embedded title + footer text.
    distractor = fig("inf_vs_id", "", [
        ("Diagram 6: Typ. drain-source on resistance", (100, 5, 540, 25)),
        ("RDS(on) [m_2]", Y_LABEL_BBOX),
        ("3.2 V", BODY_BBOX), ("10 V1", (250, 300, 330, 320)),
        ("ID [A]", X_LABEL_BBOX),
        ("RDS(on)=f(/D); T}=25 ℃; parameter: VGs", (100, 592, 540, 599)),
    ])
    assert score_figure(distractor, get_spec("rdson_vs_tj")).total_score \
        < MATCH_THRESHOLD


def test_infineon_logo_with_misattributed_caption_stays_below_threshold():
    # BSC010N04LSATMA1 fig_p9_019 — the page-header logo carrying the
    # shifted "Diagram 9" caption; previously beat the true chart's margin.
    logo = fig("inf_logo", "Diagram 9: Drain-source on-state resistance", [
        ("Infineon", (67, 47, 293, 109)),
    ])
    assert score_figure(logo, get_spec("rdson_vs_tj")).total_score < MATCH_THRESHOLD


def test_infineon_output_char_with_misattributed_caption_scores_at_most_zero():
    # BSB165N15NZ3GXUMA3 fig_p7_014 — output characteristics carrying the
    # shifted "on-state resistance" caption.
    output_char = fig("inf_output", "6 Typ. drain-source on-state resistance", [
        ("5 Typ. output characteristics Ic co", (100, 5, 540, 25)),
        ("ID [A]", Y_LABEL_BBOX),
        ("VDS [V]", X_LABEL_BBOX),
        ("ID=f( VDs); Tj=25 ℃; parameter: Vas", (100, 592, 540, 599)),
    ])
    assert score_figure(output_char, get_spec("rdson_vs_tj")).total_score <= 0


def test_case_temperature_chart_gains_no_tj_axis_credit():
    # BSC010N04LSATMA1 fig_p7_010 power dissipation — x label "Tc [ºC]"
    # must NOT be read as a junction-temperature signal.
    power = fig("inf_ptot", "Diagram 1: Power dissipation", [
        ("Ptot [W]", Y_LABEL_BBOX),
        ("Tc [ºC]", X_LABEL_BBOX),
        ("-f(T)", (100, 592, 540, 599)),
    ])
    score = score_figure(power, get_spec("rdson_vs_tj"))
    assert score.total_score < MATCH_THRESHOLD
    assert not any(s.source == "axis_x" for s in score.matched_signals)


def test_infineon_page9_end_to_end_true_chart_beats_captioned_logo():
    # The real BSC010N04LSATMA1 page-9 lineup: shifted-caption logo, the
    # true chart (caption shifted to "gate threshold voltage"), and the
    # vgsth chart (caption shifted to "capacitances").
    figures = [
        fig("logo", "Diagram 9: Drain-source on-state resistance", [
            ("Infineon", (67, 47, 293, 109)),
        ], page=9),
        fig("true_chart", "Diagram 10: Typ. gate threshold voltage", [
            ("RDS(on)[m2]", Y_LABEL_BBOX),
            ("max", BODY_BBOX), ("typ", (250, 300, 330, 320)),
            ("Tj [ºC]", X_LABEL_BBOX),
        ], page=9),
        fig("vgsth", "Diagram 11: Typ. capacitances", [
            ("VGS(th) [V]", Y_LABEL_BBOX),
            ("250 HA", BODY_BBOX),
            ("Tj [ºC]", X_LABEL_BBOX),
        ], page=9),
    ]
    result = classify_page(figures, "rdson_vs_tj")
    assert result.status is ClassificationStatus.MATCHED
    assert result.figure.figure_id == "true_chart"


def test_infineon_captionless_page_end_to_end_matches_true_chart():
    # BSC009NE2LS5-style device: everything captionless, the vs-ID chart on
    # another page used to out-score the true chart device-wide.
    figures = [infineon_true_chart("true_chart", page=8)]
    result = classify_page(figures, "rdson_vs_tj")
    assert result.status is ClassificationStatus.MATCHED
    assert result.figure.figure_id == "true_chart"


# --------------------------------------------------- capacitance must be untouched

def test_capacitance_reference_figure_does_not_match_rdson():
    cap_chart = fig("cap", "Typical Capacitance vs. Drain-to-Source Voltage", [
        ("C, Capacitance (pF)", Y_LABEL_BBOX),
        ("Ciss", BODY_BBOX),
        ("VDS, Drain-to-Source Voltage (V)", X_LABEL_BBOX),
    ])
    assert score_figure(cap_chart, get_spec("rdson_vs_tj")).total_score \
        < MATCH_THRESHOLD


def test_rdson_true_chart_does_not_match_capacitance():
    spec = get_spec("capacitance_vs_vds")
    for chart in (ir_true_chart(), infineon_true_chart()):
        assert score_figure(chart, spec).total_score < MATCH_THRESHOLD
