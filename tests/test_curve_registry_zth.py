"""Tests for the zth_vs_time Stage-4 registry entry — written FIRST
(CLAUDE.md §2, red phase). RED PHASE ONLY — the registry entry does not
exist yet.

Real corpus grounding, confirmed against actual full_extraction.json OCR
output (not invented) for TWO real chart templates seen this session:

- ROHM "ratio" template: AG087FGD3HRBTL, page 4, figure index 7 — every
  OCR line/bbox below was grep'd directly from that device's real
  full_extraction.json (the same device used throughout this session's
  classical_zth.py/hybrid_zth.py work). Caption "Fig.3 Normalized
  Transient Thermal Resistance vs. Pulse Width", y-axis "Normalized
  Transient Resistance : r(t)", x-axis "Pulse Width : PW [s]".
- ROHM "direct units" template: SCT3030ARC15, page 5, figure index 17 —
  also grep'd directly. NO caption at all: the real caption ("Fig.3
  Typical Transient Thermal Impedance vs. Pulse Width") is shifted onto
  the WRONG figure (the neighboring Safe-Operating-Area chart) — the SAME
  caption-misattribution pattern vgsth_vs_tj's own entry already
  documented for its own template (see that file's docstring). y-axis
  "ZthJC [K/W]" / "Transient Thermal Impedance :", x-axis
  "Pulse Width : PW [s]".

A THIRD template (Infineon "Diagram N: Max. transient thermal impedance",
y-axis "Z_thJC [K/W]", x-axis "t_p [s]") was reviewed VISUALLY this
session — no OCR JSON is available for those images, so its wording is
owner-confirmed real (same provenance standard if_vs_vsd's own entry
already used: see that file's docstring), not independently re-verified
via OCR grep. Kept in its own clearly-labeled fixture, separate from the
two OCR-grep-confirmed ones above.

Cross-contamination checked explicitly (same diligence as the
vgs(th)-vs-id_vs_vgs / vsd-vs-vds checks already in this test suite)
against EVERY other registered curve type, in BOTH directions: the real
zth charts must not match any other type's spec, and each other type's
own reference chart must not match zth_vs_time's spec.
"""
from src.classification.classify import MATCH_THRESHOLD, ClassificationStatus, classify_page
from src.classification.curve_registry import CurveTypeSpec, get_spec, list_registered_types
from src.classification.scoring import FigureCandidate, OcrLine, score_figure

from tests.test_curve_registry_if_vsd import if_vsd_true_chart
from tests.test_curve_registry_rdson import (
    BODY_BBOX,
    X_LABEL_BBOX,
    Y_LABEL_BBOX,
    fig,
    infineon_true_chart,
    ir_true_chart,
)
from tests.test_curve_registry_vgsth import vgsth_true_chart

# Real crop dimensions for the two OCR-grep-confirmed fixtures below (their
# OWN real image sizes, not the shared 640x600 convention the synthetic
# distractor fixtures elsewhere in this suite use).
AG087_W, AG087_H = 707.0, 694.0
SCT3030_W, SCT3030_H = 682.0, 660.0


def real_fig(figure_id, caption, lines, width, height, page=1):
    return FigureCandidate(
        figure_id=figure_id, page=page, figure_index=0,
        image_path=f"{figure_id}.png", caption=caption,
        ocr_lines=[OcrLine(text=t, bbox=b) for t, b in lines],
        figure_width=width, figure_height=height,
    )


def zth_ratio_true_chart(figure_id="zth_ratio_true", page=4):
    """AG087FGD3HRBTL, page 4, figure index 7 — exact real OCR text and
    bounding boxes (crop size 707x694), grep'd directly from
    full_extraction.json this session, not invented."""
    return real_fig(figure_id, "Fig.3 Normalized Transient Thermal Resistance vs. Pulse Width", [
        ("10", (114, 42, 144, 63)),
        ("Tc=25℃", (184, 79, 279, 107)),
        ("1", (129, 220, 142, 237)),
        ("Duty cycle", (479, 263, 574, 283)),
        ("top", (479, 285, 511, 304)),
        ("D=1", (554, 284, 591, 301)),
        ("D=0.5", (553, 305, 610, 322)),
        ("D=0.1", (555, 326, 607, 343)),
        ("D=0.05", (555, 347, 620, 363)),
        ("D=0.01", (553, 368, 620, 385)),
        ("0.1", (104, 393, 140, 414)),
        ("bottom Single", (479, 389, 611, 409)),
        ("Rth(j-c)=2.80℃/W", (265, 487, 429, 507)),
        ("Rth(j-c)(t)=r(t) × Rth(j-c)", (265, 511, 476, 533)),
        ("Normalized Transient Resistance : r(t)", (20, 76, 57, 559)),
        ("0.01", (88, 567, 142, 590)),
        ("0.0001 0.001", (129, 594, 276, 616)),
        ("0.01", (277, 594, 351, 615)),
        ("0.1", (394, 595, 425, 615)),
        ("1", (488, 596, 499, 613)),
        ("10", (562, 595, 590, 614)),
        ("100", (638, 595, 679, 615)),
        ("Pulse Width : PW [s]", (282, 642, 547, 674)),
    ], width=AG087_W, height=AG087_H, page=page)


def zth_direct_units_true_chart(figure_id="zth_direct_true", page=5):
    """SCT3030ARC15, page 5, figure index 17 — exact real OCR text and
    bounding boxes (crop size 682x660), grep'd directly. NO caption: the
    real caption is shifted onto the neighboring SOA figure (the same
    caption-misattribution pattern vgsth_vs_tj's own entry documents)."""
    return real_fig(figure_id, None, [
        ("1", (147, 21, 162, 40)),
        ("0.1", (123, 144, 162, 177)),
        ("0.01", (115, 277, 159, 296)),
        ("ZthJC [K/W]", (57, 211, 92, 368)),
        ("0.001", (102, 404, 158, 426)),
        ("Tc = 25ºC", (445, 452, 555, 482)),
        ("Single Pulse", (448, 484, 588, 511)),
        ("Transient Thermal Impedance :", (23, 87, 55, 489)),
        ("0.0001", (90, 532, 158, 554)),
        ("0.000001", (132, 558, 225, 580)),
        ("0.0001", (258, 558, 327, 579)),
        ("0.01", (387, 559, 430, 578)),
        ("1", (519, 560, 531, 577)),
        ("100", (622, 559, 659, 578)),
        ("Pulse Width : PW [s]", (283, 614, 544, 645)),
    ], width=SCT3030_W, height=SCT3030_H, page=page)


def zth_infineon_style_chart(figure_id="zth_infineon_true", page=8):
    """Infineon 'Diagram N' template, reviewed VISUALLY this session (no
    OCR JSON available for these images) — owner-confirmed real wording:
    caption "Diagram 4: Max. transient thermal impedance", y-axis
    "Z_thJC [K/W]" (OCR'd here as "ZthJC [K/W]", matching the identical
    subscript-symbol rendering already OCR-grep-confirmed on the ROHM
    direct-units template above), x-axis "t_p [s]" per the task's own
    stated signal. Same provenance standard if_vs_vsd's own entry used
    (owner-specified, not independently OCR-grepped this time)."""
    return fig(figure_id, "Diagram 4: Max. transient thermal impedance", [
        ("ZthJC [K/W]", Y_LABEL_BBOX),
        ("single pulse", BODY_BBOX),
        ("tp [s]", X_LABEL_BBOX),
    ], page=page)


# ------------------------------------------------------------- registry entry

def test_zth_vs_time_is_registered():
    assert "zth_vs_time" in list_registered_types()


def test_zth_spec_name_matches_key_and_has_both_axes():
    spec = get_spec("zth_vs_time")
    assert isinstance(spec, CurveTypeSpec)
    assert spec.name == "zth_vs_time"
    assert set(spec.axis_keywords) == {"x", "y"}


def test_no_caption_keyword_is_substring_of_another():
    keywords = [k.lower() for k in get_spec("zth_vs_time").caption_keywords]
    for i, a in enumerate(keywords):
        for j, b in enumerate(keywords):
            assert i == j or a not in b, f"{a!r} is a substring of {b!r}"


# ------------------------------------------------- the true charts must match

def test_ratio_chart_clears_match_threshold():
    score = score_figure(zth_ratio_true_chart(), get_spec("zth_vs_time"))
    assert score.total_score >= MATCH_THRESHOLD


def test_direct_units_chart_clears_match_threshold_despite_no_caption():
    result = score_figure(zth_direct_units_true_chart(), get_spec("zth_vs_time"))
    assert not any(s.source == "caption_keyword" for s in result.matched_signals)
    assert result.total_score >= MATCH_THRESHOLD


def test_infineon_style_chart_clears_match_threshold():
    score = score_figure(zth_infineon_style_chart(), get_spec("zth_vs_time"))
    assert score.total_score >= MATCH_THRESHOLD


def test_ratio_chart_y_axis_credit_comes_from_transient_resistance_not_caption_alone():
    # The y-axis label itself ("Normalized Transient Resistance : r(t)")
    # never says "thermal" -- confirms the spec's y-axis keyword actually
    # matches the REAL label text, not just riding on caption credit.
    result = score_figure(zth_ratio_true_chart(), get_spec("zth_vs_time"))
    axis_y_texts = [s.text for s in result.matched_signals if s.source == "axis_y"]
    assert "transient resistance" in axis_y_texts


def test_direct_units_chart_y_axis_credit_comes_from_zthjc_and_thermal_impedance():
    result = score_figure(zth_direct_units_true_chart(), get_spec("zth_vs_time"))
    axis_y_texts = {s.text for s in result.matched_signals if s.source == "axis_y"}
    assert "zthjc" in axis_y_texts
    assert "thermal impedance" in axis_y_texts


def test_end_to_end_page_lineup_matches_ratio_true_chart_over_soa_distractor():
    # A same-device Safe Operating Area chart is a realistic distractor:
    # it shares "single pulse"/pulse-duration wording but none of
    # zth_vs_time's own thermal-impedance/resistance anchors.
    figures = [
        fig("soa_distractor", "Fig.2 Maximum Safe Operating Area", [
            ("Drain Current", Y_LABEL_BBOX),
            ("Single Pulse", BODY_BBOX),
            ("Drain-Source Voltage", X_LABEL_BBOX),
        ], page=4),
        zth_ratio_true_chart("zth_true", page=4),
    ]
    result = classify_page(figures, "zth_vs_time")
    assert result.status is ClassificationStatus.MATCHED
    assert result.figure.figure_id == "zth_true"


# ------------------------------------------------ must not cross-match other types

def test_ratio_chart_does_not_match_capacitance():
    assert score_figure(zth_ratio_true_chart(), get_spec("capacitance_vs_vds")).total_score \
        < MATCH_THRESHOLD


def test_ratio_chart_does_not_match_rdson():
    assert score_figure(zth_ratio_true_chart(), get_spec("rdson_vs_tj")).total_score \
        < MATCH_THRESHOLD


def test_ratio_chart_does_not_match_vgsth():
    assert score_figure(zth_ratio_true_chart(), get_spec("vgsth_vs_tj")).total_score \
        < MATCH_THRESHOLD


def test_ratio_chart_does_not_match_id_vs_vgs():
    assert score_figure(zth_ratio_true_chart(), get_spec("id_vs_vgs")).total_score \
        < MATCH_THRESHOLD


def test_ratio_chart_does_not_match_if_vs_vsd():
    assert score_figure(zth_ratio_true_chart(), get_spec("if_vs_vsd")).total_score \
        < MATCH_THRESHOLD


def test_direct_units_chart_does_not_match_capacitance():
    assert score_figure(zth_direct_units_true_chart(), get_spec("capacitance_vs_vds")).total_score \
        < MATCH_THRESHOLD


def test_direct_units_chart_does_not_match_rdson():
    assert score_figure(zth_direct_units_true_chart(), get_spec("rdson_vs_tj")).total_score \
        < MATCH_THRESHOLD


def test_direct_units_chart_does_not_match_vgsth():
    assert score_figure(zth_direct_units_true_chart(), get_spec("vgsth_vs_tj")).total_score \
        < MATCH_THRESHOLD


def test_direct_units_chart_does_not_match_id_vs_vgs():
    assert score_figure(zth_direct_units_true_chart(), get_spec("id_vs_vgs")).total_score \
        < MATCH_THRESHOLD


def test_direct_units_chart_does_not_match_if_vs_vsd():
    assert score_figure(zth_direct_units_true_chart(), get_spec("if_vs_vsd")).total_score \
        < MATCH_THRESHOLD


def test_capacitance_reference_figure_does_not_match_zth():
    cap_chart = fig("cap", "Typical Capacitance vs. Drain-to-Source Voltage", [
        ("C, Capacitance (pF)", Y_LABEL_BBOX),
        ("Ciss", BODY_BBOX),
        ("VDS, Drain-to-Source Voltage (V)", X_LABEL_BBOX),
    ])
    assert score_figure(cap_chart, get_spec("zth_vs_time")).total_score < MATCH_THRESHOLD


def test_id_vs_vgs_reference_figure_does_not_match_zth():
    id_chart = fig("id_vgs", "Typical Transfer Characteristics", [
        ("ID, Drain-to-Source Current (A)", Y_LABEL_BBOX),
        ("VGS, Gate-to-Source Voltage (V)", X_LABEL_BBOX),
        ("TJ = 25℃", BODY_BBOX),
    ])
    assert score_figure(id_chart, get_spec("zth_vs_time")).total_score < MATCH_THRESHOLD


def test_vgsth_true_chart_does_not_match_zth():
    assert score_figure(vgsth_true_chart(), get_spec("zth_vs_time")).total_score < MATCH_THRESHOLD


def test_rdson_true_charts_do_not_match_zth():
    spec = get_spec("zth_vs_time")
    for chart in (ir_true_chart(), infineon_true_chart()):
        assert score_figure(chart, spec).total_score < MATCH_THRESHOLD


def test_if_vsd_true_chart_does_not_match_zth():
    assert score_figure(if_vsd_true_chart(), get_spec("zth_vs_time")).total_score < MATCH_THRESHOLD


def test_gate_charge_reference_figure_does_not_match_zth():
    qg_chart = fig("qg", "Typical Gate Charge Waveform", [
        ("VGS, Gate-to-Source Voltage (V)", Y_LABEL_BBOX),
        ("QG, Total Gate Charge (nC)", X_LABEL_BBOX),
    ])
    assert score_figure(qg_chart, get_spec("zth_vs_time")).total_score < MATCH_THRESHOLD


def test_soa_reference_figure_does_not_match_zth():
    # Same-device distractor: SOA charts often mention "pulse width"/
    # "single pulse" conditions but never zth's own thermal anchors.
    soa_chart = fig("soa", "Fig.2 Maximum Safe Operating Area", [
        ("Drain Current", Y_LABEL_BBOX),
        ("Single Pulse", BODY_BBOX),
        ("Drain-Source Voltage", X_LABEL_BBOX),
    ])
    assert score_figure(soa_chart, get_spec("zth_vs_time")).total_score < MATCH_THRESHOLD
