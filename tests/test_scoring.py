"""Tests for src.classification.scoring — written FIRST (CLAUDE.md §2).

Covers caption/axis/phrase scoring, position-aware axis-zone detection,
determinism, and graceful handling of missing/malformed OCR data.
"""
import pytest

from src.classification.curve_registry import CurveTypeSpec, get_spec
from src.classification.scoring import (
    FigureCandidate,
    OcrLine,
    figure_has_complete_axes,
    score_figure,
)


CAPACITANCE_SPEC = get_spec("capacitance_vs_vds")
TRANSFER_SPEC = get_spec("id_vs_vgs")


def make_transfer_figure(figure_id="f1", page=3, figure_index=0):
    """A realistic transfer-characteristics figure (from AUIRF1010EZS)."""
    return FigureCandidate(
        figure_id=figure_id,
        page=page,
        figure_index=figure_index,
        image_path="figures/fig_p3_009.png",
        caption="Fig. 3 Typical Transfer Characteristics",
        figure_width=800,
        figure_height=650,
        ocr_lines=[
            OcrLine(text="ID, Drain-to-Source Current (A)", bbox=(25, 108, 54, 413)),
            OcrLine(text="VGS, Gate-to-Source Voltage (V)", bbox=(177, 547, 495, 573)),
            OcrLine(text="1000", bbox=(50, 26, 96, 45)),
        ],
    )


def make_capacitance_figure(figure_id="f2", page=5, figure_index=0):
    """A realistic capacitance-vs-Vds figure."""
    return FigureCandidate(
        figure_id=figure_id,
        page=page,
        figure_index=figure_index,
        image_path="figures/fig_p5_010.png",
        caption="Fig 5. Typical Capacitance vs. Drain-to-Source Voltage",
        figure_width=800,
        figure_height=650,
        ocr_lines=[
            OcrLine(text="Capacitance (pF)", bbox=(20, 100, 50, 420)),
            OcrLine(text="VDS, Drain-to-Source Voltage (V)", bbox=(180, 550, 500, 575)),
            OcrLine(text="Ciss = Cgs + Cgd, Cds SHORTED", bbox=(300, 150, 600, 175)),
        ],
    )


def test_known_good_caption_scores_positively():
    figure = make_transfer_figure()
    result = score_figure(figure, TRANSFER_SPEC)
    assert result.total_score > 0


def test_matched_signals_include_caption_hit():
    figure = make_transfer_figure()
    result = score_figure(figure, TRANSFER_SPEC)
    sources = [s.source for s in result.matched_signals]
    assert "caption_keyword" in sources


def test_wrong_curve_type_scores_lower_than_correct_type():
    figure = make_transfer_figure()
    correct = score_figure(figure, TRANSFER_SPEC)
    wrong = score_figure(figure, CAPACITANCE_SPEC)
    assert correct.total_score > wrong.total_score


def test_negative_phrase_pulls_score_down():
    figure = make_transfer_figure()  # caption mentions "transfer characteristics"
    result = score_figure(figure, CAPACITANCE_SPEC)
    negative_signals = [s for s in result.matched_signals if s.source == "negative_phrase"]
    assert negative_signals
    assert all(s.weight < 0 for s in negative_signals)


def test_negative_phrase_can_drive_score_negative_or_low():
    figure = make_transfer_figure()
    result = score_figure(figure, CAPACITANCE_SPEC)
    # Only Ciss/Coss/Crss-style signals would score positively; a pure
    # transfer-char figure should not out-score a genuine capacitance match.
    good_match = score_figure(make_capacitance_figure(), CAPACITANCE_SPEC)
    assert result.total_score < good_match.total_score


def test_axis_keyword_in_correct_zone_scores_higher_than_wrong_zone():
    correct_zone = FigureCandidate(
        figure_id="a",
        page=1,
        figure_index=0,
        image_path="x.png",
        caption="",
        figure_width=800,
        figure_height=650,
        ocr_lines=[OcrLine(text="VGS, Gate-to-Source Voltage (V)", bbox=(180, 550, 500, 575))],
    )
    wrong_zone = FigureCandidate(
        figure_id="b",
        page=1,
        figure_index=0,
        image_path="x.png",
        caption="",
        figure_width=800,
        figure_height=650,
        # Same text, but placed in the y-axis zone (narrow & tall, left edge).
        ocr_lines=[OcrLine(text="VGS, Gate-to-Source Voltage (V)", bbox=(10, 50, 40, 600))],
    )
    correct_result = score_figure(correct_zone, TRANSFER_SPEC)
    wrong_result = score_figure(wrong_zone, TRANSFER_SPEC)
    assert correct_result.total_score > wrong_result.total_score


def test_axis_keyword_with_no_bbox_still_counts_partially():
    figure = FigureCandidate(
        figure_id="c",
        page=1,
        figure_index=0,
        image_path="x.png",
        caption="",
        ocr_lines=[OcrLine(text="VGS, Gate-to-Source Voltage (V)", bbox=None)],
    )
    result = score_figure(figure, TRANSFER_SPEC)
    assert result.total_score > 0


def test_scoring_is_deterministic():
    figure = make_transfer_figure()
    result_a = score_figure(figure, TRANSFER_SPEC)
    result_b = score_figure(figure, TRANSFER_SPEC)
    assert result_a.total_score == result_b.total_score
    assert [(s.source, s.text, s.weight) for s in result_a.matched_signals] == [
        (s.source, s.text, s.weight) for s in result_b.matched_signals
    ]


def test_empty_figure_no_caption_no_ocr_scores_zero_or_below():
    figure = FigureCandidate(
        figure_id="empty",
        page=1,
        figure_index=0,
        image_path="x.png",
        caption=None,
        ocr_lines=[],
    )
    result = score_figure(figure, TRANSFER_SPEC)
    assert result.total_score <= 0
    assert result.matched_signals == []


def test_malformed_bbox_does_not_crash_and_is_treated_as_unknown_zone():
    figure = FigureCandidate(
        figure_id="bad_bbox",
        page=1,
        figure_index=0,
        image_path="x.png",
        caption="",
        figure_width=800,
        figure_height=650,
        ocr_lines=[OcrLine(text="VGS, Gate-to-Source Voltage (V)", bbox=(1, 2, 3))],  # malformed
    )
    result = score_figure(figure, TRANSFER_SPEC)
    assert result.total_score > 0  # still partial credit, no crash


def test_positive_phrase_match_adds_signal():
    figure = make_capacitance_figure()
    result = score_figure(figure, CAPACITANCE_SPEC)
    positive_signals = [s for s in result.matched_signals if s.source == "positive_phrase"]
    assert positive_signals


def test_duplicate_axis_keyword_lines_do_not_inflate_score_unboundedly():
    figure = FigureCandidate(
        figure_id="dup",
        page=1,
        figure_index=0,
        image_path="x.png",
        caption="",
        figure_width=800,
        figure_height=650,
        ocr_lines=[
            OcrLine(text="VGS, Gate-to-Source Voltage (V)", bbox=(180, 550, 500, 575)),
            OcrLine(text="VGS, Gate-to-Source Voltage (V)", bbox=(180, 550, 500, 575)),
            OcrLine(text="VGS, Gate-to-Source Voltage (V)", bbox=(180, 550, 500, 575)),
        ],
    )
    single = FigureCandidate(
        figure_id="single",
        page=1,
        figure_index=0,
        image_path="x.png",
        caption="",
        figure_width=800,
        figure_height=650,
        ocr_lines=[OcrLine(text="VGS, Gate-to-Source Voltage (V)", bbox=(180, 550, 500, 575))],
    )
    dup_result = score_figure(figure, TRANSFER_SPEC)
    single_result = score_figure(single, TRANSFER_SPEC)
    assert dup_result.total_score == single_result.total_score


def test_figure_has_complete_axes_true_when_both_zones_present():
    figure = make_transfer_figure()  # has both a y-zone and an x-zone OCR line
    assert figure_has_complete_axes(figure) is True


def test_figure_has_complete_axes_false_when_x_zone_missing():
    figure = FigureCandidate(
        figure_id="missing_x",
        page=1,
        figure_index=0,
        image_path="x.png",
        caption="",
        figure_width=800,
        figure_height=650,
        ocr_lines=[OcrLine(text="ID, Drain-to-Source Current (A)", bbox=(25, 108, 54, 413))],
    )
    assert figure_has_complete_axes(figure) is False


def test_figure_has_complete_axes_false_when_no_ocr_lines():
    figure = FigureCandidate(
        figure_id="empty", page=1, figure_index=0, image_path="x.png",
        caption="", ocr_lines=[],
    )
    assert figure_has_complete_axes(figure) is False
