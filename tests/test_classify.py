"""Tests for src.classification.classify — written FIRST (CLAUDE.md §2).

Covers per-page classification (matched/quarantined/no_match), claimed-set
exclusion, cross-page device classification, and mutual exclusion across
curve types via the explicit claimed-set parameter/return value.
"""
import pytest

from src.classification.classify import (
    ClassificationStatus,
    classify_device,
    classify_page,
)
from src.classification.scoring import FigureCandidate, OcrLine


def transfer_figure(figure_id, page=3):
    return FigureCandidate(
        figure_id=figure_id,
        page=page,
        figure_index=0,
        image_path=f"{figure_id}.png",
        caption="Fig. 3 Typical Transfer Characteristics",
        figure_width=800,
        figure_height=650,
        ocr_lines=[
            OcrLine(text="ID, Drain-to-Source Current (A)", bbox=(25, 108, 54, 413)),
            OcrLine(text="VGS, Gate-to-Source Voltage (V)", bbox=(177, 547, 495, 573)),
        ],
    )


def capacitance_figure(figure_id, page=5):
    return FigureCandidate(
        figure_id=figure_id,
        page=page,
        figure_index=0,
        image_path=f"{figure_id}.png",
        caption="Fig 5. Typical Capacitance vs. Drain-to-Source Voltage",
        figure_width=800,
        figure_height=650,
        ocr_lines=[
            OcrLine(text="Capacitance (pF)", bbox=(20, 100, 50, 420)),
            OcrLine(text="VDS, Drain-to-Source Voltage (V)", bbox=(180, 550, 500, 575)),
            OcrLine(text="Ciss = Cgs + Cgd, Cds SHORTED", bbox=(300, 150, 600, 175)),
        ],
    )


def blank_figure(figure_id, page=1):
    return FigureCandidate(
        figure_id=figure_id, page=page, figure_index=0,
        image_path=f"{figure_id}.png", caption=None, ocr_lines=[],
    )


# ---------------------------------------------------------------- classify_page

def test_single_unambiguous_figure_matches():
    figs = [transfer_figure("f1")]
    result = classify_page(figs, "id_vs_vgs", claimed=set())
    assert result.status == ClassificationStatus.MATCHED
    assert result.figure.figure_id == "f1"


def test_two_similar_scoring_figures_are_quarantined():
    figs = [transfer_figure("f1"), transfer_figure("f2")]
    result = classify_page(figs, "id_vs_vgs", claimed=set())
    assert result.status == ClassificationStatus.QUARANTINED
    assert result.margin == 0


def test_no_matching_figure_returns_no_match():
    figs = [capacitance_figure("f1"), blank_figure("f2")]
    result = classify_page(figs, "id_vs_vgs", claimed=set())
    assert result.status == ClassificationStatus.NO_MATCH


def test_claimed_figure_excluded_from_consideration():
    figs = [transfer_figure("f1"), transfer_figure("f2")]
    # f1 already claimed elsewhere -> only f2 remains -> unambiguous match.
    result = classify_page(figs, "id_vs_vgs", claimed={"f1"})
    assert result.status == ClassificationStatus.MATCHED
    assert result.figure.figure_id == "f2"


def test_empty_figure_list_returns_no_match_without_crashing():
    result = classify_page([], "id_vs_vgs", claimed=set())
    assert result.status == ClassificationStatus.NO_MATCH
    assert result.figure is None


def test_all_figures_claimed_returns_no_match():
    figs = [transfer_figure("f1")]
    result = classify_page(figs, "id_vs_vgs", claimed={"f1"})
    assert result.status == ClassificationStatus.NO_MATCH


def test_classify_page_records_all_scores_for_audit():
    figs = [transfer_figure("f1"), capacitance_figure("f2")]
    result = classify_page(figs, "id_vs_vgs", claimed=set())
    ids = {fid for fid, _ in result.all_scores}
    assert ids == {"f1", "f2"}


def test_classify_page_reason_is_nonempty_string():
    figs = [transfer_figure("f1")]
    result = classify_page(figs, "id_vs_vgs", claimed=set())
    assert isinstance(result.reason, str) and result.reason


def test_unknown_curve_type_raises():
    with pytest.raises(KeyError):
        classify_page([transfer_figure("f1")], "not_a_real_type", claimed=set())


# -------------------------------------------------------------- classify_device

def test_classify_device_picks_best_across_multiple_pages():
    figures_by_page = {
        1: [capacitance_figure("f1", page=1)],
        3: [transfer_figure("f2", page=3)],
    }
    result, claimed = classify_device(figures_by_page, "id_vs_vgs")
    assert result.status == ClassificationStatus.MATCHED
    assert result.figure.figure_id == "f2"
    assert "f2" in claimed


def test_classify_device_mutual_exclusion_across_curve_types():
    figures_by_page = {
        3: [transfer_figure("f1", page=3)],
        5: [capacitance_figure("f2", page=5)],
    }
    result_a, claimed = classify_device(figures_by_page, "id_vs_vgs")
    assert result_a.status == ClassificationStatus.MATCHED
    assert result_a.figure.figure_id == "f1"

    result_b, claimed = classify_device(figures_by_page, "capacitance_vs_vds", claimed=claimed)
    assert result_b.status == ClassificationStatus.MATCHED
    assert result_b.figure.figure_id == "f2"
    assert claimed == {"f1", "f2"}


def test_classify_device_second_call_cannot_reclaim_already_claimed_figure():
    figures_by_page = {3: [transfer_figure("f1", page=3)]}
    result_a, claimed = classify_device(figures_by_page, "id_vs_vgs")
    assert result_a.status == ClassificationStatus.MATCHED

    # Same figure, different (mismatched) curve type: even ignoring the
    # claim, it should not match id_vs_vgs's figure against capacitance;
    # and since f1 is already claimed, it must not be reconsidered at all.
    result_b, claimed2 = classify_device(figures_by_page, "capacitance_vs_vds", claimed=claimed)
    assert result_b.status == ClassificationStatus.NO_MATCH
    assert claimed2 == claimed


def test_classify_device_no_figures_anywhere_returns_no_match():
    result, claimed = classify_device({}, "id_vs_vgs")
    assert result.status == ClassificationStatus.NO_MATCH
    assert claimed == set()


def test_classify_device_quarantined_result_does_not_claim():
    figures_by_page = {3: [transfer_figure("f1", page=3), transfer_figure("f2", page=3)]}
    result, claimed = classify_device(figures_by_page, "id_vs_vgs")
    assert result.status == ClassificationStatus.QUARANTINED
    assert claimed == set()


def test_classify_device_does_not_mutate_input_claimed_set():
    figures_by_page = {3: [transfer_figure("f1", page=3)]}
    original = {"unrelated"}
    result, claimed = classify_device(figures_by_page, "id_vs_vgs", claimed=original)
    assert original == {"unrelated"}
    assert claimed != original


# --------------------------------------------------------- axis-completeness

def x_only_axis_figure(figure_id, page=3):
    """Scores above MATCH_THRESHOLD on caption+x-axis alone, but has no y-axis line."""
    return FigureCandidate(
        figure_id=figure_id, page=page, figure_index=0, image_path=f"{figure_id}.png",
        caption="Fig. 3 Typical Transfer Characteristics",
        figure_width=800, figure_height=650,
        ocr_lines=[OcrLine(text="VGS, Gate-to-Source Voltage (V)", bbox=(177, 547, 495, 573))],
    )


def y_only_axis_figure(figure_id, page=3):
    """Scores above MATCH_THRESHOLD on caption+y-axis alone, but has no x-axis line."""
    return FigureCandidate(
        figure_id=figure_id, page=page, figure_index=0, image_path=f"{figure_id}.png",
        caption="Fig. 3 Typical Transfer Characteristics",
        figure_width=800, figure_height=650,
        ocr_lines=[OcrLine(text="ID, Drain-to-Source Current (A)", bbox=(25, 108, 54, 413))],
    )


def test_matched_figure_with_both_axes_present_stays_matched():
    result = classify_page([transfer_figure("f1")], "id_vs_vgs", claimed=set())
    assert result.status == ClassificationStatus.MATCHED


def test_matched_candidate_missing_x_axis_downgrades_to_quarantined():
    result = classify_page([y_only_axis_figure("f1")], "id_vs_vgs", claimed=set())
    assert result.status == ClassificationStatus.QUARANTINED
    assert "incomplete_axes" in result.reason


def test_matched_candidate_missing_y_axis_downgrades_to_quarantined():
    result = classify_page([x_only_axis_figure("f1")], "id_vs_vgs", claimed=set())
    assert result.status == ClassificationStatus.QUARANTINED
    assert "incomplete_axes" in result.reason


def test_axis_check_does_not_affect_already_quarantined_ambiguous_result():
    # Two complete-axis, similar-scoring figures -> quarantined for ambiguity,
    # not for incomplete_axes.
    figs = [transfer_figure("f1"), transfer_figure("f2")]
    result = classify_page(figs, "id_vs_vgs", claimed=set())
    assert result.status == ClassificationStatus.QUARANTINED
    assert "incomplete_axes" not in result.reason


def test_axis_check_does_not_affect_no_match_result():
    figs = [capacitance_figure("f1"), blank_figure("f2")]
    result = classify_page(figs, "id_vs_vgs", claimed=set())
    assert result.status == ClassificationStatus.NO_MATCH
    assert "incomplete_axes" not in result.reason


def test_classify_device_missing_ocr_and_bbox_does_not_crash():
    figs = {
        1: [
            FigureCandidate(figure_id="a", page=1, figure_index=0, image_path="a.png",
                             caption=None, ocr_lines=[]),
            FigureCandidate(figure_id="b", page=1, figure_index=1, image_path="b.png",
                             caption="Fig. 3 Typical Transfer Characteristics",
                             ocr_lines=[OcrLine(text="VGS", bbox=None)]),
        ]
    }
    result, claimed = classify_device(figs, "id_vs_vgs")
    assert result.status in (ClassificationStatus.MATCHED, ClassificationStatus.QUARANTINED,
                              ClassificationStatus.NO_MATCH)
