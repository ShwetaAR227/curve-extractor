"""Tests for src.classification.classify — written FIRST (CLAUDE.md §2).

Covers per-page classification (matched/quarantined/no_match), claimed-set
exclusion, cross-page device classification, and mutual exclusion across
curve types via the explicit claimed-set parameter/return value.
"""
import pytest

import src.classification.classify as classify_mod
from src.classification.classify import (
    DECISIVE_MARGIN,
    ClassificationStatus,
    classify_device,
    classify_page,
)
from src.classification.scoring import FigureCandidate, MatchedSignal, OcrLine, ScoreResult


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


def borderline_missing_x_axis_figure(figure_id, page=3):
    """A weaker match (single caption keyword + single y-axis keyword hit,
    real verified score 5.5 -- above MATCH_THRESHOLD/MATCH_MARGIN but below
    DECISIVE_MARGIN) with no x-axis line at all. Mirrors the real corpus
    shape of a genuinely borderline incomplete_axes case (2026-07-24
    real-data investigation, margin < 7.0 bucket)."""
    return FigureCandidate(
        figure_id=figure_id, page=page, figure_index=0, image_path=f"{figure_id}.png",
        caption="Fig. 3 Typical Transfer Characteristic",
        figure_width=800, figure_height=650,
        ocr_lines=[OcrLine(text="Drain Current (A)", bbox=(25, 108, 54, 413))],
    )


def test_matched_figure_with_both_axes_present_stays_matched():
    result = classify_page([transfer_figure("f1")], "id_vs_vgs", claimed=set())
    assert result.status == ClassificationStatus.MATCHED


# --------------------------------- DECISIVE_MARGIN (2026-07-24, owner-approved)
#
# Real-corpus investigation (126 real devices, every incomplete_axes downgrade
# scanned against each figure's own real printed caption): every real case at
# margin >= 7.0 was a genuine correct match (28/28); real wrong matches
# (a rdson_vs_tj / vgsth_vs_tj same-x-axis-keyword mixup) cluster exactly at
# margin 6.5 (35 cases). 7.0 is the tightest cutoff that excludes all of them
# while trusting every real decisive win. Below 7.0, the axis-completeness
# check still applies exactly as before (T14) -- a real true positive (a
# genuine "Typical Transfer Characteristics" chart, margin 6.5) and a real
# false positive (the same-margin rdson_vs_tj mixup) are NOT separable by
# margin alone at that exact value, so both correctly stay under review; this
# is a known, accepted limitation, not an oversight (see PROGRESS.md).

def test_decisive_margin_constant_is_seven():
    assert DECISIVE_MARGIN == 7.0


def test_decisive_margin_missing_y_axis_still_matches():
    # x_only_axis_figure's real score is 12.5 (verified) -- decisively
    # above DECISIVE_MARGIN. Was quarantined before this task; the axis
    # check is now skipped entirely and the match is trusted.
    result = classify_page([x_only_axis_figure("f1")], "id_vs_vgs", claimed=set())
    assert result.margin >= DECISIVE_MARGIN
    assert result.status == ClassificationStatus.MATCHED
    assert "incomplete_axes" not in result.reason


def test_decisive_margin_missing_x_axis_still_matches():
    # y_only_axis_figure's real score is 15.0 (verified) -- decisively
    # above DECISIVE_MARGIN. Was quarantined before this task; the axis
    # check is now skipped entirely and the match is trusted.
    result = classify_page([y_only_axis_figure("f1")], "id_vs_vgs", claimed=set())
    assert result.margin >= DECISIVE_MARGIN
    assert result.status == ClassificationStatus.MATCHED
    assert "incomplete_axes" not in result.reason


def test_borderline_margin_missing_x_axis_still_quarantined():
    # Real verified score/margin 5.5 -- above MATCH_THRESHOLD/MATCH_MARGIN
    # but below DECISIVE_MARGIN. The axis-completeness check must still
    # apply exactly as it did before this task (T14 behavior preserved for
    # genuinely borderline matches).
    result = classify_page([borderline_missing_x_axis_figure("f1")], "id_vs_vgs", claimed=set())
    assert result.margin < DECISIVE_MARGIN
    assert result.status == ClassificationStatus.QUARANTINED
    assert "incomplete_axes" in result.reason


def test_margin_exactly_at_decisive_threshold_skips_axis_check(monkeypatch):
    # Precise boundary test: margin == DECISIVE_MARGIN exactly (>=, same
    # inclusive-boundary style as the existing MATCH_THRESHOLD/MATCH_MARGIN
    # check) must skip the axis check, not just margins strictly above it.
    # score_figure is monkeypatched for exact numeric control, independent
    # of any particular keyword combination's real score.
    fake_result = ScoreResult(
        total_score=7.0, matched_signals=[MatchedSignal("caption_keyword", "fake", 7.0)],
    )
    monkeypatch.setattr(classify_mod, "score_figure", lambda figure, spec: fake_result)
    monkeypatch.setattr(classify_mod, "figure_has_complete_axes", lambda figure: False)
    result = classify_page([blank_figure("f1")], "id_vs_vgs", claimed=set())
    assert result.margin == DECISIVE_MARGIN
    assert result.status == ClassificationStatus.MATCHED
    assert "incomplete_axes" not in result.reason


def test_margin_just_below_decisive_threshold_still_applies_axis_check(monkeypatch):
    fake_result = ScoreResult(
        total_score=6.99, matched_signals=[MatchedSignal("caption_keyword", "fake", 6.99)],
    )
    monkeypatch.setattr(classify_mod, "score_figure", lambda figure, spec: fake_result)
    monkeypatch.setattr(classify_mod, "figure_has_complete_axes", lambda figure: False)
    result = classify_page([blank_figure("f1")], "id_vs_vgs", claimed=set())
    assert result.margin < DECISIVE_MARGIN
    assert result.status == ClassificationStatus.QUARANTINED
    assert "incomplete_axes" in result.reason


def test_decisive_margin_does_not_rescue_an_ambiguous_quarantine():
    # DECISIVE_MARGIN only ever short-circuits the axis-completeness check.
    # It must never rescue a result that's quarantined for the ORIGINAL
    # reason (two similar-scoring candidates -> margin below MATCH_MARGIN
    # in the first place) -- there's no "decisive" win to trust here at all.
    figs = [transfer_figure("f1"), transfer_figure("f2")]
    result = classify_page(figs, "id_vs_vgs", claimed=set())
    assert result.status == ClassificationStatus.QUARANTINED
    assert "incomplete_axes" not in result.reason


def test_low_margin_complete_axes_figure_unaffected_by_decisive_margin():
    # A low-margin match that HAS complete axes must stay matched
    # regardless of DECISIVE_MARGIN -- the new condition only ever adds
    # extra protection for the axis check, never a new way to fail.
    # No caption match at all, one weak keyword hit per axis -> real
    # verified score/margin 5.0 (above MATCH_THRESHOLD, well below
    # DECISIVE_MARGIN), with both zones present.
    complete = FigureCandidate(
        figure_id="f1", page=3, figure_index=0, image_path="f1.png",
        caption=None, figure_width=800, figure_height=650,
        ocr_lines=[
            OcrLine(text="ID,", bbox=(25, 108, 54, 413)),
            OcrLine(text="VGS", bbox=(177, 547, 495, 573)),
        ],
    )
    result = classify_page([complete], "id_vs_vgs", claimed=set())
    assert result.margin < DECISIVE_MARGIN  # sanity: genuinely borderline
    assert result.status == ClassificationStatus.MATCHED


# --------------------- real-corpus-shaped regression pins (2026-07-24)

def real_gate_threshold_mismatch_figure(figure_id, page=6):
    """Mirrors the REAL rdson_vs_tj false-positive pattern found in the
    corpus scan: a genuine 'Gate Threshold Voltage vs. Junction
    Temperature' chart (NOT an on-resistance chart) wrongly scores >=
    MATCH_THRESHOLD against rdson_vs_tj purely from shared x-axis
    ('junction temperature'/'Tj [') text -- real verified margin 6.5,
    missing the y-axis line entirely (same OCR-crop pattern the axis
    check already flags). Must stay quarantined -- DECISIVE_MARGIN must
    NOT rescue this real wrong match."""
    return FigureCandidate(
        figure_id=figure_id, page=page, figure_index=0, image_path=f"{figure_id}.png",
        caption="Fig.9 Gate Threshold Voltage vs. Junction Temperature",
        figure_width=800, figure_height=650,
        ocr_lines=[OcrLine(text="Junction Temperature : Tj [°C]", bbox=(177, 620, 495, 645))],
    )


def test_real_rdson_gate_threshold_mismatch_stays_quarantined():
    result = classify_page(
        [real_gate_threshold_mismatch_figure("f1")], "rdson_vs_tj", claimed=set())
    assert result.margin == pytest.approx(6.5)
    assert result.margin < DECISIVE_MARGIN
    assert result.status == ClassificationStatus.QUARANTINED
    assert "incomplete_axes" in result.reason


def test_matched_candidate_missing_x_axis_downgrades_to_quarantined():
    # Original T14 test, kept: outcome (QUARANTINED) is unchanged by
    # DECISIVE_MARGIN -- only the fixture changed, from y_only_axis_figure
    # (real margin 12.5, now decisive -- see
    # test_decisive_margin_missing_x_axis_still_matches) to
    # borderline_missing_x_axis_figure (real margin 5.5, genuinely still
    # borderline), so this stays a true regression pin of the ORIGINAL
    # T14 behavior for a case where it should still apply.
    result = classify_page([borderline_missing_x_axis_figure("f1")], "id_vs_vgs", claimed=set())
    assert result.status == ClassificationStatus.QUARANTINED
    assert "incomplete_axes" in result.reason


# test_matched_candidate_missing_y_axis_downgrades_to_quarantined (original
# T14 test, used x_only_axis_figure) retired, 2026-07-24: its fixture's
# real margin (12.5) is decisive under the new, owner-approved
# DECISIVE_MARGIN logic, so the outcome it pinned (QUARANTINED) is no
# longer correct -- x_only_axis_figure now genuinely SHOULD stay MATCHED.
# Superseded by test_decisive_margin_missing_y_axis_still_matches above,
# which covers the identical fixture under its correct new name/outcome
# rather than keeping a duplicate under a now-misleading name.


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


# ------------------- cross-page best-overall selection (2026-07-24) --------
#
# Real-corpus investigation (RD3G08CBKHRBTL / id_vs_vgs, owner-approved fix):
# classify_device used to exhaust every MATCHED page result before EVER
# looking at a QUARANTINED one, even when a quarantined candidate on another
# page scored higher. Fix: gather every real candidate (MATCHED +
# QUARANTINED) across every page into one pool and pick the single
# highest-scoring one overall. Ties (exact same top score) are broken in
# favor of a MATCHED ("cleanly passed") candidate over a QUARANTINED ("held
# for review") one -- owner-approved rule. Critically, the WINNING
# candidate's own status is preserved: a QUARANTINED winner still comes back
# QUARANTINED (and does not claim its figure) -- this fixes WHICH figure is
# shown for review, not whether review is skipped.

def real_id_vs_vgs_page5_weak_match(figure_id="p5fig"):
    """Real RD3G08CBKHRBTL page 5 shape: no caption on this exact figure,
    weak axis-only hits (real verified score/margin 5.0, MATCHED, complete
    axes) -- today's wrong answer for id_vs_vgs on this device."""
    return FigureCandidate(
        figure_id=figure_id, page=5, figure_index=0, image_path=f"{figure_id}.png",
        caption=None, figure_width=800, figure_height=650,
        ocr_lines=[
            OcrLine(text="ID,", bbox=(25, 108, 54, 413)),
            OcrLine(text="VGS", bbox=(177, 547, 495, 573)),
        ],
    )


def real_id_vs_vgs_page6_transfer_chart(figure_id="p6fig"):
    """Real RD3G08CBKHRBTL page 6 shape: real caption 'Fig.8 Typical
    Transfer Characteristics' + one axis-zone-unknown y-signal (real
    verified score 7.0) -- missing its x-axis OCR line entirely (a real,
    separate Stage-3 gap), so it fails figure_has_complete_axes and gets
    quarantined even though it out-scores page 5. This IS the correct
    chart for id_vs_vgs on this device."""
    return FigureCandidate(
        figure_id=figure_id, page=6, figure_index=0, image_path=f"{figure_id}.png",
        caption="Fig.8 Typical Transfer Characteristics",
        figure_width=800, figure_height=650,
        ocr_lines=[OcrLine(text="Drain Current : ID [A]", bbox=None)],
    )


def real_id_vs_vgs_page6_runner_up(figure_id="p6fig2"):
    """Real RD3G08CBKHRBTL page 6 neighbor: 'Fig.10 Forward Transfer
    Admittance vs. Drain Current' -- a weak same-page competitor (real
    verified score 0.5) that brings the page-6 winner's margin down to
    6.5, keeping it genuinely borderline (below DECISIVE_MARGIN) rather
    than decisive."""
    return FigureCandidate(
        figure_id=figure_id, page=6, figure_index=1, image_path=f"{figure_id}.png",
        caption="Fig.10 Forward Transfer Admittance vs. Drain Current",
        figure_width=800, figure_height=650,
        ocr_lines=[OcrLine(text="Drain Current : ID [A]", bbox=(177, 547, 495, 573))],
    )


def test_real_rd3g08cbkhrbtl_id_vs_vgs_page6_quarantined_beats_page5_matched():
    figures_by_page = {
        5: [real_id_vs_vgs_page5_weak_match()],
        6: [real_id_vs_vgs_page6_transfer_chart(), real_id_vs_vgs_page6_runner_up()],
    }
    # Sanity: pin the real per-page numbers this test depends on, so a
    # future scoring-weight change fails loudly here instead of silently
    # changing what this test is actually proving.
    page5_result = classify_page(figures_by_page[5], "id_vs_vgs", claimed=set())
    assert (page5_result.status, page5_result.score) == (ClassificationStatus.MATCHED, 5.0)
    page6_result = classify_page(figures_by_page[6], "id_vs_vgs", claimed=set())
    assert (page6_result.status, page6_result.score, page6_result.margin) == (
        ClassificationStatus.QUARANTINED, 7.0, 6.5,
    )

    result, claimed = classify_device(figures_by_page, "id_vs_vgs")

    # The higher-scoring page-6 candidate wins the comparison...
    assert result.figure.figure_id == "p6fig"
    assert result.score == 7.0
    # ...but its status is NOT silently upgraded just because it won: it
    # was quarantined on its own page, so the final result stays
    # quarantined (shown for human review), and does not claim its figure.
    assert result.status == ClassificationStatus.QUARANTINED
    assert claimed == set()


def real_vgsth_tie_page4_power_dissipation(figure_id="p4fig"):
    """Real RS6G122CHTB1 page 4: 'Fig.1 Power Dissipation Derating Curve'
    -- WRONG chart for vgsth_vs_tj. Matches only on the shared
    'Junction Temperature : Tj [...]' x-axis wording every chart on this
    device's temperature-axis pages uses (real verified score 5.0,
    quarantined for incomplete axes -- no y-axis line at all)."""
    return FigureCandidate(
        figure_id=figure_id, page=4, figure_index=0, image_path=f"{figure_id}.png",
        caption="Fig.1 Power Dissipation Derating Curve",
        figure_width=709, figure_height=686,
        ocr_lines=[OcrLine(text="Junction Temperature: Tj [ºC]", bbox=(224.0, 631.0, 609.0, 666.0))],
    )


def real_vgsth_tie_page5_breakdown_voltage(figure_id="p5fig"):
    """Real RS6G122CHTB1 page 5: 'Fig.7 Normalized Breakdown Voltage vs.
    Junction Temperature' -- WRONG chart for vgsth_vs_tj, but this one
    has a real (unrelated-content but geometrically present) y-axis
    label line too, so it has complete axes and cleanly MATCHES (real
    verified score 5.0) -- today's wrong answer for this device."""
    return FigureCandidate(
        figure_id=figure_id, page=5, figure_index=0, image_path=f"{figure_id}.png",
        caption="Fig.7 Normalized Breakdown Voltage vs. Junction Temperature",
        figure_width=701, figure_height=688,
        ocr_lines=[
            OcrLine(text="Junction Temperature : Tj [ºC]", bbox=(222.0, 640.0, 609.0, 674.0)),
            OcrLine(text="Normalized Breakdown Voltage : V(BR)DSS", bbox=(21.0, 46.0, 58.0, 580.0)),
        ],
    )


def real_vgsth_tie_page6_gate_threshold(figure_id="p6fig"):
    """Real RS6G122CHTB1 page 6: 'Fig.9 Gate Threshold Voltage vs.
    Junction Temperature' -- the ACTUAL correct chart for vgsth_vs_tj.
    Same x-axis-only shape as page 4 (real verified score 5.0,
    quarantined for incomplete axes), so it ties with the two wrong
    charts above rather than winning outright."""
    return FigureCandidate(
        figure_id=figure_id, page=6, figure_index=0, image_path=f"{figure_id}.png",
        caption="Fig.9 Gate Threshold Voltage vs. Junction Temperature",
        figure_width=708, figure_height=691,
        ocr_lines=[OcrLine(text="Junction Temperature : Tj [C]", bbox=(226.0, 639.0, 611.0, 672.0))],
    )


def test_real_rs6g122chtb1_vgsth_tie_breaks_toward_cleanly_passed_not_correct_chart():
    # This is the exact real 3-way tie found in investigation: all three
    # candidates score 5.0 for vgsth_vs_tj, one of the QUARANTINED ones
    # (page 6) is the actually-correct chart. The owner-agreed tie-break
    # rule prefers a "cleanly passed" (MATCHED) candidate over a
    # "held for review" (QUARANTINED) one on an EXACT tie -- so this test
    # intentionally confirms the MATCHED-but-wrong page 5 chart still wins
    # the comparison, exactly like today. This is not the "smartest"
    # possible outcome; it's the specific, deliberate rule that was agreed
    # instead of an arbitrary tie-break.
    figures_by_page = {
        4: [real_vgsth_tie_page4_power_dissipation()],
        5: [real_vgsth_tie_page5_breakdown_voltage()],
        6: [real_vgsth_tie_page6_gate_threshold()],
    }
    for page in (4, 5, 6):
        r = classify_page(figures_by_page[page], "vgsth_vs_tj", claimed=set())
        assert r.score == 5.0, f"page {page} real score drifted from 5.0 -- fixture no longer reproduces the tie"
    assert classify_page(figures_by_page[4], "vgsth_vs_tj", claimed=set()).status == ClassificationStatus.QUARANTINED
    assert classify_page(figures_by_page[5], "vgsth_vs_tj", claimed=set()).status == ClassificationStatus.MATCHED
    assert classify_page(figures_by_page[6], "vgsth_vs_tj", claimed=set()).status == ClassificationStatus.QUARANTINED

    result, claimed = classify_device(figures_by_page, "vgsth_vs_tj")

    assert result.figure.figure_id == "p5fig"
    assert result.status == ClassificationStatus.MATCHED
    assert claimed == {"p5fig"}


def real_rdson_no_change_page4(figure_id="p4fig"):
    """Real RD3G08CBKHRBTL page 4 rdson_vs_tj candidate: "Fig.1 Power
    Dissipation Derating Curve". Originally scored 6.5/MATCHED when this
    control test was written (2026-07-24, T36). A LATER same-day fix
    (T37: the "power dissipation" negative-phrase guard, added for an
    unrelated real bug on a different device, RS6G100BGTB1) legitimately
    dropped this real chart's own score to 3.5/QUARANTINED -- this
    fixture's expected score was updated to match, confirmed against the
    real corpus (see PROGRESS.md T37) that the device-level winner (page
    7, the true chart) is unaffected either way."""
    return FigureCandidate(
        figure_id=figure_id, page=4, figure_index=0, image_path=f"{figure_id}.png",
        caption="Fig.1 Power Dissipation Derating Curve",
        figure_width=709, figure_height=686,
        ocr_lines=[
            OcrLine(text="Power Dissipation : PD/Ppmax.[%]", bbox=(22.0, 82.0, 58.0, 524.0)),
            OcrLine(text="Junction Temperature : Tj [ºC]", bbox=(224.0, 631.0, 609.0, 666.0)),
        ],
    )


def real_rdson_no_change_page5(figure_id="p5fig"):
    """Real RD3G08CBKHRBTL page 5 rdson_vs_tj candidate (real verified
    score 7.5, MATCHED, complete axes)."""
    return FigureCandidate(
        figure_id=figure_id, page=5, figure_index=0, image_path=f"{figure_id}.png",
        caption="Fig.7 Normalized Breakdown Voltage vs. Junction Temperature",
        figure_width=701, figure_height=688,
        ocr_lines=[
            OcrLine(text="Normalized Breakdown Voltage : V(BR)DSS", bbox=(21.0, 46.0, 58.0, 580.0)),
            OcrLine(text="Junction Temperature : Tj [ºC]", bbox=(222.0, 640.0, 609.0, 674.0)),
        ],
    )


def real_rdson_no_change_page6(figure_id="p6fig"):
    """Real RD3G08CBKHRBTL page 6 rdson_vs_tj candidate (real verified
    score 6.5, MATCHED, complete axes)."""
    return FigureCandidate(
        figure_id=figure_id, page=6, figure_index=0, image_path=f"{figure_id}.png",
        caption="Fig.9 Gate Threshold Voltage vs. Junction Temperature",
        figure_width=708, figure_height=691,
        ocr_lines=[
            OcrLine(text="Gate Threshold Voltage : VGS(th) [V]", bbox=(23.0, 84.0, 61.0, 537.0)),
            OcrLine(text="Junction Temperature : Tj [C]", bbox=(226.0, 639.0, 611.0, 672.0)),
        ],
    )


def real_rdson_no_change_page7_correct(figure_id="p7fig"):
    """Real RD3G08CBKHRBTL page 7 rdson_vs_tj candidate -- the actual
    correct on-resistance chart (real verified score 9.0, MATCHED,
    complete axes). Already wins today (highest score, and today's
    matched-only tier has no quarantined competitor to worry about) --
    the fix must leave this exact outcome unchanged."""
    return FigureCandidate(
        figure_id=figure_id, page=7, figure_index=0, image_path=f"{figure_id}.png",
        caption="Fig.13 Static Drain - Source On - State Resistance vs. Junction Temperature",
        figure_width=716, figure_height=679,
        ocr_lines=[
            OcrLine(text="Static Drain - Source On-State Resistance : RDS(on) [m2]", bbox=(29.0, 36.0, 59.0, 579.0)),
            OcrLine(text="Junction Temperature : Tj [ºC]", bbox=(232.0, 635.0, 618.0, 668.0)),
        ],
    )


def test_real_rd3g08cbkhrbtl_rdson_vs_tj_all_matched_unaffected_by_fix():
    # Control case, as requested: confirms the T36 cross-page selection
    # fix changes nothing about this device's rdson_vs_tj winner. Page 4's
    # own status/score (QUARANTINED, 3.5) reflects the LATER T37
    # power-dissipation guard, not the T36 fix this test originally
    # targeted -- see real_rdson_no_change_page4's docstring. Either way,
    # the point of this test holds: the real correct chart (page 7) wins
    # outright, untouched by either fix.
    figures_by_page = {
        4: [real_rdson_no_change_page4()],
        5: [real_rdson_no_change_page5()],
        6: [real_rdson_no_change_page6()],
        7: [real_rdson_no_change_page7_correct()],
    }
    expected = {
        4: (ClassificationStatus.QUARANTINED, 3.5),
        5: (ClassificationStatus.MATCHED, 7.5),
        6: (ClassificationStatus.MATCHED, 6.5),
        7: (ClassificationStatus.MATCHED, 9.0),
    }
    for page, (expected_status, expected_score) in expected.items():
        r = classify_page(figures_by_page[page], "rdson_vs_tj", claimed=set())
        assert (r.status, r.score) == (expected_status, expected_score)

    result, claimed = classify_device(figures_by_page, "rdson_vs_tj")

    assert result.figure.figure_id == "p7fig"
    assert result.status == ClassificationStatus.MATCHED
    assert result.score == 9.0
    assert claimed == {"p7fig"}


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
