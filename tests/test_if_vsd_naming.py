"""Tests for src.extraction.naming.if_vs_vsd — written FIRST (CLAUDE.md §2,
red phase). RED PHASE ONLY — the module under test does not exist yet.

if_vs_vsd (body_diode) charts plot forward/reverse-diode current (I_F/I_SD)
vs. source-drain voltage (V_SD), one curve per junction temperature the
measurement was taken at ("25°C", "175°C", ...).

``count_expected_curves(ocr_lines)`` reads a chart's own temperature labels
up front to predict its curve count — same role as
``naming.vgsth_vs_tj.count_expected_curves``, but a single scheme (bare
"25°C" or prefixed "TJ = 25°C"), not two. Never guesses: returns ``None``
whenever the label set doesn't resolve to exactly one safe count — no
labels, a duplicate normalized value (owner's DUPLICATE-VALUE RULE, same
unconditional rule as vgsth_vs_tj), or a COMPOUND label (temperature +
percentile, e.g. "150°C, 98%" — seen once in the reviewed corpus, not
parsed, always ambiguous for now per owner instruction).

``name_curves_by_labels(curves, ocr_lines)`` mirrors vgsth_vs_tj's function
of the same name: 1 curve always names ``["if"]``; multi-curve names by
temperature (``if_25C``, ``if_175C``, ...), every curve independently
resolved or the whole result is ``None`` (no elimination-completion).

CORE DESIGN POINT under test — position matching uses ONLY each curve's
LOW-V_SD-region point subset (roughly the first 25% of its own column
span), NEVER a whole-curve average or whole-curve nearest-point search.
Every real if_vs_vsd chart reviewed converges/crosses at high current
(forward voltage becomes less temperature-dependent as current rises), so
a curve's high-V_SD points can end up geometrically close to a label
that's actually anchored to a DIFFERENT curve's low-V_SD segment — see
``TestNameCurvesLowVsdRegionRestriction`` below, which hand-constructs an
exact adversarial case where restricting to the low-V_SD region gives a
different (correct) answer than an unrestricted whole-curve nearest-point
search would.

All fixtures are synthetic point lists / OCR-line dicts (no images, no
GPU, no network — CLAUDE.md §2). The ``raw_ocr`` helper is imported
read-only from tests.test_rdson_two_curve (same helper vgsth_vs_tj's own
test file already reuses), never duplicated (CLAUDE.md §3).
"""
import pytest

from tests.test_rdson_two_curve import raw_ocr

from src.extraction.naming.if_vs_vsd import (
    count_expected_curves,
    name_curves_by_labels,
)


# =================================================================
# A. count_expected_curves — clean single-value labels
# =================================================================

class TestCountExpectedCurvesCleanLabels:
    def test_two_distinct_bare_temp_labels_returns_2(self):
        lines = [raw_ocr("25°C", 0, 0, 10, 10), raw_ocr("175°C", 0, 0, 10, 10)]
        assert count_expected_curves(lines) == 2

    def test_four_distinct_bare_temp_labels_returns_4(self):
        lines = [raw_ocr("-40°C", 0, 0, 10, 10), raw_ocr("25°C", 0, 0, 10, 10),
                 raw_ocr("125°C", 0, 0, 10, 10), raw_ocr("175°C", 0, 0, 10, 10)]
        assert count_expected_curves(lines) == 4

    def test_prefixed_tj_form_parses(self):
        lines = [raw_ocr("TJ = 25°C", 0, 0, 10, 10), raw_ocr("TJ = 175°C", 0, 0, 10, 10)]
        assert count_expected_curves(lines) == 2

    def test_mixed_bare_and_prefixed_forms_count_correctly(self):
        lines = [raw_ocr("Tj=25°C", 0, 0, 10, 10), raw_ocr("175°C", 0, 0, 10, 10)]
        assert count_expected_curves(lines) == 2

    def test_degree_sign_and_spacing_variants_normalize(self):
        variants = [
            [raw_ocr("25°C", 0, 0, 10, 10), raw_ocr("175 °C", 0, 0, 10, 10)],
            [raw_ocr("25ºC", 0, 0, 10, 10), raw_ocr("175 º C", 0, 0, 10, 10)],
            [raw_ocr("25 degC", 0, 0, 10, 10), raw_ocr("175 deg. C", 0, 0, 10, 10)],
        ]
        for lines in variants:
            assert count_expected_curves(lines) == 2


# =================================================================
# B. count_expected_curves — duplicate-value rule
# =================================================================

class TestCountExpectedCurvesDuplicateRule:
    def test_duplicate_identical_temp_labels_returns_none(self):
        lines = [raw_ocr("25°C", 0, 0, 10, 10), raw_ocr("25°C", 0, 0, 10, 10)]
        assert count_expected_curves(lines) is None

    def test_same_value_different_notation_is_still_a_duplicate(self):
        # "25.0°C" and "25°C" normalize to the identical value -- a
        # duplicate, ambiguous, even though these are the entire label set.
        lines = [raw_ocr("25.0°C", 0, 0, 10, 10), raw_ocr("25°C", 0, 0, 10, 10)]
        assert count_expected_curves(lines) is None

    def test_bare_and_prefixed_same_value_is_still_a_duplicate(self):
        lines = [raw_ocr("TJ = 25°C", 0, 0, 10, 10), raw_ocr("25°C", 0, 0, 10, 10)]
        assert count_expected_curves(lines) is None

    def test_two_duplicates_amid_other_distinct_values_still_none(self):
        # 175°C is distinct; the two 25°C claims are the duplicate -- the
        # unconditional rule fires regardless of what else is present.
        lines = [raw_ocr("25°C", 0, 0, 10, 10), raw_ocr("175°C", 0, 0, 10, 10),
                 raw_ocr("25°C", 0, 0, 10, 10)]
        assert count_expected_curves(lines) is None


# =================================================================
# C. count_expected_curves — compound label (owner rule: never parsed)
# =================================================================

class TestCountExpectedCurvesCompoundLabel:
    def test_compound_temp_and_percentile_label_returns_none(self):
        lines = [raw_ocr("150°C, 98%", 0, 0, 10, 10)]
        assert count_expected_curves(lines) is None

    def test_compound_label_alongside_clean_distinct_labels_still_none(self):
        # Even with an otherwise-clean, resolvable set, a compound label
        # anywhere in the OCR lines makes the whole chart ambiguous --
        # never partially parsed, never silently ignored.
        lines = [raw_ocr("25°C", 0, 0, 10, 10), raw_ocr("150°C, 98%", 0, 0, 10, 10)]
        assert count_expected_curves(lines) is None


# =================================================================
# D. count_expected_curves — ambiguous/edge cases
# =================================================================

class TestCountExpectedCurvesAmbiguous:
    def test_no_labels_at_all_returns_none(self):
        assert count_expected_curves([]) is None

    def test_stray_unrelated_text_alongside_valid_labels_is_ignored(self):
        lines = [raw_ocr("IF vs VSD", 0, 0, 10, 10),
                 raw_ocr("25°C", 0, 0, 10, 10), raw_ocr("175°C", 0, 0, 10, 10)]
        assert count_expected_curves(lines) == 2

    def test_malformed_ocr_line_missing_bounding_box_raises(self):
        lines = [{"text": "25°C"}]  # no "bounding_box" key
        with pytest.raises((KeyError, ValueError)):
            count_expected_curves(lines)


# =================================================================
# E. name_curves_by_labels — single curve
# =================================================================

class TestNameCurvesSingleCurve:
    def test_one_curve_no_labels_returns_if(self):
        curve = [(100.0, 60.0), (100.0, 300.0)]
        assert name_curves_by_labels([curve], []) == ["if"]

    def test_one_curve_with_a_label_present_still_returns_if(self):
        curve = [(100.0, 60.0), (100.0, 300.0)]
        assert name_curves_by_labels([curve], [raw_ocr("25°C", 55, 90, 75, 110)]) == ["if"]


# =================================================================
# F. name_curves_by_labels — basic multi-curve happy path (no crossing)
# =================================================================

CURVE_TOP = [(60.0, 60.0), (60.0, 130.0), (60.0, 300.0)]     # flat, well up
CURVE_BOT = [(220.0, 60.0), (220.0, 130.0), (220.0, 300.0)]  # flat, well down


class TestNameCurvesBasicHappyPath:
    def test_two_curves_two_distinct_temps_resolve_by_proximity(self):
        lines = [raw_ocr("25°C", 55, 210, 75, 230), raw_ocr("175°C", 55, 50, 75, 70)]
        names = name_curves_by_labels([CURVE_BOT, CURVE_TOP], lines)
        assert names == ["if_25C", "if_175C"]

    def test_names_align_to_input_order_not_sorted_order(self):
        lines = [raw_ocr("25°C", 55, 210, 75, 230), raw_ocr("175°C", 55, 50, 75, 70)]
        names = name_curves_by_labels([CURVE_TOP, CURVE_BOT], lines)
        assert names == ["if_175C", "if_25C"]

    def test_four_curves_four_distinct_temps(self):
        curve_a = [(30.0, 60.0), (30.0, 130.0), (30.0, 300.0)]
        curve_b = CURVE_TOP
        curve_c = [(140.0, 60.0), (140.0, 130.0), (140.0, 300.0)]
        curve_d = CURVE_BOT
        lines = [
            raw_ocr("-40°C", 55, 20, 75, 40),
            raw_ocr("25°C", 55, 50, 75, 70),
            raw_ocr("125°C", 55, 130, 75, 150),
            raw_ocr("175°C", 55, 210, 75, 230),
        ]
        names = name_curves_by_labels([curve_a, curve_b, curve_c, curve_d], lines)
        assert names == ["if_-40C", "if_25C", "if_125C", "if_175C"]

    def test_prefixed_tj_form_names_the_same_as_bare_form(self):
        lines = [raw_ocr("TJ = 25°C", 55, 210, 75, 230), raw_ocr("TJ = 175°C", 55, 50, 75, 70)]
        names = name_curves_by_labels([CURVE_BOT, CURVE_TOP], lines)
        assert names == ["if_25C", "if_175C"]

    def test_normalization_applied_before_naming_not_after(self):
        # The label says "175.0°C"; the resulting name must use the
        # NORMALIZED value ("175C"), never the raw text verbatim.
        lines = [raw_ocr("25°C", 55, 210, 75, 230), raw_ocr("175.0°C", 55, 50, 75, 70)]
        names = name_curves_by_labels([CURVE_BOT, CURVE_TOP], lines)
        assert names == ["if_25C", "if_175C"]
        assert "175.0" not in "".join(names)


# =================================================================
# G. name_curves_by_labels — LOW-V_SD-REGION-ONLY matching (core design point)
# =================================================================

class TestNameCurvesLowVsdRegionRestriction:
    # Two curves that are cleanly separated in their own low-V_SD region
    # (cols 60..132) but CROSS by high V_SD (cols 148..348) -- exact,
    # hand-picked coordinates (no interpolation) so the two possible
    # verdicts (restricted vs. unrestricted nearest-point search) are
    # deterministic, not dependent on floating-point/discretization noise.
    #
    # CURVE_A: starts HIGH row (low on the chart) at low V_SD, ends LOW row
    # (high on the chart) at high V_SD -- descends across the chart.
    CURVE_A = [(200.0, 60.0), (185.0, 90.0), (163.0, 132.0),
               (152.0, 148.0), (140.0, 180.0), (120.0, 250.0), (51.0, 348.0)]
    # CURVE_B: starts LOW row (high on the chart) at low V_SD, ends HIGH
    # row (low on the chart) at high V_SD -- the mirror image, crossing A.
    CURVE_B = [(110.0, 60.0), (130.0, 90.0), (157.0, 132.0),
               (163.0, 148.0), (200.0, 180.0), (250.0, 250.0), (298.0, 348.0)]

    def test_label_anchored_via_low_region_despite_high_vsd_convergence(self):
        # "25C" sits unambiguously near CURVE_A's own low-V_SD trace.
        # "175C" sits at (cx=150, cy=150) -- verified by hand (see module
        # docstring/session notes): restricted to each curve's low-V_SD
        # region (cols <= 132), CURVE_B's nearest point (157,132) is
        # distance ~19.3 vs. CURVE_A's nearest (163,132) at ~22.2 -- B
        # wins. But an UNRESTRICTED whole-curve search would instead find
        # CURVE_A's high-V_SD point (152,148) at distance ~2.8 -- much
        # closer than CURVE_B's whole-curve nearest (163,148) at ~13.15 --
        # A would incorrectly win. The correct, low-V_SD-anchored answer
        # is CURVE_B; this is only reachable by restricting the search.
        lines = [raw_ocr("25°C", 68, 190, 82, 200), raw_ocr("175°C", 140, 140, 160, 160)]
        names = name_curves_by_labels([self.CURVE_A, self.CURVE_B], lines)
        assert names == ["if_25C", "if_175C"]

    def test_restriction_holds_regardless_of_input_curve_order(self):
        lines = [raw_ocr("25°C", 68, 190, 82, 200), raw_ocr("175°C", 140, 140, 160, 160)]
        names = name_curves_by_labels([self.CURVE_B, self.CURVE_A], lines)
        assert names == ["if_175C", "if_25C"]


# =================================================================
# H. name_curves_by_labels — quarantine cases
# =================================================================

class TestNameCurvesQuarantine:
    def test_curve_count_exceeds_resolved_label_count_returns_none(self):
        # 3 curves but only 2 labels resolve -- no elimination-completion.
        curve_mid = [(140.0, 60.0), (140.0, 130.0), (140.0, 300.0)]
        lines = [raw_ocr("25°C", 55, 210, 75, 230), raw_ocr("175°C", 55, 50, 75, 70)]
        names = name_curves_by_labels([CURVE_BOT, curve_mid, CURVE_TOP], lines)
        assert names is None

    def test_genuine_nearest_curve_tie_returns_none(self):
        curve_a = [(100.0, 60.0), (100.0, 300.0)]
        curve_b = [(140.0, 60.0), (140.0, 300.0)]
        tied_label = raw_ocr("25°C", 170, 118, 190, 122)  # bbox center row=120, exactly midway
        assert name_curves_by_labels([curve_a, curve_b], [tied_label]) is None

    def test_no_labels_at_all_multi_curve_returns_none(self):
        assert name_curves_by_labels([CURVE_TOP, CURVE_BOT], []) is None

    def test_two_curves_resolving_to_identical_names_returns_none(self):
        # Two SEPARATE "25°C" labels, each nearest a different curve --
        # naming both curves "if_25C" would collide.
        lines = [raw_ocr("25°C", 55, 210, 75, 230), raw_ocr("25°C", 55, 50, 75, 70)]
        names = name_curves_by_labels([CURVE_BOT, CURVE_TOP], lines)
        assert names is None

    def test_compound_label_present_during_naming_returns_none(self):
        lines = [raw_ocr("25°C", 55, 210, 75, 230), raw_ocr("175°C, 98%", 55, 50, 75, 70)]
        assert name_curves_by_labels([CURVE_BOT, CURVE_TOP], lines) is None

    def test_partial_resolution_one_of_three_missing_returns_none(self):
        curve_mid = [(140.0, 60.0), (140.0, 130.0), (140.0, 300.0)]
        lines = [raw_ocr("175°C", 55, 50, 75, 70)]  # only 1 of 3 labels present
        assert name_curves_by_labels([CURVE_BOT, curve_mid, CURVE_TOP], lines) is None


# =================================================================
# I. Input validation
# =================================================================

class TestNameCurvesInputValidation:
    def test_empty_curves_list_raises_value_error(self):
        with pytest.raises(ValueError):
            name_curves_by_labels([], [])

    def test_curve_with_no_points_raises_value_error(self):
        with pytest.raises(ValueError):
            name_curves_by_labels([CURVE_TOP, []], [])
