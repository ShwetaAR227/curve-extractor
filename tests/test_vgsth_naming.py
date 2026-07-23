"""Tests for src.extraction.naming.vgsth_vs_tj — written FIRST (CLAUDE.md §2,
red phase). RED PHASE ONLY — the module under test does not exist yet.

vgsth_vs_tj charts plot gate-threshold voltage vs. junction temperature,
distinguished by which bias current the measurement was taken at. Two real
label vocabularies are anticipated (owner-specified, 2026-07-21):

- BAND scheme: curves labeled "max"/"typ"/"min" (or the percentile variant
  "98%"/"typ"/"2%", extending rdson_vs_tj's own "98%"==max convention to
  add a "2%"==min counterpart) — 1, 2, or 3 curves.
- CURRENT-VALUE scheme: curves labeled by their bias current directly
  ("I_D = 250uA", "I_D = 1.0mA", ...) — any curve count, disambiguated by
  the label's own (normalized) numeric value rather than a fixed vocabulary.

``count_expected_curves(ocr_lines)`` reads a chart's labels UP FRONT (before
detection/naming) to predict how many curves it should have — used
elsewhere as the expected-count gate, same role as rdson's fixed
``EXPECTED_CURVE_COUNT``/``TWO_CURVE_COUNT`` but data-driven per chart
instead of a constant, since vgsth's curve count isn't fixed. Never
guesses: returns ``None`` whenever the label set doesn't resolve to exactly
one safe count.

``name_curves_by_labels(curves, ocr_lines)`` mirrors rdson_vs_tj's function
of the same name (proximity-anchored labels, position never overrides
labels, contradictions -> ``None``) but generalized: variable curve count
(1-3 for band scheme, unbounded for current-value scheme), and a second
naming scheme entirely.

DUPLICATE-VALUE RULE (owner-decided, 2026-07-21): duplicate normalized
values are ALWAYS ambiguous -> None. This applies uniformly to BOTH
schemes, with no special-casing: a repeated band word (e.g. two "typ"
tokens, C.12) and two current-value labels that normalize to the identical
value (e.g. "1.0mA" and "1000uA", B.7) are treated exactly the same way —
a duplicate is never silently trusted as harmless redundant notation,
regardless of whether anything else is present on the chart. (An earlier
draft of this file special-cased "the whole label set agrees" as
unambiguous; the owner rejected that — corrected here.)

KNOWN, DELIBERATELY OUT-OF-SCOPE LIMITATION: a duplicate could legitimately
arise from OCR re-detecting the same physical text region twice (a real,
previously-observed artifact in this codebase — see T27's monochrome
"mask-fragmentation" pattern, a different mechanism but the same underlying
"OCR/detection sees one real thing twice" class of problem) rather than
from two genuinely distinct curves. Telling "OCR noise" apart from "a real
ambiguous collision" would need label-POSITION reasoning (e.g., are the two
duplicate labels' bounding boxes suspiciously close together, suggesting
one physical annotation detected twice, vs. far apart near different
curves?) — this module does not attempt that; it always quarantines on any
duplicate rather than guessing which case it is. Revisit only if this
proves too conservative on real data.

All fixtures are synthetic point lists / OCR-line dicts (no images, no
GPU, no network — CLAUDE.md §2). Curve/OCR-line construction helpers
(``line_trace``, ``raw_ocr``) are imported read-only from
tests.test_rdson_two_curve rather than duplicated (CLAUDE.md §3).
"""
import pytest

from tests.test_rdson_two_curve import line_trace, raw_ocr

from src.extraction.naming.vgsth_vs_tj import (
    count_expected_curves,
    name_curves_by_labels,
)

# ---------------------------------------------------------------- fixtures
#
# Three well-separated synthetic curves (rows chosen so "top"/"mid"/"bot"
# never overlap and proximity to a nearby label is unambiguous unless a
# test deliberately constructs a tie).

CURVE_TOP = line_trace(60, 300, 80, 60)     # rows ~60-80  (max / highest bias)
CURVE_MID = line_trace(60, 300, 150, 130)   # rows ~130-150 (typ)
CURVE_BOT = line_trace(60, 300, 220, 200)   # rows ~200-220 (min / lowest bias)


def near_top(text):
    return raw_ocr(text, 165, 55, 195, 75)


def near_mid(text):
    return raw_ocr(text, 165, 125, 195, 145)


def near_bot(text):
    return raw_ocr(text, 165, 195, 195, 215)


# =================================================================
# A. count_expected_curves — band scheme
# =================================================================

class TestCountExpectedCurvesBandScheme:
    def test_three_distinct_band_labels_returns_3(self):
        lines = [raw_ocr("max", 0, 0, 10, 10), raw_ocr("typ", 0, 0, 10, 10),
                 raw_ocr("min", 0, 0, 10, 10)]
        assert count_expected_curves(lines) == 3

    def test_three_distinct_percentile_labels_returns_3(self):
        lines = [raw_ocr("98%", 0, 0, 10, 10), raw_ocr("typ", 0, 0, 10, 10),
                 raw_ocr("2%", 0, 0, 10, 10)]
        assert count_expected_curves(lines) == 3

    def test_two_distinct_band_labels_returns_2(self):
        lines = [raw_ocr("max", 0, 0, 10, 10), raw_ocr("typ", 0, 0, 10, 10)]
        assert count_expected_curves(lines) == 2

    def test_case_and_whitespace_variants_normalize_and_count_correctly(self):
        percentile_variant = [raw_ocr("98 %", 0, 0, 10, 10), raw_ocr("TYP.", 0, 0, 10, 10)]
        assert count_expected_curves(percentile_variant) == 2
        band_variant = [raw_ocr("Max", 0, 0, 10, 10), raw_ocr("typ", 0, 0, 10, 10)]
        assert count_expected_curves(band_variant) == 2


# =================================================================
# B. count_expected_curves — current-value scheme
# =================================================================

class TestCountExpectedCurvesCurrentValueScheme:
    def test_three_distinct_ua_values_returns_3(self):
        lines = [raw_ocr("I_D = 10uA", 0, 0, 10, 10),
                 raw_ocr("I_D = 250uA", 0, 0, 10, 10),
                 raw_ocr("I_D = 1000uA", 0, 0, 10, 10)]
        assert count_expected_curves(lines) == 3

    def test_four_distinct_values_returns_4(self):
        lines = [raw_ocr("I_D = 10uA", 0, 0, 10, 10),
                 raw_ocr("I_D = 250uA", 0, 0, 10, 10),
                 raw_ocr("I_D = 1000uA", 0, 0, 10, 10),
                 raw_ocr("I_D = 2500uA", 0, 0, 10, 10)]
        assert count_expected_curves(lines) == 4

    def test_same_value_two_notations_returns_none_ambiguous(self):
        # DUPLICATE-VALUE RULE (see module docstring): "1.0mA" and
        # "1000uA" normalize to the identical real value -- a duplicate is
        # ALWAYS ambiguous, never silently deduped to a clean count, even
        # though these two labels are the entire set.
        lines = [raw_ocr("I_D = 1.0mA", 0, 0, 10, 10),
                 raw_ocr("I_D = 1000uA", 0, 0, 10, 10)]
        assert count_expected_curves(lines) is None

    def test_mixed_units_all_distinct_correct_count_after_normalization(self):
        lines = [raw_ocr("I_D = 100uA", 0, 0, 10, 10),
                 raw_ocr("I_D = 5mA", 0, 0, 10, 10),
                 raw_ocr("I_D = 2A", 0, 0, 10, 10)]
        assert count_expected_curves(lines) == 3

    def test_alternate_micro_sign_glyphs_parse_to_the_same_unit(self):
        # U+03BC (GREEK SMALL LETTER MU) vs U+00B5 (MICRO SIGN) must
        # normalize identically -- proven here by the DUPLICATE-VALUE RULE
        # firing (None): if the two glyphs were NOT recognized as the same
        # unit, these would read as two distinct values and return 2, not
        # None. Ambiguous, not deduped, same as any other duplicate.
        lines = [raw_ocr("I_D = 250μA", 0, 0, 10, 10),
                 raw_ocr("I_D = 250µA", 0, 0, 10, 10)]
        assert count_expected_curves(lines) is None


# =================================================================
# C. count_expected_curves — ambiguous/edge cases
# =================================================================

class TestCountExpectedCurvesAmbiguous:
    def test_no_labels_at_all_returns_none(self):
        assert count_expected_curves([]) is None

    def test_both_band_and_current_value_labels_present_returns_none(self):
        # Should not happen on a real chart, but must not crash or guess.
        lines = [raw_ocr("max", 0, 0, 10, 10), raw_ocr("typ", 0, 0, 10, 10),
                 raw_ocr("I_D = 250uA", 0, 0, 10, 10)]
        assert count_expected_curves(lines) is None

    def test_duplicate_identical_band_labels_returns_none(self):
        # DUPLICATE-VALUE RULE: two "typ" claims are a duplicate -> always
        # ambiguous, same unconditional rule as the current-value scheme.
        lines = [raw_ocr("typ", 0, 0, 10, 10), raw_ocr("typ", 0, 0, 10, 10)]
        assert count_expected_curves(lines) is None

    def test_two_current_values_tie_amid_other_distinct_values_returns_none(self):
        # 250uA is distinct; 1.0mA and 1000uA are the SAME value stated
        # twice -- a duplicate, so ambiguous per the unconditional rule,
        # regardless of the genuinely-distinct 250uA also being present.
        lines = [raw_ocr("I_D = 250uA", 0, 0, 10, 10),
                 raw_ocr("I_D = 1.0mA", 0, 0, 10, 10),
                 raw_ocr("I_D = 1000uA", 0, 0, 10, 10)]
        assert count_expected_curves(lines) is None

    def test_stray_unrelated_text_alongside_valid_labels_is_ignored(self):
        lines = [raw_ocr("Vgs(th) vs Tj", 0, 0, 10, 10),
                 raw_ocr("max", 0, 0, 10, 10), raw_ocr("typ", 0, 0, 10, 10),
                 raw_ocr("min", 0, 0, 10, 10)]
        assert count_expected_curves(lines) == 3

    def test_malformed_ocr_line_missing_bounding_box_raises(self):
        lines = [{"text": "max"}]  # no "bounding_box" key
        with pytest.raises((KeyError, ValueError)):
            count_expected_curves(lines)


# =================================================================
# D. name_curves_by_labels — single curve
# =================================================================

class TestNameCurvesSingleCurve:
    def test_one_curve_no_labels_returns_vgsth(self):
        assert name_curves_by_labels([CURVE_MID], []) == ["vgsth"]

    def test_one_curve_with_a_label_present_still_returns_vgsth(self):
        # No disambiguation is needed (or done) when there's only one curve.
        assert name_curves_by_labels([CURVE_MID], [near_mid("typ")]) == ["vgsth"]
        assert name_curves_by_labels(
            [CURVE_MID], [near_mid("I_D = 250uA")]) == ["vgsth"]


# =================================================================
# E. name_curves_by_labels — band scheme naming
# =================================================================

class TestNameCurvesBandScheme:
    def test_three_curves_max_typ_min_resolve_distinctly(self):
        curves = [CURVE_TOP, CURVE_MID, CURVE_BOT]
        lines = [near_top("max"), near_mid("typ"), near_bot("min")]
        assert name_curves_by_labels(curves, lines) == [
            "vgsth_max", "vgsth_typ", "vgsth_min",
        ]

    def test_three_curves_percentile_labels_98_typ_2(self):
        # Extends rdson's 98%==max convention with a 2%==min counterpart.
        curves = [CURVE_TOP, CURVE_MID, CURVE_BOT]
        lines = [near_top("98%"), near_mid("typ"), near_bot("2%")]
        assert name_curves_by_labels(curves, lines) == [
            "vgsth_max", "vgsth_typ", "vgsth_min",
        ]

    def test_two_curves_max_typ_only(self):
        curves = [CURVE_TOP, CURVE_MID]
        lines = [near_top("max"), near_mid("typ")]
        assert name_curves_by_labels(curves, lines) == ["vgsth_max", "vgsth_typ"]

    def test_names_align_to_input_curve_order_not_sorted_order(self):
        # Deliberately NOT top-to-bottom input order: index0=BOT(min),
        # index1=TOP(max), index2=MID(typ). Output must track input order.
        curves = [CURVE_BOT, CURVE_TOP, CURVE_MID]
        lines = [near_top("max"), near_mid("typ"), near_bot("min")]
        assert name_curves_by_labels(curves, lines) == [
            "vgsth_min", "vgsth_max", "vgsth_typ",
        ]


# =================================================================
# F. name_curves_by_labels — current-value naming
# =================================================================

class TestNameCurvesCurrentValueScheme:
    def test_two_curves_distinct_current_values_named_by_normalized_value(self):
        curves = [CURVE_TOP, CURVE_MID]
        lines = [near_top("I_D = 250uA"), near_mid("I_D = 1.0mA")]
        assert name_curves_by_labels(curves, lines) == [
            "vgsth_id_250uA", "vgsth_id_1000uA",
        ]

    def test_four_curves_four_distinct_current_values(self):
        curve_a = line_trace(60, 300, 40, 20)
        curve_b = CURVE_TOP
        curve_c = CURVE_MID
        curve_d = CURVE_BOT
        lines = [
            raw_ocr("I_D = 10uA", 165, 15, 195, 35),
            near_top("I_D = 250uA"),
            near_mid("I_D = 1000uA"),
            near_bot("I_D = 2500uA"),
        ]
        names = name_curves_by_labels([curve_a, curve_b, curve_c, curve_d], lines)
        assert names == [
            "vgsth_id_10uA", "vgsth_id_250uA", "vgsth_id_1000uA", "vgsth_id_2500uA",
        ]

    def test_normalization_applied_before_naming_not_after(self):
        # The label says "1.0mA"; the resulting name must use the
        # NORMALIZED value (1000uA), never the raw "1.0mA" text verbatim.
        curves = [CURVE_TOP, CURVE_MID]
        lines = [near_top("I_D = 1.0mA"), near_mid("I_D = 10uA")]
        names = name_curves_by_labels(curves, lines)
        assert names == ["vgsth_id_1000uA", "vgsth_id_10uA"]
        assert "1.0mA" not in "".join(names)


# =================================================================
# G. name_curves_by_labels — quarantine cases (Option-A safety net)
# =================================================================

class TestNameCurvesQuarantine:
    def test_curve_count_exceeds_resolved_label_count_returns_none(self):
        # 3 curves but only 2 labels resolve (no "min"/3rd label at all).
        # Unlike rdson's 2-curve elimination trick, vgsth's 3-curve case
        # does NOT auto-complete the missing role by elimination -- every
        # curve needs its own resolved label, or the whole result quarantines.
        curves = [CURVE_TOP, CURVE_MID, CURVE_BOT]
        lines = [near_top("max"), near_mid("typ")]
        assert name_curves_by_labels(curves, lines) is None

    def test_genuine_nearest_curve_tie_returns_none(self):
        # A label positioned exactly equidistant between two curves' nearest
        # points -- name_curves_by_labels must refuse to guess which curve
        # it belongs to (distinct from nearest_curve_index's own low-level
        # deterministic tie-break -- this is a higher-level ambiguity check).
        curve_a = [(100.0, 60.0), (100.0, 300.0)]
        curve_b = [(140.0, 60.0), (140.0, 300.0)]
        tied_label = raw_ocr("max", 170, 118, 190, 122)  # bbox center row=120, exactly midway
        assert name_curves_by_labels([curve_a, curve_b], [tied_label]) is None

    def test_no_labels_at_all_multi_curve_returns_none(self):
        assert name_curves_by_labels([CURVE_TOP, CURVE_MID], []) is None

    def test_two_curves_resolving_to_identical_names_returns_none(self):
        # Two SEPARATE "I_D = 250uA" labels, each nearest a different curve
        # -- naming both curves "vgsth_id_250uA" would collide; must not be
        # silently overwritten/deduped.
        curves = [CURVE_TOP, CURVE_MID]
        lines = [near_top("I_D = 250uA"), near_mid("I_D = 250uA")]
        assert name_curves_by_labels(curves, lines) is None


# =================================================================
# H. Input validation
# =================================================================

class TestNameCurvesInputValidation:
    def test_empty_curves_list_raises_value_error(self):
        with pytest.raises(ValueError):
            name_curves_by_labels([], [])

    def test_curve_with_no_points_raises_value_error(self):
        with pytest.raises(ValueError):
            name_curves_by_labels([CURVE_TOP, []], [])
