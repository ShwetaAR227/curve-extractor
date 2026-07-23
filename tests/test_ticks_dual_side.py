"""Tests for dual-side axis-tick calibration — written FIRST (CLAUDE.md §2,
RED PHASE ONLY). The capability under test does NOT exist in
src.calibration.ticks yet; every test that exercises it is expected to fail.

============================================================================
READ FIRST: src/calibration/ticks.py's full docstring/history was read
before writing this file (CLAUDE.md §2/§4 instruction). Relevant carried-
over context this file must not regress:
- Caveat #1 (module docstring): "Tick zoning (bottom 30% / left 30% /
  tight-corner 15%) assumes a bottom x-axis and left y-axis — right-hand
  or dual axes will mis-bucket." THIS is the exact caveat this task
  resolves (additively) — see the new docstring gap this task also adds
  (detect_y_axis_units still only scans left, deliberately unfixed here).
- T15/T16 exponent-repair history and T18 stray-zero-drop history
  (_repair_exponent_ticks / _drop_stray_zero_on_log_axis) — Section D below
  confirms these keep firing correctly on the NEW top/right candidate
  lists, not just bottom/left.
- fit_axis's documented 4-tuple contract ``(slope, intercept, used,
  is_log)`` — reused UNCHANGED, never reimplemented, per the approved
  design ("Run the existing, unchanged fit_axis... independently on each
  candidate side").
============================================================================

IMPLEMENTATION-FACING API DECISIONS MADE FOR THIS RED PHASE (the approved
design describes BEHAVIOR, not function signatures — these two additions
are this session's reasonable, minimal-disruption choice of how to expose
that behavior; flagged prominently for owner review before green phase):

1. ``parse_numeric_ticks`` itself is left COMPLETELY UNTOUCHED (same
   signature, same 2-tuple return, same ~80 existing tests in
   test_ticks.py keep passing unmodified) — this is what makes the change
   "purely additive" rather than a rewrite of a frozen, heavily-tested
   function. A NEW sibling function,
   ``parse_numeric_ticks_dual_side(ocr_lines, img_w, img_h) ->
   (x_ticks_bottom, x_ticks_top, y_ticks_left, y_ticks_right)``,
   is introduced instead: ``x_ticks_bottom``/``y_ticks_left`` are (by
   contract) byte-identical to what ``parse_numeric_ticks`` already
   returns for the same input; ``x_ticks_top``/``y_ticks_right`` are the
   NEW candidates from the mirrored zone (top 30% / right 30%), through
   the SAME per-line parsing/exponent-repair/stray-zero-drop pipeline.
2. ``fit_axis`` itself is left COMPLETELY UNTOUCHED. A new sibling,
   ``fit_axis_dual_side(default_ticks, opposite_ticks, min_n=2,
   inlier_threshold=15.0) -> Optional[Tuple[float, float, List[Tick],
   bool]]`` (same 4-tuple-or-None shape as ``fit_axis``), runs
   ``fit_axis`` independently on each side and applies the selection rule
   (more RANSAC inliers wins; exact tie -> default/bottom/left wins).
   ``default_ticks`` is always the bottom/left side, ``opposite_ticks``
   the top/right side — one function serves both axes.
3. ``derive_calibration`` (signature UNCHANGED) is the only frozen
   function whose INTERNALS change: it will call
   ``parse_numeric_ticks_dual_side`` + ``fit_axis_dual_side`` per axis
   instead of ``parse_numeric_ticks`` + ``fit_axis`` directly. Its own
   return shape is untouched (same keys), which is exactly why every
   existing consumer of ``derive_calibration`` (classical.py,
   classical_vgsth.py, model_if_vsd.py, pipeline.py, and their full test
   suites) needs zero changes.

DOCUMENTED, DELIBERATELY-UNFIXED GAP (to be written into ticks.py's own
docstring during the green phase, per instruction — noted here so the red
phase itself has the right test shape around it): ``detect_y_axis_units``
only scans the LEFT y-zone (``cx / img_w < 0.30``). If a right-side y-axis
wins under this new dual-side logic, unit detection would still miss it
and return None (ambiguous/undetected), same as today for any chart with
no left-zone unit text. NOT being fixed in this task (no real chart
observed yet needs it) — Section F below explicitly does NOT assert
anything about units for the right-axis-wins case, consistent with this
gap being left alone, not silently patched over.

Every coordinate fixture below places ticks on a SINGLE shared canvas
(IMG_W=1400, IMG_H=3000) via the ``x_line``/``y_line`` helpers, chosen so
that: (a) bottom/top x-zone ticks and left/right y-zone ticks can coexist
in one figure without crossing into each other's zone or the existing
tight-corner special case (verified by hand against the exact zone
predicates in parse_numeric_ticks/_place); (b) inlier-count fixtures
(clean/tie/stray) were EMPIRICALLY verified against the current, unchanged
``fit_axis`` (not hand-computed) before being written into these tests —
see session notes. A uniform pixel-position shift never changes RANSAC
inlier counts (translation-invariant), so shifting the empirically-
verified (value, pixel) fixtures into zone-safe absolute coordinates below
preserves their verified inlier counts exactly.
"""
import logging

import pytest

from src.calibration.ticks import derive_calibration, fit_axis, parse_numeric_ticks

# Will fail at collection time until implemented -- exactly the "missing
# capability" red failure this phase is meant to produce, not a fixture bug.
try:
    from src.calibration.ticks import fit_axis_dual_side, parse_numeric_ticks_dual_side
except ImportError:
    fit_axis_dual_side = None
    parse_numeric_ticks_dual_side = None


IMG_W, IMG_H = 1400, 3000
# Zone thresholds actually in force for this canvas (mirroring the existing
# parse_numeric_ticks predicates): bottom cy>2100, top cy<900,
# left cx<420, right cx>980. Fixture coordinates below are chosen with
# comfortable margin inside each band.


def ocr_line(text, x1, y1, x2, y2):
    return {"text": text, "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def x_line(text, along_px, side="bottom"):
    """One OCR line for an x-axis tick candidate. ``along_px`` becomes the
    tick's pixel position (cx); ``side`` picks the bottom (today's zone,
    cy=2500) or top (new, mirrored zone, cy=200) vertical band."""
    cy = 2500 if side == "bottom" else 200
    cx = along_px + 200  # keeps every fixture clear of the tight-corner (<210) cutoff
    return ocr_line(text, cx - 20, cy - 10, cx + 20, cy + 10)


def y_line(text, along_px, side="left"):
    """One OCR line for a y-axis tick candidate. ``along_px`` becomes the
    tick's pixel position (cy); ``side`` picks the left (today's zone,
    cx=150) or right (new, mirrored zone, cx=1250) horizontal band."""
    cx = 150 if side == "left" else 1250
    cy = along_px + 950  # keeps every fixture inside the (900,2100) x-zone-safe band
    return ocr_line(text, cx - 20, cy - 10, cx + 20, cy + 10)


# Empirically-verified (against today's real fit_axis) tick-value fixtures,
# expressed as (value, along_px) pairs -- along_px is later placed via
# x_line/y_line's uniform-shift-safe helper.
CLEAN4 = [(0.0, 100.0), (10.0, 200.0), (20.0, 300.0), (30.0, 400.0)]          # 4/4 inliers
CLEAN5 = [(0.0, 100.0), (5.0, 150.0), (10.0, 200.0), (20.0, 300.0), (30.0, 400.0)]  # 5/5 inliers
TIE_B = [(0.0, 700.0), (10.0, 650.0), (20.0, 600.0), (30.0, 550.0)]          # 4/4 inliers, different line than CLEAN4
STRAY6 = [(5.0, 50.0), (12.0, 900.0), (18.0, 150.0), (24.0, 700.0), (29.0, 300.0), (33.0, 950.0)]  # 2/6 inliers
DEGENERATE = [(5.0, 100.0), (5.0, 200.0), (5.0, 300.0)]                       # fit_axis -> None
VALID3 = [(0.0, 50.0), (10.0, 150.0), (20.0, 250.0)]                          # 3/3 inliers


def lines_for(fixture, side, axis, text_fn=None):
    """Build OCR lines for a whole (value, along_px) fixture on one side."""
    place = x_line if axis == "x" else y_line
    text_fn = text_fn or (lambda v: str(int(v)) if v == int(v) else str(v))
    return [place(text_fn(v), px, side=side) for v, px in fixture]


def _skip_if_not_implemented():
    if fit_axis_dual_side is None or parse_numeric_ticks_dual_side is None:
        pytest.fail(
            "parse_numeric_ticks_dual_side / fit_axis_dual_side do not exist yet "
            "-- expected RED-phase failure (missing capability), not a fixture bug."
        )


# =================================================================
# A. Backward compatibility (the non-negotiable bar)
# =================================================================

class TestBackwardCompatibility:
    def test_normal_chart_bottom_left_only_matches_todays_derive_calibration(self):
        # A1: byte-identical to today for a chart with ticks only
        # below/left -- the golden value is derived from TODAY's real,
        # unchanged parse_numeric_ticks + fit_axis composition at test-run
        # time (never hardcoded magic numbers), so this pins actual
        # current behavior, not a guess.
        lines = (
            lines_for(CLEAN4, "bottom", "x")
            + lines_for(CLEAN5, "left", "y")
        )
        x_ticks, y_ticks = parse_numeric_ticks(lines, IMG_W, IMG_H)
        x_fit = fit_axis(x_ticks)
        y_fit = fit_axis(y_ticks)
        assert x_fit is not None and y_fit is not None
        expected = {
            "plot_bbox": {
                "left": sorted(p for _, p in x_fit[2])[0],
                "right": sorted(p for _, p in x_fit[2])[-1],
                "top": sorted(p for _, p in y_fit[2])[0],
                "bottom": sorted(p for _, p in y_fit[2])[-1],
            },
            "x_slope": x_fit[0], "x_intercept": x_fit[1],
            "y_slope": y_fit[0], "y_intercept": y_fit[1],
            "x_log": x_fit[3], "y_log": y_fit[3],
        }
        assert derive_calibration(lines, IMG_W, IMG_H) == expected

    def test_no_opposite_side_ticks_dual_side_matches_single_side_result(self):
        # A2: no top/right ticks exist at all -> dual-side result must
        # equal the plain single-side fit_axis result exactly (no wasted
        # work, no different answer).
        _skip_if_not_implemented()
        lines = lines_for(CLEAN4, "bottom", "x")
        x_bottom, x_top, y_left, y_right = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        assert x_top == []
        assert x_bottom == parse_numeric_ticks(lines, IMG_W, IMG_H)[0]
        assert fit_axis_dual_side(x_bottom, x_top) == fit_axis(x_bottom)

    # A3 (full existing regression suite passes unchanged) is not a
    # single assertion -- verified separately by running `pytest tests/`
    # and reported alongside this file's own red-phase output (see
    # session report). Trivially true right now (nothing implemented
    # yet); its role is as a standing guardrail for the green phase.


# =================================================================
# B. New detection -- x-axis (below vs. above)
# =================================================================

class TestXAxisBelowVsAbove:
    def test_only_below_unchanged_path(self):
        # B4
        _skip_if_not_implemented()
        lines = lines_for(CLEAN4, "bottom", "x")
        x_bottom, x_top, _, _ = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        assert x_top == []
        assert fit_axis_dual_side(x_bottom, x_top) == fit_axis(x_bottom)

    def test_only_above_detected_new_capability(self):
        # B5: nothing below at all -- above must still be found and used.
        _skip_if_not_implemented()
        lines = lines_for(CLEAN4, "top", "x")
        x_bottom, x_top, _, _ = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        assert x_bottom == []
        assert len(x_top) == 4
        result = fit_axis_dual_side(x_bottom, x_top)
        assert result is not None
        assert result == fit_axis(x_top)

    def test_only_above_full_chart_calibrates(self):
        # B5, end-to-end: a full chart with x-ticks ONLY on top (plus a
        # normal left y-axis) must still calibrate successfully.
        _skip_if_not_implemented()
        lines = lines_for(CLEAN4, "top", "x") + lines_for(CLEAN5, "left", "y")
        cal = derive_calibration(lines, IMG_W, IMG_H)
        assert cal is not None
        # Expected values derived from the ACTUAL placed ticks (x_line's
        # coordinate offset shifts intercept, not slope) -- not the raw,
        # unshifted fixture.
        x_bottom, x_top, _, _ = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        expected_x = fit_axis(x_top)
        assert cal["x_slope"] == pytest.approx(expected_x[0])
        assert cal["x_intercept"] == pytest.approx(expected_x[1])

    def test_both_present_below_wins_more_inliers(self):
        # B6: below=CLEAN5 (5 inliers) beats above=STRAY6 (2 inliers).
        _skip_if_not_implemented()
        lines = lines_for(CLEAN5, "bottom", "x") + lines_for(STRAY6, "top", "x")
        x_bottom, x_top, _, _ = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        result = fit_axis_dual_side(x_bottom, x_top)
        assert result == fit_axis(x_bottom)
        assert len(result[2]) == 5

    def test_both_present_above_wins_more_inliers(self):
        # B7: above=CLEAN5 (5 inliers) beats below=STRAY6 (2 inliers) --
        # proves selection isn't hardcoded to prefer bottom.
        _skip_if_not_implemented()
        lines = lines_for(STRAY6, "bottom", "x") + lines_for(CLEAN5, "top", "x")
        x_bottom, x_top, _, _ = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        result = fit_axis_dual_side(x_bottom, x_top)
        assert result == fit_axis(x_top)
        assert len(result[2]) == 5

    def test_exact_tie_bottom_wins(self):
        # B8: CLEAN4 (4 inliers) on both sides, different lines -> tie ->
        # bottom's fit wins (documented default tie-break).
        _skip_if_not_implemented()
        lines = lines_for(CLEAN4, "bottom", "x") + lines_for(TIE_B, "top", "x")
        x_bottom, x_top, _, _ = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        assert len(fit_axis(x_bottom)[2]) == len(fit_axis(x_top)[2]) == 4
        result = fit_axis_dual_side(x_bottom, x_top)
        assert result == fit_axis(x_bottom)


# =================================================================
# C. New detection -- y-axis (left vs. right)
# =================================================================

class TestYAxisLeftVsRight:
    def test_only_left_unchanged(self):
        # C9
        _skip_if_not_implemented()
        lines = lines_for(CLEAN4, "left", "y")
        _, _, y_left, y_right = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        assert y_right == []
        assert fit_axis_dual_side(y_left, y_right) == fit_axis(y_left)

    def test_only_right_detected_new_capability(self):
        # C10: nothing on the left at all -- right must still be found.
        _skip_if_not_implemented()
        lines = lines_for(CLEAN4, "right", "y")
        _, _, y_left, y_right = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        assert y_left == []
        assert len(y_right) == 4
        assert fit_axis_dual_side(y_left, y_right) == fit_axis(y_right)

    def test_only_right_full_chart_calibrates(self):
        _skip_if_not_implemented()
        lines = lines_for(CLEAN4, "bottom", "x") + lines_for(CLEAN4, "right", "y")
        cal = derive_calibration(lines, IMG_W, IMG_H)
        assert cal is not None
        _, _, y_left, y_right = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        expected_y = fit_axis(y_right)
        assert cal["y_slope"] == pytest.approx(expected_y[0])
        assert cal["y_intercept"] == pytest.approx(expected_y[1])

    def test_both_present_right_wins_more_inliers(self):
        # C11
        _skip_if_not_implemented()
        lines = lines_for(STRAY6, "left", "y") + lines_for(CLEAN5, "right", "y")
        _, _, y_left, y_right = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        result = fit_axis_dual_side(y_left, y_right)
        assert result == fit_axis(y_right)
        assert len(result[2]) == 5

    def test_exact_tie_left_wins(self):
        # C12
        _skip_if_not_implemented()
        lines = lines_for(CLEAN4, "left", "y") + lines_for(TIE_B, "right", "y")
        _, _, y_left, y_right = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        assert len(fit_axis(y_left)[2]) == len(fit_axis(y_right)[2]) == 4
        result = fit_axis_dual_side(y_left, y_right)
        assert result == fit_axis(y_left)


# =================================================================
# D. Interaction with existing repair logic (D13/D14)
# =================================================================

class TestRepairLogicAppliesOnEitherSide:
    def test_exponent_repair_applies_when_above_wins_x_axis(self):
        # D13 (x-axis): real T16 exponent-space-form labels, placed ONLY
        # on top -- must resolve to 1000/100/10, not be left as literal
        # "10 3"/"10 2"/"10 1" tokens or dropped.
        _skip_if_not_implemented()
        lines = [
            x_line("10 3", 100, side="top"),
            x_line("10 2", 300, side="top"),
            x_line("10 1", 500, side="top"),
        ]
        x_bottom, x_top, _, _ = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        assert x_bottom == []
        values = sorted(v for v, _ in x_top)
        assert values == [10.0, 100.0, 1000.0]

    def test_exponent_repair_applies_when_right_wins_y_axis(self):
        # D13 (y-axis mirror): same real exponent labels, placed ONLY on
        # the right y-zone.
        _skip_if_not_implemented()
        lines = [
            y_line("10 3", 100, side="right"),
            y_line("10 2", 300, side="right"),
            y_line("10 1", 500, side="right"),
        ]
        _, _, y_left, y_right = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        assert y_left == []
        values = sorted(v for v, _ in y_right)
        assert values == [10.0, 100.0, 1000.0]

    def test_stray_zero_drop_applies_when_above_wins_x_axis(self):
        # D14 (x-axis): real T18 AUIRL3705N-style pattern (0/1/10/100, the
        # 0 is noise on a log axis), placed ONLY on top.
        _skip_if_not_implemented()
        lines = [
            x_line("0", 50, side="top"), x_line("1", 100, side="top"),
            x_line("10", 400, side="top"), x_line("100", 700, side="top"),
        ]
        x_bottom, x_top, _, _ = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        assert x_bottom == []
        values = sorted(v for v, _ in x_top)
        assert values == [1.0, 10.0, 100.0]  # the stray 0 is dropped

    def test_stray_zero_drop_applies_when_right_wins_y_axis(self):
        # D14 (y-axis mirror)
        _skip_if_not_implemented()
        lines = [
            y_line("0", 50, side="right"), y_line("1", 100, side="right"),
            y_line("10", 400, side="right"), y_line("100", 700, side="right"),
        ]
        _, _, y_left, y_right = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        assert y_left == []
        values = sorted(v for v, _ in y_right)
        assert values == [1.0, 10.0, 100.0]


# =================================================================
# E. Adversarial / stray-annotation defense
# =================================================================

class TestAdversarialDefense:
    def test_stray_label_column_with_more_raw_ticks_still_loses(self):
        # E15: the "losing" side (top) has MORE raw candidate ticks (6,
        # STRAY6) than the real axis (bottom, 4, CLEAN4) -- but its
        # RANSAC inlier count (2) is worse, so it must still lose.
        # Selection is fit-quality-based, not raw-tick-count-based.
        _skip_if_not_implemented()
        lines = lines_for(CLEAN4, "bottom", "x") + lines_for(STRAY6, "top", "x")
        x_bottom, x_top, _, _ = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        assert len(x_top) == 6 and len(x_bottom) == 4  # top DOES have more raw ticks
        result = fit_axis_dual_side(x_bottom, x_top)
        assert result == fit_axis(x_bottom)  # bottom still wins

    def test_both_sides_below_min_tick_count_returns_none_no_crash(self):
        # E16
        _skip_if_not_implemented()
        default_ticks = [(1.0, 100.0)]  # 1 tick, < min_n=2
        opposite_ticks = [(2.0, 200.0)]  # 1 tick, < min_n=2
        assert fit_axis_dual_side(default_ticks, opposite_ticks) is None
        assert fit_axis_dual_side([], []) is None

    def test_one_side_fewer_than_two_distinct_values_only_valid_side_used(self):
        # E17: default (bottom) is degenerate (all same value, fit_axis
        # returns None); opposite (top) is a valid, distinct-value set --
        # must use the valid side without crashing on the None comparison.
        _skip_if_not_implemented()
        assert fit_axis(DEGENERATE) is None
        assert fit_axis(VALID3) is not None
        result = fit_axis_dual_side(DEGENERATE, VALID3)
        assert result == fit_axis(VALID3)
        # ... and symmetrically, opposite degenerate / default valid:
        result2 = fit_axis_dual_side(VALID3, DEGENERATE)
        assert result2 == fit_axis(VALID3)


# =================================================================
# F. The actual motivating case
# =================================================================

class TestMotivatingCase:
    def test_synthetic_quadrant3_right_y_axis_calibrates_correctly(self):
        # F18: a chart with a normal bottom x-axis but Y-AXIS LABELS ONLY
        # ON THE RIGHT (no left-side y ticks at all) -- BEFORE this task's
        # implementation this chart failed to calibrate entirely (y_ticks
        # == [], derive_calibration returned None; this was hand-confirmed
        # during the red phase -- see session notes). Must now calibrate
        # correctly using the right side; this is the real motivating case,
        # not a hypothetical.
        _skip_if_not_implemented()
        lines = lines_for(CLEAN4, "bottom", "x") + lines_for(CLEAN5, "right", "y")
        cal = derive_calibration(lines, IMG_W, IMG_H)
        assert cal is not None
        _, _, y_left, y_right = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        assert y_left == []
        expected_y = fit_axis(y_right)
        assert cal["y_slope"] == pytest.approx(expected_y[0])
        assert cal["y_intercept"] == pytest.approx(expected_y[1])
        assert cal["y_log"] == expected_y[3]

    def test_if_vs_vsd_existing_real_device_tests_do_not_regress(self):
        # F19: not a new-capability test -- if_vs_vsd's own suites
        # (test_curve_registry_if_vsd.py, test_model_if_vsd.py,
        # test_if_vsd_naming.py) are all standard left-axis charts in the
        # real corpus reviewed so far; this dual-side change is forward-
        # looking, not fixing an observed if_vs_vsd bug. Verified by
        # running those suites directly (see session report) rather than
        # duplicated here -- this is a placeholder marking the intent so
        # it isn't silently skipped from the checklist.
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q",
             "tests/test_curve_registry_if_vsd.py", "tests/test_model_if_vsd.py",
             "tests/test_if_vsd_naming.py"],
            cwd=__file__.rsplit("/tests/", 1)[0], capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout[-2000:]


# =================================================================
# G. Integration
# =================================================================

class TestIntegration:
    def test_derive_calibration_shape_and_plot_bbox_from_winning_side(self):
        # G20: TOP wins for x, RIGHT wins for y (both non-default sides
        # winning simultaneously) -- derive_calibration's key set must be
        # unchanged, and plot_bbox must reflect the ACTUAL winning ticks'
        # pixel extents, not a leftover/default reference.
        _skip_if_not_implemented()
        lines = (
            lines_for(CLEAN5, "top", "x") + lines_for(STRAY6, "bottom", "x")
            + lines_for(CLEAN5, "right", "y") + lines_for(STRAY6, "left", "y")
        )
        cal = derive_calibration(lines, IMG_W, IMG_H)
        assert cal is not None
        assert set(cal.keys()) == {
            "plot_bbox", "x_slope", "x_intercept", "y_slope", "y_intercept",
            "x_log", "y_log",
        }
        assert set(cal["plot_bbox"].keys()) == {"left", "right", "top", "bottom"}
        x_bottom, x_top, y_left, y_right = parse_numeric_ticks_dual_side(lines, IMG_W, IMG_H)
        expected_x = fit_axis(x_top)     # top wins (5 inliers vs bottom's 2)
        expected_y = fit_axis(y_right)   # right wins (5 inliers vs left's 2)
        x_px = sorted(p for _, p in expected_x[2])
        y_px = sorted(p for _, p in expected_y[2])
        assert cal["plot_bbox"]["left"] == pytest.approx(x_px[0])
        assert cal["plot_bbox"]["right"] == pytest.approx(x_px[-1])
        assert cal["plot_bbox"]["top"] == pytest.approx(y_px[0])
        assert cal["plot_bbox"]["bottom"] == pytest.approx(y_px[-1])

    def test_pixel_to_data_and_data_to_pixel_signatures_unchanged(self):
        # G21: zero changes needed downstream -- pin the exact current
        # signatures of both functions so any accidental edit is caught.
        # (Expected GREEN already: nothing about this task touches either
        # function; this is a standing regression guard, not a new-
        # capability test.)
        import inspect

        from src.calibration.ticks import data_to_pixel, pixel_to_data

        assert [p.name for p in inspect.signature(pixel_to_data).parameters.values()] == [
            "px", "py", "calibration",
        ]
        assert [p.name for p in inspect.signature(data_to_pixel).parameters.values()] == [
            "x", "y", "calibration",
        ]


# =================================================================
# H. Logging
# =================================================================

class TestLogging:
    def test_distinct_log_line_when_non_default_side_wins(self, caplog):
        # H22: a non-default (top/right) side winning must log distinctly
        # -- never silent. And when default (bottom/left) wins normally,
        # that specific "non-default side won" message must NOT appear.
        _skip_if_not_implemented()
        caplog.set_level(logging.INFO)

        # Non-default (top) wins:
        x_bottom, x_top = STRAY6, CLEAN5
        fit_axis_dual_side(x_bottom, x_top)
        non_default_logged = any(
            ("top" in r.message.lower() or "opposite" in r.message.lower()
             or "non-default" in r.message.lower())
            for r in caplog.records
        )
        assert non_default_logged, "no distinct log line when the non-default side won"

        caplog.clear()
        # Default (bottom) wins normally -- no such message.
        fit_axis_dual_side(CLEAN5, STRAY6)
        non_default_logged_again = any(
            ("top" in r.message.lower() or "opposite" in r.message.lower()
             or "non-default" in r.message.lower())
            for r in caplog.records
        )
        assert not non_default_logged_again, "spurious non-default-side log line when default won"
