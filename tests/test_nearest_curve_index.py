"""Tests for src.extraction.curve_detection.nearest_curve_index — written
FIRST (CLAUDE.md §2, red phase). RED PHASE ONLY — the function does not
exist in curve_detection.py yet (it's currently rdson_vs_tj.py's private
``_nearest_curve_index``).

This is the shared generic version of that helper: given a list of curves
(point lists) and a query pixel ``(cx, cy)``, return the index of the
nearest curve. Moving it to ``curve_detection.py`` (the module-with-zero-
curve_type-specific-logic established by the earlier detect_curve_classical/
detect_curve_monochrome extraction) lets both rdson_vs_tj's and
vgsth_vs_tj's naming modules share ONE implementation instead of
duplicating it (CLAUDE.md §3).

Separately (not new test code — a read-only regression check, see the
session report): rdson_vs_tj's own naming tests (tests/test_rdson_two_curve.py)
must keep passing, unchanged, once rdson_vs_tj.py is rewired to import
``nearest_curve_index`` from here instead of keeping its own private copy
— that rewiring is implementation work, out of scope for this red-phase
session (and would touch an existing frozen-adjacent file), so it is
verified as a currently-passing BASELINE now, to be re-verified after the
future move.
"""
from src.extraction.curve_detection import nearest_curve_index


class TestNearestCurveIndexImportable:
    def test_importable_from_curve_detection_not_private_to_rdson(self):
        assert callable(nearest_curve_index)


class TestNearestCurveIndexBehavior:
    def test_point_closest_to_curve_2_of_3_returns_index_1(self):
        curve0 = [(60.0, 100.0), (60.0, 200.0)]     # far above
        curve1 = [(150.0, 100.0), (150.0, 200.0)]   # the target
        curve2 = [(240.0, 100.0), (240.0, 200.0)]   # far below
        # Query point sits right next to curve1 (index 1, 0-indexed).
        index = nearest_curve_index([curve0, curve1, curve2], cx=150.0, cy=151.0)
        assert index == 1

    def test_genuine_distance_tie_is_deterministic_not_random(self):
        # Two curves with points exactly equidistant from the query point.
        # Documented tie-break (matches the algorithm being moved
        # verbatim from rdson_vs_tj.py's strict "<" comparison): the
        # first curve encountered in iteration order wins -- lower index,
        # here curve index 0 -- not whichever the platform/hash order
        # happens to produce.
        curve0 = [(100.0, 150.0)]
        curve1 = [(200.0, 150.0)]
        # cy=150 is exactly 50px from both curve0 (row 100) and curve1 (row 200).
        result_a = nearest_curve_index([curve0, curve1], cx=150.0, cy=150.0)
        result_b = nearest_curve_index([curve0, curve1], cx=150.0, cy=150.0)
        assert result_a == result_b == 0, "tie must deterministically favor the lower curve index"
