"""Tests for src.extraction.naming — written FIRST (CLAUDE.md §2).

Per-curve-type pluggable naming, mirroring the Stage 4 curve_registry
pattern: a lookup-by-curve_type registry (src.extraction.naming.get_naming_fn)
plus one position-based rule for capacitance_vs_vds specifically
(Ciss/Coss/Crss by top/middle/bottom mean y-pixel position). Curve types 2+
(e.g. id_vs_vgs's temperature-based naming) get their own module later —
nothing here is hardcoded into the general pipeline.
"""
import pytest

from src.extraction.naming import get_naming_fn, list_registered_naming_types
from src.extraction.naming.capacitance_vs_vds import name_curves


def curve(mean_row, n_points=5, col_start=0):
    return [(mean_row, float(col_start + i)) for i in range(n_points)]


def test_name_curves_assigns_ciss_coss_crss_by_position_top_to_bottom():
    curves = [curve(50.0), curve(10.0), curve(90.0)]  # middle, top, bottom
    names = name_curves(curves)
    assert names[1] == "Ciss"   # topmost (smallest row)
    assert names[0] == "Coss"   # middle
    assert names[2] == "Crss"   # bottommost (largest row)


def test_name_curves_result_aligned_to_input_order_not_sorted_order():
    curves = [curve(90.0), curve(50.0), curve(10.0)]  # bottom, middle, top
    names = name_curves(curves)
    assert names == ["Crss", "Coss", "Ciss"]


def test_name_curves_already_sorted_input():
    curves = [curve(10.0), curve(50.0), curve(90.0)]
    names = name_curves(curves)
    assert names == ["Ciss", "Coss", "Crss"]


def test_name_curves_handles_tied_mean_positions_without_crashing():
    curves = [curve(50.0), curve(50.0), curve(10.0)]
    names = name_curves(curves)
    assert len(names) == 3
    assert set(names) == {"Ciss", "Coss", "Crss"}


def test_name_curves_wrong_count_raises_value_error():
    with pytest.raises(ValueError):
        name_curves([curve(10.0), curve(50.0)])
    with pytest.raises(ValueError):
        name_curves([curve(10.0), curve(20.0), curve(30.0), curve(40.0)])


def test_name_curves_empty_curve_points_raises_value_error():
    with pytest.raises(ValueError):
        name_curves([curve(10.0), [], curve(90.0)])


def test_name_curves_uses_row_not_col_for_position():
    # Two curves with identical row ranges but different col spans should
    # still be orderable purely by row (position naming is vertical only).
    curves = [
        [(30.0, 0.0), (30.0, 100.0)],
        [(10.0, 5.0), (10.0, 6.0)],
        [(60.0, 0.0), (60.0, 200.0)],
    ]
    names = name_curves(curves)
    assert names == ["Coss", "Ciss", "Crss"]


def test_get_naming_fn_returns_capacitance_naming_function():
    fn = get_naming_fn("capacitance_vs_vds")
    assert fn is name_curves


def test_get_naming_fn_unknown_curve_type_raises_key_error():
    with pytest.raises(KeyError):
        get_naming_fn("not_a_real_curve_type")


def test_get_naming_fn_error_message_lists_registered_types():
    with pytest.raises(KeyError) as exc_info:
        get_naming_fn("bogus")
    assert "capacitance_vs_vds" in str(exc_info.value)


def test_list_registered_naming_types_contains_capacitance():
    assert "capacitance_vs_vds" in list_registered_naming_types()


# --------------------------------------------------------- vgsth_vs_tj
# placeholder naming (owner-approved frozen-file addition, 2026-07-21):
# exists ONLY so process_detections's unconditional get_naming_fn(curve_type)
# lookup doesn't KeyError for vgsth_vs_tj -- its real naming is entirely
# label-driven (name_curves_by_labels needs ocr_lines, which this
# registry's plain NamingFn signature can't carry), so there's no
# meaningful position-only naming to register the way rdson_vs_tj has.
# classical_vgsth.py's wrapper is required to override this on every
# "ok"-status path; it may only surface as-is inside a needs_review result.

def test_get_naming_fn_returns_a_callable_for_vgsth_vs_tj():
    fn = get_naming_fn("vgsth_vs_tj")
    assert callable(fn)


def test_vgsth_placeholder_names_follow_curve_n_pattern_aligned_to_input_order():
    fn = get_naming_fn("vgsth_vs_tj")
    names = fn([[(0.0, 0.0)], [(1.0, 1.0)], [(2.0, 2.0)]])
    assert names == ["curve_0", "curve_1", "curve_2"]


def test_vgsth_placeholder_never_raises_on_zero_curves():
    fn = get_naming_fn("vgsth_vs_tj")
    assert fn([]) == []


def test_vgsth_placeholder_never_raises_on_a_curve_with_no_points():
    # Unlike every real naming function in this registry, the placeholder
    # doesn't inspect point content at all -- it can't raise here.
    fn = get_naming_fn("vgsth_vs_tj")
    assert fn([[], [(0.0, 0.0)]]) == ["curve_0", "curve_1"]


def test_vgsth_placeholder_docstring_flags_itself_as_non_authoritative():
    # Documentation IS part of the contract for a function this dangerous
    # to trust by accident -- lock in that the warning stays present.
    fn = get_naming_fn("vgsth_vs_tj")
    assert fn.__doc__ is not None
    assert "placeholder" in fn.__doc__.lower()
    assert "override" in fn.__doc__.lower()


def test_list_registered_naming_types_now_includes_vgsth_vs_tj():
    assert "vgsth_vs_tj" in list_registered_naming_types()


# --------------------------------------------------------- if_vs_vsd
# placeholder naming (owner-approved frozen-file addition, 2026-07-22
# follow-up session): same disposable, never-trusted, generic-curve_N-names
# pattern already built for vgsth_vs_tj -- if_vs_vsd's real naming is
# entirely label-driven (name_curves_by_labels needs ocr_lines, which this
# registry's plain NamingFn signature can't carry). model_if_vsd.py's
# wrapper is required to override this on every "ok"-status path; it may
# only surface as-is inside a needs_review result.

def test_get_naming_fn_returns_a_callable_for_if_vs_vsd():
    fn = get_naming_fn("if_vs_vsd")
    assert callable(fn)


def test_if_vsd_placeholder_names_follow_curve_n_pattern_aligned_to_input_order():
    fn = get_naming_fn("if_vs_vsd")
    names = fn([[(0.0, 0.0)], [(1.0, 1.0)], [(2.0, 2.0)]])
    assert names == ["curve_0", "curve_1", "curve_2"]


def test_if_vsd_placeholder_never_raises_on_zero_curves():
    fn = get_naming_fn("if_vs_vsd")
    assert fn([]) == []


def test_if_vsd_placeholder_never_raises_on_a_curve_with_no_points():
    fn = get_naming_fn("if_vs_vsd")
    assert fn([[], [(0.0, 0.0)]]) == ["curve_0", "curve_1"]


def test_if_vsd_placeholder_docstring_flags_itself_as_non_authoritative():
    fn = get_naming_fn("if_vs_vsd")
    assert fn.__doc__ is not None
    assert "placeholder" in fn.__doc__.lower()
    assert "override" in fn.__doc__.lower()


def test_if_vsd_placeholder_is_a_distinct_function_from_vgsth_placeholder():
    # Two independent throwaway placeholders, not one shared object
    # accidentally reused across curve types.
    assert get_naming_fn("if_vs_vsd") is not get_naming_fn("vgsth_vs_tj")


def test_list_registered_naming_types_now_includes_if_vs_vsd():
    assert "if_vs_vsd" in list_registered_naming_types()


def test_if_vs_vsd_has_no_expected_names_entry_names_not_fixed():
    # Mirrors vgsth_vs_tj: curve names aren't a fixed set (temperature-
    # driven), so there's nothing truthful to register in
    # _EXPECTED_NAMES/get_expected_names for this curve type either.
    from src.extraction.naming import get_expected_names
    with pytest.raises(KeyError):
        get_expected_names("if_vs_vsd")
