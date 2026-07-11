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
