"""Tests for src.extraction.skeletonize — written FIRST (CLAUDE.md §2).

Mask -> single-pixel-wide centerline -> ordered (row, col) point list.
Uses skimage.skeletonize (consistent with LineFormer's own infer.py
approach per LEGACY_REVIEW.md §5), but averages multi-pixel skeleton runs
per column on the already-thin skeleton (not the raw thick mask — avoiding
the documented legacy "averaging window on raw mask" flaw) and keeps every
column (no x-deduplication, unlike the documented data-lossy legacy
interpolate() step).
"""
import numpy as np
import pytest

from src.extraction.skeletonize import mask_to_points


def horizontal_band(shape, row, thickness, col_start, col_end):
    mask = np.zeros(shape, dtype=bool)
    mask[row : row + thickness, col_start:col_end] = True
    return mask


def test_horizontal_line_produces_one_point_per_column():
    mask = horizontal_band((50, 50), 20, 1, 5, 45)
    points = mask_to_points(mask)
    cols = [c for _, c in points]
    assert cols == sorted(cols)
    assert len(points) == 40


def test_points_are_ordered_by_ascending_column():
    mask = horizontal_band((50, 50), 20, 1, 5, 45)
    points = mask_to_points(mask)
    cols = [c for _, c in points]
    assert cols == list(range(5, 45))


def test_thick_band_skeletonizes_to_single_row_per_column():
    mask = horizontal_band((50, 50), 15, 7, 5, 45)  # 7px-thick band
    points = mask_to_points(mask)
    # Skeletonizing a uniform-thickness band collapses to ~1 row per column.
    assert len(points) > 0
    rows = [r for r, _ in points]
    assert max(rows) - min(rows) <= 2  # roughly flat, allowing skeleton edge noise


def test_diagonal_line_row_increases_with_column():
    mask = np.zeros((50, 50), dtype=bool)
    for i in range(40):
        mask[i, i] = True
    points = mask_to_points(mask)
    assert len(points) > 0
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    assert cols == sorted(cols)
    assert rows[0] < rows[-1]


def test_empty_mask_returns_empty_list():
    mask = np.zeros((50, 50), dtype=bool)
    assert mask_to_points(mask) == []


def test_single_pixel_mask_returns_one_point_no_crash():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10, 10] = True
    points = mask_to_points(mask)
    assert len(points) == 1
    assert points[0] == (10.0, 10.0)


def test_tiny_two_pixel_mask_does_not_crash():
    mask = np.zeros((10, 10), dtype=bool)
    mask[3, 3] = True
    mask[3, 4] = True
    points = mask_to_points(mask)
    assert len(points) >= 1


def test_column_with_multiple_skeleton_rows_is_averaged():
    # A vertical stub at one column alongside a horizontal line -> that
    # column's skeleton pixels should collapse to one averaged point.
    mask = horizontal_band((50, 50), 20, 1, 5, 45)
    mask[15:25, 20] = True  # vertical stub through column 20
    points = mask_to_points(mask)
    cols = [c for _, c in points]
    assert len(cols) == len(set(cols))  # exactly one point per column


def test_disjoint_segments_combined_and_ordered():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10, 5:10] = True
    mask[30, 30:35] = True
    points = mask_to_points(mask)
    cols = [c for _, c in points]
    assert cols == sorted(cols)
    assert len(points) == 10


def test_returns_row_col_tuples_of_floats():
    mask = horizontal_band((50, 50), 20, 1, 5, 10)
    points = mask_to_points(mask)
    for row, col in points:
        assert isinstance(row, float)
        assert isinstance(col, float)
