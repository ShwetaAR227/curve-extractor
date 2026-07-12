"""Tests for src/ingest/unit_conversion.py — unit normalization.

Deliberate change from legacy (D:\\Extractor\\1_csv_input\\unit_conversion.py):
an unrecognized source unit returns None (flagged, logged) instead of the
legacy behavior of silently assuming a 1:1 conversion.
"""
import logging

import pytest

from src.ingest.unit_conversion import UNIT_CONVERSIONS, normalize_unit


class TestKnownConversions:
    @pytest.mark.parametrize("value,source,target,expected", [
        (5.0, "Ohm", "mOhm", 5000.0),
        (5.0, "mOhm", "mOhm", 5.0),
        (150.0, "uOhm", "mOhm", 0.15),
        (2.4, "nF", "pF", 2400.0),
        (1.2, "uC", "nC", 1200.0),
        (500.0, "mV", "V", 0.5),
        (200.0, "mA", "A", 0.2),
        (1.1, "K/W", "C/W", 1.1),
        (0.5, "J", "mJ", 500.0),
        (1.5, "us", "ns", 1500.0),
        (400.0, "mW", "W", 0.4),
    ])
    def test_conversion(self, value, source, target, expected):
        assert normalize_unit(value, source, target) == pytest.approx(expected)

    def test_case_insensitive_source(self):
        assert normalize_unit(5.0, "OHM", "mOhm") == 5000.0

    def test_unicode_omega(self):
        assert normalize_unit(5.0, "Ω", "mOhm") == 5000.0


class TestNoneAndBadInput:
    def test_none_value(self):
        assert normalize_unit(None, "Ohm", "mOhm") is None

    def test_non_numeric_value(self):
        assert normalize_unit("abc", "Ohm", "mOhm") is None


class TestUnknownUnitsFlaggedNotGuessed:
    def test_unknown_source_returns_none(self, caplog):
        """LEGACY BUG NOT PORTED: legacy assumed 1:1 for unknown source units."""
        with caplog.at_level(logging.WARNING):
            result = normalize_unit(5.0, "furlongs", "mOhm")
        assert result is None
        assert any("furlongs" in r.message for r in caplog.records)

    def test_unknown_target_raises(self):
        with pytest.raises(KeyError) as exc:
            normalize_unit(5.0, "Ohm", "parsecs")
        assert "parsecs" in str(exc.value)

    def test_registry_has_expected_targets(self):
        for target in ("mOhm", "pF", "nC", "V", "A", "C/W", "mJ", "ns", "W"):
            assert target in UNIT_CONVERSIONS
