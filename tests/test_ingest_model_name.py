"""Tests for src/ingest/model_name_utils.py — part-number normalization.

Legacy behavior preserved: only DigiKey-specific suffixes (-TRDKR, -DKR, -ND)
are stripped; manufacturer suffixes (-TR, -CT, -PbF, -E3) encode real part
variants and are kept.
"""
import pytest

from src.ingest.model_name_utils import normalize_model_name, sanitize_for_filesystem


class TestNormalizeModelName:
    @pytest.mark.parametrize("raw,expected", [
        ("IRF540N-ND", "IRF540N"),
        ("BSC042N03MS-DKR", "BSC042N03MS"),
        ("STP55NF06-TRDKR", "STP55NF06"),
        ("irf540n-nd", "irf540n"),          # case-insensitive suffix match
    ])
    def test_digikey_suffixes_stripped(self, raw, expected):
        assert normalize_model_name(raw) == expected

    @pytest.mark.parametrize("name", [
        "IRF540NPBF-TR",     # manufacturer suffix kept
        "BSS138-CT",
        "SUD50N04-PbF",
        "SIHF540-E3",
    ])
    def test_manufacturer_suffixes_kept(self, name):
        assert normalize_model_name(name) == name

    def test_whitespace_stripped(self):
        assert normalize_model_name("  IRF540N-ND  ") == "IRF540N"

    def test_plain_name_unchanged(self):
        assert normalize_model_name("IPB017N10N5") == "IPB017N10N5"

    def test_suffix_only_at_end(self):
        # "-ND" mid-string must not be stripped
        assert normalize_model_name("AB-ND-XYZ") == "AB-ND-XYZ"


class TestSanitizeForFilesystem:
    @pytest.mark.parametrize("raw,expected", [
        ("BSC042N03MS G ATMA1", "BSC042N03MS_G_ATMA1"),
        ("IRF540 (TO-220)", "IRF540_TO-220"),
        ("a/b\\c", "a_b_c"),
        ("a  b", "a_b"),
        ("__x__", "x"),
    ])
    def test_sanitize(self, raw, expected):
        assert sanitize_for_filesystem(raw) == expected

    def test_idempotent(self):
        once = sanitize_for_filesystem("a (b)/c")
        assert sanitize_for_filesystem(once) == once
