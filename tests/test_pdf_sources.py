"""Tests for src/pdf_download/sources/ — pluggable PDF-source registry.

No network: find_url implementations that would touch the network
(mouser_api) are exercised only for their no-credential short-circuit here;
their HTTP behavior is tested in test_pdf_downloader.py with mocks.
"""
import pytest

from src.pdf_download.sources import get_source, list_sources, iter_sources
from src.pdf_download.sources.csv_url import CsvUrlSource
from src.pdf_download.sources.direct_mfr import DirectMfrSource, MFR_URL_PATTERNS
from src.pdf_download.sources.mouser_api import MouserApiSource


DEVICE = {
    "ModelName": "IRF540N",
    "manufacturer": "Infineon",
    "pdf_url": "https://example.com/irf540n.pdf",
}


class TestRegistry:
    def test_known_sources_resolve(self):
        for name in ("csv_url", "direct_mfr", "mouser_api"):
            assert get_source(name).name == name

    def test_unknown_source_raises_with_available_list(self):
        with pytest.raises(KeyError) as exc:
            get_source("carrier_pigeon")
        assert "csv_url" in str(exc.value)

    def test_list_sources_sorted(self):
        assert list_sources() == sorted(list_sources())

    def test_iter_sources_priority_order(self):
        priorities = [s.priority for s in iter_sources()]
        assert priorities == sorted(priorities)

    def test_csv_url_is_highest_priority(self):
        assert iter_sources()[0].name == "csv_url"


class TestCsvUrlSource:
    def test_returns_device_url(self):
        assert CsvUrlSource().find_url(DEVICE) == "https://example.com/irf540n.pdf"

    def test_missing_url_returns_none(self):
        assert CsvUrlSource().find_url({"ModelName": "X"}) is None

    @pytest.mark.parametrize("bad", ["", "-", "   "])
    def test_blank_url_returns_none(self, bad):
        assert CsvUrlSource().find_url({"pdf_url": bad}) is None

    def test_protocol_relative_url_normalized(self):
        dev = {"pdf_url": "//example.com/a.pdf"}
        assert CsvUrlSource().find_url(dev) == "https://example.com/a.pdf"


class TestDirectMfrSource:
    def test_st_pattern(self):
        dev = {"ModelName": "STP55NF06", "manufacturer": "STMicroelectronics"}
        url = DirectMfrSource().find_url(dev)
        assert url == "https://www.st.com/resource/en/datasheet/stp55nf06.pdf"

    def test_onsemi_pattern(self):
        dev = {"ModelName": "NTD5867NL", "manufacturer": "onsemi"}
        url = DirectMfrSource().find_url(dev)
        assert url == "https://www.onsemi.com/download/data-sheet/pdf/ntd5867nl-d.pdf"

    def test_unknown_manufacturer_returns_none(self):
        dev = {"ModelName": "X123", "manufacturer": "ACME Widgets"}
        assert DirectMfrSource().find_url(dev) is None

    def test_missing_model_name_returns_none(self):
        assert DirectMfrSource().find_url({"manufacturer": "onsemi"}) is None

    def test_patterns_are_data(self):
        # Registry-pattern guarantee: adding a manufacturer is a data edit.
        assert isinstance(MFR_URL_PATTERNS, list)
        for matcher, template in MFR_URL_PATTERNS:
            assert callable(matcher) and isinstance(template, str)


class TestMouserApiSource:
    def test_no_api_key_returns_none_without_network(self, monkeypatch):
        monkeypatch.delenv("MOUSER_API_KEY", raising=False)
        called = []
        monkeypatch.setattr(
            "src.pdf_download.sources.mouser_api.urlopen",
            lambda *a, **k: called.append(1),
        )
        assert MouserApiSource().find_url(DEVICE) is None
        assert called == []  # never touched the network
