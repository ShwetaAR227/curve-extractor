"""Tests for src/ingest/column_mapping.py — data-driven column registry.

Header aliases below are real DigiKey export column names taken from the
legacy repo's mapping tables (D:\\Extractor\\1_csv_input\\column_mapping.py),
not invented.
"""
import pytest

from src.ingest.column_mapping import (
    DEVICE_TYPE_COLUMNS,
    find_column,
    map_columns,
    list_device_types,
)


class TestRegistryShape:
    def test_known_device_types_registered(self):
        for dt in ("Si-MOSFET", "SiC-MOSFET", "GaN-HEMT", "IGBT"):
            assert dt in DEVICE_TYPE_COLUMNS

    def test_every_device_type_has_modelname_and_manufacturer(self):
        for dt, table in DEVICE_TYPE_COLUMNS.items():
            assert "ModelName" in table, dt
            assert "manufacturer" in table, dt

    def test_aliases_are_nonempty_lists(self):
        for dt, table in DEVICE_TYPE_COLUMNS.items():
            for field, aliases in table.items():
                assert isinstance(aliases, list) and aliases, f"{dt}.{field}"

    def test_list_device_types_sorted(self):
        types = list_device_types()
        assert types == sorted(types)
        assert "Si-MOSFET" in types


class TestFindColumn:
    HEADERS = [
        "Mfr Part #",
        "Mfr",
        "Drain to Source Voltage (Vdss)",
        "Rds On (Max) @ Id, Vgs",
        "Package / Case",
    ]

    def test_first_alias_match(self):
        assert find_column(self.HEADERS, "ModelName", "Si-MOSFET") == 0

    def test_later_alias_match(self):
        # "Part Number" absent; "Mfr Part #" is the first alias that matches
        headers = ["Part Number", "Manufacturer"]
        assert find_column(headers, "ModelName", "Si-MOSFET") == 0

    def test_case_insensitive(self):
        headers = ["MFR PART #", "mfr"]
        assert find_column(headers, "ModelName", "Si-MOSFET") == 0
        assert find_column(headers, "manufacturer", "Si-MOSFET") == 1

    def test_whitespace_stripped(self):
        headers = ["  Mfr Part #  "]
        assert find_column(headers, "ModelName", "Si-MOSFET") == 0

    def test_missing_field_returns_none(self):
        assert find_column(["Foo", "Bar"], "ModelName", "Si-MOSFET") is None

    def test_unknown_device_type_raises_with_valid_list(self):
        with pytest.raises(KeyError) as exc:
            find_column(self.HEADERS, "ModelName", "Flux-Capacitor")
        assert "Si-MOSFET" in str(exc.value)


class TestMapColumns:
    def test_maps_known_headers(self):
        headers = ["Mfr Part #", "Mfr", "Drain to Source Voltage (Vdss)"]
        mapping = map_columns(headers, "Si-MOSFET")
        assert mapping["ModelName"] == 0
        assert mapping["manufacturer"] == 1
        assert mapping["Vdss"] == 2

    def test_unknown_headers_ignored_not_crashed(self):
        headers = ["Mfr Part #", "Some Random Column", "Another Junk Header"]
        mapping = map_columns(headers, "Si-MOSFET")
        assert mapping == {"ModelName": 0}

    def test_igbt_specific_aliases(self):
        headers = ["Mfr Part #", "Vces", "Vce(sat)"]
        mapping = map_columns(headers, "IGBT")
        assert mapping["Vdss"] == 1
        assert mapping["Vce_sat"] == 2

    def test_sic_shares_mosfet_aliases(self):
        headers = ["Mfr Part #", "Rds On (Max) @ Id, Vgs"]
        assert map_columns(headers, "SiC-MOSFET")["Rdson"] == 1

    def test_empty_headers(self):
        assert map_columns([], "Si-MOSFET") == {}
