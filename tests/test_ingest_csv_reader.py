"""End-to-end tests for src/ingest/csv_reader.py.

Fixtures use real DigiKey column-header names (from the legacy mapping
tables) with deliberate variation between the two files, plus malformed
rows to exercise skip-and-continue behavior.
"""
from pathlib import Path

import pytest

from src.ingest.csv_reader import read_csv, IngestResult, SkippedRow

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"
DIGIKEY_CSV = FIXTURES / "mosfet_digikey.csv"
VARIANT_CSV = FIXTURES / "mosfet_variant_headers.csv"


class TestHappyPath:
    def test_returns_ingest_result(self):
        result = read_csv(DIGIKEY_CSV, "Si-MOSFET")
        assert isinstance(result, IngestResult)

    def test_good_rows_parsed(self):
        result = read_csv(DIGIKEY_CSV, "Si-MOSFET")
        names = [r["ModelName"] for r in result.records]
        assert "IRF540N" in names          # -ND suffix stripped
        assert "BSC042N03MS" in names      # -DKR suffix stripped

    def test_values_in_canonical_units(self):
        result = read_csv(DIGIKEY_CSV, "Si-MOSFET")
        rec = next(r for r in result.records if r["ModelName"] == "IRF540N")
        assert rec["Vdss"] == 100.0            # volts
        assert rec["Id"] == 33.0               # amps
        assert rec["Rdson"] == 44.0            # milliohms
        assert rec["Ciss"] == 1960.0           # pF
        assert rec["Tjmin"] == -55.0
        assert rec["Tjmax"] == 175.0
        assert rec["Cost"] == 1.02
        assert rec["package"] == "TO-220-3"
        assert rec["manufacturer"] == "Infineon"

    def test_raw_part_number_retained(self):
        result = read_csv(DIGIKEY_CSV, "Si-MOSFET")
        rec = next(r for r in result.records if r["ModelName"] == "IRF540N")
        assert rec["_raw_part_number"] == "IRF540N-ND"


class TestColumnVariation:
    def test_variant_headers_resolve_to_same_fields(self):
        result = read_csv(VARIANT_CSV, "Si-MOSFET")
        rec = next(r for r in result.records if r["ModelName"] == "STP55NF06")
        assert rec["Vdss"] == 60.0
        assert rec["Id"] == 50.0
        assert rec["Rdson"] == 18.0
        assert rec["Ciss"] == 1350.0

    def test_both_fixture_schemas_produce_same_field_keys(self):
        keys_a = set(read_csv(DIGIKEY_CSV, "Si-MOSFET").records[0])
        keys_b = set(read_csv(VARIANT_CSV, "Si-MOSFET").records[0])
        # canonical core fields present in both despite different headers
        for field in ("ModelName", "manufacturer", "Vdss", "Id", "Rdson", "Ciss"):
            assert field in keys_a and field in keys_b


class TestMalformedRows:
    def test_missing_model_name_skipped_with_reason(self):
        result = read_csv(DIGIKEY_CSV, "Si-MOSFET")
        assert len(result.skipped) == 1
        skip = result.skipped[0]
        assert isinstance(skip, SkippedRow)
        assert skip.row_num == 4                  # 1-based incl. header
        assert "ModelName" in skip.reason

    def test_unparseable_values_become_none_row_kept(self):
        # SUD50N04 row has garbage in every numeric column; the row survives
        # with None values rather than crashing or being dropped.
        result = read_csv(DIGIKEY_CSV, "Si-MOSFET")
        rec = next(r for r in result.records if r["ModelName"] == "SUD50N04")
        assert rec["Id"] is None
        assert rec["Rdson"] is None
        assert rec["Ciss"] is None

    def test_batch_continues_after_bad_rows(self):
        result = read_csv(DIGIKEY_CSV, "Si-MOSFET")
        assert len(result.records) == 3           # 4 data rows, 1 skipped


class TestReaderContract:
    def test_missing_modelname_column_raises(self, tmp_path):
        bad = tmp_path / "no_model.csv"
        bad.write_text("Foo,Bar\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError) as exc:
            read_csv(bad, "Si-MOSFET")
        assert "ModelName" in str(exc.value)

    def test_unknown_device_type_raises(self):
        with pytest.raises(KeyError):
            read_csv(DIGIKEY_CSV, "Flux-Capacitor")

    def test_limit(self):
        result = read_csv(DIGIKEY_CSV, "Si-MOSFET", limit=1)
        assert len(result.records) == 1

    def test_utf8_bom_handled(self, tmp_path):
        p = tmp_path / "bom.csv"
        p.write_text("Mfr Part #,Mfr\nIRF540N,Infineon\n", encoding="utf-8-sig")
        result = read_csv(p, "Si-MOSFET")
        assert result.records[0]["ModelName"] == "IRF540N"

    def test_empty_csv_returns_empty_result(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("Mfr Part #,Mfr\n", encoding="utf-8")
        result = read_csv(p, "Si-MOSFET")
        assert result.records == []
        assert result.skipped == []
