"""Tests for src/ingest/parsers.py — value parsing to canonical units.

Example raw values are real DigiKey cell formats documented in the legacy
parsers (e.g. "79mOhm @ 13.2A, 15V", "49A (Tc)", "-40 C ~ 175 C (TJ)").
"""
import pytest

from src.ingest.parsers import (
    parse_numeric,
    parse_voltage,
    parse_current,
    parse_rdson_mohm,
    parse_capacitance,
    parse_charge,
    parse_power,
    parse_energy,
    parse_timing,
    parse_thermal_resistance,
    parse_price,
    parse_temperature_range,
)


EMPTYish = ["", "-", "N/A", "n/a", None]


class TestParseNumeric:
    @pytest.mark.parametrize("raw", EMPTYish)
    def test_empty_variants_return_none(self, raw):
        assert parse_numeric(raw) is None

    def test_plain_number(self):
        assert parse_numeric("42.5") == 42.5

    def test_thousands_comma(self):
        assert parse_numeric("1,621") == 1621.0

    def test_range_takes_first(self):
        assert parse_numeric("1.5 ~ 2.0") == 1.5

    def test_conditions_after_at_ignored(self):
        assert parse_numeric("79 @ 13.2A, 15V") == 79.0

    def test_garbage_returns_none(self):
        assert parse_numeric("not a number") is None


class TestParseVoltage:
    @pytest.mark.parametrize("raw,expected", [
        ("650 V", 650.0),
        ("25V", 25.0),
        ("1.2kV", 1200.0),
        ("500mV", 0.5),
        ("±20V", 20.0),           # ±20V -> absolute
        ("3.6V @ 4.84mA", 3.6),        # Vgs(th)-style with conditions
        ("-", None),
        ("", None),
    ])
    def test_voltage(self, raw, expected):
        assert parse_voltage(raw) == expected


class TestParseCurrent:
    def test_simple_amps(self):
        assert parse_current("49A (Tc)") == 49.0

    def test_milliamps(self):
        assert parse_current("200mA (Ta)") == 0.2

    def test_prefers_ta_over_tc(self):
        assert parse_current("13.5A (Ta), 22A (Tc)") == 13.5

    def test_tc_only_fallback(self):
        assert parse_current("22A (Tc)") == 22.0

    def test_empty(self):
        assert parse_current("-") is None


class TestParseRdson:
    @pytest.mark.parametrize("raw,expected", [
        ("79mOhm @ 13.2A, 15V", 79.0),
        ("5Ohm @ 500mA", 5000.0),
        ("150uOhm", 0.15),
        ("0.079", 0.079),              # bare number assumed mOhm (legacy behavior)
        ("12mΩ", 12.0),           # Unicode omega
        ("-", None),
    ])
    def test_rdson(self, raw, expected):
        assert parse_rdson_mohm(raw) == expected


class TestParseCapacitanceChargePower:
    def test_pf(self):
        assert parse_capacitance("1621 pF @ 400 V") == 1621.0

    def test_nf_to_pf(self):
        assert parse_capacitance("2.4nF") == 2400.0

    def test_charge_nc(self):
        assert parse_charge("59 nC @ 15 V") == 59.0

    def test_charge_uc_to_nc(self):
        assert parse_charge("1.2uC") == 1200.0

    def test_power_watts(self):
        assert parse_power("164W (Tc)") == 164.0

    def test_power_prefers_ta(self):
        assert parse_power("400mW (Ta), 2W (Tc)") == 0.4


class TestParseEnergyTimingThermal:
    def test_energy_mj(self):
        assert parse_energy("1.2mJ @ 400V, 40A") == 1.2

    def test_energy_uj_to_mj(self):
        assert parse_energy("500uJ") == 0.5

    def test_timing_ns(self):
        assert parse_timing("35ns @ 400V, 40A") == 35.0

    def test_timing_us_to_ns(self):
        assert parse_timing("1.5us") == 1500.0

    def test_thermal_resistance(self):
        assert parse_thermal_resistance("0.65 °C/W") == 0.65

    def test_thermal_resistance_kw(self):
        assert parse_thermal_resistance("1.1 K/W") == 1.1


class TestParsePrice:
    def test_dollar_sign(self):
        assert parse_price("$15.51") == 15.51

    def test_bare(self):
        assert parse_price("15.51") == 15.51

    def test_comma(self):
        assert parse_price("$1,234.56") == 1234.56

    def test_garbage(self):
        assert parse_price("Call") is None


class TestParseTemperatureRange:
    def test_full_range(self):
        assert parse_temperature_range("-40 C ~ 175 C (TJ)") == (-40.0, 175.0)

    def test_degree_symbol(self):
        assert parse_temperature_range("-55°C ~ 150°C (TJ)") == (-55.0, 150.0)

    def test_single_value_is_tjmax(self):
        assert parse_temperature_range("175 C (TJ)") == (None, 175.0)

    def test_empty(self):
        assert parse_temperature_range("-") == (None, None)
