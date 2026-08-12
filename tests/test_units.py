"""
tests/test_units.py
~~~~~~~~~~~~~~~~~~~
Tests for ergane.units.Units.

Covers:
  - Construction and property derivation
  - Units.code() / Units.cgs() / Units.si() class methods
  - Units.from_params() integration with parsed athinput
  - scale() factor lookup for all standard field names
  - label() string generation
  - Repr
"""

from __future__ import annotations

import math

import pytest

from ergane.units import Units


# ── Construction helpers ──────────────────────────────────────────────────────

LENGTH   = 3.086e18   # 1 pc in cm
DENSITY  = 1.67e-24   # 1 proton mass / cc in g/cm^3
VELOCITY = 1.0e5      # 1 km/s in cm/s


# ── Units.code() ──────────────────────────────────────────────────────────────

class TestCodeUnits:
    def test_all_scales_unity(self):
        u = Units.code()
        assert u.length   == pytest.approx(1.0)
        assert u.density  == pytest.approx(1.0)
        assert u.velocity == pytest.approx(1.0)
        assert u.time     == pytest.approx(1.0)
        assert u.pressure == pytest.approx(1.0)

    def test_system_name(self):
        assert Units.code().system == "code"

    def test_scale_returns_one_for_all_fields(self):
        u = Units.code()
        for field in ("density", "pressure", "velx", "vely", "velz", "bx", "by", "bz"):
            assert u.scale(field) == pytest.approx(1.0), f"Field {field!r} scale != 1"

    def test_scale_returns_one_for_unknown_field(self):
        assert Units.code().scale("nonexistent_field") == pytest.approx(1.0)

    def test_label_includes_code_tag(self):
        u = Units.code()
        assert "[code]" in u.label("density")


# ── Units.cgs() ───────────────────────────────────────────────────────────────

class TestCgsUnits:
    @pytest.fixture
    def cgs(self):
        return Units.cgs(length=LENGTH, density=DENSITY, velocity=VELOCITY)

    def test_system_name(self, cgs):
        assert cgs.system == "CGS"

    def test_length_scale(self, cgs):
        assert cgs.length == pytest.approx(LENGTH)

    def test_density_scale(self, cgs):
        assert cgs.density == pytest.approx(DENSITY)

    def test_velocity_scale(self, cgs):
        assert cgs.velocity == pytest.approx(VELOCITY)

    def test_derived_time(self, cgs):
        expected = LENGTH / VELOCITY
        assert cgs.time == pytest.approx(expected, rel=1e-6)

    def test_derived_pressure(self, cgs):
        """
        Units.cgs() auto-derives pressure as density * v_cgs**2.
        VELOCITY < 100 → treated as km/s and converted to cm/s internally.
        """
        v_cgs = VELOCITY * 1e5 if VELOCITY < 100.0 else VELOCITY
        expected = DENSITY * (v_cgs ** 2)
        assert cgs.pressure == pytest.approx(expected, rel=1e-6)

    def test_derived_magnetic(self, cgs):
        expected = math.sqrt(DENSITY * VELOCITY ** 2)
        assert cgs.magnetic == pytest.approx(expected, rel=1e-6)

    def test_scale_density(self, cgs):
        assert cgs.scale("density") == pytest.approx(DENSITY)

    def test_scale_velocity_fields(self, cgs):
        for f in ("velx", "vely", "velz"):
            assert cgs.scale(f) == pytest.approx(VELOCITY), f"scale({f!r}) mismatch"

    def test_scale_bfield_fields(self, cgs):
        expected = math.sqrt(DENSITY * VELOCITY ** 2)
        for f in ("bx", "by", "bz"):
            assert cgs.scale(f) == pytest.approx(expected, rel=1e-6)


# ── Units.si() ────────────────────────────────────────────────────────────────

class TestSiUnits:
    def test_system_name(self):
        u = Units.si(length=3.086e16, density=1.67e-27, velocity=1e3)
        assert u.system == "SI"

    def test_scales_stored(self):
        u = Units.si(length=1.0, density=2.0, velocity=3.0)
        assert u.length   == pytest.approx(1.0)
        assert u.density  == pytest.approx(2.0)
        assert u.velocity == pytest.approx(3.0)


# ── Units() direct constructor ────────────────────────────────────────────────

class TestUnitsDirectConstructor:
    def test_time_override(self):
        u = Units(length=1.0, density=1.0, velocity=1.0, time=42.0)
        assert u.time == pytest.approx(42.0)

    def test_pressure_override(self):
        u = Units(length=1.0, density=1.0, velocity=1.0, pressure=99.0)
        assert u.pressure == pytest.approx(99.0)

    def test_magnetic_override(self):
        u = Units(length=1.0, density=1.0, velocity=1.0, magnetic=7.0)
        assert u.magnetic == pytest.approx(7.0)

    def test_custom_system_name(self):
        u = Units(system="MySystem")
        assert u.system == "MySystem"

    def test_mu_stored(self):
        u = Units(mu=1.0)
        assert u.mu == pytest.approx(1.0)

    def test_default_mu(self):
        u = Units.code()
        assert u.mu == pytest.approx(0.62)


# ── label() ───────────────────────────────────────────────────────────────────

class TestLabel:
    @pytest.fixture
    def cgs(self):
        return Units.cgs(length=LENGTH, density=DENSITY, velocity=VELOCITY)

    def test_explicit_label_overrides(self):
        u = Units(labels={"density": "g/cc"})
        assert u.label("density") == "g/cc"

    def test_default_density_label(self, cgs):
        lbl = cgs.label("density")
        assert "ρ" in lbl

    def test_unknown_field_returns_field_name(self, cgs):
        lbl = cgs.label("my_custom_field")
        assert "my_custom_field" in lbl

    def test_system_appears_in_label(self, cgs):
        lbl = cgs.label("velx")
        assert "CGS" in lbl


# ── Units.from_params() ───────────────────────────────────────────────────────

class TestFromParams:
    def test_returns_code_units_when_no_units_section(self):
        params = {"mesh": {"nx1": "256"}}
        u = Units.from_params(params)
        assert u.system == "code"

    def test_returns_code_units_for_empty_params(self):
        u = Units.from_params({})
        assert u.system == "code"

    def test_parses_units_section(self):
        params = {
            "units": {
                "length_cgs": "3.08568e18",
                "mass_cgs":   "4.91417e31",
                "time_cgs":   "3.15576e13",
                "mu":         "0.62",
            }
        }
        u = Units.from_params(params)
        assert u.system == "CGS"
        assert u.length == pytest.approx(3.08568e18, rel=1e-4)

    def test_falls_back_to_code_on_bad_values(self):
        params = {"units": {"length_cgs": "not_a_number"}}
        u = Units.from_params(params)
        assert u.system == "code"

    def test_real_athinput_has_no_units_section(self, athinput_path):
        """The KH2D athinput has no <units> block → code units."""
        from ergane.athinput_parser import parse_athinput
        params = parse_athinput(athinput_path)
        u = Units.from_params(params)
        assert u.system == "code"


# ── Repr ──────────────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr_contains_system(self):
        u = Units.cgs(LENGTH, DENSITY, VELOCITY)
        r = repr(u)
        assert "CGS" in r

    def test_repr_is_string(self):
        assert isinstance(repr(Units.code()), str)
