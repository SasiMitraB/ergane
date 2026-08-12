"""
tests/test_mhd_simulation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for ergane with MHD (Orszag-Tang vortex) data.

Uses the real Orszag-Tang outputs in orszang_tang_vortex/outputs/vtk/:
  - 200 matched mhd_w + mhd_bcc frame pairs (frames 1-200)
  - 400x400 grid, physics='mhd', tlim=2.0

Also uses fast synthetic MHD VTK pairs (8x8) for unit-level tests
that don't depend on disk I/O.

Organised into:
  TestMhdSimConstruction      – file discovery, physics detection
  TestMhdSimProperties        – grid, physics type, gamma, basename, fields
  TestMhdFrameNumbers         – frame_numbers, n_frames, first/last frame
  TestMhdFieldAccessors       – bx/by/bz present; hydro-only fields absent
  TestMhdFrameObject          – Frame with B-field arrays, temperature
  TestMhdTimeQueries          – times, frame_at, frames_between
  TestMhdUnitIntegration      – set_units scales B-fields correctly
  TestMhdSyntheticSimulation  – fast synthetic MHD tests
  TestMhdAthinput             – parsing <mhd> section
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ergane import Frame, SimulationData
from ergane.units import Units
from tests.conftest import make_synthetic_mhd_vtk_pair, make_synthetic_vtk


# ─────────────────────────────────────────────────────────────────────────────
# Construction & File Discovery
# ─────────────────────────────────────────────────────────────────────────────

class TestMhdSimConstruction:
    def test_constructs_from_ot_directory(self, mhd_sim):
        assert mhd_sim is not None

    def test_physics_is_mhd(self, mhd_sim):
        assert mhd_sim.physics == "mhd"

    def test_n_frames_200(self, mhd_sim):
        assert mhd_sim.n_frames == 200

    def test_params_loaded(self, mhd_sim):
        assert "mesh" in mhd_sim.params
        assert "mhd" in mhd_sim.params

    def test_synthetic_mhd_physics_detected(self, synthetic_mhd_sim):
        assert synthetic_mhd_sim.physics == "mhd"

    def test_synthetic_mhd_n_frames(self, synthetic_mhd_sim):
        assert synthetic_mhd_sim.n_frames == 1


# ─────────────────────────────────────────────────────────────────────────────
# Grid / Physics Properties
# ─────────────────────────────────────────────────────────────────────────────

class TestMhdSimProperties:
    def test_nx_400(self, mhd_sim):
        assert mhd_sim.nx == 400

    def test_ny_400(self, mhd_sim):
        assert mhd_sim.ny == 400

    def test_x1min(self, mhd_sim):
        assert mhd_sim.x1min == pytest.approx(-0.5)

    def test_x1max(self, mhd_sim):
        assert mhd_sim.x1max == pytest.approx(0.5)

    def test_x2min(self, mhd_sim):
        assert mhd_sim.x2min == pytest.approx(-0.5)

    def test_x2max(self, mhd_sim):
        assert mhd_sim.x2max == pytest.approx(0.5)

    def test_gamma_from_mhd_section(self, mhd_sim):
        assert mhd_sim.gamma == pytest.approx(1.666666667, rel=1e-6)

    def test_basename(self, mhd_sim):
        assert mhd_sim.basename == "OrszagTang"

    def test_fields_available_includes_bfield(self, mhd_sim):
        for f in ("bx", "by"):
            assert f in mhd_sim.fields_available

    def test_fields_available_includes_hydro(self, mhd_sim):
        for f in ("density", "pressure", "velx", "vely"):
            assert f in mhd_sim.fields_available

    def test_units_default_code(self, mhd_sim):
        assert mhd_sim.units.system == "code"

    def test_repr_shows_mhd(self, mhd_sim):
        assert "mhd" in repr(mhd_sim)


# ─────────────────────────────────────────────────────────────────────────────
# Frame Numbers
# ─────────────────────────────────────────────────────────────────────────────

class TestMhdFrameNumbers:
    def test_frame_numbers_sorted(self, mhd_sim):
        nums = mhd_sim.frame_numbers
        assert nums == sorted(nums)

    def test_first_frame_number_is_1(self, mhd_sim):
        """OT outputs start at 00001.vtk (no frame 0)."""
        assert mhd_sim.frame_numbers[0] == 1

    def test_last_frame_number_is_200(self, mhd_sim):
        assert mhd_sim.frame_numbers[-1] == 200

    def test_get_frame_1_returns_frame(self, mhd_sim):
        f = mhd_sim.get_frame(1)
        assert isinstance(f, Frame)

    def test_get_frame_200(self, mhd_sim):
        f = mhd_sim.get_frame(200)
        assert f.number == 200

    def test_get_frame_0_raises(self, mhd_sim):
        """Frame 0 doesn't exist in the OT dataset."""
        with pytest.raises(KeyError):
            mhd_sim.get_frame(0)

    def test_negative_indexing_is_last(self, mhd_sim):
        arr_neg  = mhd_sim.density[-1]
        arr_last = mhd_sim.density[200]
        np.testing.assert_array_equal(arr_neg, arr_last)


# ─────────────────────────────────────────────────────────────────────────────
# MHD Field Accessors
# ─────────────────────────────────────────────────────────────────────────────

class TestMhdFieldAccessors:
    def test_bx_returns_array(self, mhd_sim):
        arr = mhd_sim.bx[1]
        assert isinstance(arr, np.ndarray)

    def test_by_returns_array(self, mhd_sim):
        arr = mhd_sim.by[1]
        assert isinstance(arr, np.ndarray)

    def test_bx_shape(self, mhd_sim):
        """B-field arrays must be 2-D with shape (ny, nx)."""
        arr = mhd_sim.bx[1]
        assert arr.ndim == 2
        assert arr.shape == (400, 400)

    def test_by_shape(self, mhd_sim):
        assert mhd_sim.by[1].shape == (400, 400)

    def test_density_shape(self, mhd_sim):
        assert mhd_sim.density[1].shape == (400, 400)

    def test_pressure_shape(self, mhd_sim):
        assert mhd_sim.pressure[1].shape == (400, 400)

    def test_velx_shape(self, mhd_sim):
        assert mhd_sim.velx[1].shape == (400, 400)

    def test_vely_shape(self, mhd_sim):
        assert mhd_sim.vely[1].shape == (400, 400)

    def test_bx_slice(self, mhd_sim):
        result = mhd_sim.bx[0:3]   # positional slice → 3 arrays
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(a, np.ndarray) for a in result)

    def test_bx_negative_index(self, mhd_sim):
        arr_neg  = mhd_sim.bx[-1]
        arr_last = mhd_sim.bx[200]
        np.testing.assert_array_equal(arr_neg, arr_last)

    def test_bx_at_time(self, mhd_sim):
        arr = mhd_sim.bx.at_time(0.0)
        assert isinstance(arr, np.ndarray)

    def test_bx_between(self, mhd_sim):
        result = mhd_sim.bx.between(0.0, 0.05)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_hydro_only_field_absent_on_hydro_sim(self, synthetic_sim):
        """Cross-check: bx on a hydro sim must raise KeyError."""
        with pytest.raises(KeyError):
            _ = synthetic_sim.bx[0]

    def test_bz_accessor_exists(self, mhd_sim):
        """bz accessor is available even if the 2-D field is zero."""
        acc = mhd_sim.bz
        assert acc is not None

    def test_bx_bad_frame_raises(self, mhd_sim):
        with pytest.raises(KeyError):
            _ = mhd_sim.bx[9999]


# ─────────────────────────────────────────────────────────────────────────────
# Frame Object
# ─────────────────────────────────────────────────────────────────────────────

class TestMhdFrameObject:
    def test_frame_has_bx(self, mhd_first_frame):
        assert mhd_first_frame.bx is not None

    def test_frame_has_by(self, mhd_first_frame):
        assert mhd_first_frame.by is not None

    def test_bx_shape_on_frame(self, mhd_first_frame):
        assert mhd_first_frame.bx.shape == (400, 400)

    def test_by_shape_on_frame(self, mhd_first_frame):
        assert mhd_first_frame.by.shape == (400, 400)

    def test_frame_has_density(self, mhd_first_frame):
        assert mhd_first_frame.density is not None

    def test_frame_has_pressure(self, mhd_first_frame):
        assert mhd_first_frame.pressure is not None

    def test_frame_has_velx(self, mhd_first_frame):
        assert mhd_first_frame.velx is not None

    def test_frame_has_vely(self, mhd_first_frame):
        assert mhd_first_frame.vely is not None

    def test_frame_number_is_1(self, mhd_first_frame):
        assert mhd_first_frame.number == 1

    def test_frame_time_positive(self, mhd_first_frame):
        """Frame 1 must have t > 0 (OT output starts after t=0)."""
        assert mhd_first_frame.time > 0.0

    def test_frame_x_has_401_nodes(self, mhd_first_frame):
        assert len(mhd_first_frame.x) == 401

    def test_frame_y_has_401_nodes(self, mhd_first_frame):
        assert len(mhd_first_frame.y) == 401

    def test_xc_length(self, mhd_first_frame):
        assert len(mhd_first_frame.xc) == 400

    def test_yc_length(self, mhd_first_frame):
        assert len(mhd_first_frame.yc) == 400

    def test_temperature_available(self, mhd_first_frame):
        T = mhd_first_frame.temperature
        assert T is not None
        assert T.shape == (400, 400)

    def test_temperature_positive(self, mhd_first_frame):
        assert (mhd_first_frame.temperature > 0).all()

    def test_density_positive(self, mhd_first_frame):
        assert (mhd_first_frame.density > 0).all()

    def test_pressure_positive(self, mhd_first_frame):
        assert (mhd_first_frame.pressure > 0).all()

    def test_frame_repr_shows_bx(self, mhd_first_frame):
        r = repr(mhd_first_frame)
        assert "bx" in r

    def test_bx_by_finite(self, mhd_first_frame):
        assert np.all(np.isfinite(mhd_first_frame.bx))
        assert np.all(np.isfinite(mhd_first_frame.by))

    def test_frame_units_code(self, mhd_first_frame):
        assert mhd_first_frame.units.system == "code"


# ─────────────────────────────────────────────────────────────────────────────
# Time Queries
# ─────────────────────────────────────────────────────────────────────────────

class TestMhdTimeQueries:
    def test_times_length(self, mhd_sim):
        assert len(mhd_sim.times) == 200

    def test_times_monotonically_increasing(self, mhd_sim):
        assert np.all(np.diff(mhd_sim.times) >= 0)

    def test_times_start_positive(self, mhd_sim):
        """First OT frame is at t > 0."""
        assert mhd_sim.times[0] > 0.0

    def test_times_end_near_tlim(self, mhd_sim):
        """Last frame should be close to tlim=2.0."""
        assert mhd_sim.times[-1] == pytest.approx(2.0, abs=0.02)

    def test_frame_at_early_time(self, mhd_sim):
        f = mhd_sim.frame_at(t=mhd_sim.times[0])
        assert f.number == mhd_sim.frame_numbers[0]

    def test_frame_at_late_time(self, mhd_sim):
        f = mhd_sim.frame_at(t=mhd_sim.times[-1])
        assert f.number == mhd_sim.frame_numbers[-1]

    def test_frames_between_returns_frames(self, mhd_sim):
        t0, t1 = mhd_sim.times[0], mhd_sim.times[4]
        frames = mhd_sim.frames_between(t0, t1)
        assert len(frames) == 5

    def test_frames_between_all_in_range(self, mhd_sim):
        t0, t1 = 0.5, 1.0
        frames = mhd_sim.frames_between(t0, t1)
        for f in frames:
            assert t0 <= f.time <= t1

    def test_frames_between_empty_range(self, mhd_sim):
        frames = mhd_sim.frames_between(-100.0, -50.0)
        assert frames == []

    def test_bx_at_time(self, mhd_sim):
        arr = mhd_sim.bx.at_time(1.0)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (400, 400)

    def test_bx_between_include_times(self, mhd_sim):
        t0, t1 = mhd_sim.times[0], mhd_sim.times[2]
        result = mhd_sim.bx.between(t0, t1, include_times=True)
        assert len(result) == 3
        for t, arr in result:
            assert isinstance(t, float)
            assert isinstance(arr, np.ndarray)


# ─────────────────────────────────────────────────────────────────────────────
# Unit Integration (MHD-specific: B-field scaling)
# ─────────────────────────────────────────────────────────────────────────────

class TestMhdUnitIntegration:
    LENGTH   = 3.086e18
    DENSITY  = 1.67e-24
    VELOCITY = 1.0e5

    def _cgs(self):
        return Units.cgs(self.LENGTH, self.DENSITY, self.VELOCITY)

    def test_set_units_changes_system(self, mhd_sim):
        try:
            mhd_sim.set_units(self._cgs())
            assert mhd_sim.units.system == "CGS"
        finally:
            mhd_sim.set_units(Units.code())

    def test_bx_scale_applied(self, mhd_sim):
        """bx in code units vs bx in CGS should differ by the magnetic scale factor."""
        bx_code = mhd_sim.bx[1].copy()
        try:
            u = self._cgs()
            mhd_sim.set_units(u)
            bx_cgs = mhd_sim.bx[1]
            expected_scale = u.magnetic
            np.testing.assert_allclose(bx_cgs, bx_code * expected_scale, rtol=1e-5)
        finally:
            mhd_sim.set_units(Units.code())

    def test_by_scale_applied(self, mhd_sim):
        by_code = mhd_sim.by[1].copy()
        try:
            u = self._cgs()
            mhd_sim.set_units(u)
            by_cgs = mhd_sim.by[1]
            np.testing.assert_allclose(by_cgs, by_code * u.magnetic, rtol=1e-5)
        finally:
            mhd_sim.set_units(Units.code())

    def test_magnetic_scale_correct(self):
        u = self._cgs()
        expected = math.sqrt(self.DENSITY * self.VELOCITY ** 2)
        assert u.magnetic == pytest.approx(expected, rel=1e-6)

    def test_reset_to_code(self, mhd_sim):
        mhd_sim.set_units(self._cgs())
        mhd_sim.set_units(Units.code())
        assert mhd_sim.units.system == "code"
        # B-field values should be of order 1 in code units
        bx = mhd_sim.bx[1]
        assert bx.max() < 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic MHD (fast unit tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestMhdSyntheticSimulation:
    def test_n_frames_one(self, synthetic_mhd_sim):
        assert synthetic_mhd_sim.n_frames == 1

    def test_physics_mhd(self, synthetic_mhd_sim):
        assert synthetic_mhd_sim.physics == "mhd"

    def test_density_shape(self, synthetic_mhd_sim):
        arr = synthetic_mhd_sim.density[0]
        assert arr.shape == (8, 8)

    def test_bx_shape(self, synthetic_mhd_sim):
        arr = synthetic_mhd_sim.bx[0]
        assert arr.shape == (8, 8)

    def test_by_shape(self, synthetic_mhd_sim):
        arr = synthetic_mhd_sim.by[0]
        assert arr.shape == (8, 8)

    def test_density_positive(self, synthetic_mhd_sim):
        assert (synthetic_mhd_sim.density[0] > 0).all()

    def test_pressure_positive(self, synthetic_mhd_sim):
        assert (synthetic_mhd_sim.pressure[0] > 0).all()

    def test_bx_finite(self, synthetic_mhd_sim):
        assert np.all(np.isfinite(synthetic_mhd_sim.bx[0]))

    def test_frame_has_bx_and_by(self, synthetic_mhd_sim):
        f = synthetic_mhd_sim.get_frame(0)
        assert f.bx is not None
        assert f.by is not None

    def test_bx_values_match_input(self, tmp_path):
        """Round-trip: synthetic Bcc written → read back via bx should equal Bcc[...,0]."""
        rng = np.random.default_rng(99)
        bcc = rng.uniform(-1.0, 1.0, (4, 4, 3)).astype(np.float32)
        rho = rng.uniform(0.5, 2.0, (4, 4)).astype(np.float32)
        w_bytes = make_synthetic_vtk(ncx=4, ncy=4, fields={"rho": rho})
        bcc_bytes = make_synthetic_vtk(ncx=4, ncy=4, fields={"Bcc": bcc})
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        (vtk_dir / "r.mhd_w.00000.vtk").write_bytes(w_bytes)
        (vtk_dir / "r.mhd_bcc.00000.vtk").write_bytes(bcc_bytes)
        s = SimulationData(datafolder=str(tmp_path))
        f = s.get_frame(0)
        np.testing.assert_allclose(f.bx, bcc[..., 0].astype(np.float64), rtol=1e-5)
        np.testing.assert_allclose(f.by, bcc[..., 1].astype(np.float64), rtol=1e-5)

    def test_multi_frame_mhd_synthetic(self, tmp_path):
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        for i in (1, 2, 3):
            w, bcc = make_synthetic_mhd_vtk_pair(time=float(i) * 0.01)
            (vtk_dir / f"r.mhd_w.{i:05d}.vtk").write_bytes(w)
            (vtk_dir / f"r.mhd_bcc.{i:05d}.vtk").write_bytes(bcc)
        s = SimulationData(datafolder=str(tmp_path))
        assert s.n_frames == 3
        assert s.physics == "mhd"
        assert s.frame_numbers == [1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# MHD Athinput
# ─────────────────────────────────────────────────────────────────────────────

class TestMhdAthinput:
    def test_mhd_section_present(self, ot_athinput_path):
        from ergane.athinput_parser import parse_athinput
        params = parse_athinput(ot_athinput_path)
        assert "mhd" in params

    def test_gamma_in_mhd_section(self, ot_athinput_path):
        from ergane.athinput_parser import parse_athinput
        params = parse_athinput(ot_athinput_path)
        assert float(params["mhd"]["gamma"]) == pytest.approx(1.666666667, rel=1e-6)

    def test_grid_400x400(self, ot_athinput_path):
        from ergane.athinput_parser import parse_athinput
        params = parse_athinput(ot_athinput_path)
        assert int(params["mesh"]["nx1"]) == 400
        assert int(params["mesh"]["nx2"]) == 400

    def test_tlim_2(self, ot_athinput_path):
        from ergane.athinput_parser import parse_athinput
        params = parse_athinput(ot_athinput_path)
        assert float(params["time"]["tlim"]) == pytest.approx(2.0)

    def test_basename_ot(self, ot_athinput_path):
        from ergane.athinput_parser import parse_athinput
        params = parse_athinput(ot_athinput_path)
        assert params["job"]["basename"] == "OrszagTang"

    def test_mhd_sim_gamma_matches_athinput(self, mhd_sim):
        assert mhd_sim.gamma == pytest.approx(
            float(mhd_sim.params["mhd"]["gamma"]), rel=1e-6
        )
