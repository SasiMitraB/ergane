"""
tests/test_simulation_data.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for ergane.SimulationData and ergane.Frame (2-D only).

Organised into:
  TestSimulationDataConstruction   – __init__, file discovery, params
  TestSimulationDataProperties     – grid, physics, basename, gamma, times
  TestFrameNumberIndexing          – frame_numbers, n_frames, get_frame
  TestFieldAccessors               – density/pressure/vel accessors + slice/neg
  TestFieldAccessorTimeBased       – .at_time(), .between()
  TestFrameObject                  – Frame dataclass, xc/yc, temperature
  TestFrameScalarFields            – passive scalars via Frame.scalars
  TestTimeQueries                  – frame_at, frames_between
  TestUnitIntegration              – set_units, scaled field values
  TestErrorHandling                – missing files, bad frame numbers, etc.
  TestSyntheticSimulation          – fast unit tests using in-memory VTK
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ergane import Frame, SimulationData
from ergane.units import Units

# Expose helper for variant synthetic sims
from tests.conftest import make_synthetic_vtk


# ─────────────────────────────────────────────────────────────────────────────
# Construction & File Discovery
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulationDataConstruction:
    def test_constructs_with_athinp_and_datafolder(self, sim):
        assert sim is not None

    def test_constructs_without_athinp(self, tmp_path, synthetic_vtk_bytes):
        """No athinput provided: SimulationData should still load."""
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        (vtk_dir / "run.hydro_w.00000.vtk").write_bytes(synthetic_vtk_bytes)
        s = SimulationData(datafolder=str(tmp_path))
        assert s is not None

    def test_nested_vtk_subdir_discovered(self, tmp_path, synthetic_vtk_bytes):
        """VTK files in a vtk/ subdirectory are found automatically."""
        (tmp_path / "vtk").mkdir()
        (tmp_path / "vtk" / "run.hydro_w.00000.vtk").write_bytes(synthetic_vtk_bytes)
        s = SimulationData(datafolder=str(tmp_path))
        assert s.n_frames == 1

    def test_flat_vtk_dir_discovered(self, tmp_path, synthetic_vtk_bytes):
        """VTK files directly in datafolder are also accepted."""
        (tmp_path / "run.hydro_w.00000.vtk").write_bytes(synthetic_vtk_bytes)
        s = SimulationData(datafolder=str(tmp_path))
        assert s.n_frames == 1

    def test_raises_if_no_files_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SimulationData(datafolder=str(tmp_path))

    def test_params_loaded_from_athinp(self, sim):
        assert "mesh" in sim.params
        assert "hydro" in sim.params

    def test_params_empty_without_athinp(self, tmp_path, synthetic_vtk_bytes):
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        (vtk_dir / "run.hydro_w.00000.vtk").write_bytes(synthetic_vtk_bytes)
        s = SimulationData(datafolder=str(tmp_path))
        assert isinstance(s.params, dict)  # may be empty but not None


# ─────────────────────────────────────────────────────────────────────────────
# Grid / Physics Properties
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulationDataProperties:
    def test_nx_from_athinp(self, sim):
        assert sim.nx == 256

    def test_ny_from_athinp(self, sim):
        assert sim.ny == 512

    def test_x1min(self, sim):
        assert sim.x1min == pytest.approx(-0.5)

    def test_x1max(self, sim):
        assert sim.x1max == pytest.approx(0.5)

    def test_x2min(self, sim):
        assert sim.x2min == pytest.approx(-1.0)

    def test_x2max(self, sim):
        assert sim.x2max == pytest.approx(1.0)

    def test_physics_hydro(self, sim):
        assert sim.physics == "hydro"

    def test_gamma_from_athinp(self, sim):
        assert sim.gamma == pytest.approx(1.666667, rel=1e-5)

    def test_gamma_is_none_without_athinp(self, tmp_path, synthetic_vtk_bytes):
        (tmp_path / "vtk").mkdir()
        (tmp_path / "vtk" / "r.hydro_w.00000.vtk").write_bytes(synthetic_vtk_bytes)
        s = SimulationData(datafolder=str(tmp_path))
        assert s.gamma is None

    def test_basename_from_athinp(self, sim):
        assert sim.basename == "KH"

    def test_n_frames_301(self, sim):
        assert sim.n_frames == 301

    def test_units_default_code(self, sim):
        assert sim.units.system == "code"

    def test_fields_available_includes_density(self, sim):
        assert "density" in sim.fields_available

    def test_fields_available_includes_pressure(self, sim):
        assert "pressure" in sim.fields_available

    def test_fields_available_includes_velocities(self, sim):
        for f in ("velx", "vely"):
            assert f in sim.fields_available

    def test_fields_available_includes_temperature(self, sim):
        assert "temperature" in sim.fields_available

    def test_repr_is_string(self, sim):
        r = repr(sim)
        assert isinstance(r, str)
        assert "SimulationData" in r

    def test_nx_fallback_without_athinp(self, tmp_path, synthetic_vtk_bytes):
        """Without athinput, nx should be inferred from the first VTK frame."""
        (tmp_path / "vtk").mkdir()
        (tmp_path / "vtk" / "r.hydro_w.00000.vtk").write_bytes(synthetic_vtk_bytes)
        s = SimulationData(datafolder=str(tmp_path))
        # Synthetic has ncx=8
        assert s.nx == 8

    def test_ny_fallback_without_athinp(self, tmp_path, synthetic_vtk_bytes):
        (tmp_path / "vtk").mkdir()
        (tmp_path / "vtk" / "r.hydro_w.00000.vtk").write_bytes(synthetic_vtk_bytes)
        s = SimulationData(datafolder=str(tmp_path))
        assert s.ny == 16


# ─────────────────────────────────────────────────────────────────────────────
# Frame Numbers
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameNumberIndexing:
    def test_frame_numbers_is_sorted_list(self, sim):
        nums = sim.frame_numbers
        assert nums == sorted(nums)

    def test_frame_numbers_start_at_zero(self, sim):
        assert sim.frame_numbers[0] == 0

    def test_frame_numbers_end_at_300(self, sim):
        assert sim.frame_numbers[-1] == 300

    def test_get_frame_returns_frame_object(self, sim):
        f = sim.get_frame(0)
        assert isinstance(f, Frame)

    def test_get_frame_number_attribute(self, sim):
        f = sim.get_frame(0)
        assert f.number == 0

    def test_get_frame_last(self, sim):
        f = sim.get_frame(300)
        assert f.number == 300

    def test_get_frame_nonexistent_raises(self, sim):
        with pytest.raises(KeyError):
            sim.get_frame(9999)


# ─────────────────────────────────────────────────────────────────────────────
# Field Accessors
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldAccessors:
    def test_density_integer_index(self, sim):
        arr = sim.density[0]
        assert isinstance(arr, np.ndarray)

    def test_density_shape_2d(self, sim):
        arr = sim.density[0]
        assert arr.ndim == 2
        assert arr.shape == (512, 256)

    def test_pressure_shape_matches_density(self, sim):
        assert sim.pressure[0].shape == sim.density[0].shape

    def test_velx_shape(self, sim):
        assert sim.velx[0].shape == (512, 256)

    def test_vely_shape(self, sim):
        assert sim.vely[0].shape == (512, 256)

    def test_negative_index_returns_last_frame(self, sim):
        arr_neg  = sim.density[-1]
        arr_last = sim.density[300]
        np.testing.assert_array_equal(arr_neg, arr_last)

    def test_slice_indexing_returns_list(self, sim):
        result = sim.density[0:3]
        assert isinstance(result, list)
        assert len(result) == 3

    def test_slice_elements_are_arrays(self, sim):
        result = sim.density[0:3]
        for arr in result:
            assert isinstance(arr, np.ndarray)

    def test_invalid_index_type_raises(self, sim):
        with pytest.raises(TypeError):
            _ = sim.density["bad"]

    def test_mhd_field_on_hydro_sim_raises(self, sim):
        """Requesting bx on a hydro simulation must raise KeyError."""
        with pytest.raises(KeyError):
            _ = sim.bx[0]

    def test_accessor_repr_is_string(self, sim):
        r = repr(sim.density)
        assert isinstance(r, str)


# ─────────────────────────────────────────────────────────────────────────────
# Time-Based Field Access
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldAccessorTimeBased:
    def test_at_time_returns_array(self, sim):
        arr = sim.density.at_time(0.0)
        assert isinstance(arr, np.ndarray)

    def test_at_time_nearest_frame0(self, sim):
        arr = sim.density.at_time(0.0)
        np.testing.assert_array_equal(arr, sim.density[0])

    def test_at_time_nearest_last(self, sim):
        last_time = sim.times[-1]
        arr = sim.density.at_time(last_time + 999)   # beyond end → clamp to last
        np.testing.assert_array_equal(arr, sim.density[-1])

    def test_between_returns_list(self, sim):
        result = sim.density.between(0.0, 0.1)
        assert isinstance(result, list)

    def test_between_all_arrays(self, sim):
        result = sim.density.between(0.0, 0.1)
        for arr in result:
            assert isinstance(arr, np.ndarray)

    def test_between_empty_range_returns_empty(self, sim):
        # Before any data
        result = sim.density.between(-100.0, -50.0)
        assert result == []

    def test_between_include_times(self, sim):
        result = sim.density.between(0.0, 0.1, include_times=True)
        for item in result:
            t, arr = item
            assert isinstance(t, float)
            assert isinstance(arr, np.ndarray)


# ─────────────────────────────────────────────────────────────────────────────
# Frame Object
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameObject:
    def test_frame_has_density(self, first_frame):
        assert first_frame.density is not None

    def test_frame_has_pressure(self, first_frame):
        assert first_frame.pressure is not None

    def test_frame_has_velx(self, first_frame):
        assert first_frame.velx is not None

    def test_frame_has_vely(self, first_frame):
        assert first_frame.vely is not None

    def test_frame_bx_none_for_hydro(self, first_frame):
        assert first_frame.bx is None

    def test_frame_time_is_float(self, first_frame):
        assert isinstance(first_frame.time, float)

    def test_frame_time_frame0_zero(self, first_frame):
        assert first_frame.time == pytest.approx(0.0, abs=1e-9)

    def test_frame_x_array(self, first_frame):
        assert isinstance(first_frame.x, np.ndarray)

    def test_frame_y_array(self, first_frame):
        assert isinstance(first_frame.y, np.ndarray)

    def test_xc_length(self, first_frame):
        """Cell-centre x coords have one fewer point than face coords."""
        assert len(first_frame.xc) == len(first_frame.x) - 1

    def test_yc_length(self, first_frame):
        assert len(first_frame.yc) == len(first_frame.y) - 1

    def test_xc_values_between_faces(self, first_frame):
        """Cell centres should lie strictly between face coords."""
        assert np.all(first_frame.xc >= first_frame.x[:-1])
        assert np.all(first_frame.xc <= first_frame.x[1:])

    def test_temperature_property_returns_array(self, first_frame):
        T = first_frame.temperature
        assert T is not None
        assert isinstance(T, np.ndarray)

    def test_temperature_shape_matches_density(self, first_frame):
        assert first_frame.temperature.shape == first_frame.density.shape

    def test_temperature_positive(self, first_frame):
        T = first_frame.temperature
        assert (T > 0).all()

    def test_temperature_none_when_no_pressure(self):
        """Frame without pressure must return None for temperature."""
        f = Frame(
            number=0,
            time=0.0,
            x=np.array([0.0, 1.0]),
            y=np.array([0.0, 1.0]),
            density=np.ones((1, 1)),
            pressure=None,
        )
        assert f.temperature is None

    def test_frame_repr_is_string(self, first_frame):
        r = repr(first_frame)
        assert isinstance(r, str)
        assert "Frame" in r

    def test_getattr_missing_raises(self, first_frame):
        with pytest.raises(AttributeError):
            _ = first_frame.nonexistent_field_xyz


# ─────────────────────────────────────────────────────────────────────────────
# Passive Scalars
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameScalarFields:
    def test_scalar_00_in_frame(self, first_frame):
        """KH2D athinput has nscalars=1, so scalar_00 must exist."""
        assert "scalar_00" in first_frame.scalars

    def test_scalar_accessible_as_attribute(self, first_frame):
        arr = first_frame.scalar_00
        assert isinstance(arr, np.ndarray)

    def test_scalar_shape_matches_density(self, first_frame):
        assert first_frame.scalar_00.shape == first_frame.density.shape

    def test_scalar_00_in_fields_available(self, sim):
        assert "scalar_00" in sim.fields_available


# ─────────────────────────────────────────────────────────────────────────────
# Time-Based Frame Queries
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeQueries:
    def test_times_array_length(self, sim):
        assert len(sim.times) == sim.n_frames

    def test_times_start_at_zero(self, sim):
        assert sim.times[0] == pytest.approx(0.0, abs=1e-9)

    def test_times_monotonically_increasing(self, sim):
        assert np.all(np.diff(sim.times) >= 0)

    def test_frame_at_returns_frame(self, sim):
        f = sim.frame_at(t=0.0)
        assert isinstance(f, Frame)

    def test_frame_at_time_zero(self, sim):
        f = sim.frame_at(t=0.0)
        assert f.number == 0

    def test_frame_at_last_time(self, sim):
        f = sim.frame_at(t=sim.times[-1])
        assert f.number == sim.frame_numbers[-1]

    def test_frames_between_returns_list(self, sim):
        frames = sim.frames_between(0.0, 0.1)
        assert isinstance(frames, list)

    def test_frames_between_sorted(self, sim):
        frames = sim.frames_between(0.0, 1.0)
        times = [f.time for f in frames]
        assert times == sorted(times)

    def test_frames_between_empty_range(self, sim):
        frames = sim.frames_between(-100.0, -50.0)
        assert frames == []

    def test_frames_between_all_in_range(self, sim):
        t0, t1 = 0.0, 0.5
        frames = sim.frames_between(t0, t1)
        for f in frames:
            assert t0 <= f.time <= t1


# ─────────────────────────────────────────────────────────────────────────────
# Unit Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestUnitIntegration:
    LENGTH   = 3.086e18
    DENSITY  = 1.67e-24
    VELOCITY = 1.0e5

    def test_set_units_changes_system(self, sim):
        try:
            u = Units.cgs(self.LENGTH, self.DENSITY, self.VELOCITY)
            sim.set_units(u)
            assert sim.units.system == "CGS"
        finally:
            sim.set_units(Units.code())

    def test_set_units_scales_density(self, sim):
        try:
            u = Units.cgs(self.LENGTH, self.DENSITY, self.VELOCITY)
            sim.set_units(u)
            arr = sim.density[0]
            # All values should be in rough CGS density range (order 1e-24 to 1e-23)
            assert arr.min() > 0
            assert arr.max() < 1.0   # should not be of order 1 in CGS
        finally:
            sim.set_units(Units.code())

    def test_reset_to_code_units(self, sim):
        sim.set_units(Units.cgs(self.LENGTH, self.DENSITY, self.VELOCITY))
        sim.set_units(Units.code())
        assert sim.units.system == "code"
        # Code-unit density values should be of order 1
        arr = sim.density[0]
        assert arr.mean() == pytest.approx(1.5, abs=0.5)

    def test_frame_time_scaled(self, sim):
        try:
            u = Units.cgs(self.LENGTH, self.DENSITY, self.VELOCITY)
            sim.set_units(u)
            f = sim.get_frame(0)
            # t=0 in code → 0 in any units
            assert f.time == pytest.approx(0.0, abs=1e-30)
        finally:
            sim.set_units(Units.code())

    def test_set_units_clears_accessor_cache(self, sim):
        """After set_units, a freshly fetched accessor should reflect new units."""
        try:
            u = Units.cgs(self.LENGTH, self.DENSITY, self.VELOCITY)
            sim.set_units(u)
            acc = sim.density
            assert acc._sim.units.system == "CGS"
        finally:
            sim.set_units(Units.code())


# ─────────────────────────────────────────────────────────────────────────────
# Error Handling
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_invalid_datafolder_raises(self):
        with pytest.raises(FileNotFoundError):
            SimulationData(datafolder="/tmp/__nonexistent_dir_xyz__")

    def test_get_frame_bad_number_raises(self, sim):
        with pytest.raises(KeyError):
            sim.get_frame(99999)

    def test_density_bad_integer_raises(self, sim):
        with pytest.raises(KeyError):
            _ = sim.density[99999]

    def test_accessor_bad_type_raises(self, sim):
        with pytest.raises(TypeError):
            _ = sim.density[1.5]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Simulation (fast unit tests, no real I/O)
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticSimulation:
    def test_n_frames_single(self, synthetic_sim):
        assert synthetic_sim.n_frames == 1

    def test_frame_numbers(self, synthetic_sim):
        assert synthetic_sim.frame_numbers == [0]

    def test_density_shape(self, synthetic_sim):
        arr = synthetic_sim.density[0]
        assert arr.shape == (16, 8)

    def test_velx_shape(self, synthetic_sim):
        arr = synthetic_sim.velx[0]
        assert arr.shape == (16, 8)

    def test_vely_shape(self, synthetic_sim):
        arr = synthetic_sim.vely[0]
        assert arr.shape == (16, 8)

    def test_density_positive(self, synthetic_sim):
        arr = synthetic_sim.density[0]
        assert (arr > 0).all()

    def test_multi_frame_synthetic(self, tmp_path):
        """Check frame ordering when multiple synthetic frames are present."""
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        for i in (0, 5, 10):
            blob = make_synthetic_vtk(time=float(i) * 0.5)
            (vtk_dir / f"run.hydro_w.{i:05d}.vtk").write_bytes(blob)
        s = SimulationData(datafolder=str(tmp_path))
        assert s.n_frames == 3
        assert s.frame_numbers == [0, 5, 10]

    def test_time_index_synthetic(self, tmp_path):
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        times = [0.0, 1.0, 2.0]
        for i, t in enumerate(times):
            blob = make_synthetic_vtk(time=t)
            (vtk_dir / f"run.hydro_w.{i:05d}.vtk").write_bytes(blob)
        s = SimulationData(datafolder=str(tmp_path))
        np.testing.assert_allclose(s.times, np.array(times), atol=1e-4)

    def test_frame_at_synthetic(self, tmp_path):
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        blob0 = make_synthetic_vtk(time=0.0)
        blob1 = make_synthetic_vtk(time=1.0)
        (vtk_dir / "run.hydro_w.00000.vtk").write_bytes(blob0)
        (vtk_dir / "run.hydro_w.00001.vtk").write_bytes(blob1)
        s = SimulationData(datafolder=str(tmp_path))
        f = s.frame_at(t=0.9)
        assert f.number == 1    # nearest to 0.9 is frame 1 (t=1.0)
