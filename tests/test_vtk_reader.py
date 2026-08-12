"""
tests/test_vtk_reader.py
~~~~~~~~~~~~~~~~~~~~~~~~
Tests for ergane.vtk_reader:
  - parse_athena_vtk      (full parse)
  - read_vtk_time         (header-only fast read)
  - extract_frame_num     (filename integer extraction)
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import pytest

from ergane.vtk_reader import extract_frame_num, parse_athena_vtk, read_vtk_time


# ── Helpers ────────────────────────────────────────────────────────────────────

# conftest.make_synthetic_vtk is imported via conftest's module-level alias.
# We access it through the pytest fixture `synthetic_vtk_bytes` and also
# call the helper directly for parametrised variant tests.
from tests.conftest import make_synthetic_vtk


# ── extract_frame_num ──────────────────────────────────────────────────────────

class TestExtractFrameNum:
    @pytest.mark.parametrize("filename, expected", [
        ("KH.hydro_w.00000.vtk",  0),
        ("KH.hydro_w.00042.vtk", 42),
        ("KH.hydro_w.00300.vtk", 300),
        ("KH.mhd_w.00001.vtk",    1),
        ("KH.mhd_bcc.00007.vtk",  7),
        ("run.hydro_w.00099.bin", 99),   # also handles .bin
    ])
    def test_known_filenames(self, filename, expected):
        assert extract_frame_num(Path(filename)) == expected

    def test_no_match_returns_minus_one(self):
        assert extract_frame_num(Path("nopadding.vtk")) == -1

    def test_accepts_path_with_directory(self):
        p = Path("/some/dir/KH.hydro_w.00005.vtk")
        assert extract_frame_num(p) == 5


# ── read_vtk_time ─────────────────────────────────────────────────────────────

class TestReadVtkTime:
    def test_returns_float(self, vtk_dir):
        first = sorted(vtk_dir.glob("*.vtk"))[0]
        t = read_vtk_time(first)
        assert isinstance(t, float)

    def test_frame_zero_time_is_zero(self, vtk_dir):
        frame0 = vtk_dir / "KH.hydro_w.00000.vtk"
        t = read_vtk_time(frame0)
        assert t == pytest.approx(0.0, abs=1e-9)

    def test_time_increases_monotonically(self, vtk_dir):
        files = sorted(vtk_dir.glob("KH.hydro_w.*.vtk"))[:10]
        times = [read_vtk_time(f) for f in files]
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))

    def test_synthetic_time_parsed_correctly(self, tmp_path):
        blob = make_synthetic_vtk(time=3.14159)
        p = tmp_path / "t.hydro_w.00001.vtk"
        p.write_bytes(blob)
        assert read_vtk_time(p) == pytest.approx(3.14159, rel=1e-4)

    def test_accepts_string_path(self, vtk_dir):
        first = str(sorted(vtk_dir.glob("*.vtk"))[0])
        t = read_vtk_time(first)
        assert isinstance(t, float)

    def test_returns_zero_if_no_time_token(self, tmp_path):
        # Write a VTK-like file whose comment line has no 'time=' token
        blob = make_synthetic_vtk(time=0.0)
        # Replace second line with one that has no time=
        lines = blob.split(b"\n")
        lines[1] = b"CONSERVED vars  no_time_here"
        p = tmp_path / "notime.hydro_w.00000.vtk"
        p.write_bytes(b"\n".join(lines))
        assert read_vtk_time(p) == pytest.approx(0.0)


# ── parse_athena_vtk ──────────────────────────────────────────────────────────

class TestParseAthenaVtk:
    """Full parse of synthetic 8×16 blobs and a real example frame."""

    # ── Return structure ──────────────────────────────────────────────────────

    def test_returns_dict_with_expected_keys(self, synthetic_vtk_path):
        result = parse_athena_vtk(synthetic_vtk_path)
        assert set(result.keys()) >= {"time", "x", "y", "fields"}

    def test_time_value(self, synthetic_vtk_path):
        result = parse_athena_vtk(synthetic_vtk_path)
        assert result["time"] == pytest.approx(1.23, rel=1e-4)

    # ── Coordinate arrays ─────────────────────────────────────────────────────

    def test_x_coord_length(self, synthetic_vtk_path):
        """x has ncx+1 nodes (node coords, not cell-centre coords)."""
        result = parse_athena_vtk(synthetic_vtk_path)
        # synth vtk has ncx=8, so x should have 9 nodes
        assert len(result["x"]) == 9

    def test_y_coord_length(self, synthetic_vtk_path):
        """y has ncy+1 nodes."""
        result = parse_athena_vtk(synthetic_vtk_path)
        assert len(result["y"]) == 17

    def test_x_monotonically_increasing(self, synthetic_vtk_path):
        result = parse_athena_vtk(synthetic_vtk_path)
        assert np.all(np.diff(result["x"]) > 0)

    def test_y_monotonically_increasing(self, synthetic_vtk_path):
        result = parse_athena_vtk(synthetic_vtk_path)
        assert np.all(np.diff(result["y"]) > 0)

    def test_x_range(self, synthetic_vtk_path):
        result = parse_athena_vtk(synthetic_vtk_path)
        assert result["x"][0] == pytest.approx(-0.5, abs=1e-6)

    def test_y_range(self, synthetic_vtk_path):
        result = parse_athena_vtk(synthetic_vtk_path)
        assert result["y"][0] == pytest.approx(-1.0, abs=1e-6)

    # ── Fields ────────────────────────────────────────────────────────────────

    def test_fields_is_dict(self, synthetic_vtk_path):
        result = parse_athena_vtk(synthetic_vtk_path)
        assert isinstance(result["fields"], dict)

    def test_scalar_field_shape(self, synthetic_vtk_path):
        """rho (density scalar) must be 2-D with shape (ncy, ncx)."""
        result = parse_athena_vtk(synthetic_vtk_path)
        assert result["fields"]["rho"].shape == (16, 8)

    def test_vector_field_shape(self, synthetic_vtk_path):
        """vel (velocity vector) must be 3-D with shape (ncy, ncx, 3)."""
        result = parse_athena_vtk(synthetic_vtk_path)
        assert result["fields"]["vel"].shape == (16, 8, 3)

    def test_dtype_default_float64(self, synthetic_vtk_path):
        result = parse_athena_vtk(synthetic_vtk_path)
        for arr in result["fields"].values():
            assert arr.dtype == np.float64

    def test_dtype_float32_respected(self, synthetic_vtk_path):
        result = parse_athena_vtk(synthetic_vtk_path, dtype=np.float32)
        for arr in result["fields"].values():
            assert arr.dtype == np.float32

    def test_values_in_reasonable_range(self, synthetic_vtk_path):
        result = parse_athena_vtk(synthetic_vtk_path)
        rho = result["fields"]["rho"]
        assert rho.min() > 0, "Density must be positive"
        assert rho.max() < 10.0, "Density value out of expected range"

    # ── Round-trip value fidelity ─────────────────────────────────────────────

    def test_scalar_roundtrip(self, tmp_path):
        """Values written to the synthetic VTK should survive the round-trip."""
        rng = np.random.default_rng(0)
        rho_expected = rng.uniform(1.0, 5.0, size=(4, 4)).astype(np.float32)
        blob = make_synthetic_vtk(ncx=4, ncy=4, fields={"rho": rho_expected})
        p = tmp_path / "rtrip.hydro_w.00000.vtk"
        p.write_bytes(blob)
        result = parse_athena_vtk(p, dtype=np.float32)
        np.testing.assert_allclose(result["fields"]["rho"], rho_expected, rtol=1e-5)

    # ── Real-data smoke tests ─────────────────────────────────────────────────

    def test_real_frame0_time(self, vtk_dir):
        frame0 = vtk_dir / "KH.hydro_w.00000.vtk"
        result = parse_athena_vtk(frame0)
        assert result["time"] == pytest.approx(0.0, abs=1e-9)

    def test_real_frame0_grid_shape(self, vtk_dir):
        """KH2D: 256 x 512 cells → x has 257 nodes, y has 513 nodes."""
        frame0 = vtk_dir / "KH.hydro_w.00000.vtk"
        result = parse_athena_vtk(frame0)
        assert len(result["x"]) == 257
        assert len(result["y"]) == 513

    def test_real_frame0_has_density(self, vtk_dir):
        frame0 = vtk_dir / "KH.hydro_w.00000.vtk"
        result = parse_athena_vtk(frame0)
        # May come as 'rho' or 'dens'
        assert ("rho" in result["fields"]) or ("dens" in result["fields"])

    def test_real_density_all_positive(self, vtk_dir):
        frame0 = vtk_dir / "KH.hydro_w.00000.vtk"
        result = parse_athena_vtk(frame0)
        rho = result["fields"].get("rho", result["fields"].get("dens"))
        assert (rho > 0).all()
