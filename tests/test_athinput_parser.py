"""
tests/test_athinput_parser.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for ergane.athinput_parser.parse_athinput and typed().
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ergane.athinput_parser import parse_athinput, typed


# ── Basic parse of the real KH2D athinput ─────────────────────────────────────

class TestParseAthinputRealFile:
    """Tests against the bundled kh2d-sin.athinput example."""

    def test_returns_dict(self, athinput_path):
        params = parse_athinput(athinput_path)
        assert isinstance(params, dict)

    def test_sections_present(self, athinput_path):
        params = parse_athinput(athinput_path)
        for section in ("job", "mesh", "time", "hydro", "problem"):
            assert section in params, f"Section <{section}> missing"

    def test_section_keys_are_lowercase(self, athinput_path):
        params = parse_athinput(athinput_path)
        for section in params:
            assert section == section.lower()

    def test_mesh_grid_size(self, athinput_path):
        params = parse_athinput(athinput_path)
        assert params["mesh"]["nx1"] == "256"
        assert params["mesh"]["nx2"] == "512"

    def test_mesh_domain_bounds(self, athinput_path):
        params = parse_athinput(athinput_path)
        assert float(params["mesh"]["x1min"]) == pytest.approx(-0.5)
        assert float(params["mesh"]["x1max"]) == pytest.approx(0.5)
        assert float(params["mesh"]["x2min"]) == pytest.approx(-1.0)
        assert float(params["mesh"]["x2max"]) == pytest.approx(1.0)

    def test_time_limit(self, athinput_path):
        params = parse_athinput(athinput_path)
        assert float(params["time"]["tlim"]) == pytest.approx(6.0)

    def test_hydro_gamma(self, athinput_path):
        params = parse_athinput(athinput_path)
        assert float(params["hydro"]["gamma"]) == pytest.approx(1.666667, rel=1e-5)

    def test_job_basename(self, athinput_path):
        params = parse_athinput(athinput_path)
        assert params["job"]["basename"] == "KH"

    def test_inline_comments_stripped(self, athinput_path):
        """Values must not contain '#' characters."""
        params = parse_athinput(athinput_path)
        for section, kv in params.items():
            for key, val in kv.items():
                assert "#" not in val, f"Inline comment not stripped: [{section}]{key}={val!r}"


# ── Synthetic in-memory parsing ────────────────────────────────────────────────

class TestParseAthinputSynthetic:
    """Use tmp_path to write ad-hoc athinput files and verify edge cases."""

    def _write(self, tmp_path, content: str) -> Path:
        p = tmp_path / "test.athinput"
        p.write_text(content, encoding="utf-8")
        return p

    def test_empty_file_returns_empty_dict(self, tmp_path):
        p = self._write(tmp_path, "")
        assert parse_athinput(p) == {}

    def test_comment_only_file(self, tmp_path):
        p = self._write(tmp_path, "# This is a comment\n# Another comment\n")
        assert parse_athinput(p) == {}

    def test_single_section(self, tmp_path):
        content = "<mesh>\nnx1 = 64\nnx2 = 128\n"
        p = self._write(tmp_path, content)
        params = parse_athinput(p)
        assert params["mesh"]["nx1"] == "64"
        assert params["mesh"]["nx2"] == "128"

    def test_value_after_inline_comment(self, tmp_path):
        content = "<mesh>\nnx1 = 64  # number of zones\n"
        p = self._write(tmp_path, content)
        params = parse_athinput(p)
        assert params["mesh"]["nx1"] == "64"

    def test_multiple_sections(self, tmp_path):
        content = "<mesh>\nnx1 = 32\n<hydro>\ngamma = 1.4\n"
        p = self._write(tmp_path, content)
        params = parse_athinput(p)
        assert "mesh" in params
        assert "hydro" in params

    def test_section_keys_with_whitespace(self, tmp_path):
        content = "<mesh>\n  nx1   =   64  \n"
        p = self._write(tmp_path, content)
        params = parse_athinput(p)
        assert params["mesh"]["nx1"] == "64"

    def test_string_path_accepted(self, athinput_path):
        """parse_athinput must accept str as well as Path."""
        params = parse_athinput(str(athinput_path))
        assert "mesh" in params

    def test_section_name_case_normalised(self, tmp_path):
        content = "<Mesh>\nnx1 = 10\n"
        p = self._write(tmp_path, content)
        params = parse_athinput(p)
        assert "mesh" in params


# ── typed() helper ─────────────────────────────────────────────────────────────

class TestTyped:
    def test_int_cast(self, athinput_path):
        params = parse_athinput(athinput_path)
        assert typed(params, "mesh", "nx1", int) == 256

    def test_float_cast(self, athinput_path):
        params = parse_athinput(athinput_path)
        assert typed(params, "mesh", "x1min", float) == pytest.approx(-0.5)

    def test_str_default(self, athinput_path):
        params = parse_athinput(athinput_path)
        assert typed(params, "job", "basename") == "KH"

    def test_missing_section_raises(self, athinput_path):
        params = parse_athinput(athinput_path)
        with pytest.raises(KeyError):
            typed(params, "nonexistent_section", "key")

    def test_missing_key_raises(self, athinput_path):
        params = parse_athinput(athinput_path)
        with pytest.raises(KeyError):
            typed(params, "mesh", "nonexistent_key")
