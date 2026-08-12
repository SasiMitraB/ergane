"""
tests/conftest.py
~~~~~~~~~~~~~~~~~
Shared pytest fixtures for the ergane test suite.

Hydro (KH2D) fixtures
---------------------
athinput_path      – Path to the KH2D kh2d-sin.athinput file.
vtk_dir            – Path to the KH2D VTK frame directory.
sim                – Session-scoped SimulationData (hydro, 256x512, 301 frames).
first_frame        – Frame #0 from the KH2D sim.

MHD (Orszag-Tang) fixtures
--------------------------
ot_athinput_path   – Path to the OT athinput.orszag-tang file.
ot_vtk_dir         – Path to the OT VTK frame directory.
mhd_sim            – Session-scoped SimulationData (mhd, 400x400, 200 frames).
mhd_first_frame    – Frame #1 (first available) from the OT sim.

Synthetic fixtures
------------------
synthetic_vtk_bytes  – Raw bytes of a minimal AthenaK hydro VTK blob.
synthetic_vtk_path   – That blob written to a tmp file.
synthetic_sim        – SimulationData backed by a single synthetic hydro frame.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import pytest

# ── Paths ──────────────────────────────────────────────────────────────────────

_REPO_ROOT    = Path(__file__).parent.parent
_EXAMPLE_DATA = _REPO_ROOT / "kh2d"
_ATHINPUT     = _EXAMPLE_DATA / "kh2d-sin.athinput"
_VTK_DIR      = _EXAMPLE_DATA / "outputs" / "vtk"

# Orszag-Tang MHD
_OT_ROOT      = _REPO_ROOT / "orszang_tang_vortex"
_OT_ATHINPUT  = _OT_ROOT / "athinput.orszag-tang"
_OT_VTK_DIR   = _OT_ROOT / "outputs" / "vtk"


# ── Real-data fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def athinput_path() -> Path:
    """Absolute path to the bundled kh2d-sin.athinput file."""
    assert _ATHINPUT.exists(), f"athinput not found at {_ATHINPUT}"
    return _ATHINPUT


@pytest.fixture(scope="session")
def vtk_dir() -> Path:
    """Directory that contains the KH2D VTK frames."""
    assert _VTK_DIR.is_dir(), f"VTK directory not found: {_VTK_DIR}"
    return _VTK_DIR


@pytest.fixture(scope="session")
def sim():
    """
    A fully initialised SimulationData over the KH2D example set.
    Scoped to the *session* so the expensive file-discovery step runs only once.
    """
    from ergane import SimulationData
    return SimulationData(
        datafolder=str(_EXAMPLE_DATA / "outputs"),
        athinp=str(_ATHINPUT),
    )


@pytest.fixture(scope="session")
def first_frame(sim):
    """Frame object for the very first KH2D snapshot (frame number 0)."""
    return sim.get_frame(sim.frame_numbers[0])


# ── MHD (Orszag-Tang) real-data fixtures ──────────────────────────────────────

@pytest.fixture(scope="session")
def ot_athinput_path() -> Path:
    """Absolute path to the Orszag-Tang athinput.orszag-tang file."""
    assert _OT_ATHINPUT.exists(), f"OT athinput not found: {_OT_ATHINPUT}"
    return _OT_ATHINPUT


@pytest.fixture(scope="session")
def ot_vtk_dir() -> Path:
    """Directory containing Orszag-Tang VTK frames (mhd_w + mhd_bcc)."""
    assert _OT_VTK_DIR.is_dir(), f"OT VTK dir not found: {_OT_VTK_DIR}"
    return _OT_VTK_DIR


@pytest.fixture(scope="session")
def mhd_sim():
    """
    Session-scoped SimulationData over the Orszag-Tang MHD example.
    physics='mhd', grid=(400x400), frames 1-200.
    """
    from ergane import SimulationData
    return SimulationData(
        datafolder=str(_OT_ROOT / "outputs"),
        athinp=str(_OT_ATHINPUT),
    )


@pytest.fixture(scope="session")
def mhd_first_frame(mhd_sim):
    """Frame object for the first available OT snapshot (frame number 1)."""
    return mhd_sim.get_frame(mhd_sim.frame_numbers[0])


# ── Synthetic VTK helpers ──────────────────────────────────────────────────────

def _make_synthetic_vtk(
    ncx: int = 8,
    ncy: int = 16,
    origin: tuple[float, float] = (-0.5, -1.0),
    dx: float = 0.125,
    dy: float = 0.125,
    time: float = 1.23,
    fields: dict | None = None,
) -> bytes:
    """
    Build a minimal AthenaK-style STRUCTURED_POINTS VTK binary blob in memory.

    The 2-D grid has *ncx* x *ncy* cells.  Each field in *fields* must be a
    numpy array of shape ``(ncy, ncx)`` (scalars) or ``(ncy, ncx, 3)``
    (vectors).  If *fields* is None, a default density + velocity field is
    created automatically.

    Returns
    -------
    bytes
        Raw VTK binary blob that can be written to a .vtk file or wrapped in
        ``io.BytesIO`` for in-memory parsing.
    """
    nx, ny, nz = ncx + 1, ncy + 1, 2   # node counts (cells + 1 each dim; z nodes = 2)
    ncz = 1
    n_cells = ncx * ncy * ncz
    ox, oy = origin

    if fields is None:
        rng = np.random.default_rng(42)
        rho = rng.uniform(0.5, 2.0, size=(ncy, ncx)).astype(np.float32)
        vel = rng.uniform(-1.0, 1.0, size=(ncy, ncx, 3)).astype(np.float32)
        fields = {"rho": rho, "vel": vel}

    buf = io.BytesIO()
    w = buf.write

    def line(s: str) -> None:
        w((s + "\n").encode("ascii"))

    line("# vtk DataFile Version 2.0")
    line(f"CONSERVED vars  time= {time:.6f}  cycle=0  variables=")
    line("BINARY")
    line("DATASET STRUCTURED_POINTS")
    line(f"DIMENSIONS {nx} {ny} {nz}")
    line(f"ORIGIN {ox} {oy} -0.5")
    line(f"SPACING {dx} {dy} 1.0")
    line(f"CELL_DATA {n_cells}")

    for name, arr in fields.items():
        flat = arr.flatten(order="C").astype(">f4")
        if arr.ndim == 2:
            # Scalar
            line(f"SCALARS {name} float")
            line("LOOKUP_TABLE default")
            w(flat.tobytes())
            w(b"\n")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            # Vector
            line(f"VECTORS {name} float")
            w(flat.tobytes())
            w(b"\n")

    return buf.getvalue()


@pytest.fixture
def synthetic_vtk_bytes() -> bytes:
    """Raw bytes of a valid minimal AthenaK VTK file (8x16 cells, 2-D)."""
    return _make_synthetic_vtk()


@pytest.fixture
def synthetic_vtk_path(tmp_path, synthetic_vtk_bytes) -> Path:
    """A synthetic VTK file written to *tmp_path*."""
    p = tmp_path / "synth.hydro_w.00000.vtk"
    p.write_bytes(synthetic_vtk_bytes)
    return p


@pytest.fixture
def synthetic_sim(tmp_path, synthetic_vtk_bytes):
    """
    A SimulationData backed by a single synthetic 8x16 VTK frame.
    No athinput file -- exercises the fallback grid-detection path.
    """
    from ergane import SimulationData

    vtk_dir = tmp_path / "vtk"
    vtk_dir.mkdir()
    (vtk_dir / "synth.hydro_w.00000.vtk").write_bytes(synthetic_vtk_bytes)
    return SimulationData(datafolder=str(tmp_path))


# ── Synthetic MHD VTK pair helper ─────────────────────────────────────────────

def _make_synthetic_mhd_vtk_pair(
    ncx: int = 8,
    ncy: int = 8,
    time: float = 0.5,
) -> tuple[bytes, bytes]:
    """
    Build a matched pair of (mhd_w, mhd_bcc) VTK blobs for a synthetic 2-D MHD sim.

    mhd_w  contains: rho (scalar), press (scalar), vel (vector)
    mhd_bcc contains: Bcc (vector)

    Returns
    -------
    (mhd_w_bytes, mhd_bcc_bytes)
    """
    rng = np.random.default_rng(7)
    rho   = rng.uniform(0.5, 2.0,  size=(ncy, ncx)).astype(np.float32)
    press = rng.uniform(0.1, 1.0,  size=(ncy, ncx)).astype(np.float32)
    vel   = rng.uniform(-1.0, 1.0, size=(ncy, ncx, 3)).astype(np.float32)
    bcc   = rng.uniform(-0.5, 0.5, size=(ncy, ncx, 3)).astype(np.float32)

    mhd_w_bytes   = _make_synthetic_vtk(ncx=ncx, ncy=ncy, time=time,
                                         fields={"rho": rho, "press": press, "vel": vel})
    mhd_bcc_bytes = _make_synthetic_vtk(ncx=ncx, ncy=ncy, time=time,
                                         fields={"Bcc": bcc})
    return mhd_w_bytes, mhd_bcc_bytes


@pytest.fixture
def synthetic_mhd_sim(tmp_path):
    """
    A SimulationData backed by a single synthetic 8x8 MHD VTK frame pair.
    physics='mhd', no athinput.
    """
    from ergane import SimulationData

    vtk_dir = tmp_path / "vtk"
    vtk_dir.mkdir()
    w_bytes, bcc_bytes = _make_synthetic_mhd_vtk_pair()
    (vtk_dir / "synth.mhd_w.00000.vtk").write_bytes(w_bytes)
    (vtk_dir / "synth.mhd_bcc.00000.vtk").write_bytes(bcc_bytes)
    return SimulationData(datafolder=str(tmp_path))


# ── Expose helpers so individual test modules can reuse them ───────────────────
make_synthetic_vtk          = _make_synthetic_vtk
make_synthetic_mhd_vtk_pair = _make_synthetic_mhd_vtk_pair
