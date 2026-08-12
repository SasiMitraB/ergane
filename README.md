# ergane

A Python library for loading, inspecting, and visualising output data from
[AthenaK](https://github.com/IAS-Astrophysics/athenak) and
[Athena++](https://github.com/PrincetonUniversity/athena) simulations.

The library is built around one principle: **do a lot with very little code.**
Point it at an output directory and an input file, and it handles file
discovery, VTK/BIN parsing, field normalisation, physical unit conversion, and
interactive visualisation — all lazily, so only the data you actually request is
ever read from disk.

> **Name**: *Ergane* (Ἐργάνη) is an epithet of Athena in her role as goddess
> of craft, skill, and tools — a fitting name for a library built to work with
> Athena simulation data.

---

## Project Layout

```
AthenaWrapper/
├── ergane/                      # Library source
│   ├── __init__.py              # Public API exports
│   ├── simulation_data.py       # SimulationData, Frame, _FieldAccessor
│   ├── visualization.py         # Visualization (fastplotlib / matplotlib / Jupyter)
│   ├── units.py                 # Units — physical unit system
│   ├── vtk_reader.py            # Binary VTK parser (AthenaK + Athena++ formats)
│   ├── athinput_parser.py       # Athena input file parser
│   └── bin_reader.py            # Binary .bin file reader (AthenaK native format)
│
├── kh2d/                        # Kelvin-Helmholtz 2D (hydro) example
│   ├── kh2d-sin.athinput        # Simulation parameters (256×512 grid)
│   ├── outputs/
│   │   ├── vtk/                 # 301 VTK snapshots (t = 0 → 6)
│   │   └── KH.hydro.hst         # History file
│   └── visualize.py             # Example visualisation script
│
├── orszang_tang_vortex/         # Orszag-Tang vortex (MHD) example
│   ├── athinput.orszag-tang     # Simulation parameters (400×400 grid)
│   ├── outputs/
│   │   └── vtk/                 # 200 matched mhd_w + mhd_bcc frame pairs
│   └── visualize.py             # Example visualisation script
│
├── tests/                       # Pytest test suite (284 tests, all passing)
│   ├── conftest.py              # Shared fixtures (hydro + MHD + synthetic data)
│   ├── test_athinput_parser.py
│   ├── test_vtk_reader.py
│   ├── test_units.py
│   ├── test_simulation_data.py  # KH2D hydro tests
│   └── test_mhd_simulation.py  # Orszag-Tang MHD tests
│
├── requirements.txt             # numpy, h5py, pytest
└── pytest.ini                   # Test configuration
```

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` covers the core runtime and test dependencies:

```
numpy
h5py
pytest
```

For interactive visualisation, install one of the supported backends separately:

```bash
pip install fastplotlib   # GPU-accelerated (default backend)
# or
pip install matplotlib    # for the matplotlib backend
```

### 3. Make the library importable

Run scripts from the repo root, or add it to your path:

```python
import sys
sys.path.insert(0, "/path/to/AthenaWrapper")
```

---

## Quick Start

```python
from ergane import SimulationData

# Kelvin-Helmholtz instability (hydro)
sim = SimulationData(
    datafolder = "kh2d/outputs",
    athinp     = "kh2d/kh2d-sin.athinput",
)

# Orszag-Tang vortex (MHD)
ot = SimulationData(
    datafolder = "orszang_tang_vortex/outputs",
    athinp     = "orszang_tang_vortex/athinput.orszag-tang",
)

print(sim)
# <SimulationData  basename='KH'  physics='hydro'  grid=(256×512)  n_frames=301  units='code'>

print(sim.nx, sim.ny)     # 256  512
print(sim.n_frames)       # 301
print(sim.gamma)          # 1.666667
print(sim.times[:5])      # [0.  0.02 0.04 0.06 0.08]

# ── Field access ──────────────────────────────────────────────────────────────
rho   = sim.density[0]       # np.ndarray shape (512, 256) — frame 0
rho   = sim.density[300]     # frame 300 (file suffix, not position)
rho   = sim.density[-1]      # last frame (negative positional)
slc   = sim.density[0:5]     # list of 5 arrays (positional slice)

# ── Time-based access ─────────────────────────────────────────────────────────
arr   = sim.density.at_time(3.0)                      # nearest frame to t=3.0
arrs  = sim.density.between(1.0, 2.0)                 # list of arrays
pairs = sim.density.between(1.0, 2.0, include_times=True)  # [(t, arr), ...]

# ── Frame objects (all fields at once) ────────────────────────────────────────
frame = sim.get_frame(150)          # Frame #150  t=3.0  fields=[...]
frame = sim.frame_at(t=3.0)        # nearest frame to t=3.0
frames = sim.frames_between(1.0, 2.0)  # list of Frame objects

print(frame.density.shape)  # (512, 256)
print(frame.xc.shape)       # (256,)  — cell-centre x coordinates
print(frame.yc.shape)       # (512,)  — cell-centre y coordinates
print(frame.temperature)    # derived from P/ρ × μ mH / kB
print(frame.scalar_00)      # passive scalar field (if present)

# ── Visualisation ─────────────────────────────────────────────────────────────
sim.visualize().show()
sim.visualize(fields=["density", "pressure"], backend="matplotlib").show()
```

---

## File Format Support

| Format | Extension | Detected by |
|---|---|---|
| AthenaK VTK (structured points) | `.vtk` | `ORIGIN` + `SPACING` header |
| Athena++ VTK (rectilinear grid) | `.vtk` | `X_COORDINATES` header |
| AthenaK binary | `.bin` | extension |

File discovery is automatic. The library searches `datafolder` and its `vtk/`
or `bin/` subdirectories, then identifies the physics type from the filename:

| Pattern | Physics |
|---|---|
| `*.hydro_w.*.vtk / .bin` | `"hydro"` |
| `*.mhd_w.*.vtk / .bin` + `*.mhd_bcc.*` | `"mhd"` |
| Anything else | `"hydro"` (fallback) |

---

## SimulationData API

### Constructor

```python
SimulationData(datafolder, athinp=None, dtype=np.float64)
```

| Parameter | Type | Description |
|---|---|---|
| `datafolder` | `str\|Path` | Output directory (or its parent if files are in `vtk/`/`bin/`) |
| `athinp` | `str\|Path` | Path to the `.athinput` parameter file (optional but recommended) |
| `dtype` | `np.dtype` | Floating-point precision for loaded arrays (default `float64`) |

### Properties

| Property | Type | Description |
|---|---|---|
| `n_frames` | `int` | Total snapshots available |
| `frame_numbers` | `list[int]` | Sorted file-suffix integers |
| `physics` | `str` | `"hydro"` or `"mhd"` |
| `fields_available` | `list[str]` | Normalised field names for this run |
| `times` | `np.ndarray` | Simulation time per frame (code units, cached) |
| `nx`, `ny` | `int` | Cell counts in X1, X2 |
| `x1min/max`, `x2min/max` | `float` | Domain bounds (code units) |
| `gamma` | `float\|None` | Adiabatic index (from athinput) |
| `basename` | `str\|None` | Job name (from athinput or inferred) |
| `params` | `dict[str, dict[str, str]]` | Full parsed athinput parameters |
| `units` | `Units` | Active unit system |

### Field accessors

```python
sim.density    sim.pressure   sim.temperature   # computed: P/ρ × μ mH/kB
sim.velx       sim.vely       sim.velz
sim.bx         sim.by         sim.bz            # MHD only
```

Each accessor supports:

```python
accessor[n]                         # frame number n (file suffix)
accessor[-1]                        # last frame (positional)
accessor[0:5]                       # positional slice → list of arrays
accessor.at_time(t)                 # nearest frame to simulation time t
accessor.between(t0, t1)            # all frames in [t0, t1] → list
accessor.between(t0, t1, include_times=True)  # → list of (float, array)
```

### Frame-level methods

```python
sim.get_frame(num)             # → Frame  (by file suffix)
sim.frame_at(t)                # → Frame  (nearest to simulation time t)
sim.frames_between(t0, t1)     # → list[Frame]
```

---

## The `Frame` Object

`get_frame()` and `frame_at()` return a `Frame` dataclass:

```python
frame.number      # int   — file suffix
frame.time        # float — simulation time (scaled if units are set)
frame.x           # 1-D node coords along X (scaled)
frame.y           # 1-D node coords along Y (scaled)
frame.xc          # 1-D cell-centre coords along X
frame.yc          # 1-D cell-centre coords along Y
frame.units       # active Units object

frame.density     # np.ndarray | None
frame.pressure    # np.ndarray | None
frame.velx        # np.ndarray | None
frame.vely        # np.ndarray | None
frame.velz        # np.ndarray | None
frame.bx          # np.ndarray | None  (MHD only)
frame.by          # np.ndarray | None  (MHD only)
frame.bz          # np.ndarray | None  (MHD only)
frame.temperature # np.ndarray | None  (derived, requires density + pressure)
frame.scalars     # dict[str, np.ndarray] — passive scalar fields
frame.scalar_00   # shorthand attribute access for passive scalars
```

---

## Physical Units

All arrays are returned in **code units** by default. Attach a `Units` object to
convert everything automatically:

```python
from ergane import Units

cgs = Units.cgs(
    length   = 3.086e18,   # cm  — 1 pc
    density  = 1.67e-24,   # g/cm³ — 1 proton mass / cc
    velocity = 1.0e5,      # cm/s — 1 km/s
    labels   = {
        "density":  "g cm⁻³",
        "pressure": "dyn cm⁻²",
        "velx":     "km s⁻¹",
    },
)

sim.set_units(cgs)

rho   = sim.density[0]           # array in g/cm³
p     = sim.pressure[0]          # array in dyn/cm²
frame = sim.get_frame(50)
print(frame.time)                 # seconds
print(frame.x)                    # cm

print(sim.units.label("density")) # "g cm⁻³"
print(sim.units.label("bx"))      # "B_x [CGS]"  (auto-generated)

sim.set_units(Units.code())       # reset to code units
```

### Derived scales

| Quantity | Default derivation | Override via |
|---|---|---|
| `units.time` | `length / velocity` | `time=` kwarg |
| `units.pressure` | `density × velocity²` | `pressure=` kwarg |
| `units.magnetic` | `√(density × velocity²)` | `magnetic=` kwarg |

### Units constructors

```python
Units.code()                                          # all scale factors = 1
Units.cgs(length, density, velocity, **kwargs)        # system="CGS"
Units.si(length, density, velocity, **kwargs)         # system="SI"
Units(length, density, velocity, system="custom", labels={}, mu=0.62)

u.scale("density")   # float — multiply code array by this to get physical units
u.label("velx")      # str   — unit label for plots
```

---

## Low-level API

These functions are exported directly from `ergane` for one-off use:

```python
from ergane import parse_athena_vtk, read_vtk_time, parse_athinput

# Parse a single VTK file
data = parse_athena_vtk("example_data/outputs/vtk/KH.hydro_w.00300.vtk")
# → {"time": 6.0, "x": array(...), "y": array(...), "fields": {"rho": ..., "vel": ...}}

# Read only the simulation time from the header (reads 2 lines — very fast)
t = read_vtk_time("example_data/outputs/vtk/KH.hydro_w.00300.vtk")

# Parse an athinput file
params = parse_athinput("example_data/kh2d-sin.athinput")
params["mesh"]["nx1"]    # "256"
params["time"]["tlim"]   # "6.0"
```

---

## Visualisation

```python
viz = sim.visualize()
viz = sim.visualize(fields=["density", "pressure"])
viz = sim.visualize(fields=["density"], cmaps={"density": "plasma"})
viz = sim.visualize(backend="matplotlib")    # or "fastplotlib", "jupyter", "auto"
viz.show()
```

Supported backends:

| Backend | Use case |
|---|---|
| `"fastplotlib"` | GPU-accelerated interactive window (default) |
| `"matplotlib"` | Animated matplotlib window |
| `"jupyter"` | Inline ipywidgets in Jupyter notebooks |
| `"auto"` | Selects `"jupyter"` inside a kernel, `"fastplotlib"` otherwise |

---

## Testing

The test suite lives in `tests/` and covers all ergane modules at 190 tests.

```bash
# Run the full suite
.venv/bin/python -m pytest

# Run with verbose output
.venv/bin/python -m pytest -v

# Run a specific module
.venv/bin/python -m pytest tests/test_simulation_data.py
```

### Test architecture

| Module | What it tests |
|---|---|
| `tests/conftest.py` | Shared fixtures: session-scoped `sim` (KH2D hydro) and `mhd_sim` (OT MHD), plus fast synthetic in-memory VTK blobs for both physics types |
| `tests/test_athinput_parser.py` | `parse_athinput`, `typed()`, edge cases (empty files, inline comments, whitespace) |
| `tests/test_vtk_reader.py` | `parse_athena_vtk` (shapes, dtypes, round-trips), `read_vtk_time`, `extract_frame_num` |
| `tests/test_units.py` | All `Units` constructors, scale factors, override kwargs, `label()`, `from_params()` |
| `tests/test_simulation_data.py` | `SimulationData` + `Frame` for hydro: construction, grid props, field accessors, slicing, time queries, passive scalars, unit integration, error handling |
| `tests/test_mhd_simulation.py` | `SimulationData` + `Frame` for MHD: B-field accessors, B-field unit scaling, OT athinput `<mhd>` section, synthetic MHD round-trips |

Synthetic tests use an in-memory VTK byte generator (`conftest.make_synthetic_vtk`) so they
run without touching the large example dataset.

---

## Performance Notes

- **Lazy loading** — only files you index are parsed; no field data is held in memory
  across frames.
- **Fast time index** — `sim.times` and all time-based queries read only the
  2-line ASCII header of each VTK file (~60 ms for 301 frames, cached after the
  first call).
- **Instant construction** — `SimulationData(...)` does no file I/O beyond a
  `glob`; no VTK data is opened until you request a frame.
