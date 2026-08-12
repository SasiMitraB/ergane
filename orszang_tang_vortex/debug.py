import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────
FRAME = 0  # <-- change this to the frame you want to plot
OUT_DIR = (
    pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("outputs/vtk")
)
SAVE_PATH = "frame.png"  # set to None to just display
# ───────────────────────────────────────────────────────────────


def _read_newline(f):
    b = f.read(1)
    if b != b"\n":
        f.seek(-1, 1)


def parse_athena_vtk(path):
    fields = {}
    with open(path, "rb") as f:
        f.readline()  # # vtk DataFile Version
        comment = f.readline().decode("ascii", errors="replace").strip()
        f.readline()  # BINARY
        f.readline()  # DATASET ...

        dim_line = f.readline().decode("ascii").strip()
        nx, ny, nz = map(int, dim_line.split()[1:])

        ncx, ncy = max(1, nx - 1), max(1, ny - 1)
        ncz = max(1, nz - 1)
        n_cells = ncx * ncy * ncz

        next_line = f.readline().decode("ascii").strip()

        if next_line.startswith("ORIGIN"):
            ox, oy, oz = map(float, next_line.split()[1:])
            spacing_line = f.readline().decode("ascii").strip()
            dx, dy, dz = map(float, spacing_line.split()[1:])
            x = np.linspace(ox, ox + dx * ncx, nx, dtype="f4")
            y = np.linspace(oy, oy + dy * ncy, ny, dtype="f4")
            z = np.linspace(oz, oz + dz * ncz, nz, dtype="f4")
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("CELL_DATA"):
                    break

        elif next_line.startswith("X_COORDINATES"):
            n_x = int(next_line.split()[1])
            x = np.frombuffer(f.read(n_x * 4), dtype=">f4").astype("f4").copy()
            _read_newline(f)
            y_hdr = f.readline().decode("ascii").strip()
            n_y = int(y_hdr.split()[1])
            y = np.frombuffer(f.read(n_y * 4), dtype=">f4").astype("f4").copy()
            _read_newline(f)
            z_hdr = f.readline().decode("ascii").strip()
            n_z = int(z_hdr.split()[1])
            z = np.frombuffer(f.read(n_z * 4), dtype=">f4").astype("f4").copy()
            _read_newline(f)
            f.readline()  # CELL_DATA

        while True:
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="ignore").strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue

            if parts[0] == "SCALARS":
                name = parts[1]
                f.readline()  # LOOKUP_TABLE default
                arr = (
                    np.frombuffer(f.read(n_cells * 4), dtype=">f4").astype("f4").copy()
                )
                arr = arr.reshape(ncy, ncx) if ncz == 1 else arr.reshape(ncz, ncy, ncx)
                _read_newline(f)
                fields[name] = arr

            elif parts[0] == "VECTORS":
                name = parts[1]
                arr = (
                    np.frombuffer(f.read(n_cells * 3 * 4), dtype=">f4")
                    .astype("f4")
                    .copy()
                )
                arr = (
                    arr.reshape(ncy, ncx, 3)
                    if ncz == 1
                    else arr.reshape(ncz, ncy, ncx, 3)
                )
                _read_newline(f)
                fields[name] = arr

    m = re.search(r"time=(\S+)", comment)
    time_val = float(m.group(1)) if m else 0.0
    return {"time": time_val, "x": x, "y": y, "fields": fields}


def extract_num(path):
    m = re.search(r"\.(\d+)\.vtk$", path.name)
    return int(m.group(1)) if m else -1


# ── Find and match files ────────────────────────────────────────
hydro_files = sorted(OUT_DIR.glob("*.mhd_w.*.vtk"), key=lambda p: extract_num(p))
bfield_files = sorted(OUT_DIR.glob("*.mhd_bcc.*.vtk"), key=lambda p: extract_num(p))

hydro_nums = {extract_num(p): p for p in hydro_files}
bfield_nums = {extract_num(p): p for p in bfield_files}
common_nums = sorted(set(hydro_nums.keys()) & set(bfield_nums.keys()))

if not common_nums:
    print("ERROR: No matching timestep pairs found")
    sys.exit(1)

if FRAME < 0 or FRAME >= len(common_nums):
    print(f"ERROR: FRAME={FRAME} is out of range (0 to {len(common_nums) - 1})")
    sys.exit(1)

num = common_nums[FRAME]
print(f"Loading frame {FRAME} (file number {num:05d})")

raw_hydro = parse_athena_vtk(str(hydro_nums[num]))
raw_bfield = parse_athena_vtk(str(bfield_nums[num]))
f = {**raw_hydro["fields"], **raw_bfield["fields"]}

# ── Resolve field names (Athena++ vs AthenaK) ─────────────────
rho = f.get("rho") or f.get("dens")
press = f.get("press") or f.get("eint")

if "vel" in f:
    vx = f["vel"][:, :, 0]
    vy = f["vel"][:, :, 1]
else:
    vx = f["velx"]
    vy = f["vely"]

if "Bcc" in f:
    bx = f["Bcc"][:, :, 0]
    by = f["Bcc"][:, :, 1]
else:
    bx = f["bcc1"]
    by = f["bcc2"]

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
fig.suptitle(
    f"Orszag-Tang Vortex  |  Frame {FRAME}  |  t = {raw_hydro['time']:.3f}", fontsize=14
)

plots = [
    (rho, "Density", "inferno"),
    (press, "Pressure / eint", "inferno"),
    (bx, r"$B_x$", "bwr"),
    (vx, r"$v_x$", "seismic"),
    (vy, r"$v_y$", "seismic"),
    (by, r"$B_y$", "bwr"),
]

for ax, (data, title, cmap) in zip(axes.flat, plots):
    im = ax.imshow(
        data,
        origin="lower",
        cmap=cmap,
        aspect="auto",
        extent=[
            raw_hydro["x"].min(),
            raw_hydro["x"].max(),
            raw_hydro["y"].min(),
            raw_hydro["y"].max(),
        ],
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

if SAVE_PATH:
    plt.savefig(SAVE_PATH, dpi=150)
    print(f"Saved to {SAVE_PATH}")
else:
    plt.show()
