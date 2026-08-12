import pathlib
import re
import sys

import fastplotlib as fpl
import numpy as np


# ── VTK reader (same as before) ───────────────────────────────
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

        # Check the next header to determine the grid type
        next_line = f.readline().decode("ascii").strip()

        if next_line.startswith("ORIGIN"):
            # --- AthenaK Style (STRUCTURED_POINTS) ---
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
            # --- Legacy Athena++ Style (RECTILINEAR_GRID) ---
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

            f.readline()  # Consume CELL_DATA line

        # --- Parse the actual binary field data ---
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


# ── Field name resolution helper ──────────────────────────────
def resolve_fields(fields):
    """
    Normalize Athena++ / AthenaK field names for pure hydro.
    Returns a dict with standard keys: rho, press, velx, vely
    """
    resolved = {}

    # Density
    if "rho" in fields:
        resolved["rho"] = fields["rho"]
    elif "dens" in fields:
        resolved["rho"] = fields["dens"]
    else:
        raise KeyError("No density field found (tried 'rho', 'dens')")

    # Pressure / Internal energy
    if "press" in fields:
        resolved["press"] = fields["press"]
    elif "eint" in fields:
        resolved["press"] = fields["eint"]
    else:
        raise KeyError(
            "No pressure/internal energy field found (tried 'press', 'eint')"
        )

    # Velocity x
    if "vel" in fields:
        resolved["velx"] = fields["vel"][:, :, 0]
    elif "velx" in fields:
        resolved["velx"] = fields["velx"]
    else:
        raise KeyError("No x-velocity field found (tried 'vel', 'velx')")

    # Velocity y
    if "vel" in fields:
        resolved["vely"] = fields["vel"][:, :, 1]
    elif "vely" in fields:
        resolved["vely"] = fields["vely"]
    else:
        raise KeyError("No y-velocity field found (tried 'vel', 'vely')")

    return resolved


# ── File matching helpers ─────────────────────────────────────
def extract_num(path):
    """Extract the numeric suffix from an Athena filename."""
    m = re.search(r"\.(\d+)\.vtk$", path.name)
    return int(m.group(1)) if m else -1


out_dir = (
    pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("outputs/vtk")
)

# Athena-K pure hydro test writes to hydro_w
hydro_files = sorted(out_dir.glob("*.hydro_w.*.vtk"), key=lambda p: extract_num(p))

if not hydro_files:
    print(f"ERROR: No hydro .vtk files (e.g. *.hydro_w.*.vtk) found in {out_dir}")
    sys.exit(1)

n_frames = len(hydro_files)
print(f"Found {n_frames} timestep files")

# ── Load first frame to set up figure ─────────────────────────
load_frame_number = -1
print(f"Loading frame {load_frame_number}...", end=" ", flush=True)

# Parse the single hydro file
raw_hydro = parse_athena_vtk(str(hydro_files[load_frame_number]))
fields = resolve_fields(raw_hydro["fields"])

rho0 = fields["rho"]
print("done.")

# ── Build figure (Updated for 2x2 Hydro) ────────────────────────
figure = fpl.Figure(
    shape=(2, 2),
    size=(1000, 1000),
    names=[["Density", "Pressure"], ["v_x", "v_y"]],
    controller_ids="sync",
)

image_density = figure[0, 0].add_image(data=rho0, name="density")
image_density.cmap = "inferno"

image_pressure = figure[0, 1].add_image(data=fields["press"], name="pressure")
image_pressure.cmap = "inferno"

image_velx = figure[1, 0].add_image(data=fields["velx"], name="velx")
image_velx.cmap = "seismic"

image_vely = figure[1, 1].add_image(data=fields["vely"], name="vely")
image_vely.cmap = "seismic"

for subplot in figure:
    subplot.toolbar = False

# ── Animation callback ─────────────────────────────────────────
frame_idx = 0  # first frame already displayed


def update_data(figure_instance):
    global frame_idx
    frame_idx = (frame_idx + 1) % n_frames

    # Only parse the hydro file now
    raw_hydro = parse_athena_vtk(str(hydro_files[frame_idx]))
    fields = resolve_fields(raw_hydro["fields"])

    figure_instance[0, 0]["density"].data = fields["rho"]
    figure_instance[0, 1]["pressure"].data = fields["press"]
    figure_instance[1, 0]["velx"].data = fields["velx"]
    figure_instance[1, 1]["vely"].data = fields["vely"]


figure.add_animations(update_data)

figure.show()
if __name__ == "__main__":
    fpl.loop.run()
