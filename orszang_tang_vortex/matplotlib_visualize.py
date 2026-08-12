"""
save_density_movie.py
=====================
Simple script to render the density field from Athena++ VTK files to an MP4.

Usage:
    python save_density_movie.py [outputs_dir] [output.mp4]

Defaults:
    outputs_dir = ./outputs
    output.mp4  = density_movie.mp4
"""

import pathlib
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

matplotlib.use("Agg")  # Headless backend

# ── VTK reader (same as visualize.py) ──────────────────────────


def _read_newline(f):
    b = f.read(1)
    if b != b"\n":
        f.seek(-1, 1)


def parse_athena_vtk(path):
    fields = {}
    with open(path, "rb") as f:
        f.readline()
        comment = f.readline().decode("ascii", errors="replace").strip()
        f.readline()
        f.readline()
        dim_line = f.readline().decode("ascii").strip()
        nx, ny, nz = map(int, dim_line.split()[1:])

        x_hdr = f.readline().decode("ascii").strip()
        n_x = int(x_hdr.split()[1])
        x = np.frombuffer(f.read(n_x * 4), dtype=">f4").astype("f4").copy()
        _read_newline(f)

        y_hdr = f.readline().decode("ascii").strip()
        n_y = int(y_hdr.split()[1])
        y = np.frombuffer(f.read(n_y * 4), dtype=">f4").astype("f4").copy()
        _read_newline(f)

        z_hdr = f.readline().decode("ascii").strip()
        n_z = int(z_hdr.split()[1])
        _z = np.frombuffer(f.read(n_z * 4), dtype=">f4")
        _read_newline(f)

        ncx, ncy = nx - 1, ny - 1
        ncz = max(1, nz - 1)
        n_cells = ncx * ncy * ncz

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
                arr = arr.reshape(ncy, ncx)
                _read_newline(f)
                fields[name] = arr
            elif parts[0] == "VECTORS":
                name = parts[1]
                arr = (
                    np.frombuffer(f.read(n_cells * 3 * 4), dtype=">f4")
                    .astype("f4")
                    .copy()
                )
                arr = arr.reshape(ncy, ncx, 3)
                _read_newline(f)
                fields[name] = arr

    m = re.search(r"time=(\S+)", comment)
    time_val = float(m.group(1)) if m else 0.0
    return {"time": time_val, "x": x, "y": y, "fields": fields}


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = (
        pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("outputs")
    )
    mp4_path = sys.argv[2] if len(sys.argv) > 2 else "density_movie.mp4"

    vtk_files = sorted(out_dir.glob("*.vtk"))
    if not vtk_files:
        print(f"ERROR: No .vtk files found in {out_dir}")
        sys.exit(1)

    n_frames = len(vtk_files)
    print(f"Found {n_frames} VTK files")

    # Load first frame to set up figure
    print("Loading frame 0...", end=" ", flush=True)
    raw0 = parse_athena_vtk(str(vtk_files[0]))
    rho0 = raw0["fields"]["rho"]
    print("done.")

    vmin, vmax = float(rho0.min()), float(rho0.max())
    print(f"Density range: {vmin:.4f} -- {vmax:.4f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Density  ρ", fontsize=14, fontweight="bold", pad=8)

    im = ax.imshow(
        rho0,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("ρ", fontsize=12)

    # Time label
    time_text = ax.text(
        0.02,
        0.98,
        f"t = {raw0['time']:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        color="white",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
    )

    fig.tight_layout()

    # Writer setup
    writer = FFMpegWriter(
        fps=15, metadata={"title": "Orszag-Tang Density", "artist": "Athena++"}
    )

    print(f"Rendering to {mp4_path} ...")
    with writer.saving(fig, mp4_path, dpi=150):
        for i, vtk_path in enumerate(vtk_files):
            raw = parse_athena_vtk(str(vtk_path))
            rho = raw["fields"]["rho"]

            im.set_data(rho)
            time_text.set_text(f"t = {raw['time']:.4f}")

            writer.grab_frame()

            if (i + 1) % 10 == 0 or i == n_frames - 1:
                print(f"  {i + 1}/{n_frames} frames written")

    plt.close(fig)
    print(f"Done! Saved to {mp4_path}")
