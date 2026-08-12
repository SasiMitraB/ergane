#!/usr/bin/env python3
"""
plot_history.py
================================================================================
A premium visualization script for Athena++ / AthenaK history (.hst) files.
Parses global conserved quantities and energy components, then generates
publication-quality plots showing physical evolution and conservation properties.

Usage:
    python plot_history.py [path_to_hst_file] [output_image_path]

Defaults:
    path_to_hst_file  = outputs/OrszagTang.mhd.hst
    output_image_path = outputs/history_plots.png
================================================================================
"""

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

def parse_hst_file(filepath):
    """
    Parses an Athena history file, dynamically mapping columns based on the
    header block containing indices like [1]=time, [2]=dt, etc.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"History file not found at: {filepath}")

    columns = {}
    data_lines = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                # Dynamically map headers like [1]=time, [2]=dt
                matches = re.findall(r'\[(\d+)\]=([\w\-]+)', line)
                if matches:
                    columns = {int(idx) - 1: name for idx, name in matches}
                continue

            # Parse numeric lines
            parts = [float(x) for x in line.split()]
            if parts:
                data_lines.append(parts)

    data = np.array(data_lines)
    if data.size == 0:
        raise ValueError(f"No data parsed from file: {filepath}")

    # Fallback to standard columns if no header description was found
    if not columns:
        print("Warning: Header map not found in comments. Using default columns.")
        default_names = [
            "time", "dt", "mass", "1-mom", "2-mom", "3-mom",
            "tot-E", "1-KE", "2-KE", "3-KE", "1-ME", "2-ME", "3-ME"
        ]
        columns = {i: name for i, name in enumerate(default_names)}

    data_dict = {}
    for idx, name in columns.items():
        if idx < data.shape[1]:
            data_dict[name] = data[:, idx]
        else:
            print(f"Warning: Column index {idx} ({name}) out of bounds for data shape {data.shape}")

    return data_dict

def main():
    # ── Resolve Paths Relative to Script Directory ────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    hst_path = os.path.join(script_dir, "outputs", "OrszagTang.mhd.hst")
    if len(sys.argv) > 1:
        hst_path = sys.argv[1]

    output_path = os.path.join(script_dir, "outputs", "history_plots.png")
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    print(f"Reading Athena history file: {hst_path}")
    try:
        data = parse_hst_file(hst_path)
    except Exception as e:
        print(f"Error parsing history file: {e}")
        sys.exit(1)

    print("Successfully parsed columns:", ", ".join(data.keys()))
    time = data.get("time")
    if time is None:
        print("Error: 'time' column not found in history data.")
        sys.exit(1)

    # ── Derived Physics Calculations ──────────────────────────────────────────
    # Total Kinetic Energy (sum of directions)
    ke_1 = data.get("1-KE", np.zeros_like(time))
    ke_2 = data.get("2-KE", np.zeros_like(time))
    ke_3 = data.get("3-KE", np.zeros_like(time))
    total_ke = ke_1 + ke_2 + ke_3

    # Total Magnetic Energy (sum of directions)
    me_1 = data.get("1-ME", np.zeros_like(time))
    me_2 = data.get("2-ME", np.zeros_like(time))
    me_3 = data.get("3-ME", np.zeros_like(time))
    total_me = me_1 + me_2 + me_3

    # Total Energy
    tot_e = data.get("tot-E")
    
    # Estimate Thermal/Internal Energy if total energy is available
    if tot_e is not None:
        thermal_e = tot_e - total_ke - total_me
    else:
        thermal_e = None

    # Mass and Momentum
    mass = data.get("mass")
    mom1 = data.get("1-mom", np.zeros_like(time))
    mom2 = data.get("2-mom", np.zeros_like(time))
    mom3 = data.get("3-mom", np.zeros_like(time))
    dt = data.get("dt")

    # ── Premium Aesthetic Configuration ──────────────────────────────────────
    # Modern dark style with rich colors
    plt.style.use('dark_background')
    
    # Custom color palette (Slate / Zinc style)
    bg_color = "#0f172a"      # Sleek Slate 900 background
    panel_color = "#1e293b"   # Slate 800 panels
    text_color = "#f8fafc"    # Off-white Slate 50
    grid_color = "#334155"    # Slate 700 grid lines

    # Color codes for plots
    c_tot_e = "#38bdf8"       # Bright Sky Blue
    c_therm = "#f43f5e"       # Vibrant Rose
    c_ke = "#10b981"          # Emerald Green
    c_me = "#f59e0b"          # Amber Orange
    
    c_x = "#f43f5e"           # Rose for x-axis components
    c_y = "#3b82f6"           # Royal Blue for y-axis components
    c_z = "#10b981"           # Emerald for z-axis components

    # Apply configuration to Matplotlib runtime rcParams
    plt.rcParams.update({
        'figure.facecolor': bg_color,
        'axes.facecolor': bg_color,
        'axes.edgecolor': grid_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'grid.color': grid_color,
        'grid.alpha': 0.5,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'legend.facecolor': panel_color,
        'legend.edgecolor': grid_color,
        'legend.shadow': True
    })

    # Create 2x2 multi-panel layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    fig.suptitle(f"Athena++ Orszag-Tang Vortex Simulation History\nFile: {os.path.basename(hst_path)}",
                 fontsize=16, color=text_color, fontweight='bold', y=0.97)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 1: Energy Evolution (Top Left)
    # ──────────────────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_title("Energy Components Evolution", fontsize=12, fontweight='bold', pad=10, color=text_color)
    
    if tot_e is not None:
        ax.plot(time, tot_e, label="Total Energy ($E_{tot}$)", color=c_tot_e, linewidth=2.5)
    if thermal_e is not None:
        ax.plot(time, thermal_e, label="Thermal Energy ($E_{th}$)", color=c_therm, linewidth=2)
    ax.plot(time, total_ke, label="Kinetic Energy ($E_{k}$)", color=c_ke, linewidth=2)
    ax.plot(time, total_me, label="Magnetic Energy ($E_{m}$)", color=c_me, linewidth=2)
    
    ax.set_xlabel("Time ($t$)", fontsize=10)
    ax.set_ylabel("Energy Density / Volume Integrated", fontsize=10)
    ax.grid(True, linestyle=':')
    ax.legend(loc="best", framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 2: Energy Fractions & Partitioning (Top Right)
    # ──────────────────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_title("Kinetic vs Magnetic Energy Components", fontsize=12, fontweight='bold', pad=10, color=text_color)
    
    # Plot components to show spatial anisotropy
    ax.plot(time, ke_1, label="KE - $v_x$", color=c_x, linestyle="-", linewidth=1.5)
    ax.plot(time, ke_2, label="KE - $v_y$", color=c_y, linestyle="-", linewidth=1.5)
    ax.plot(time, me_1, label="ME - $B_x$", color=c_x, linestyle="--", linewidth=1.5)
    ax.plot(time, me_2, label="ME - $B_y$", color=c_y, linestyle="--", linewidth=1.5)
    
    # If 3D, include z-components (usually zero for Orszag-Tang, but good to show)
    if np.any(ke_3 > 1e-10) or np.any(me_3 > 1e-10):
        ax.plot(time, ke_3, label="KE - $v_z$", color=c_z, linestyle="-", linewidth=1.5)
        ax.plot(time, me_3, label="ME - $B_z$", color=c_z, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time ($t$)", fontsize=10)
    ax.set_ylabel("Energy Component Value", fontsize=10)
    ax.grid(True, linestyle=':')
    ax.legend(loc="best", framealpha=0.9, ncol=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 3: Conservation properties & Momentum (Bottom Left)
    # ──────────────────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.set_title("Relative Conservation Errors & Momentum", fontsize=12, fontweight='bold', pad=10, color=text_color)
    
    # Relative Mass error
    if mass is not None and len(mass) > 0:
        mass_rel_err = (mass - mass[0]) / mass[0]
        line1 = ax.plot(time, mass_rel_err, label=r"Mass error $\Delta M / M_0$", color="#8b5cf6", linewidth=1.8)
    
    # Relative Total Energy error
    if tot_e is not None and len(tot_e) > 0:
        energy_rel_err = (tot_e - tot_e[0]) / tot_e[0]
        line2 = ax.plot(time, energy_rel_err, label=r"Energy error $\Delta E / E_0$", color=c_tot_e, linewidth=1.8)

    ax.set_xlabel("Time ($t$)", fontsize=10)
    ax.set_ylabel("Relative Error", fontsize=10)
    ax.grid(True, linestyle=':')
    
    # Format relative errors nicely with scientific notation
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))

    # Add second y-axis for Momentum components (since they are absolute values close to 0)
    ax2 = ax.twinx()
    line3 = ax2.plot(time, mom1, label="Mom-x", color="#e2e8f0", linestyle=":", alpha=0.7)
    line4 = ax2.plot(time, mom2, label="Mom-y", color="#94a3b8", linestyle=":", alpha=0.7)
    ax2.set_ylabel("Total Momentum (Absolute)", color="#94a3b8", fontsize=10)
    ax2.tick_params(axis='y', labelcolor="#94a3b8")
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))

    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right", framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 4: Simulation Time Step (dt) Evolution (Bottom Right)
    # ──────────────────────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.set_title("Solver Adaptive Timestep ($dt$)", fontsize=12, fontweight='bold', pad=10, color=text_color)
    
    if dt is not None:
        ax.plot(time, dt, label="Timestep ($dt$)", color="#34d399", linewidth=1.8)
        # Add moving average or trend line to see overall behavior
        if len(dt) > 10:
            dt_smooth = np.convolve(dt, np.ones(10)/10, mode='valid')
            time_smooth = time[len(time)-len(dt_smooth):]
            ax.plot(time_smooth, dt_smooth, label="10-step Moving Avg", color="#059669", linestyle="-.", linewidth=1.5)

    ax.set_xlabel("Time ($t$)", fontsize=10)
    ax.set_ylabel("Time Step ($dt$)", fontsize=10)
    ax.grid(True, linestyle=':')
    ax.legend(loc="best", framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))

    # Optimize spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the output image
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor=bg_color)
    plt.close()
    print(f"Visualization saved to: {output_path}")

    # Print summary statistics to the console
    print("\nSimulation History Summary Statistics:")
    print("-" * 50)
    print(f"Time Range:         {time[0]:.4e} to {time[-1]:.4e}")
    print(f"Number of Records:  {len(time)}")
    if mass is not None:
        mass_err = np.max(np.abs(mass - mass[0])) / mass[0]
        print(f"Initial Mass:       {mass[0]:.6e}")
        print(f"Mass Conservation:  Max Rel Err = {mass_err:.4e}")
    if tot_e is not None:
        e_err = np.max(np.abs(tot_e - tot_e[0])) / tot_e[0]
        print(f"Initial Total E:    {tot_e[0]:.6e}")
        print(f"Energy Conservation: Max Rel Err = {e_err:.4e}")
    if dt is not None:
        print(f"Timestep dt range:  {np.min(dt):.4e} to {np.max(dt):.4e}")
    print("-" * 50)

if __name__ == "__main__":
    main()
