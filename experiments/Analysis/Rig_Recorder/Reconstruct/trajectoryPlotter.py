# -*- coding: utf-8 -*-
"""
Extended analysis & visualisation script for pipette, stage and resistance data
==============================================================================

Changes (May 29 2025)
---------------------
* **Plot 5** perfectly matches the static 3-D scatter look:
  * Viridis colour-bar added.
  * Time-progressive colouring now works (thanks to cmap/vmin/vmax).
  * Still no legend or per-demo labels.
* Animation defaults: ~6 Ã— real-time (60 fps * multiplier 6).

Toggle plots and GIF export with the flags below. Pillow is required only when
`save_animation=True`.
"""

from __future__ import annotations

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from pathlib import Path

# ====== CONFIGURATION FLAGS ======
plot_pipette_3d = True   # Static 3-D scatter (Plot 1)
plot_resistance = False          # Plot 2
plot_stage_3d = False            # Plot 3
plot_time_courses = True        # Plot 4
plot_time_velocity_courses = True    # Plot 4b (pipette/stage velocities)
plot_pipette_animation = False    # Animated scatter (Plot 5)

# Animation tuning -----------------------------------------------------------
animation_speed_multiplier = 1   # ~6 Ã— faster than real-time
base_animation_fps = 60          # logical FPS before speed-up
save_animation = False           # Export GIF (requires Pillow)
animation_path = Path(r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Datasets\pipette_paths.gif")
# ===========================================================================

file_path = Path(r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Datasets\PatcherBot_dataset_v0_740\PatcherBot_dataset_v0_740_find_pipette.hdf5")

pipette_positions_data: dict[str, np.ndarray] = {}
resistance_data: dict[str, np.ndarray] = {}
stage_positions_data: dict[str, np.ndarray] = {}

# ---------------------------------------------------------------------------
# 1. Load data ---------------------------------------------------------------
# ---------------------------------------------------------------------------
with h5py.File(file_path, "r") as hdf:
    for demo_key in hdf["data"].keys():
        demo_group = hdf["data"][demo_key]["obs"]
        if "pipette_positions" in demo_group:
            pipette_positions_data[demo_key] = demo_group["pipette_positions"][:]
        if "resistance" in demo_group:
            resistance_data[demo_key] = demo_group["resistance"][:]
        if "stage_positions" in demo_group:
            stage_positions_data[demo_key] = demo_group["stage_positions"][:]

# ---------------------------------------------------------------------------
# 2. STATIC PLOTS (1-4) â€“ original style ------------------------------------
# ---------------------------------------------------------------------------
if plot_pipette_3d:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    cmap = cm.viridis
    for positions in pipette_positions_data.values():
        t = np.arange(positions.shape[0])
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c=cmap(t / positions.shape[0]), marker="o", s=15)
        ax.set(xlim=[-100, 100], ylim=[-100, 100], zlim=[-50, 50],
           title="3-D Pipette Motion with Time-Colour Encoding",
           xlabel="X position", ylabel="Y position", zlabel="Z position (top-down)")
    mappable = cm.ScalarMappable(cmap=cmap); mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("Time steps"); cbar.ax.invert_yaxis()

if plot_resistance:
    fig, ax = plt.subplots(figsize=(12, 4))
    for resistance in resistance_data.values():
        ax.plot(resistance)
    ax.set(title="Resistance Change over Time", xlabel="Time steps", ylabel="Resistance (Î©)")

if plot_stage_3d:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    cmap = cm.viridis
    for positions in stage_positions_data.values():
        t = np.arange(positions.shape[0])
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c=cmap(t / positions.shape[0]), marker="o", s=15)
    ax.set(xlim=[-40, 40], ylim=[-40, 40], zlim=[-50, 50],
           title="3-D Stage Motion with Time-Colour Encoding",
           xlabel="X position", ylabel="Y position", zlabel="Z position (top-down)")
    cb = plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, pad=0.1, shrink=0.6)
    cb.set_label("Time steps"); cb.ax.invert_yaxis()

if plot_time_courses:
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for pos in pipette_positions_data.values():
        t = np.arange(pos.shape[0])
        axs[0].plot(t, pos[:, 0]); axs[1].plot(t, pos[:, 1]); axs[2].plot(t, pos[:, 2])
    for pos in stage_positions_data.values():
        t = np.arange(pos.shape[0])
        axs[0].plot(t, pos[:, 0]); axs[1].plot(t, pos[:, 1]); axs[2].plot(t, pos[:, 2])
    axs[0].set_title("Pipette & Stage Positions Time Courses (X, Y, Z)")
    axs[0].set_ylabel("X Position"); axs[1].set_ylabel("Y Position"); axs[2].set_ylabel("Z Position")
    axs[2].set_xlabel("Time steps"); plt.tight_layout()

if plot_time_velocity_courses:
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for pos in pipette_positions_data.values():
        vel = np.diff(pos, axis=0)  # simple finite difference for velocity
        if vel.size == 0:
            continue
        t = np.arange(vel.shape[0])
        axs[0].plot(t, vel[:, 0]); axs[1].plot(t, vel[:, 1]); axs[2].plot(t, vel[:, 2])
    for pos in stage_positions_data.values():
        vel = np.diff(pos, axis=0)
        if vel.size == 0:
            continue
        t = np.arange(vel.shape[0])
        axs[0].plot(t, vel[:, 0]); axs[1].plot(t, vel[:, 1]); axs[2].plot(t, vel[:, 2])
    axs[0].set_title('Pipette and Stage Velocity Time Courses (X, Y, Z)')
    axs[0].set_ylabel('X Velocity'); axs[1].set_ylabel('Y Velocity'); axs[2].set_ylabel('Z Velocity')
    axs[2].set_xlabel('Time steps'); plt.tight_layout()
# ---------------------------------------------------------------------------
# 3. ANIMATED 3-D PIPETTE MOTION (Plot 5) -----------------------------------
# ---------------------------------------------------------------------------
if plot_pipette_animation and pipette_positions_data:

    fig_anim = plt.figure(figsize=(12, 8))
    ax_anim = fig_anim.add_subplot(111, projection="3d")
    ax_anim.grid(False)
    ax.set(xlim=[-100, 100], ylim=[-100, 100], zlim=[-50, 50],
                title="Animated 3-D Pipette Motion",
                xlabel="X position", ylabel="Y position", zlabel="Z position (top-down)")

    cmap = cm.viridis

    # Add viridis colour-bar (mirrors static plot)
    mappable_anim = cm.ScalarMappable(cmap=cmap); mappable_anim.set_array([])
    cbar_anim = plt.colorbar(mappable_anim, ax=ax_anim, pad=0.1, shrink=0.6)
    cbar_anim.set_label("Time steps"); cbar_anim.ax.invert_yaxis()

    # Prepare one scatter artist per trajectory
    scatters: list[tuple[animation.Artist, np.ndarray]] = []
    for pos in pipette_positions_data.values():
        scatters.append((ax_anim.scatter([], [], [], c=[], cmap=cmap, vmin=0, vmax=1,
                                         marker="o", s=15), pos))

    max_len = max(pos.shape[0] for pos in pipette_positions_data.values())

    def _init():
        for sc, _ in scatters:
            sc._offsets3d = ([], [], [])
            sc.set_array(np.array([]))
        return [sc for sc, _ in scatters]

    def _update(frame: int):
        for sc, pos in scatters:
            if frame < pos.shape[0]:
                x, y, z = pos[:frame + 1].T
                sc._offsets3d = (x, y, z)
                sc.set_array(np.linspace(0, 1, frame + 1))  # progressive colour gradient
        return [sc for sc, _ in scatters]

    interval_ms = 1000 / (base_animation_fps * animation_speed_multiplier)
    anim = animation.FuncAnimation(fig_anim, _update, init_func=_init,
                                   frames=max_len, interval=interval_ms, blit=False)

    if save_animation:
        try:
            anim.save(animation_path, writer=animation.PillowWriter(
                fps=base_animation_fps * animation_speed_multiplier))
            print(f"[INFO] Animation saved â†’ {animation_path.resolve()}")
        except Exception as exc:
            print("[WARNING] GIF not saved â€“", exc)

# ---------------------------------------------------------------------------
# 4. SHOW ALL FIGURES --------------------------------------------------------
# ---------------------------------------------------------------------------
plt.show()

