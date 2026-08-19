"""Visualization: pendulum animation and diagnostic plots.

All functions accept a :class:`~npendulum.simulate.SimulationResult` and work
for arbitrary N.  Links are colored with a fixed categorical palette for up
to 8 links; beyond that a perceptually uniform ordinal ramp (viridis) is
sampled, since link index is an ordered quantity.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .simulate import SimulationResult

# Categorical palette (validated fixed slot order) and chart chrome
_CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"


def link_colors(n: int) -> List[str]:
    """One color per link: fixed categorical slots for n <= 8, ordinal ramp after."""
    if n <= len(_CATEGORICAL):
        return _CATEGORICAL[:n]
    cmap = plt.get_cmap("viridis")
    return [matplotlib.colors.to_hex(cmap(v)) for v in np.linspace(0.05, 0.85, n)]


def _new_axes(xlabel: str, ylabel: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.set_xlabel(xlabel, color=_INK_SECONDARY)
    ax.set_ylabel(ylabel, color=_INK_SECONDARY)
    ax.set_title(title, color=_INK, loc="left", fontsize=12)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.tick_params(colors=_MUTED)
    for spine in ax.spines.values():
        spine.set_color(_BASELINE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def _finish(fig, ax, path: Optional[Path], n_series: int) -> plt.Figure:
    if n_series >= 2:
        ax.legend(loc="upper right", framealpha=0.9, edgecolor=_GRID,
                  labelcolor=_INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, facecolor=_SURFACE)
        plt.close(fig)
    return fig


def plot_angles(result: SimulationResult, path: "Path | str | None" = None):
    """Angular position of every link versus time."""
    n = result.config.num_links
    fig, ax = _new_axes("time [s]", "angle [rad]", "Angular position")
    for i, color in enumerate(link_colors(n)):
        ax.plot(result.t, result.theta[:, i], color=color, linewidth=1.6,
                label=f"link {i + 1}")
    return _finish(fig, ax, _as_path(path), n)


def plot_velocities(result: SimulationResult, path: "Path | str | None" = None):
    """Angular velocity of every link versus time."""
    n = result.config.num_links
    fig, ax = _new_axes("time [s]", "angular velocity [rad/s]",
                        "Angular velocity")
    for i, color in enumerate(link_colors(n)):
        ax.plot(result.t, result.omega[:, i], color=color, linewidth=1.6,
                label=f"link {i + 1}")
    return _finish(fig, ax, _as_path(path), n)


def plot_energy(result: SimulationResult, path: "Path | str | None" = None):
    """Kinetic, potential and total mechanical energy versus time."""
    fig, ax = _new_axes("time [s]", "energy [J]", "Mechanical energy")
    series = [
        ("kinetic", result.kinetic),
        ("potential", result.potential),
        ("total", result.energy),
    ]
    for (label, values), color in zip(series, _CATEGORICAL):
        ax.plot(result.t, values, color=color, linewidth=1.6, label=label)
        ax.annotate(label, (result.t[-1], values[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    color=_INK_SECONDARY, fontsize=8, va="center")
    return _finish(fig, ax, _as_path(path), len(series))


def plot_energy_error(result: SimulationResult,
                      path: "Path | str | None" = None):
    """Energy-balance residual E(t) - E(0) + E_dissipated(t) versus time.

    For undamped runs this is exactly the energy drift; with damping it also
    verifies that the lost energy matches the joint dissipation.
    """
    fig, ax = _new_axes("time [s]", "energy residual [J]",
                        "Energy-balance error (drift + unaccounted dissipation)")
    ax.plot(result.t, result.energy_error, color=_CATEGORICAL[0],
            linewidth=1.6)
    ax.axhline(0.0, color=_BASELINE, linewidth=1.0)
    return _finish(fig, ax, _as_path(path), 1)


def plot_phase_space(result: SimulationResult,
                     path: "Path | str | None" = None):
    """Phase-space trajectory (theta_i, omega_i) of every link."""
    n = result.config.num_links
    fig, ax = _new_axes("angle [rad]", "angular velocity [rad/s]",
                        "Phase space")
    for i, color in enumerate(link_colors(n)):
        ax.plot(result.theta[:, i], result.omega[:, i], color=color,
                linewidth=1.0, alpha=0.85, label=f"link {i + 1}")
    return _finish(fig, ax, _as_path(path), n)


def plot_phase_space_per_link(result: SimulationResult,
                              path: "Path | str | None" = None):
    """Phase-space trajectory (theta_i, omega_i) of each link in its own subplot."""
    n = result.config.num_links
    colors = link_colors(n)
    ncols = min(3, n)
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows),
                             dpi=120, squeeze=False)
    fig.patch.set_facecolor(_SURFACE)
    fig.suptitle("Phase space per link", color=_INK, x=0.02, ha="left",
                fontsize=12)

    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i >= n:
            ax.set_visible(False)
            continue
        ax.set_facecolor(_SURFACE)
        ax.plot(result.theta[:, i], result.omega[:, i], color=colors[i],
                linewidth=1.0, alpha=0.85)
        ax.set_title(f"link {i + 1}", color=_INK, loc="left", fontsize=10)
        ax.set_xlabel("angle [rad]", color=_INK_SECONDARY)
        ax.set_ylabel("angular velocity [rad/s]", color=_INK_SECONDARY)
        ax.grid(True, color=_GRID, linewidth=0.8)
        ax.tick_params(colors=_MUTED)
        for spine in ax.spines.values():
            spine.set_color(_BASELINE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    resolved = _as_path(path)
    if resolved is not None:
        fig.savefig(resolved, facecolor=_SURFACE)
        plt.close(fig)
    return fig


def animate_pendulum(
    result: SimulationResult,
    path: "Path | str | None" = None,
    fps: int = 30,
    max_frames: int = 600,
    trail_seconds: float = 1.5,
) -> FuncAnimation:
    """Animate the pendulum motion for arbitrary N.

    The trajectory is subsampled to at most ``max_frames`` frames.  The last
    bob leaves a fading trail.  If ``path`` is given the animation is saved
    as a GIF; otherwise the (unsaved) ``FuncAnimation`` is returned.
    """
    config = result.config
    n = config.num_links
    x, y = result.positions()

    stride = max(1, len(result.t) // max_frames)
    frames = range(0, len(result.t), stride)
    trail_pts = max(1, int(round(trail_seconds / config.dt_output)))

    reach = float(np.sum(config.lengths_array)) * 1.05
    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.tick_params(colors=_MUTED)
    for spine in ax.spines.values():
        spine.set_color(_BASELINE)
    ax.set_title(f"{n}-link pendulum", color=_INK, loc="left", fontsize=12)

    rods, = ax.plot([], [], color=_INK_SECONDARY, linewidth=1.6, zorder=2)
    trail, = ax.plot([], [], color=_MUTED, linewidth=1.0, alpha=0.6, zorder=1)
    # Marker area proportional to mass so heavier bobs read as heavier
    sizes = 40.0 * config.masses_array / np.max(config.masses_array)
    bobs = ax.scatter(x[0], y[0], s=sizes, c=link_colors(n), zorder=3)
    time_label = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                         color=_INK_SECONDARY, fontsize=9, va="top")

    def update(frame: int):
        rods.set_data(np.concatenate([[0.0], x[frame]]),
                      np.concatenate([[0.0], y[frame]]))
        start = max(0, frame - trail_pts)
        trail.set_data(x[start:frame + 1, -1], y[start:frame + 1, -1])
        bobs.set_offsets(np.column_stack([x[frame], y[frame]]))
        time_label.set_text(f"t = {result.t[frame]:5.2f} s")
        return rods, trail, bobs, time_label

    animation = FuncAnimation(fig, update, frames=frames,
                              interval=1000.0 / fps, blit=True)
    if path is not None:
        animation.save(str(path), writer=PillowWriter(fps=fps))
        plt.close(fig)
    return animation


def save_all(result: SimulationResult, output_dir: "Path | str",
             animation: bool = True) -> List[Path]:
    """Write every diagnostic figure (and optionally the animation) to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, plot in [
        ("angles.svg", plot_angles),
        ("velocities.svg", plot_velocities),
        ("energy.svg", plot_energy),
        ("energy_error.svg", plot_energy_error),
        ("phase_space.svg", plot_phase_space),
        ("phase_space_per_link.svg", plot_phase_space_per_link),
    ]:
        target = output_dir / name
        plot(result, target)
        written.append(target)
    if animation:
        target = output_dir / "animation.gif"
        animate_pendulum(result, target)
        written.append(target)
    return written


def _as_path(path: "Path | str | None") -> Optional[Path]:
    return None if path is None else Path(path)
