#!/usr/bin/env python
"""Timestep-convergence study of the fixed-step integrators.

Runs the RK4 and implicit-midpoint integrators over a range of internal
timesteps, measures the final-state error against a tight-tolerance DOP853
reference, and writes a log-log convergence plot annotated with the fitted
orders (expected: ~4 for RK4, ~2 for implicit midpoint).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from npendulum import SimulationConfig, convergence_study

OUTPUT = Path(__file__).resolve().parent.parent / "output"


def main() -> None:
    config = SimulationConfig(
        num_links=2,
        masses=(1.0, 0.8),
        lengths=(1.0, 0.7),
        initial_angles=(0.5, -0.3),
        initial_velocities=(0.0, 0.2),
        t_final=2.0,
        dt_output=0.05,
    )
    dts = [0.02, 0.01, 0.005, 0.0025]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for integrator, color in [("rk4", "#2a78d6"), ("midpoint", "#eb6834")]:
        study = convergence_study(config, dts, integrator=integrator)
        print(f"{integrator}: observed order {study.observed_order:.2f}")
        ax.loglog(study.dts, study.errors, "o-", color=color, linewidth=1.6,
                  markersize=5,
                  label=f"{integrator} (order {study.observed_order:.2f})")

    ax.set_xlabel("internal timestep dt [s]", color="#52514e")
    ax.set_ylabel("final-state max error vs DOP853 reference", color="#52514e")
    ax.set_title("Timestep convergence", color="#0b0b0b", loc="left")
    ax.grid(True, which="both", color="#e1e0d9", linewidth=0.8)
    ax.legend(labelcolor="#52514e", edgecolor="#e1e0d9")
    ax.tick_params(colors="#898781")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "convergence.svg"
    fig.tight_layout()
    fig.savefig(target, facecolor="#fcfcfb")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
