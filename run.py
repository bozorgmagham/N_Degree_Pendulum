#!/usr/bin/env python
"""Edit the settings below, then run: python run.py

Every SimulationConfig field can be set here. Per-link values (masses,
lengths, damping, initial_angles, initial_velocities) accept either a single
number (broadcast to all links) or a tuple with one value per link.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from npendulum import SimulationConfig, energy_diagnostics, run_simulation
from npendulum.visualize import save_all

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

config = SimulationConfig(
    num_links=3,
    masses=1.0,
    lengths=(1.0, 1.5, 2.0),
    gravity=9.81,
    damping=(1.2, 0.5, 0.02),
    initial_angles=1.0 * math.pi / 3.0,    # 120 degrees from vertical
    initial_velocities=0.0,
    t_final=30.0,
    dt_output=0.01,
    integrator="dop853",                   # "dop853" | "midpoint" | "rk4"
    dt_internal=None,                      # fixed-step integrators only
    rtol=1e-10,
    atol=1e-10,
)

output_dir = Path(__file__).resolve().parent / "output"
write_animation = True                     # animation.gif is slow to render

# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Simulating {config.num_links}-link pendulum "
          f"({config.integrator}, t_final={config.t_final}s) ...")
    result = run_simulation(config)

    diag = energy_diagnostics(result)
    print(f"  initial energy      : {diag.initial_energy: .6f} J")
    print(f"  final energy        : {diag.final_energy: .6f} J")
    print(f"  dissipated energy   : {diag.dissipated_energy: .6f} J")
    print(f"  max |energy error|  : {diag.max_abs_error: .3e} J "
          f"({diag.max_relative_error:.3e} relative)")

    output_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(output_dir / "config.json")
    np.savez(output_dir / "trajectory.npz", t=result.t,
             theta=result.theta, omega=result.omega, energy=result.energy,
             kinetic=result.kinetic, potential=result.potential,
             dissipated=result.dissipated)
    written = save_all(result, output_dir, animation=write_animation)
    for path in [output_dir / "config.json", output_dir / "trajectory.npz"] + written:
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
