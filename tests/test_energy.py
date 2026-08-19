"""Energy conservation, drift and dissipation diagnostics."""

import numpy as np

from npendulum.config import SimulationConfig
from npendulum.diagnostics import energy_diagnostics
from npendulum.simulate import run_simulation


def _config(**overrides):
    defaults = dict(
        num_links=3,
        masses=(1.0, 0.8, 1.2),
        lengths=(1.0, 0.9, 0.6),
        initial_angles=(1.5, -0.8, 2.2),
        initial_velocities=(0.0, 0.4, -0.3),
        t_final=20.0,
        dt_output=0.01,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def test_undamped_energy_conserved_dop853():
    result = run_simulation(_config(rtol=1e-11, atol=1e-11))
    diag = energy_diagnostics(result)
    assert diag.dissipated_energy == 0.0
    assert diag.max_relative_error < 1e-8


def test_undamped_energy_bounded_midpoint():
    """Symplectic midpoint: energy error stays bounded, no secular drift.

    The error bound is O(dt^2) but does not grow with simulation time —
    the defining long-term property of a symplectic scheme.
    """
    result = run_simulation(
        _config(initial_angles=(0.8, -0.5, 0.6), integrator="midpoint",
                dt_internal=0.002, t_final=50.0)
    )
    rel = np.abs(result.relative_energy_error)
    assert np.max(rel) < 2e-4
    # No secular growth: the last quarter is no worse than the second one.
    q = len(rel) // 4
    assert np.max(rel[-q:]) < 3 * np.max(rel[q:2 * q])


def test_damped_energy_decreases_and_balances():
    result = run_simulation(_config(damping=0.4, rtol=1e-11, atol=1e-11))
    energy = result.energy
    # Monotonically non-increasing total energy (tiny tolerance for output
    # sampling of an oscillatory decay)
    assert np.all(np.diff(energy) <= 1e-10)
    assert energy[-1] < energy[0]
    # Lost energy is fully accounted for by the joint dissipation integral
    diag = energy_diagnostics(result)
    assert diag.dissipated_energy > 0
    assert diag.max_relative_error < 1e-6


def test_damped_pendulum_settles_to_rest():
    result = run_simulation(
        _config(damping=2.0, t_final=120.0, initial_angles=(0.6, 0.2, -0.4))
    )
    assert np.max(np.abs(result.omega[-1])) < 1e-3
    assert np.max(np.abs(result.theta[-1])) < 1e-3  # hangs straight down


def test_dop853_energy_error_tracks_tolerance():
    loose = run_simulation(_config(rtol=1e-6, atol=1e-6, t_final=10.0))
    tight = run_simulation(_config(rtol=1e-12, atol=1e-12, t_final=10.0))
    loose_err = np.max(np.abs(loose.relative_energy_error))
    tight_err = np.max(np.abs(tight.relative_energy_error))
    assert tight_err < loose_err
    assert tight_err < 1e-9
