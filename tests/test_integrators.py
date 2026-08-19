"""Convergence and cross-consistency of the integrators."""

import numpy as np
import pytest

from npendulum.config import SimulationConfig
from npendulum.diagnostics import convergence_study
from npendulum.simulate import run_simulation


def _base_config(**overrides):
    defaults = dict(
        num_links=2,
        masses=(1.0, 0.8),
        lengths=(1.0, 0.7),
        initial_angles=(0.9, -0.4),
        initial_velocities=(0.0, 0.5),
        t_final=2.0,
        dt_output=0.05,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def test_rk4_fourth_order_convergence():
    # Mild amplitudes and a dt range inside the asymptotic regime (but above
    # the error floor of the DOP853 reference) give a clean order estimate.
    config = _base_config(initial_angles=(0.5, -0.3),
                          initial_velocities=(0.0, 0.2))
    study = convergence_study(config, dts=[0.02, 0.01, 0.005],
                              integrator="rk4")
    assert 3.4 < study.observed_order < 4.6
    assert np.all(np.diff(study.errors) < 0)  # errors shrink with dt


def test_midpoint_second_order_convergence():
    study = convergence_study(_base_config(), dts=[0.02, 0.01, 0.005, 0.0025],
                              integrator="midpoint")
    assert 1.7 < study.observed_order < 2.3
    assert np.all(np.diff(study.errors) < 0)


@pytest.mark.parametrize("integrator,theta_tol,omega_tol", [
    ("midpoint", 2e-5, 2e-4),  # 2nd order: larger trajectory error at fixed dt
    ("rk4", 1e-6, 1e-5),
])
def test_integrators_agree_with_dop853(integrator, theta_tol, omega_tol):
    """All methods must converge to the same trajectory on a short horizon."""
    reference = run_simulation(_base_config(rtol=1e-12, atol=1e-12))
    other = run_simulation(
        _base_config(integrator=integrator, dt_internal=0.001)
    )
    assert np.max(np.abs(other.theta - reference.theta)) < theta_tol
    assert np.max(np.abs(other.omega - reference.omega)) < omega_tol


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # deliberate blow-up
def test_midpoint_diverges_gracefully_on_huge_step():
    config = _base_config(integrator="midpoint", dt_internal=0.05,
                          initial_velocities=(50.0, -50.0))
    with pytest.raises(RuntimeError, match="did not converge"):
        run_simulation(config)
