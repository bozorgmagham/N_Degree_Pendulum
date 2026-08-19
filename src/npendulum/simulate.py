"""Simulation driver: turn a :class:`SimulationConfig` into a trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .dynamics import NPendulumDynamics
from .integrators import integrate_dop853, integrate_midpoint, integrate_rk4


@dataclass(frozen=True)
class SimulationResult:
    """Trajectory and derived diagnostics of one simulation run.

    Attributes
    ----------
    config:
        The exact configuration used (makes the run reproducible).
    t:
        Output times, shape ``(T,)``.
    theta, omega:
        Angles and angular velocities, shape ``(T, N)``.
    kinetic, potential, energy:
        Kinetic, potential and total mechanical energy, shape ``(T,)``.
    dissipated:
        Cumulative energy removed by joint damping, shape ``(T,)``,
        integrated alongside the state so it carries the same numerical
        accuracy as the trajectory.
    """

    config: SimulationConfig
    t: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    kinetic: np.ndarray
    potential: np.ndarray
    energy: np.ndarray
    dissipated: np.ndarray

    @property
    def energy_error(self) -> np.ndarray:
        """Energy-balance residual E(t) - E(0) + E_dissipated(t).

        Zero in exact arithmetic, for both damped and undamped systems;
        its magnitude measures the numerical error of the integration.
        """
        return self.energy - self.energy[0] + self.dissipated

    @property
    def relative_energy_error(self) -> np.ndarray:
        """Energy-balance residual normalized by the initial energy scale."""
        scale = max(abs(self.energy[0]), 1e-12)
        return self.energy_error / scale

    def positions(self) -> "tuple[np.ndarray, np.ndarray]":
        """Cartesian bob trajectories, each of shape ``(T, N)``."""
        dyn = NPendulumDynamics.from_config(self.config)
        return dyn.positions(self.theta)


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Integrate the N-pendulum defined by ``config`` and package the result.

    Deterministic: the same configuration always produces the same result.
    """
    dynamics = NPendulumDynamics.from_config(config)
    t = config.output_times
    # Augmented state [theta, omega, W]: W accumulates the dissipated energy
    # with the same accuracy as the trajectory (see rhs_augmented).
    y0 = np.concatenate([config.initial_state, [0.0]])

    if config.integrator == "dop853":
        states = integrate_dop853(
            dynamics.rhs_augmented, y0, t, rtol=config.rtol, atol=config.atol
        )
    elif config.integrator == "midpoint":
        states = integrate_midpoint(
            dynamics.rhs_augmented, y0, t, config.dt_internal
        )
    elif config.integrator == "rk4":
        states = integrate_rk4(dynamics.rhs_augmented, y0, t,
                               config.dt_internal)
    else:  # pragma: no cover - guarded by SimulationConfig validation
        raise ValueError(f"unknown integrator {config.integrator!r}")

    theta = states[:, : config.num_links]
    omega = states[:, config.num_links : 2 * config.num_links]
    dissipated = states[:, -1]

    kinetic = dynamics.kinetic_energy(theta, omega)
    potential = dynamics.potential_energy(theta)

    return SimulationResult(
        config=config,
        t=t,
        theta=theta,
        omega=omega,
        kinetic=kinetic,
        potential=potential,
        energy=kinetic + potential,
        dissipated=dissipated,
    )
