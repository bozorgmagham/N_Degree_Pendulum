"""Numerical-quality diagnostics: energy drift and timestep convergence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .config import SimulationConfig
from .simulate import SimulationResult, run_simulation


@dataclass(frozen=True)
class EnergyDiagnostics:
    """Summary statistics of the energy-balance residual of one run."""

    initial_energy: float
    final_energy: float
    dissipated_energy: float
    max_abs_error: float
    max_relative_error: float
    final_relative_error: float


def energy_diagnostics(result: SimulationResult) -> EnergyDiagnostics:
    """Summarize how well the run honors the energy balance dE/dt = -P."""
    err = result.energy_error
    rel = result.relative_energy_error
    return EnergyDiagnostics(
        initial_energy=float(result.energy[0]),
        final_energy=float(result.energy[-1]),
        dissipated_energy=float(result.dissipated[-1]),
        max_abs_error=float(np.max(np.abs(err))),
        max_relative_error=float(np.max(np.abs(rel))),
        final_relative_error=float(rel[-1]),
    )


@dataclass(frozen=True)
class ConvergenceStudy:
    """Result of a timestep-refinement study for a fixed-step integrator."""

    integrator: str
    dts: np.ndarray
    errors: np.ndarray      # max-norm final-state error vs. the reference
    observed_order: float   # log-log slope fitted over the dt range


def convergence_study(
    config: SimulationConfig,
    dts: Sequence[float],
    integrator: "str | None" = None,
) -> ConvergenceStudy:
    """Measure how the final-state error shrinks as the timestep decreases.

    Runs the fixed-step ``integrator`` (default: the one in ``config``) with
    each internal step in ``dts`` and compares the final state against a
    tightly-toleranced DOP853 reference solution of the same problem.  The
    fitted log-log slope estimates the integrator's convergence order.
    """
    integrator = integrator or config.integrator
    if integrator not in ("midpoint", "rk4"):
        raise ValueError("convergence_study requires a fixed-step integrator")

    reference = run_simulation(
        replace(config, integrator="dop853", rtol=1e-12, atol=1e-12)
    )
    ref_final = np.concatenate([reference.theta[-1], reference.omega[-1]])

    errors = []
    for dt in dts:
        result = run_simulation(
            replace(config, integrator=integrator, dt_internal=float(dt))
        )
        final = np.concatenate([result.theta[-1], result.omega[-1]])
        errors.append(np.max(np.abs(final - ref_final)))

    dts_arr = np.asarray(dts, dtype=float)
    errors_arr = np.asarray(errors)
    slope = float(np.polyfit(np.log(dts_arr), np.log(errors_arr), 1)[0])
    return ConvergenceStudy(
        integrator=integrator, dts=dts_arr, errors=errors_arr,
        observed_order=slope,
    )
