"""Numerical integrators for the pendulum equations of motion.

Three methods are provided behind one interface (``integrate(f, y0, t_eval)``
returning the states at the requested output times):

``dop853``
    Adaptive 8th-order Dormand-Prince (via :func:`scipy.integrate.solve_ivp`).
    Gives rigorous local error control (``rtol``/``atol``), which is what
    keeps individual chaotic trajectories accurate over the simulated span.

``midpoint``
    Fixed-step implicit midpoint rule.  The N-pendulum Hamiltonian is
    *non-separable* (the mass matrix depends on the angles), so explicit
    symplectic schemes such as leapfrog/Verlet do not apply.  The implicit
    midpoint rule is symplectic and time-reversible for general Hamiltonian
    systems, so its energy error stays bounded instead of drifting
    secularly — the right tool for long undamped runs.

``rk4``
    Classical fixed-step 4th-order Runge-Kutta; a well-understood baseline
    used in the convergence study.

Explicit Euler is deliberately not offered: it is neither symplectic nor
A-stable and exhibits unbounded energy growth on oscillatory systems.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

RHS = Callable[[float, np.ndarray], np.ndarray]


def integrate_dop853(
    f: RHS,
    y0: np.ndarray,
    t_eval: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> np.ndarray:
    """Adaptive integration; returns states of shape (len(t_eval), len(y0))."""
    sol = solve_ivp(
        f,
        (t_eval[0], t_eval[-1]),
        np.asarray(y0, dtype=float),
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"DOP853 integration failed: {sol.message}")
    return sol.y.T


def _fixed_step_grid(t_eval: np.ndarray, dt_internal: "float | None"):
    """Split each output interval into an integer number of internal steps."""
    dt_out = float(t_eval[1] - t_eval[0])
    if not np.allclose(np.diff(t_eval), dt_out):
        raise ValueError("fixed-step integrators require a uniform output grid")
    if dt_internal is None:
        dt_internal = dt_out
    substeps = max(1, int(round(dt_out / dt_internal)))
    return substeps, dt_out / substeps


def integrate_rk4(
    f: RHS,
    y0: np.ndarray,
    t_eval: np.ndarray,
    dt_internal: "float | None" = None,
) -> np.ndarray:
    """Classical fixed-step 4th-order Runge-Kutta."""
    substeps, h = _fixed_step_grid(t_eval, dt_internal)
    out = np.empty((len(t_eval), len(y0)))
    y = np.asarray(y0, dtype=float).copy()
    out[0] = y
    t = float(t_eval[0])
    for i in range(1, len(t_eval)):
        for _ in range(substeps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        out[i] = y
    return out


def integrate_midpoint(
    f: RHS,
    y0: np.ndarray,
    t_eval: np.ndarray,
    dt_internal: "float | None" = None,
    fixed_point_tol: float = 1e-13,
    max_iterations: int = 100,
) -> np.ndarray:
    """Symplectic implicit midpoint rule.

    Each step solves ``z = y_n + (h/2) f(t + h/2, z)`` for the midpoint state
    ``z`` by fixed-point iteration, then sets ``y_{n+1} = 2 z - y_n``.  The
    iteration converges for step sizes small compared to the fastest system
    timescale; a persistent failure signals that ``dt_internal`` is too large.
    """
    substeps, h = _fixed_step_grid(t_eval, dt_internal)
    out = np.empty((len(t_eval), len(y0)))
    y = np.asarray(y0, dtype=float).copy()
    out[0] = y
    t = float(t_eval[0])
    diverged = RuntimeError(
        "implicit midpoint fixed-point iteration did not converge; "
        "reduce dt_internal"
    )
    for i in range(1, len(t_eval)):
        for _ in range(substeps):
            t_mid = t + h / 2
            z = y + (h / 2) * f(t, y)  # explicit Euler predictor
            for _ in range(max_iterations):
                try:
                    z_new = y + (h / 2) * f(t_mid, z)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    raise diverged from None  # iterate blew up inside f
                if not np.all(np.isfinite(z_new)):
                    raise diverged
                scale = np.maximum(1.0, np.abs(z_new))
                if np.max(np.abs(z_new - z) / scale) < fixed_point_tol:
                    z = z_new
                    break
                z = z_new
            else:
                raise diverged
            y = 2.0 * z - y
            t += h
        out[i] = y
    return out
