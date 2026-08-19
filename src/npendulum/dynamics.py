"""Physics of the N-link pendulum.

The model is a chain of N point masses connected by rigid massless rods,
suspended from a fixed pivot.  Generalized coordinates are the *absolute*
angles ``theta_i`` of each link measured from the downward vertical.

With the tail-mass sums ``mu_i = sum_{k>=i} m_k`` and
``a_ij = mu_max(i,j)``, the Lagrangian ``L = T - V`` with

    T = 1/2 * sum_ij a_ij l_i l_j cos(theta_i - theta_j) w_i w_j
    V = -g * sum_i mu_i l_i cos(theta_i)

yields, via the Euler-Lagrange equations (see docs/DERIVATION.md), the
closed-form linear system for the angular accelerations:

    M(theta) @ thetadd = b(theta, w)

    M_ij = a_ij l_i l_j cos(theta_i - theta_j)
    b_i  = - sum_j a_ij l_i l_j sin(theta_i - theta_j) w_j**2
           - g mu_i l_i sin(theta_i) + Q_i

where ``Q_i`` are the generalized damping forces.  Every expression is built
with vectorized NumPy for arbitrary N; nothing is hard-coded per N.

Damping model: each joint ``i`` exerts a torque proportional to the relative
angular velocity ``w_i - w_{i-1}`` of the links it connects (``w_0 = 0`` for
the fixed support).  This is the Rayleigh dissipation function
``R = 1/2 * sum_i c_i (w_i - w_{i-1})**2`` with ``Q_i = -dR/dw_i``, so the
mechanical energy obeys ``dE/dt = -2R <= 0``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .config import SimulationConfig


class NPendulumDynamics:
    """Equations of motion and energy functions for an N-link pendulum.

    Parameters mirror the physical part of :class:`SimulationConfig`;
    :meth:`from_config` builds an instance directly from one.
    """

    def __init__(
        self,
        masses: np.ndarray,
        lengths: np.ndarray,
        gravity: float,
        damping: np.ndarray,
    ) -> None:
        self.m = np.asarray(masses, dtype=float)
        self.l = np.asarray(lengths, dtype=float)
        self.g = float(gravity)
        self.c = np.asarray(damping, dtype=float)
        self.n = self.m.size
        if not (self.l.size == self.c.size == self.n):
            raise ValueError("masses, lengths and damping must have equal length")

        # mu_i = sum_{k>=i} m_k (tail sums); a_ij = mu_{max(i,j)}
        self.mu = np.cumsum(self.m[::-1])[::-1]
        idx = np.arange(self.n)
        self.a = self.mu[np.maximum.outer(idx, idx)]
        # Precomputed coefficient matrix a_ij * l_i * l_j
        self._A = self.a * np.outer(self.l, self.l)

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "NPendulumDynamics":
        return cls(
            config.masses_array,
            config.lengths_array,
            config.gravity,
            config.damping_array,
        )

    # -- equations of motion ----------------------------------------------------

    def mass_matrix(self, theta: np.ndarray) -> np.ndarray:
        """Generalized (symmetric positive definite) mass matrix M(theta)."""
        diff = np.subtract.outer(theta, theta)
        return self._A * np.cos(diff)

    def damping_forces(self, omega: np.ndarray) -> np.ndarray:
        """Generalized forces Q from the Rayleigh dissipation function."""
        rel = np.diff(omega, prepend=0.0)  # w_i - w_{i-1}, with w_0 = 0
        torque = self.c * rel
        return -torque + np.append(torque[1:], 0.0)

    def accelerations(self, theta: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Solve M(theta) @ thetadd = b(theta, omega) for the accelerations."""
        diff = np.subtract.outer(theta, theta)
        M = self._A * np.cos(diff)
        S = self._A * np.sin(diff)
        b = (
            -S @ (omega**2)
            - self.g * self.mu * self.l * np.sin(theta)
            + self.damping_forces(omega)
        )
        try:
            return cho_solve(cho_factor(M), b)
        except np.linalg.LinAlgError:
            # M is SPD in exact arithmetic; fall back for near-singular cases
            return np.linalg.solve(M, b)

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """First-order ODE right-hand side for y = [theta, omega]."""
        theta, omega = np.split(y, 2)
        return np.concatenate([omega, self.accelerations(theta, omega)])

    def rhs_augmented(self, t: float, y: np.ndarray) -> np.ndarray:
        """RHS for y = [theta, omega, W] where dW/dt = dissipated power.

        Integrating the dissipated energy W alongside the state (instead of
        post-hoc quadrature of sampled power) keeps the energy-balance
        diagnostic as accurate as the trajectory itself.  W does not feed
        back into the dynamics, so the pendulum states are unaffected.
        """
        theta = y[: self.n]
        omega = y[self.n : 2 * self.n]
        return np.concatenate([
            omega,
            self.accelerations(theta, omega),
            [self.dissipated_power(omega)],
        ])

    # -- geometry ---------------------------------------------------------------

    def positions(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cartesian bob positions.

        Accepts a single state ``(N,)`` or a trajectory ``(T, N)``; returns
        ``(x, y)`` arrays of the same shape, with the pivot at the origin and
        y pointing upward.
        """
        theta = np.asarray(theta)
        x = np.cumsum(self.l * np.sin(theta), axis=-1)
        y = -np.cumsum(self.l * np.cos(theta), axis=-1)
        return x, y

    # -- energies (accept single states or trajectories) ------------------------

    def kinetic_energy(self, theta: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """T = 1/2 w^T M(theta) w, vectorized over a leading time axis."""
        theta = np.asarray(theta)
        omega = np.asarray(omega)
        diff = theta[..., :, None] - theta[..., None, :]
        M = self._A * np.cos(diff)
        return 0.5 * np.einsum("...i,...ij,...j->...", omega, M, omega)

    def potential_energy(self, theta: np.ndarray) -> np.ndarray:
        """V = -g sum_i mu_i l_i cos(theta_i)  (zero at the pivot height)."""
        theta = np.asarray(theta)
        return -self.g * np.sum(self.mu * self.l * np.cos(theta), axis=-1)

    def total_energy(self, theta: np.ndarray, omega: np.ndarray) -> np.ndarray:
        return self.kinetic_energy(theta, omega) + self.potential_energy(theta)

    def dissipated_power(self, omega: np.ndarray) -> np.ndarray:
        """Instantaneous dissipated power P = 2R = sum_i c_i (w_i - w_{i-1})^2.

        The energy balance is dE/dt = -P.
        """
        omega = np.asarray(omega)
        rel = np.diff(omega, axis=-1, prepend=0.0)
        return np.sum(self.c * rel**2, axis=-1)
