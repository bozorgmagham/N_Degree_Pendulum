"""Simulation configuration.

All physical parameters and numerical settings for an N-link pendulum run are
collected in :class:`SimulationConfig`, an immutable dataclass.  Per-link
quantities (masses, lengths, damping, initial conditions) may be given either
as a scalar (broadcast to all links) or as a sequence of length ``num_links``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence, Union

import numpy as np

ScalarOrSequence = Union[float, int, Sequence[float]]

_INTEGRATORS = ("dop853", "midpoint", "rk4")


def _broadcast(value: ScalarOrSequence, n: int, name: str) -> "tuple[float, ...]":
    """Broadcast a scalar to length ``n`` or validate a length-``n`` sequence."""
    if isinstance(value, (int, float)):
        return (float(value),) * n
    values = tuple(float(v) for v in value)
    if len(values) != n:
        raise ValueError(
            f"{name} must be a scalar or a sequence of length {n}, "
            f"got length {len(values)}"
        )
    return values


@dataclass(frozen=True)
class SimulationConfig:
    """Physical and numerical parameters of an N-link pendulum simulation.

    Parameters
    ----------
    num_links:
        Number of links / point masses (N >= 1).
    masses:
        Mass of each bob [kg].  Scalar or length-N sequence.
    lengths:
        Length of each rigid link [m].  Scalar or length-N sequence.
    gravity:
        Gravitational acceleration [m/s^2].
    damping:
        Joint damping coefficients ``c_i`` [N*m*s].  Joint ``i`` (1-based)
        connects link ``i`` to link ``i-1``; joint 1 connects link 1 to the
        fixed support.  The dissipative torque at joint ``i`` is proportional
        to the *relative* angular velocity of the connected links.  Scalar or
        length-N sequence; set the first entry to 0 to disable damping
        against the fixed support.
    initial_angles:
        Initial angle of each link measured from the downward vertical [rad].
        Scalar or length-N sequence.
    initial_velocities:
        Initial angular velocity of each link [rad/s].  Scalar or length-N.
    t_final:
        Simulation duration [s].
    dt_output:
        Sampling interval of the stored trajectory [s].
    integrator:
        One of ``"dop853"`` (adaptive 8th-order Runge-Kutta, default),
        ``"midpoint"`` (fixed-step symplectic implicit midpoint) or
        ``"rk4"`` (fixed-step classical Runge-Kutta).
    dt_internal:
        Internal step size for the fixed-step integrators [s].  If ``None``
        it defaults to ``dt_output``.  The actual step is rounded so that an
        integer number of internal steps fits in one output interval.
    rtol, atol:
        Relative/absolute tolerances for the adaptive integrator.
    """

    num_links: int
    masses: ScalarOrSequence = 1.0
    lengths: ScalarOrSequence = 1.0
    gravity: float = 9.81
    damping: ScalarOrSequence = 0.0
    initial_angles: ScalarOrSequence = 0.0
    initial_velocities: ScalarOrSequence = 0.0
    t_final: float = 10.0
    dt_output: float = 0.01
    integrator: str = "dop853"
    dt_internal: "float | None" = field(default=None)
    rtol: float = 1e-10
    atol: float = 1e-10

    def __post_init__(self) -> None:
        if not isinstance(self.num_links, int) or self.num_links < 1:
            raise ValueError("num_links must be a positive integer")
        for name in ("masses", "lengths", "damping", "initial_angles",
                     "initial_velocities"):
            object.__setattr__(
                self, name, _broadcast(getattr(self, name), self.num_links, name)
            )
        if any(m <= 0 for m in self.masses):
            raise ValueError("all masses must be positive")
        if any(l <= 0 for l in self.lengths):
            raise ValueError("all lengths must be positive")
        if any(c < 0 for c in self.damping):
            raise ValueError("damping coefficients must be non-negative")
        if self.t_final <= 0:
            raise ValueError("t_final must be positive")
        if self.dt_output <= 0:
            raise ValueError("dt_output must be positive")
        if self.dt_output > self.t_final:
            raise ValueError("dt_output must not exceed t_final")
        if self.integrator not in _INTEGRATORS:
            raise ValueError(f"integrator must be one of {_INTEGRATORS}")
        if self.dt_internal is not None and self.dt_internal <= 0:
            raise ValueError("dt_internal must be positive")
        if self.rtol <= 0 or self.atol <= 0:
            raise ValueError("rtol and atol must be positive")

    # -- convenient array views -------------------------------------------------

    @property
    def masses_array(self) -> np.ndarray:
        return np.asarray(self.masses)

    @property
    def lengths_array(self) -> np.ndarray:
        return np.asarray(self.lengths)

    @property
    def damping_array(self) -> np.ndarray:
        return np.asarray(self.damping)

    @property
    def initial_state(self) -> np.ndarray:
        """Initial state vector ``y0 = [theta_1..theta_N, omega_1..omega_N]``."""
        return np.concatenate([self.initial_angles, self.initial_velocities])

    @property
    def output_times(self) -> np.ndarray:
        """Uniform output grid from 0 to (approximately) ``t_final``."""
        n_steps = int(round(self.t_final / self.dt_output))
        return np.linspace(0.0, n_steps * self.dt_output, n_steps + 1)

    # -- serialization ----------------------------------------------------------

    def to_json(self, path: "str | Path") -> None:
        """Write the configuration to a JSON file (fully reproducible)."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: "str | Path") -> "SimulationConfig":
        """Load a configuration previously written by :meth:`to_json`."""
        return cls(**json.loads(Path(path).read_text()))
