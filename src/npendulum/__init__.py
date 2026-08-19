"""npendulum: Lagrangian simulation of a chaotic N-link pendulum."""

from .config import SimulationConfig
from .diagnostics import ConvergenceStudy, convergence_study, energy_diagnostics
from .dynamics import NPendulumDynamics
from .simulate import SimulationResult, run_simulation

__all__ = [
    "SimulationConfig",
    "NPendulumDynamics",
    "SimulationResult",
    "run_simulation",
    "energy_diagnostics",
    "convergence_study",
    "ConvergenceStudy",
]

__version__ = "0.1.0"
