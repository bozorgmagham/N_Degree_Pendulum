# npendulum — Chaotic N-Link Pendulum Simulator

A self-contained, extensible physics simulation of a planar pendulum made of
**N point masses connected by rigid massless rods**, with configurable joint
damping. The equations of motion are derived from the Lagrangian formulation
and evaluated in a closed form valid for *arbitrary* N — nothing is
hard-coded per link count.

![animation](output/animation.gif)

## Model and assumptions

- N point masses `m_1 … m_N` on rigid massless rods `l_1 … l_N`, hanging from
  a fixed frictionless pivot; motion is planar; gravity `g` acts downward.
- Generalized coordinates: absolute angles `theta_i` of each link from the
  downward vertical.
- Optional viscous damping at every joint, proportional to the **relative
  angular velocity of the connected links** (joint 1 damps link 1 against the
  fixed support; set its coefficient to 0 to disable that).

With the tail-mass sums `mu_i = m_i + m_{i+1} + … + m_N` and
`a_ij = mu_max(i,j)`, the Euler–Lagrange equations reduce to the linear system

```
M(theta) @ thetadd = b(theta, omega)

M_ij = a_ij l_i l_j cos(theta_i - theta_j)                (symmetric positive definite)
b_i  = - sum_j a_ij l_i l_j sin(theta_i - theta_j) omega_j^2
       - g mu_i l_i sin(theta_i)  +  Q_i                  (Q_i = joint damping forces)
```

solved each step by Cholesky factorization. The full derivation — kinetic and
potential energy, Lagrangian, Euler–Lagrange equations, Rayleigh dissipation,
and the exact mapping of every term to the code — is in
[docs/DERIVATION.md](docs/DERIVATION.md).

**Validation:** an independent SymPy module
([src/npendulum/symbolic.py](src/npendulum/symbolic.py)) re-derives the
equations symbolically from first principles; the tests confirm both agree to
machine precision for N = 1, 2, 3, and that N = 1 / N = 2 match the analytic
single- and textbook double-pendulum equations.

## Numerical integration — why these methods

For a nonlinear Hamiltonian system with chaotic trajectories, the integrator
choice matters more than usual. Explicit Euler is excluded outright: it is
neither symplectic nor stable on oscillatory problems and pumps energy in
unboundedly. Three methods are provided
([src/npendulum/integrators.py](src/npendulum/integrators.py)):

| method | type | order | why it's here |
|---|---|---|---|
| `dop853` (default) | adaptive Runge–Kutta (SciPy) | 8 | rigorous local error control (`rtol`/`atol`) — the right tool for accurate individual chaotic trajectories over finite horizons |
| `midpoint` | implicit midpoint, fixed step | 2 | **symplectic and time-reversible for non-separable Hamiltonians.** The N-pendulum's mass matrix depends on the angles, so the Hamiltonian is non-separable and explicit symplectic schemes (leapfrog/Verlet) do not apply. Implicit midpoint preserves the symplectic structure, so its energy error stays *bounded* over arbitrarily long undamped runs instead of drifting secularly |
| `rk4` | classical Runge–Kutta, fixed step | 4 | transparent baseline for the timestep-convergence study |

Practical guidance: use `dop853` for production runs and plots (tolerance-
controlled error), and `midpoint` for long-horizon studies of the undamped
system where structure preservation matters more than pointwise accuracy.
Chaos caveat: no integrator tracks an exact chaotic trajectory for long —
nearby solutions separate exponentially — but the symplectic method keeps the
*energy surface* (and the statistics that depend on it) right, which is the
meaningful notion of long-term correctness here.

The implicit midpoint equation `z = y_n + (h/2) f(t_mid, z)` is solved by
fixed-point iteration to near machine precision each step; a non-converging
iteration raises an error telling you to reduce `dt_internal` rather than
silently degrading.

**Energy accounting:** the cumulative dissipated energy `W(t)` is integrated
as an augmented ODE state (`dW/dt = P_dissipated`), so the energy-balance
residual `E(t) − E(0) + W(t)` — which is exactly zero for the true solution,
damped or not — measures pure numerical error. With the defaults it stays at
the 1e-9 relative level over 20 s of violent 3-link chaos.

## Project layout

```
src/npendulum/
  config.py       SimulationConfig: all physical + numerical parameters, validation,
                  scalar→per-link broadcasting, JSON round-trip
  dynamics.py     NPendulumDynamics: mass matrix, accelerations, energies,
                  damping forces, dissipated power, Cartesian positions
  symbolic.py     independent SymPy first-principles derivation (validation)
  integrators.py  DOP853 wrapper, implicit midpoint, RK4
  simulate.py     run_simulation(config) -> SimulationResult
  diagnostics.py  energy diagnostics, timestep-convergence study
  visualize.py    animation (any N) + angle/velocity/energy/error/phase-space plots
tests/            33 unit tests (physics vs SymPy & textbook forms, convergence
                  orders, energy conservation/dissipation, config validation,
                  reproducibility)
run.py            edit-and-run entry point (the settings block is the config)
examples/         convergence_study.py
docs/             DERIVATION.md — full Lagrangian derivation
output/           generated plots, animation, trajectory data
```

## Setup

Requires Python ≥ 3.9. A ready-to-use virtualenv is included at `.venv`; to
recreate it from scratch:

```bash
python3 -m venv .venv && .venv/bin/pip install -e . pytest
```

## Usage

### Quick start

Open [run.py](run.py), edit the settings block (number of links, masses,
lengths, damping, initial conditions, integrator, ...), then run it:

```bash
.venv/bin/python run.py
```

This writes the animation, all diagnostic plots, `config.json` and
`trajectory.npz` into `output/`. Per-link fields (`masses`, `lengths`,
`damping`, `initial_angles`, `initial_velocities`) accept either a single
number, broadcast to every link, or a tuple with one value per link, e.g.
`masses=(1.0, 0.8, 0.6, 0.4)` for a 4-link pendulum.

For a long-horizon symplectic run, set `integrator="midpoint"` and
`dt_internal=0.001`. For the timestep-convergence study:

```bash
.venv/bin/python examples/convergence_study.py
```

### Python API

```python
import numpy as np
from npendulum import SimulationConfig, run_simulation, energy_diagnostics
from npendulum.visualize import save_all

config = SimulationConfig(
    num_links=4,
    masses=(1.0, 0.8, 0.6, 0.4),      # or a scalar, broadcast to all links
    lengths=1.0,
    damping=0.02,                      # joint damping; damping=(0, .1, .1, .1)
    initial_angles=np.pi / 2,          # radians from downward vertical
    initial_velocities=0.0,
    t_final=30.0,
    dt_output=0.01,
    integrator="dop853",               # or "midpoint" / "rk4" (+ dt_internal)
)
result = run_simulation(config)        # deterministic & reproducible

print(energy_diagnostics(result))      # drift / dissipation summary
save_all(result, "output/")            # animation + all plots

result.theta, result.omega             # (T, N) trajectories
result.energy, result.energy_error     # (T,) diagnostics
```

Reproducibility: a run is a pure function of its `SimulationConfig`;
`config.to_json()` / `SimulationConfig.from_json()` round-trip the exact
configuration alongside the saved trajectory.

## Visualization

`npendulum.visualize` produces, for any N:

- `animation.gif` — the physical motion with mass-scaled bobs and a trail,
- `angles.svg`, `velocities.svg` — angular position / velocity vs time,
- `energy.svg` — kinetic, potential and total energy vs time,
- `energy_error.svg` — energy-balance residual (drift) vs time,
- `phase_space.svg` — (theta_i, omega_i) phase portraits per link,
- `convergence.svg` (from `examples/convergence_study.py`) — log-log error
  vs timestep with fitted orders (~4 for RK4, ~2 for implicit midpoint).

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

The suite validates, among other things:

- numeric accelerations ≡ SymPy first-principles Euler–Lagrange (N = 1–3,
  with damping) and ≡ the textbook double-pendulum closed form,
- mass-matrix symmetry/positive-definiteness; energies vs raw Cartesian sums,
- observed convergence orders (RK4 ≈ 4, midpoint ≈ 2) and cross-integrator
  agreement,
- undamped energy conservation (DOP853 tracks its tolerance; midpoint error
  bounded with no secular growth over 50 s),
- damped runs: energy decreases monotonically, matches the integrated joint
  dissipation exactly, and the pendulum settles to rest,
- config validation, broadcasting, JSON round-trip, bitwise reproducibility,
  and an N = 10 smoke test against hard-coding.
