"""Validate the vectorized dynamics against independent derivations."""

import numpy as np
import pytest

from npendulum.dynamics import NPendulumDynamics
from npendulum.symbolic import derive_acceleration_function


def test_single_pendulum_closed_form():
    """N=1 must reduce to thetadd = -(g/l) sin(theta) - c w / (m l^2)."""
    m, l, g, c = 2.0, 1.5, 9.81, 0.3
    dyn = NPendulumDynamics([m], [l], g, [c])
    theta, omega = 0.7, -1.2
    acc = dyn.accelerations(np.array([theta]), np.array([omega]))[0]
    expected = -(g / l) * np.sin(theta) - c * omega / (m * l * l)
    assert acc == pytest.approx(expected, rel=1e-12)


def test_double_pendulum_textbook_form():
    """N=2 (undamped) must match the standard double-pendulum equations."""
    m1, m2, l1, l2, g = 1.3, 0.7, 1.1, 0.9, 9.81
    th1, th2, w1, w2 = 1.2, -0.5, 0.4, 2.0
    delta = th1 - th2
    den = m1 + m2 * np.sin(delta) ** 2
    a1 = (
        -(m1 + m2) * g * np.sin(th1)
        - m2 * l2 * w2**2 * np.sin(delta)
        - m2 * np.cos(delta) * (l1 * w1**2 * np.sin(delta) - g * np.sin(th2))
    ) / (l1 * den)
    a2 = (
        (m1 + m2)
        * (l1 * w1**2 * np.sin(delta) - g * np.sin(th2)
           + g * np.sin(th1) * np.cos(delta))
        + m2 * l2 * w2**2 * np.sin(delta) * np.cos(delta)
    ) / (l2 * den)

    dyn = NPendulumDynamics([m1, m2], [l1, l2], g, [0.0, 0.0])
    acc = dyn.accelerations(np.array([th1, th2]), np.array([w1, w2]))
    assert acc == pytest.approx([a1, a2], rel=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_matches_sympy_euler_lagrange(n):
    """The closed form must match the symbolic first-principles derivation."""
    rng = np.random.default_rng(42 + n)
    symbolic = derive_acceleration_function(n)
    m = rng.uniform(0.5, 2.0, n)
    l = rng.uniform(0.5, 2.0, n)
    c = rng.uniform(0.0, 0.5, n)
    g = 9.81
    dyn = NPendulumDynamics(m, l, g, c)
    for _ in range(5):
        theta = rng.uniform(-np.pi, np.pi, n)
        omega = rng.uniform(-3.0, 3.0, n)
        numeric = dyn.accelerations(theta, omega)
        expected = np.array(symbolic(theta, omega, m, l, g, c))
        assert numeric == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_mass_matrix_symmetric_positive_definite():
    rng = np.random.default_rng(7)
    n = 6
    dyn = NPendulumDynamics(rng.uniform(0.5, 2, n), rng.uniform(0.5, 2, n),
                            9.81, np.zeros(n))
    for _ in range(5):
        M = dyn.mass_matrix(rng.uniform(-np.pi, np.pi, n))
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(M) > 0)


def test_energies_match_cartesian_definitions():
    """T and V from the mass matrix must equal sum of per-bob Cartesian terms."""
    rng = np.random.default_rng(3)
    n = 5
    m = rng.uniform(0.5, 2, n)
    l = rng.uniform(0.5, 2, n)
    g = 9.81
    dyn = NPendulumDynamics(m, l, g, np.zeros(n))
    theta = rng.uniform(-np.pi, np.pi, n)
    omega = rng.uniform(-3, 3, n)

    # Cartesian velocities: xd_i = sum_{j<=i} l_j cos(th_j) w_j, etc.
    xd = np.cumsum(l * np.cos(theta) * omega)
    yd = np.cumsum(l * np.sin(theta) * omega)
    T_cart = 0.5 * np.sum(m * (xd**2 + yd**2))
    y = -np.cumsum(l * np.cos(theta))
    V_cart = np.sum(m * g * y)

    assert dyn.kinetic_energy(theta, omega) == pytest.approx(T_cart, rel=1e-12)
    assert dyn.potential_energy(theta) == pytest.approx(V_cart, rel=1e-12)


def test_damping_forces_dissipate_energy():
    """Power of the damping forces must equal -sum c_i (w_i - w_{i-1})^2."""
    rng = np.random.default_rng(11)
    n = 4
    c = rng.uniform(0.1, 1.0, n)
    dyn = NPendulumDynamics(np.ones(n), np.ones(n), 9.81, c)
    omega = rng.uniform(-3, 3, n)
    power_from_forces = float(dyn.damping_forces(omega) @ omega)
    assert power_from_forces == pytest.approx(-dyn.dissipated_power(omega),
                                              rel=1e-12)
    assert power_from_forces <= 0.0


def test_small_oscillation_frequency_single_pendulum():
    """Linearized single pendulum: period 2 pi sqrt(l/g)."""
    from npendulum.config import SimulationConfig
    from npendulum.simulate import run_simulation

    l, g = 1.0, 9.81
    config = SimulationConfig(
        num_links=1, lengths=l, gravity=g, initial_angles=0.01,
        t_final=2 * np.pi * np.sqrt(l / g), dt_output=0.001,
    )
    result = run_simulation(config)
    # After exactly one linear period the angle returns to its start
    assert result.theta[-1, 0] == pytest.approx(0.01, rel=1e-3)
