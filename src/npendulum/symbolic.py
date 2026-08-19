"""Symbolic Euler-Lagrange derivation with SymPy (validation / documentation).

This module derives the equations of motion of the N-link pendulum *from
first principles* — Cartesian kinematics, kinetic and potential energy,
Lagrangian, Euler-Lagrange equations with Rayleigh damping — without using
the closed-form mass-matrix expressions of :mod:`npendulum.dynamics`.

It is intentionally independent of the numeric implementation so the test
suite can cross-validate the two.  It is practical for small N (symbolic
solving scales poorly); the runtime simulation never uses it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import sympy as sp


@lru_cache(maxsize=None)
def derive_acceleration_function(n: int) -> Callable:
    """Symbolically derive the angular accelerations of an n-link pendulum.

    Returns a function ``f(theta, omega, masses, lengths, gravity, damping)``
    (sequences of length ``n``, scalars for gravity) evaluating the angular
    accelerations, obtained by:

    1. writing bob positions from the link angles,
    2. forming T, V and the Lagrangian L = T - V,
    3. applying d/dt(dL/dw_i) - dL/dtheta_i = Q_i with the Rayleigh damping
       forces Q_i = -dR/dw_i,  R = 1/2 sum_i c_i (w_i - w_{i-1})^2,
    4. solving the resulting linear system for the second derivatives.
    """
    t = sp.Symbol("t")
    g = sp.Symbol("g", positive=True)
    m = sp.symbols(f"m1:{n + 1}", positive=True)
    l = sp.symbols(f"l1:{n + 1}", positive=True)
    c = sp.symbols(f"c1:{n + 1}", nonnegative=True)
    th = [sp.Function(f"theta{i + 1}")(t) for i in range(n)]
    w = [sp.diff(q, t) for q in th]
    wd = [sp.diff(q, t, 2) for q in th]

    # Cartesian kinematics (pivot at origin, y upward)
    x = [sum(l[j] * sp.sin(th[j]) for j in range(i + 1)) for i in range(n)]
    y = [-sum(l[j] * sp.cos(th[j]) for j in range(i + 1)) for i in range(n)]

    T = sum(
        sp.Rational(1, 2)
        * m[i]
        * (sp.diff(x[i], t) ** 2 + sp.diff(y[i], t) ** 2)
        for i in range(n)
    )
    V = sum(m[i] * g * y[i] for i in range(n))
    L = sp.expand(T - V)

    # Rayleigh dissipation: joint i connects link i to link i-1 (w_0 = 0)
    rel = [w[0]] + [w[i] - w[i - 1] for i in range(1, n)]
    R = sp.Rational(1, 2) * sum(c[i] * rel[i] ** 2 for i in range(n))
    Q = [-sp.diff(R, w[i]) for i in range(n)]

    # Euler-Lagrange residuals (== 0) with generalized forces; they are
    # linear in the second derivatives, so extract A @ wd = rhs and solve.
    residuals = [
        sp.expand(sp.diff(sp.diff(L, w[i]), t) - sp.diff(L, th[i]) - Q[i])
        for i in range(n)
    ]
    A, rhs = sp.linear_eq_to_matrix(residuals, wd)
    accels = list(A.LUsolve(rhs))

    # Replace time-dependent functions by plain symbols for lambdify.
    # Derivatives must be substituted before the functions they derive.
    th_s = sp.symbols(f"theta1:{n + 1}")
    w_s = sp.symbols(f"omega1:{n + 1}")
    accels = [
        a.subs(dict(zip(w, w_s))).subs(dict(zip(th, th_s))) for a in accels
    ]

    func = sp.lambdify((th_s, w_s, m, l, g, c), accels, modules="numpy")

    def acceleration(theta, omega, masses, lengths, gravity, damping):
        return func(tuple(theta), tuple(omega), tuple(masses),
                    tuple(lengths), gravity, tuple(damping))

    return acceleration
