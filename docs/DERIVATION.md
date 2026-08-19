# Lagrangian Derivation of the N-Link Pendulum Equations of Motion

This document derives the equations of motion implemented in
[`src/npendulum/dynamics.py`](../src/npendulum/dynamics.py) and states exactly
how each symbol maps onto the code. An independent, fully symbolic version of
the same derivation (SymPy, first principles, no closed forms) lives in
[`src/npendulum/symbolic.py`](../src/npendulum/symbolic.py); the test suite
verifies that both agree to machine precision for N = 1, 2, 3.

## 1. System and generalized coordinates

The system is a chain of $N$ point masses $m_1, \dots, m_N$ connected by rigid
massless rods of lengths $l_1, \dots, l_N$, suspended from a fixed frictionless
pivot at the origin. Gravity $g$ acts downward. Model assumptions:

- rods are rigid and massless (all inertia is in the point-mass bobs),
- motion is planar,
- the only non-conservative forces are viscous torques at the joints.

As generalized coordinates we choose the **absolute angles** $\theta_i$, the
angle of link $i$ measured from the *downward* vertical ($\theta = 0$ means
hanging straight down; positive is counterclockwise). Absolute angles give a
mass matrix that depends only on differences $\theta_i - \theta_j$, which is
what makes a clean closed form possible for arbitrary $N$.

The Cartesian position of bob $i$ (with $y$ pointing up) is

$$x_i = \sum_{j \le i} l_j \sin\theta_j, \qquad
  y_i = -\sum_{j \le i} l_j \cos\theta_j .$$

## 2. Kinetic and potential energy

Differentiating the positions,

$$\dot x_i = \sum_{j\le i} l_j \cos\theta_j\,\dot\theta_j,\qquad
  \dot y_i = \sum_{j\le i} l_j \sin\theta_j\,\dot\theta_j .$$

The kinetic energy is

$$T = \tfrac12 \sum_{i=1}^{N} m_i\left(\dot x_i^2 + \dot y_i^2\right)
    = \tfrac12 \sum_{i=1}^{N} m_i \sum_{j\le i}\sum_{k\le i}
      l_j l_k \cos(\theta_j - \theta_k)\,\dot\theta_j\dot\theta_k ,$$

using $\cos\theta_j\cos\theta_k + \sin\theta_j\sin\theta_k =
\cos(\theta_j-\theta_k)$. Swapping the summation order (summing over which
bobs $i$ contain the pair $(j,k)$, i.e. $i \ge \max(j,k)$) and defining the
**tail-mass sums**

$$\mu_j = \sum_{k \ge j} m_k, \qquad a_{jk} = \mu_{\max(j,k)},$$

we obtain the compact quadratic form

$$\boxed{\,T = \tfrac12\sum_{j,k} a_{jk}\, l_j l_k \cos(\theta_j-\theta_k)\,
  \dot\theta_j \dot\theta_k \;=\; \tfrac12\,\dot{\boldsymbol\theta}^{\!\top}
  M(\boldsymbol\theta)\,\dot{\boldsymbol\theta}\,},
  \qquad M_{jk} = a_{jk}\, l_j l_k \cos(\theta_j - \theta_k).$$

$M(\boldsymbol\theta)$ is the generalized mass matrix: symmetric and positive
definite (it is a sum over bobs of rank-≤2 positive semidefinite Cartesian
contributions whose total is nondegenerate for distinct links).

The potential energy is

$$V = \sum_i m_i g\, y_i = -g \sum_i m_i \sum_{j \le i} l_j \cos\theta_j
    = \boxed{-g\sum_j \mu_j\, l_j \cos\theta_j\,}$$

again by swapping the summation order (zero level at the pivot).

## 3. Damping (Rayleigh dissipation)

Joint $i$ connects link $i$ to link $i-1$ (joint 1 connects link 1 to the
fixed support, whose angular velocity is zero). Each joint exerts a viscous
torque proportional to the **relative angular velocity** of the links it
connects, described by the Rayleigh dissipation function

$$\mathcal R = \tfrac12 \sum_{i=1}^{N} c_i\,
   (\dot\theta_i - \dot\theta_{i-1})^2, \qquad \dot\theta_0 \equiv 0,$$

with the generalized forces

$$Q_i = -\frac{\partial \mathcal R}{\partial \dot\theta_i}
      = -c_i(\dot\theta_i - \dot\theta_{i-1})
        + c_{i+1}(\dot\theta_{i+1} - \dot\theta_i)$$

(the $c_{N+1}$ term is absent for $i = N$). Setting $c_1 = 0$ removes the
damping of the first link against the support, leaving only inter-link
damping. The mechanical energy then obeys the exact balance

$$\frac{dE}{dt} = \sum_i Q_i \dot\theta_i = -2\mathcal R
   = -\sum_i c_i (\dot\theta_i - \dot\theta_{i-1})^2 \le 0 .$$

## 4. Euler–Lagrange equations

With $L = T - V$, the equations of motion are

$$\frac{d}{dt}\frac{\partial L}{\partial \dot\theta_i}
  - \frac{\partial L}{\partial \theta_i} = Q_i .$$

Working through the three terms with $M_{ij} = a_{ij} l_i l_j
\cos(\theta_i - \theta_j)$:

$$\frac{\partial L}{\partial \dot\theta_i} = \sum_j M_{ij}\dot\theta_j,$$

$$\frac{d}{dt}\frac{\partial L}{\partial \dot\theta_i}
  = \sum_j M_{ij}\ddot\theta_j
  - \sum_j a_{ij} l_i l_j \sin(\theta_i-\theta_j)
    (\dot\theta_i - \dot\theta_j)\dot\theta_j,$$

$$\frac{\partial L}{\partial \theta_i}
  = -\sum_j a_{ij} l_i l_j \sin(\theta_i - \theta_j)\,
     \dot\theta_i \dot\theta_j
    - g \mu_i l_i \sin\theta_i .$$

The $\dot\theta_i\dot\theta_j$ cross terms cancel between the last two
expressions, leaving the remarkably clean linear system for the accelerations:

$$\boxed{\;\sum_j M_{ij}(\boldsymbol\theta)\,\ddot\theta_j
  = -\sum_j a_{ij}\, l_i l_j \sin(\theta_i - \theta_j)\,\dot\theta_j^{\,2}
    - g\,\mu_i l_i \sin\theta_i + Q_i \;}$$

Sanity checks:

- **N = 1**: $m l^2 \ddot\theta = -m g l\sin\theta - c\dot\theta$, the damped
  pendulum equation.
- **N = 2**: eliminating $\mu_1 = m_1 + m_2$, $\mu_2 = m_2$ reproduces the
  standard textbook double-pendulum equations (checked numerically in
  `tests/test_dynamics.py::test_double_pendulum_textbook_form`).

## 5. Mapping to the implementation

| Mathematics | Code (`NPendulumDynamics`) |
|---|---|
| $\mu_i = \sum_{k\ge i} m_k$ | `self.mu = np.cumsum(self.m[::-1])[::-1]` |
| $a_{ij} = \mu_{\max(i,j)}$ | `self.a = self.mu[np.maximum.outer(idx, idx)]` |
| $a_{ij} l_i l_j$ | `self._A = self.a * np.outer(self.l, self.l)` (precomputed) |
| $M_{ij}$ | `mass_matrix()`: `self._A * np.cos(diff)` with `diff[i,j] = theta_i - theta_j` |
| centrifugal term $\sum_j a_{ij} l_i l_j \sin(\theta_i-\theta_j)\dot\theta_j^2$ | `S @ omega**2` with `S = self._A * np.sin(diff)` |
| gravity term $g\mu_i l_i \sin\theta_i$ | `self.g * self.mu * self.l * np.sin(theta)` |
| $Q_i$ | `damping_forces()` |
| solve $M\ddot{\boldsymbol\theta} = \mathbf b$ | `cho_solve(cho_factor(M), b)` (Cholesky — M is SPD) |
| $T$, $V$ | `kinetic_energy()`, `potential_energy()` |
| $2\mathcal R$ | `dissipated_power()` |

Everything is expressed with $N\times N$ vectorized NumPy arrays, so the same
code handles any $N$; per-evaluation cost is $O(N^2)$ to build the system and
$O(N^3)$ for the Cholesky solve.

## 6. First-order form and the dissipation quadrature

For the integrators the second-order system is rewritten as
$\dot y = f(t, y)$ with $y = (\boldsymbol\theta, \boldsymbol\omega)$ and
$f = (\boldsymbol\omega, M^{-1}\mathbf b)$. The simulation actually integrates
the *augmented* state $(\boldsymbol\theta, \boldsymbol\omega, W)$ with
$\dot W = 2\mathcal R$, so the cumulative dissipated energy $W(t)$ carries the
same numerical accuracy as the trajectory. The energy-balance residual

$$\varepsilon(t) = E(t) - E(0) + W(t)$$

is identically zero for the exact solution — damped or not — so $|\varepsilon|$
is a direct measure of numerical error (reported by
`SimulationResult.energy_error` and `diagnostics.energy_diagnostics`).
