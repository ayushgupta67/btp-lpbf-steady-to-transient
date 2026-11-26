"""1D demonstration data: a laser moving along a rod, max steady vs max transient temperature.

Same idea as the bead task, small enough to generate hundreds of runs in seconds.

Rod of length L (mm) on N nodes, both ends held at T0 = 65 C. A Gaussian heat source of power P
moves from x = 1 mm to x = 9 mm at speed v. The rod also loses heat to its surroundings
(a cooling rate beta, like convection from the sides). Heat equation, per unit time:

    dT/dt = alpha * d2T/dx2  -  beta * (T - T0)  +  source(x - x_laser)

* transient run: solved in time (implicit Euler), the laser moving at v; record the maximum
  temperature each node ever reaches (scan + 1 s of cooling).
* steady run (as in SS_new): the laser held still at each of 81 positions along the scan and the
  equation solved with dT/dt = 0; record the maximum over the positions at each node.
  The steady result does not depend on v (no time), exactly as in the bead data.

Material/cooling (alpha, beta) and spot radius r vary between runs but are NOT given to the model
as numbers: they are only visible through the steady field. P and v are given as numbers.
"""

import numpy as np
from scipy.linalg import solve_banded

L, N = 10.0, 101
X = np.linspace(0.0, L, N)
DX = X[1] - X[0]
T0 = 65.0
S_PER_W = 140.0  # source strength per watt (C mm / s per W); sets realistic peaks of 10^2..10^4 C
SCAN = (1.0, 9.0)
COOL_S = 1.0

RANGES = {"P": (20.0, 200.0), "v": (2.0, 50.0), "r": (0.15, 0.40), "alpha": (3.0, 12.0), "beta": (0.5, 5.0)}


def source(xc, P, r):
    g = np.exp(-0.5 * ((X - xc) / r) ** 2)
    g /= g.sum() * DX  # unit integral on the grid
    return S_PER_W * P * g


def _operator(alpha, beta):
    """Banded matrix of  -alpha d2/dx2 + beta  on interior nodes (Dirichlet ends)."""
    n = N - 2
    ab = np.zeros((3, n))
    ab[0, 1:] = -alpha / DX ** 2
    ab[1, :] = 2 * alpha / DX ** 2 + beta
    ab[2, :-1] = -alpha / DX ** 2
    return ab


def steady_max(P, r, alpha, beta, n_pos=81):
    ab = _operator(alpha, beta)
    best = np.full(N, -np.inf)
    for xc in np.linspace(*SCAN, n_pos):
        u = np.zeros(N)
        u[1:-1] = solve_banded((1, 1), ab, source(xc, P, r)[1:-1])
        best = np.maximum(best, u)
    return T0 + best


def transient_max(P, v, r, alpha, beta):
    dt = 0.5 * DX / v  # laser moves half a cell per step
    ab = _operator(alpha, beta) * dt
    ab[1, :] += 1.0  # (I + dt*A) u_new = u_old + dt*s
    u = np.zeros(N)
    best = np.zeros(N)
    t_scan = (SCAN[1] - SCAN[0]) / v
    n_scan = int(np.ceil(t_scan / dt))
    for k in range(1, n_scan + 1):
        xc = SCAN[0] + v * min(k * dt, t_scan)
        u[1:-1] = solve_banded((1, 1), ab, u[1:-1] + dt * source(xc, P, r)[1:-1])
        best = np.maximum(best, u)
    dtc = 0.01  # cooling: no source, larger steps
    abc = _operator(alpha, beta) * dtc
    abc[1, :] += 1.0
    for _ in range(int(COOL_S / dtc)):
        u[1:-1] = solve_banded((1, 1), abc, u[1:-1])
        best = np.maximum(best, u)
    return T0 + best


def sample_params(n, seed):
    rng = np.random.default_rng(seed)
    lo = {k: np.log(a) for k, (a, b) in RANGES.items()}
    hi = {k: np.log(b) for k, (a, b) in RANGES.items()}
    return {k: np.exp(rng.uniform(lo[k], hi[k], n)) for k in RANGES}  # log-uniform


def generate(n, seed=0):
    p = sample_params(n, seed)
    ss = np.stack([steady_max(p["P"][i], p["r"][i], p["alpha"][i], p["beta"][i]) for i in range(n)])
    tr = np.stack([transient_max(p["P"][i], p["v"][i], p["r"][i], p["alpha"][i], p["beta"][i]) for i in range(n)])
    return p, ss, tr


if __name__ == "__main__":
    import time
    t = time.time()
    p, ss, tr = generate(5, 1)
    for i in range(5):
        print(f"P={p['P'][i]:6.1f} v={p['v'][i]:5.1f} r={p['r'][i]:.2f} a={p['alpha'][i]:5.2f} b={p['beta'][i]:4.2f} | "
              f"steady max {ss[i].max():7.1f}  transient max {tr[i].max():7.1f}  ratio {(tr[i].max() - T0) / (ss[i].max() - T0):.3f}")
    print(f"{time.time() - t:.2f} s for 5 runs")
