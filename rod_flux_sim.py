"""Case 1 as specified by the supervisor: a 1D rod with a heat flux at the top and a fixed
temperature at the bottom (the rod of Ranjan et al. 2020, Sec. 3.2), with no scan speed.

    python rod_flux_sim.py      (quick check of a few runs)

Rod of length L = 10 mm on N = 101 nodes; x = 0 is the bottom, held at T0 = 65 C (Dirichlet, the
heat sink); x = L is the top, where a heat flux q (W/mm^2) enters during the heating time t_h and
stops afterwards (top then insulated). Conductivity k0 everywhere except a 1 mm patch of lower
conductivity f * k0 centred at height c (a 1D stand-in for powder or a void below the top).
Volumetric heat capacity rho*c fixed, so alpha = k / (rho c).

* transient: implicit Euler in time through heating (t_h) and 2 s of cooling; the maximum
  temperature each node ever reaches is recorded.
* steady  : the same rod with the flux on and dT/dt = 0 (t = infinity), solved once.

k0, c and f are hidden from the model; it sees the steady field, ln q and ln t_h.
"""
import numpy as np
from scipy.linalg import solve_banded

L, N = 10.0, 101
X = np.linspace(0.0, L, N)
DX = X[1] - X[0]
T0 = 65.0
RHO_C = 2.5e-3          # J / (mm^3 K), about Ti-6Al-4V
PATCH_W = 1.0           # mm
COOL_S = 2.0
RANGES = {"q": (0.05, 1.0), "t_h": (0.1, 2.0), "k0": (0.005, 0.03), "c": (0.5, 9.5), "f": (0.05, 0.5)}
LOG = {"q", "t_h", "k0", "f"}


def conductivity(k0, c, f):
    k = np.full(N, k0)
    k[np.abs(X - c) <= PATCH_W / 2] *= f
    return k


def _system(k):
    """Conductance matrix on nodes 1..N-1 (node 0 is Dirichlet), banded, and node volumes."""
    kf = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:])      # harmonic mean on the N-1 faces
    g = kf / DX                                       # conductance per unit area
    n = N - 1
    ab = np.zeros((3, n))
    diag = np.zeros(n)
    diag += g[:]                                      # face below each unknown node
    diag[:-1] += g[1:]                                # face above (none above the top node)
    ab[1] = diag
    ab[0, 1:] = -g[1:]
    ab[2, :-1] = -g[1:]
    vol = np.full(n, DX); vol[-1] = DX / 2
    return ab, vol


def steady(q, k0, c, f):
    ab, _ = _system(conductivity(k0, c, f))
    b = np.zeros(N - 1); b[-1] = q
    u = np.zeros(N); u[1:] = solve_banded((1, 1), ab, b)
    return T0 + u


def transient_max(q, t_h, k0, c, f, steps=200):
    ab, vol = _system(conductivity(k0, c, f))
    cap = RHO_C * vol
    u = np.zeros(N - 1); best = np.zeros(N - 1)
    for dt, n, flux in ((t_h / steps, steps, q), (COOL_S / steps, steps, 0.0)):
        A = ab * dt; A[1] += cap
        b_src = np.zeros(N - 1); b_src[-1] = flux * dt
        for _ in range(n):
            u = solve_banded((1, 1), A, cap * u + b_src)
            best = np.maximum(best, u)
    return T0 + np.concatenate([[0.0], best])


def sample_params(n, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for k, (a, b) in RANGES.items():
        out[k] = np.exp(rng.uniform(np.log(a), np.log(b), n)) if k in LOG else rng.uniform(a, b, n)
    return out


def generate(n, seed=0):
    p = sample_params(n, seed)
    ss = np.stack([steady(p["q"][i], p["k0"][i], p["c"][i], p["f"][i]) for i in range(n)])
    tr = np.stack([transient_max(p["q"][i], p["t_h"][i], p["k0"][i], p["c"][i], p["f"][i]) for i in range(n)])
    return p, ss, tr


if __name__ == "__main__":
    import time
    t = time.time()
    p, ss, tr = generate(6, 1)
    for i in range(6):
        lc = np.sqrt(p["k0"][i] / RHO_C * p["t_h"][i])
        print(f"q={p['q'][i]:.3f} t_h={p['t_h'][i]:.2f} k0={p['k0'][i]:.4f} patch c={p['c'][i]:.1f} f={p['f'][i]:.2f} Lc={lc:.2f} | "
              f"steady top {ss[i, -1]:8.1f}  transient max {tr[i].max():7.1f}  ratio {(tr[i].max() - T0) / (ss[i, -1] - T0):.3f}")
    print(f"{time.time() - t:.2f} s")
