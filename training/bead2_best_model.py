"""The best model found on the updated dataset: amplitude x shape, no neural network.

    python training/bead2_best_model.py --fit                     # fit on all usable runs, save to results/case2_snapshot_task/model/best_simple_model.npz
    python training/bead2_best_model.py --P 75 --v 50 --hatch 0.12 --t 2

field(t) = 65 + A(P/hatch, v, t) * shape(P/hatch, v, t)
* A: ln(peak rise) is quadratic in (ln q, ln v), q = P / hatch, one fit per whole-number time t in {2, 3}
* shape (peak = 1): linear in ln q, fitted separately for each speed v seen in training
* ambient (65 C everywhere) once the cooling window is over: t - (2 + 3.2/v) >= 1.0
Cross-validated RMSE on unseen settings: 0.22 C over all nodes (0.55 C at t = 2). Prediction for a speed
that was not in training uses the nearest trained speed's shape, and is less reliable.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / d) for d in ("model", "simulators", "data_loaders", "training")]

import argparse

import numpy as np

import bead2_data as bd
from bead2_hybrid import amp_fit, amp_predict, has_signal, shapes_of

OUT = REPO / "results" / "case2_snapshot_task" / "model"
B = bd.BASE_T


def fit():
    data = bd.load_cases()
    P = bd.build_pairs(data)
    mask = bd.active_node_mask(data)
    P["steady"], P["y"] = P["steady"][:, mask], P["y"][:, mask]
    rows = np.flatnonzero(has_signal(P["params"], P["t"]))
    fits = amp_fit(P, rows)
    shp, _ = shapes_of(P, rows)
    out = {"mask": mask}
    for t, f in fits.items():
        out[f"amp_{t}_keep"], out[f"amp_{t}_mu"], out[f"amp_{t}_sd"], out[f"amp_{t}_c"] = f["keep"], f["mu"], f["sd"], f["c"]
    for t in sorted(set(P["t_target"][rows])):
        for v in sorted(set(P["params"][rows][:, 1])):
            sel = (P["t_target"][rows] == t) & (P["params"][rows][:, 1] == v)
            if sel.sum() < 2:
                continue
            lq = np.log(P["params"][rows][sel][:, 0] / P["params"][rows][sel][:, 2])
            A = np.column_stack([np.ones(len(lq)), lq - lq.mean()])
            W = np.linalg.lstsq(A, shp[sel], rcond=None)[0]
            out[f"shape_{int(t)}_{int(v)}_W"], out[f"shape_{int(t)}_{int(v)}_lqmean"] = W, lq.mean()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "best_simple_model.npz", **out)
    print("saved", OUT / "best_simple_model.npz", f"({len(out)} arrays)")


def predict(Pw, v, hatch, t_actual, t_target=None):
    """Full 5,166-node temperature field (C) for one setting at one time."""
    m = np.load(OUT / "best_simple_model.npz")
    t_target = int(round(t_actual)) if t_target is None else t_target
    params = np.array([[Pw, v, hatch]])
    if not has_signal(params, np.array([t_actual]))[0]:
        return np.full(bd.N_NODES, B)
    f = {"keep": m[f"amp_{t_target}_keep"], "mu": m[f"amp_{t_target}_mu"], "sd": m[f"amp_{t_target}_sd"], "c": m[f"amp_{t_target}_c"]}
    amp = amp_predict({t_target: f}, params, np.array([t_target]))[0]
    vs = sorted(int(k.split("_")[2]) for k in m.files if k.startswith(f"shape_{t_target}_") and k.endswith("_W"))
    v_use = min(vs, key=lambda x: abs(np.log(x / v)))
    W, lqm = m[f"shape_{t_target}_{v_use}_W"], float(m[f"shape_{t_target}_{v_use}_lqmean"])
    s = np.clip(np.array([1.0, np.log(Pw / hatch) - lqm]) @ W, -0.2, 1.5)
    s = s / max(s.max(), 1e-6)
    full = np.full(bd.N_NODES, B)
    full[m["mask"]] = B + amp * s
    return full


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--P", type=float)
    ap.add_argument("--v", type=float)
    ap.add_argument("--hatch", type=float)
    ap.add_argument("--t", type=float, default=2.0)
    a = ap.parse_args()
    if a.fit:
        fit()
    else:
        f = predict(a.P, a.v, a.hatch, a.t)
        print(f"peak {f.max():.1f} C, mean {f.mean():.2f} C, hot nodes (>65.5): {int((f > 65.5).sum())}")
