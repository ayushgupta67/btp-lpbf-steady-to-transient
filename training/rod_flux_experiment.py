"""Case 1 on the supervisor's rod (rod_flux_sim.py): max steady field + ln q, ln t_h -> max transient field.

    python -u training/rod_flux_experiment.py      -> results/case1_rod/

1. 1,600 training and 300 unseen test runs.
2. Diffusion model (model/diffusion_model.py) with the steady field; the same without it (q, t_h only);
   and with every physical input given as a number (q, t_h, k0, c, f), with and without the steady field.
3. The steady field used directly as the answer.
4. Local domain in 1D: how much the hidden low-conductivity patch changes the steady and the transient
   peak, against the patch depth below the top measured in characteristic lengths Lc = sqrt(alpha t_h).
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / d) for d in ("model", "simulators", "data_loaders", "training")]

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch

import rod_flux_sim as sim
from diffusion_model import Config, Model

OUT = REPO / "results" / "case1_rod"
UPDATES, BATCH = 8000, 32
_D = {}


def metrics(pred, true):
    d = pred - true
    pk = (pred.max(1) - true.max(1)) / (true.max(1) - sim.T0) * 100
    return {"rmse_C": float(np.sqrt((d ** 2).mean())), "mae_C": float(np.abs(d).mean()),
            "mean_abs_peak_err_pct": float(np.abs(pk).mean()), "median_abs_peak_err_pct": float(np.median(np.abs(pk))),
            "p95_abs_peak_err_pct": float(np.percentile(np.abs(pk), 95))}


def _init(d):
    torch.set_num_threads(3)
    _D.update(d)


def _job(job):
    name, keys, use_ss, n = job
    D = _D
    sc_tr = np.column_stack([D["p_train"][k][:n] for k in keys]); sc_te = np.column_stack([D["p_test"][k] for k in keys])
    ss_tr = D["ss_train"][:n] if use_ss else np.full_like(D["ss_train"][:n], sim.T0)
    ss_te = D["ss_test"] if use_ss else np.full_like(D["ss_test"], sim.T0)
    act = np.arange(1, sim.N)          # node 0 is the fixed bottom
    m = Model(Config(n_nodes=len(act), n_scalars=len(keys), hidden=256, depth=2, code=64), "cpu")
    logk = [k in sim.LOG for k in keys]
    m.scalar_fn = lambda p: np.column_stack([np.log(p[:, j]) if lg else p[:, j] for j, lg in enumerate(logk)])
    t0 = time.time()
    m.fit(ss_tr, sc_tr, D["tr_train"][:n], act, epochs=int(np.ceil(UPDATES / np.ceil(n / BATCH))), batch=BATCH, seed=0)
    pred, spread = m.predict(ss_te, sc_te, ddim_steps=50, draws=8, seed=0)
    res = {"name": name, "n_train": n, "steady_input": use_ss, "scalars": keys, "params": m.n_params(),
           "train_s": time.time() - t0, "draw_spread_C": spread, **metrics(pred, D["tr_test"])}
    if name == "with_ss" and n == 1600:
        np.save(OUT / "pred_test.npy", pred)
    return res


def patch_effect(p, n=300):
    """Peak change caused by the patch (patch on vs off), steady and transient, against depth / Lc."""
    rows = []
    for i in range(n):
        q, th, k0, c, f = (p[k][i] for k in ("q", "t_h", "k0", "c", "f"))
        lc = np.sqrt(k0 / sim.RHO_C * th)
        s_on, s_off = sim.steady(q, k0, c, f)[-1], sim.steady(q, k0, c, 1.0)[-1]
        t_on, t_off = sim.transient_max(q, th, k0, c, f).max(), sim.transient_max(q, th, k0, c, 1.0).max()
        rows.append({"depth_over_Lc": (sim.L - c) / lc, "steady_change_pct": (s_on - s_off) / (s_off - sim.T0) * 100,
                     "transient_change_pct": (t_on - t_off) / (t_off - sim.T0) * 100})
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    T = time.time()
    p_tr, ss_tr, tr_tr = sim.generate(1600, seed=1)
    p_te, ss_te, tr_te = sim.generate(300, seed=2)
    t_gen = time.time() - T
    np.savez_compressed(OUT / "rodflux_data.npz", **{f"train_{k}": v for k, v in p_tr.items()}, **{f"test_{k}": v for k, v in p_te.items()},
                        ss_train=ss_tr, tr_train=tr_tr, ss_test=ss_te, tr_test=tr_te, x=sim.X)
    gap = metrics(ss_te, tr_te)
    over = (ss_te.max(1) - tr_te.max(1)) / (tr_te.max(1) - sim.T0) * 100
    gap.update(mean_peak_overestimate_pct=float(over.mean()), median_peak_overestimate_pct=float(np.median(over)), max_peak_overestimate_pct=float(over.max()))
    print(f"generated 1600 + 300 runs in {t_gen:.1f} s; steady as the answer:", json.dumps(gap), flush=True)
    data = {"p_train": p_tr, "p_test": p_te, "ss_train": ss_tr, "tr_train": tr_tr, "ss_test": ss_te, "tr_test": tr_te}
    base, allk = ["q", "t_h"], ["q", "t_h", "k0", "c", "f"]
    jobs = [("with_ss", base, True, 1600), ("no_ss", base, False, 1600), ("all_params+ss", allk, True, 1600),
            ("all_params_only", allk, False, 1600), ("with_ss", base, True, 200), ("with_ss", base, True, 50)]
    res = []
    with ProcessPoolExecutor(6, mp_context=get_context("spawn"), initializer=_init, initargs=(data,)) as ex:
        for r in ex.map(_job, jobs):
            res.append(r)
            print(f"[{r['name']:<16} N={r['n_train']:5d}] {r['train_s']:4.0f}s  RMSE {r['rmse_C']:6.2f} C  "
                  f"peak err mean {r['mean_abs_peak_err_pct']:5.2f}%  median {r['median_abs_peak_err_pct']:5.2f}%  p95 {r['p95_abs_peak_err_pct']:5.2f}%", flush=True)
    pe = patch_effect(p_te)
    with open(OUT / "patch_effect.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pe[0])); w.writeheader(); w.writerows(pe)
    d = np.array([r["depth_over_Lc"] for r in pe]); sc = np.array([r["steady_change_pct"] for r in pe]); tc = np.array([r["transient_change_pct"] for r in pe])
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 100)):
        m = (d >= lo) & (d < hi)
        print(f"patch depth {lo}-{hi} Lc: n={m.sum():3d}  steady peak +{np.median(sc[m]):6.1f}%  transient peak +{np.median(tc[m]):6.1f}% (medians)")
    (OUT / "summary.json").write_text(json.dumps({"steady_as_answer": gap, "runs": res, "generate_s": t_gen,
                                                   "total_s": time.time() - T, "updates": UPDATES, "batch": BATCH}, indent=1))
    print(f"total {(time.time() - T) / 60:.1f} min")


if __name__ == "__main__":
    main()
