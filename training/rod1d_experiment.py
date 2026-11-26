"""1D demonstration of the max steady -> max transient mapping, with enough data.

    python -u training/rod1d_experiment.py

1. Generate 1,600 training runs and 300 test runs (rod1d_sim.py), save them (npz + CSV).
2. Train the same diffusion model as the bead task (model/diffusion_model.py) on N = 25 .. 1,600 runs,
   always the same number of updates, and test on the 300 unseen runs: error vs amount of data.
3. Same, but with the steady field removed from the input (only P, v): does the steady input help?
4. The steady field used directly as the answer (the cheap model on its own): how big is the gap?
Outputs in results/case1_rod_moving_laser/.
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

import rod1d_sim as sim
from diffusion_model import Config, Model

OUT = REPO / "results" / "case1_rod_moving_laser"
UPDATES, BATCH = 8000, 32
SIZES = [25, 50, 100, 200, 400, 800, 1600]
NO_SS_SIZES = [100, 400, 1600]
_D = {}


def log_features(p):
    return np.log(np.asarray(p, dtype=np.float64))


def metrics(pred, true):
    d = pred - true
    rise = true - sim.T0
    pk = (pred.max(1) - true.max(1)) / (true.max(1) - sim.T0) * 100
    return {"rmse_C": float(np.sqrt((d ** 2).mean())), "mae_C": float(np.abs(d).mean()),
            "mean_abs_peak_err_pct": float(np.abs(pk).mean()), "median_abs_peak_err_pct": float(np.median(np.abs(pk))),
            "p95_abs_peak_err_pct": float(np.percentile(np.abs(pk), 95)),
            "rel_l2": float(np.mean(np.linalg.norm(d, axis=1) / np.linalg.norm(rise, axis=1)))}


def _init(data, threads):
    torch.set_num_threads(threads)
    _D.update(data)


def _job(job):
    name, n, use_ss, seed = job
    D = _D
    ss_tr = D["ss_train"][:n] if use_ss else np.full_like(D["ss_train"][:n], sim.T0)
    ss_te = D["ss_test"] if use_ss else np.full_like(D["ss_test"], sim.T0)
    act = np.arange(1, sim.N - 1)  # the two end nodes are fixed at 65 C
    m = Model(Config(n_nodes=len(act), n_scalars=2, hidden=256, depth=2, code=64), "cpu")
    m.scalar_fn = log_features
    epochs = int(np.ceil(UPDATES / np.ceil(n / BATCH)))
    t0 = time.time()
    curve = m.fit(ss_tr, D["sc_train"][:n], D["tr_train"][:n], act, epochs=epochs, batch=BATCH, seed=seed)
    tt = time.time() - t0
    pred, spread = m.predict(ss_te, D["sc_test"], ddim_steps=50, draws=8, seed=seed)
    res = {"name": name, "n_train": n, "steady_input": use_ss, "train_s": tt, "final_loss": curve[-1],
           "draw_spread_C": spread, **metrics(pred, D["tr_test"])}
    if name == "with_ss" and n == SIZES[-1]:
        m.save(OUT / "rod1d_diffusion.pt", {"n_train": n, "updates": UPDATES, "features": "ln P, ln v"})
        np.save(OUT / "pred_test_N1600.npy", pred)
    return res


def main():
    OUT.mkdir(exist_ok=True)
    T = time.time()
    p_tr, ss_tr, tr_tr = sim.generate(SIZES[-1], seed=1)
    p_te, ss_te, tr_te = sim.generate(300, seed=2)
    t_gen = time.time() - T
    keys = list(sim.RANGES)
    np.savez_compressed(OUT / "rod1d_data.npz", **{f"train_{k}": v for k, v in p_tr.items()}, **{f"test_{k}": v for k, v in p_te.items()},
                        ss_train=ss_tr, tr_train=tr_tr, ss_test=ss_te, tr_test=tr_te, x=sim.X)
    with open(OUT / "rod1d_runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "run"] + keys + ["steady_max_peak", "transient_max_peak"])
        for split, p, ss, tr in (("train", p_tr, ss_tr, tr_tr), ("test", p_te, ss_te, tr_te)):
            for i in range(len(ss)):
                w.writerow([split, i] + [f"{p[k][i]:.4f}" for k in keys] + [f"{ss[i].max():.2f}", f"{tr[i].max():.2f}"])
    print(f"generated {len(ss_tr)} + {len(ss_te)} runs in {t_gen:.1f} s")

    # steady field used directly as the answer
    gap = metrics(ss_te, tr_te)
    over = (ss_te.max(1) - tr_te.max(1)) / (tr_te.max(1) - sim.T0) * 100
    gap.update(mean_peak_overestimate_pct=float(over.mean()), max_peak_overestimate_pct=float(over.max()), min_peak_overestimate_pct=float(over.min()))
    print("steady field as the answer:", json.dumps(gap, indent=1))

    data = {"ss_train": ss_tr, "tr_train": tr_tr, "sc_train": np.column_stack([p_tr["P"], p_tr["v"]]),
            "ss_test": ss_te, "tr_test": tr_te, "sc_test": np.column_stack([p_te["P"], p_te["v"]])}
    jobs = [("with_ss", n, True, 0) for n in SIZES] + [("no_ss", n, False, 0) for n in NO_SS_SIZES]
    res = []
    with ProcessPoolExecutor(4, mp_context=get_context("spawn"), initializer=_init, initargs=(data, 3)) as ex:
        for r in ex.map(_job, jobs):
            res.append(r)
            print(f"[{r['name']:<8} N={r['n_train']:5d}] {r['train_s']:5.0f}s  RMSE {r['rmse_C']:7.2f} C  "
                  f"peak err mean {r['mean_abs_peak_err_pct']:5.2f}%  p95 {r['p95_abs_peak_err_pct']:5.2f}%  relL2 {r['rel_l2']:.3f}", flush=True)
    with open(OUT / "rod1d_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    (OUT / "rod1d_summary.json").write_text(json.dumps({"steady_as_answer": gap, "runs": res, "generate_s": t_gen,
                                                        "total_s": time.time() - T, "updates": UPDATES, "batch": BATCH}, indent=1))
    print(f"total {(time.time() - T) / 60:.1f} min")


if __name__ == "__main__":
    main()
