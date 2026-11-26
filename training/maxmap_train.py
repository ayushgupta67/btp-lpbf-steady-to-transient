"""Train and evaluate the max-temperature diffusion model.

    python -u training/maxmap_train.py                    # final model + 24-fold cross-validation
    python -u training/maxmap_train.py --skip-cv          # final model only
    python -u training/maxmap_train.py --transient-root DIR --ss-root DIR   # any other / larger dataset

Input : per-node max of the steady-state run (SS_new) + P, v, hatch
Output: per-node max of the transient run
One pair per run. The final model (all pairs) is saved to results/case2_single_bead/model/maxmap_diffusion.pt.
Honest test: leave-one-group-out CV, a group = runs with the same (P/hatch, v), which are
physically identical, so a twin can never sit in training while the other is tested.
Errors in deg C over all 5,166 nodes and over the modelled (active) nodes; peak error in %.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / d) for d in ("model", "simulators", "data_loaders", "training")]

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch

import maxmap_data as md
from diffusion_model import BASE_T, Config, Model

_G = {}


def metrics(pred, true, n_full):
    d = pred - true
    rise_t, rise_p = true - BASE_T, pred - BASE_T
    hot_t = rise_t >= 0.5 * rise_t.max()  # hot-spot: at least half of the true peak rise
    hot_p = rise_p >= 0.5 * rise_t.max()
    jac = float((hot_t & hot_p).sum() / max((hot_t | hot_p).sum(), 1))
    return {"rmse_all": float(np.sqrt((d ** 2).sum() / n_full)), "mae_all": float(np.abs(d).sum() / n_full),
            "max_abs_err": float(np.abs(d).max()), "peak_true": float(true.max()), "peak_pred": float(pred.max()),
            "peak_err_pct": float((pred.max() - true.max()) / (true.max() - BASE_T) * 100), "hotspot_jaccard": jac,
            "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(rise_t))}


def active_nodes(pairs, tol=1e-3):
    a = np.concatenate([pairs["ss_max"], pairs["tr_max"]])
    return np.flatnonzero((a.max(0) - BASE_T > tol) | (a.min(0) < BASE_T - tol))


def _init(pairs, args, threads):
    torch.set_num_threads(threads)
    _G.update(pairs=pairs, args=args)


def _fold(job):
    f, grp, test, train = job
    P, a = _G["pairs"], _G["args"]
    m = Model(Config(n_nodes=len(active_nodes({k: v[train] for k, v in P.items() if k in ("ss_max", "tr_max")})), n_scalars=4), "cpu")
    act = active_nodes({k: v[train] for k, v in P.items() if k in ("ss_max", "tr_max")})
    t0 = time.time()
    curve = m.fit(P["ss_max"][train], P["params"][train], P["tr_max"][train], act, epochs=a["epochs"], batch=a["batch"], seed=a["seed"] + 1 + f)
    tt = time.time() - t0
    pred, spread = m.predict(P["ss_max"][test], P["params"][test], ddim_steps=a["ddim"], draws=a["draws"], seed=a["seed"] + f)
    rows = [{"held_out": f"q={grp[0]:g}_v={grp[1]:g}", "run": P["name"][i], "P": float(P["params"][i][0]), "v": float(P["params"][i][1]),
             "hatch": float(P["params"][i][2]), **metrics(pred[j], P["tr_max"][i], P["tr_max"].shape[1])} for j, i in enumerate(test)]
    return f, rows, pred, test, {"fold": f, "train_s": tt, "final_loss": curve[-1], "draw_spread_C": spread}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transient-root", default=str(md.TRANSIENT_ROOT))
    ap.add_argument("--ss-root", default=str(md.SS_ROOT))
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ddim", type=int, default=50)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-cv", action="store_true")
    a = ap.parse_args()
    out = md.OUT
    (out / "model").mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    tr = md.load_runs(a.transient_root, "transient")
    st = md.load_runs(a.ss_root, "steady")
    pairs, skipped = md.build_pairs(tr, st)
    n = len(pairs["name"])
    groups = [md.group_of(p) for p in pairs["params"]]
    act = active_nodes(pairs)
    print(f"{n} pairs ({len(set(groups))} distinct (P/hatch, v) groups); skipped {skipped}")
    print(f"modelled nodes {len(act)} of {pairs['tr_max'].shape[1]}; device {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

    # ---- final model on all pairs ---------------------------------------------------------
    torch.set_num_threads(8)
    m = Model(Config(n_nodes=len(act), n_scalars=4))
    print(f"[final] {m.n_params():,} parameters, {a.epochs} epochs, batch {a.batch} ({-(-n // a.batch)} updates/epoch)")
    t0 = time.time()
    curve = m.fit(pairs["ss_max"], pairs["params"], pairs["tr_max"], act, epochs=a.epochs, batch=a.batch, seed=a.seed, log_every=500)
    t_final = time.time() - t0
    pred_in, _ = m.predict(pairs["ss_max"], pairs["params"], ddim_steps=a.ddim, draws=a.draws, seed=a.seed)
    ins = float(np.mean([metrics(p, t, t.size)["rmse_all"] for p, t in zip(pred_in, pairs["tr_max"])]))
    meta = {"runs": pairs["name"], "epochs": a.epochs, "batch": a.batch, "train_seconds": t_final, "final_loss": curve[-1],
            "in_sample_rmse_all": ins, "ddim": a.ddim, "draws": a.draws, "task": "max SS field + (P, v, hatch) -> max transient field"}
    m.save(out / "model" / "maxmap_diffusion.pt", meta)
    with open(out / "model" / "training_loss.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "loss"])
        w.writerows((i + 1, v) for i, v in enumerate(curve))
    print(f"[final] {t_final:.0f} s; loss {curve[0]:.4f} -> {curve[-1]:.6f}; in-sample RMSE {ins:.2f} C (seen data); saved\n")
    if a.skip_cv:
        return

    # ---- cross-validation -------------------------------------------------------------------
    jobs = []
    for f, g in enumerate(sorted(set(groups))):
        test = np.array([i for i, x in enumerate(groups) if x == g])
        jobs.append((f, g, test, np.array([i for i, x in enumerate(groups) if x != g])))
    print(f"[cv] {len(jobs)} folds, {a.workers} workers x {a.threads} threads")
    res, folds, held = [], [], np.zeros_like(pairs["tr_max"])
    t_cv = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(pairs, vars(a), a.threads)) as ex:
        for f, rows, pred, test, info in ex.map(_fold, jobs):
            res += rows
            folds.append(info)
            held[test] = pred
            print(f"[cv {f + 1:2d}/{len(jobs)}] {jobs[f][1]}  {info['train_s']:.0f}s  peak err "
                  + ", ".join(f"{r['peak_err_pct']:+.1f}%" for r in rows), flush=True)
    t_cv = time.time() - t_cv
    with open(out / "cv_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    np.savez_compressed(out / "cv_predictions.npz", pred=held, true=pairs["tr_max"], ss=pairs["ss_max"],
                        params=pairs["params"], name=np.array(pairs["name"]))
    k = lambda key: np.array([r[key] for r in res])
    summ = {"pairs": n, "folds": len(jobs), "rmse_all_C": float(np.sqrt(np.mean(k("rmse_all") ** 2))),
            "mae_all_C": float(np.mean(k("mae_all"))), "mean_abs_peak_err_pct": float(np.mean(np.abs(k("peak_err_pct")))),
            "median_abs_peak_err_pct": float(np.median(np.abs(k("peak_err_pct")))), "max_abs_peak_err_pct": float(np.max(np.abs(k("peak_err_pct")))),
            "mean_hotspot_jaccard": float(np.mean(k("hotspot_jaccard"))), "mean_rel_l2": float(np.mean(k("rel_l2"))),
            "in_sample_rmse_all_C": ins, "final_train_s": t_final, "cv_wall_s": t_cv,
            "cv_fold_train_s_mean": float(np.mean([x["train_s"] for x in folds])), "total_s": time.time() - T0,
            "parameters": m.n_params(), "epochs": a.epochs, "batch": a.batch, "cpu_threads": os.cpu_count()}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
