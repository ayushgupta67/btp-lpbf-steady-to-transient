"""Train the steady -> transient diffusion model on the UPDATED single-bead dataset and
evaluate it honestly.

    python -u training/bead2_train.py                 # whole job, about 15-25 min on this laptop CPU
    python -u training/bead2_train.py --skip-cv       # only the final model (about 2 min)

Task: input  = steady-state field (1,094 active nodes) + P, v, hatch + time t
      output = transient temperature field (1,094 active nodes) at t = 2, 3, 4, 5, 6, 7
Only whole-number times are used (nearest stored frame); the decimal frames inside the scan
carry the solver's numerical undershoot (see bead2_analyze.py).

1. Final model trained on every usable (run, time) row, saved to results/case2_snapshot_task/model/.
2. Leave-one-setting-out cross-validation: runs with the same (P/hatch, v) are bit-identical
   (e.g. P60/h0.10 and P90/h0.15), so they are held out together; training on one and testing
   on its twin would only measure memorisation. 24 settings -> 24 retrainings, run in parallel.
3. Four baselines that use no deep learning, scored on the same held-out rows.

Errors are RMSE in degrees C, over all 5,166 nodes and over the 1,094 active nodes
(the 4,072 air nodes are exactly 65 C in truth and in every prediction).
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

import bead2_data as bd
import bead_model as bm

OUT = REPO / "results" / "case2_snapshot_task"
BASE_T = bd.BASE_T
N_ALL = bd.N_NODES
_G = {}  # per-process copy of the arrays, set by _init


def z_clip_for(z):
    return (float(z.min()) - 0.2, float(z.max()) + 0.2)


def make_xy(pairs, rows):
    cond = bm.cond_vector(pairs["steady"][rows], pairs["params"][rows], pairs["t"][rows])
    return cond, bm.to_z(pairs["y"][rows]).astype(np.float32)


def group_key(params):
    P, v, h = params
    return (round(P / h, 1), float(v))


# ---- baselines ----------------------------------------------------------------------------
def baselines(pairs, train, test):
    y, tt, par = pairs["y"], pairs["t_target"], pairs["params"]
    out = {"ambient (65 everywhere)": np.full((len(test), y.shape[1]), BASE_T)}
    out["mean of training runs"] = np.stack([y[train][tt[train] == tt[i]].mean(0) for i in test])

    def feats(p):
        q, v = p[0] / p[2], p[1]
        return np.array([1.0, np.log(q), np.log(v)])

    near, ridge, ampshape = [], [], []
    for i in test:
        cand = train[tt[train] == tt[i]]
        x = feats(par[i])[1:]
        d = np.array([np.abs(feats(par[j])[1:] - x).sum() for j in cand])
        near.append(y[cand[d.argmin()]])
        # ridge on [1, ln q, ln v] -> node temperature, one model per target time
        X = np.stack([feats(par[j]) for j in cand])
        mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-9
        Xs = np.column_stack([np.ones(len(X)), (X[:, 1:] - mu) / sd])
        W = np.linalg.solve(Xs.T @ Xs + 1e-2 * np.diag([0, 1, 1]), Xs.T @ (y[cand] - BASE_T))
        ridge.append(BASE_T + np.concatenate([[1.0], (x - mu) / sd]) @ W)
        # amplitude x shape: ln(peak rise) linear in ln q, ln v; shape = mean normalised field
        rise = y[cand] - BASE_T
        amp = rise.max(1)
        if (amp > 0.5).all():
            c, *_ = np.linalg.lstsq(Xs, np.log(amp), rcond=None)
            a_new = np.exp(np.concatenate([[1.0], (x - mu) / sd]) @ c)
            ampshape.append(BASE_T + a_new * (rise / amp[:, None]).mean(0))
        else:  # t >= 3 for fast runs: the truth is ambient, so no shape to scale
            ampshape.append(BASE_T + rise.mean(0))
    out["nearest setting (ln q, ln v)"] = np.stack(near)
    out["ridge (ln q, ln v)"] = np.stack(ridge)
    out["amplitude x shape"] = np.stack(ampshape)
    return out


def rmse_pair(pred, true):
    d = pred - true
    act = float(np.sqrt((d ** 2).mean()))
    return {"rmse_active": act, "rmse_all5166": act * np.sqrt(d.shape[0] / N_ALL),
            "mae_active": float(np.abs(d).mean()), "max_abs_err": float(np.abs(d).max()),
            "peak_true": float(true.max()), "peak_pred": float(pred.max()),
            "rms_rise_true": float(np.sqrt(((true - BASE_T) ** 2).mean()))}


# ---- one fold (runs in a worker process) --------------------------------------------------
def _init(pairs, epochs, draws, ddim, seed, threads):
    torch.set_num_threads(threads)
    _G.update(pairs=pairs, epochs=epochs, draws=draws, ddim=ddim, seed=seed)


def _fold(args):
    f, held_key, test, train = args
    P = _G["pairs"]
    torch.manual_seed(_G["seed"] + 1 + f)
    c_tr, z_tr = make_xy(P, train)
    c_te, _ = make_xy(P, test)
    m = bm.BeadDiffusion(P["y"].shape[1])
    t0 = time.time()
    curve = bm.train(m, c_tr, z_tr, epochs=_G["epochs"])
    t_train = time.time() - t0
    t0 = time.time()
    draws = bm.sample(m, c_te, z_clip=z_clip_for(z_tr), n_steps=_G["ddim"], n_draws=_G["draws"], seed=_G["seed"] + f)
    t_sample = time.time() - t0
    preds = {"diffusion (mean of draws)": draws.mean(0)}
    preds.update(baselines(P, train, test))
    rows = []
    for name, pred in preds.items():
        for j, i in enumerate(test):
            rows.append({"held_out_setting": f"q={held_key[0]}_v={held_key[1]:g}", "run": P["case"][i],
                         "P": P["params"][i][0], "v": P["params"][i][1], "hatch": P["params"][i][2],
                         "t_target": int(P["t_target"][i]), "t_actual": float(P["t"][i]), "method": name,
                         **rmse_pair(pred[j], P["y"][i])})
    return f, rows, {"fold": f, "setting": held_key, "train_s": t_train, "sample_s": t_sample,
                     "final_loss": curve[-1], "draw_spread_C": float(draws.std(0).mean())}, draws.mean(0), test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2, help="torch threads per CV worker")
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()

    model_dir = OUT / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    data = bd.load_cases()
    pairs = bd.build_pairs(data)
    mask = bd.active_node_mask(data)
    n_nodes = int(mask.sum())
    pairs["steady"], pairs["y"] = pairs["steady"][:, mask], pairs["y"][:, mask]
    runs = sorted(set(pairs["case"]))
    n = len(pairs["case"])
    keys = [group_key(p) for p in pairs["params"]]
    settings = sorted(set(keys))
    print(f"{n} rows from {len(runs)} runs ({len(settings)} distinct (P/hatch, v) settings); excluded: {sorted(bd.EXCLUDED)}")
    print(f"active nodes {n_nodes}; cpu threads {os.cpu_count()}; torch {torch.__version__}\n")

    # ---- 1. final model on everything, saved first --------------------------------------
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    cond, z = make_xy(pairs, np.arange(n))
    clip = z_clip_for(z)
    model = bm.BeadDiffusion(n_nodes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[final] training on all {n} rows, {args.epochs} epochs, {n_params:,} parameters")
    t0 = time.time()
    curve = bm.train(model, cond, z, epochs=args.epochs, log=250)
    t_final = time.time() - t0
    meta = {
        "trained_on_runs": runs, "rows": n, "epochs": args.epochs, "updates": args.epochs * (-(-n // 16)),
        "batch": 16, "seed": args.seed, "z_clip": clip, "ddim_steps": args.ddim_steps, "draws": args.draws,
        "final_loss": curve[-1], "param_scale": bm.PARAM_SCALE.tolist(), "asinh_knee": bm.ASINH_KNEE,
        "z_div": bm.Z_DIV, "t_offset": bm.T_OFFSET, "t_span": bm.T_SPAN, "steps": bm.STEPS, "n_nodes": n_nodes,
        "n_parameters": n_params, "train_seconds": t_final, "device": "cpu",
        "active_node_indices": np.flatnonzero(mask).tolist(),
        "note": "inputs: steady field of the run's (P, hatch) + P, v, hatch + frame time; outputs: transient field",
    }
    bm.save_checkpoint(model_dir / "bead2_diffusion.pt", model, meta)
    with open(model_dir / "training_loss.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "loss"])
        w.writerows((i + 1, v) for i, v in enumerate(curve))
    (model_dir / "training_meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "active_node_indices"}, indent=2))
    print(f"[final] {t_final:.0f}s ({t_final / 60:.1f} min); loss {curve[0]:.4f} -> {curve[-1]:.6f}; saved {model_dir / 'bead2_diffusion.pt'}")

    # in-sample error of the saved model: memorisation, NOT generalisation
    draws = bm.sample(model, cond, z_clip=clip, n_steps=args.ddim_steps, n_draws=args.draws, seed=args.seed)
    pred = draws.mean(0)
    ins = np.sqrt(((pred - pairs["y"]) ** 2).mean())
    print(f"[final] in-sample RMSE (active nodes) {ins:.2f} C (seen data, not a test)\n")
    if args.skip_cv:
        return

    # ---- 2. leave-one-setting-out cross-validation, in parallel ---------------------------
    jobs = []
    for f, key in enumerate(settings):
        test = np.array([i for i, k in enumerate(keys) if k == key])
        train = np.array([i for i, k in enumerate(keys) if k != key])
        jobs.append((f, key, test, train))
    print(f"[cv] {len(jobs)} folds on {args.workers} workers x {args.threads} threads")
    t_cv = time.time()
    results, fold_info, held_preds = [], [], {}
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(pairs, args.epochs, args.draws, args.ddim_steps, args.seed, args.threads)) as ex:
        for f, rows, info, p_mean, test in ex.map(_fold, jobs):
            results += rows
            fold_info.append(info)
            held_preds[f] = (test, p_mean)
            print(f"[cv {f + 1:2d}/{len(jobs)}] held out q={info['setting'][0]:g} v={info['setting'][1]:g}: "
                  f"train {info['train_s']:.0f}s, loss {info['final_loss']:.4f}", flush=True)
    t_cv = time.time() - t_cv
    print(f"[cv] wall time {t_cv / 60:.1f} min")

    with open(OUT / "cv_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    with open(OUT / "cv_folds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fold_info[0]))
        w.writeheader()
        w.writerows(fold_info)
    pred_all = np.zeros_like(pairs["y"])
    for f, (test, p_mean) in held_preds.items():
        pred_all[test] = p_mean
    np.savez_compressed(OUT / "cv_predictions.npz", pred=pred_all, true=pairs["y"], t_target=pairs["t_target"],
                        params=pairs["params"], run=np.array(pairs["case"]), steady=pairs["steady"])
    timing = {"final_train_s": t_final, "cv_wall_s": t_cv, "cv_fold_train_s_mean": float(np.mean([i["train_s"] for i in fold_info])),
              "cv_fold_sample_s_mean": float(np.mean([i["sample_s"] for i in fold_info])), "total_wall_s": time.time() - t_all,
              "in_sample_rmse_active": float(ins), "epochs": args.epochs, "rows": n, "params": n_params,
              "cpu": "12th Gen Intel i5-12500H (16 logical cores), no GPU"}
    (OUT / "timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2))
    summarise(results)


def summarise(results):
    methods = list(dict.fromkeys(r["method"] for r in results))
    ts = sorted({r["t_target"] for r in results})

    def agg(rows, key):
        return float(np.sqrt(np.mean([r[key] ** 2 for r in rows])))
    print("\nCROSS-VALIDATED RMSE (deg C), all 5,166 nodes; rows are held-out runs")
    print(f"{'method':<30}" + "".join(f"{'t=' + str(t):>9}" for t in ts) + f"{'ALL':>9}")
    for m in methods:
        rows = [r for r in results if r["method"] == m]
        print(f"{m:<30}" + "".join(f"{agg([r for r in rows if r['t_target'] == t], 'rmse_all5166'):>9.2f}" for t in ts)
              + f"{agg(rows, 'rmse_all5166'):>9.2f}")
    print("\nsame, active nodes only (1,094)")
    for m in methods:
        rows = [r for r in results if r["method"] == m]
        print(f"{m:<30}" + "".join(f"{agg([r for r in rows if r['t_target'] == t], 'rmse_active'):>9.2f}" for t in ts)
              + f"{agg(rows, 'rmse_active'):>9.2f}")


if __name__ == "__main__":
    main()
