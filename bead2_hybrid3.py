"""Hybrid v3: quadratic amplitude x (regression shape + diffusion RESIDUAL).

    python -u bead2_hybrid3.py

The diffusion model no longer has to generate the whole shape from 35 rows. The shape is first
predicted by a simple regression (linear in ln q within the same speed and time), and the
diffusion model, conditioned on the steady field and (P, v, hatch, t), generates only the
correction to it. Training residuals are leave-one-setting-out inside the training set, so the
model sees the size of correction it will really face on a new setting.
Same 24 leave-one-setting-out folds as bead2_train.py and bead2_hybrid.py.
"""

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

import bead2_data as bd
import bead_model as bm
from bead2_hybrid import amp_fit, amp_predict, has_signal, shapes_of
from bead2_train import group_key, make_xy, rmse_pair, z_clip_for

OUT = Path(__file__).parent / "bead2_outputs"
BASE_T = bd.BASE_T
RES_SCALE = 1000.0  # residual shape (about +-0.01) -> +-10 C so the asinh code sees it
_G = {}


def base_shape(P, shp_all, pool, i):
    """Regression shape for row i from the rows `pool` (indices into the signal-row arrays)."""
    sel = pool[(P["t_target"][pool] == P["t_target"][i]) & (P["params"][pool][:, 1] == P["params"][i][1])]
    if len(sel) >= 2:
        lq = np.log(P["params"][sel][:, 0] / P["params"][sel][:, 2])
        A = np.column_stack([np.ones(len(lq)), lq - lq.mean()])
        W = np.linalg.lstsq(A, shp_all[sel], rcond=None)[0]
        return np.array([1.0, np.log(P["params"][i][0] / P["params"][i][2]) - lq.mean()]) @ W
    sel = pool[P["t_target"][pool] == P["t_target"][i]]
    return shp_all[sel].mean(0)


def fit_predict(P, train, test, epochs, draws, ddim, seed):
    sig_tr = train[has_signal(P["params"][train], P["t"][train])]
    fits = amp_fit(P, sig_tr)
    shp, _ = shapes_of(P, sig_tr)
    shp_all = np.zeros_like(P["y"])
    shp_all[sig_tr] = shp
    keys = [group_key(p) for p in P["params"]]
    res = np.zeros_like(shp)
    for n, i in enumerate(sig_tr):
        pool = np.array([j for j in sig_tr if keys[j] != keys[i]])
        res[n] = shp[n] - base_shape(P, shp_all, pool, i)
    cond = bm.cond_vector(P["steady"][sig_tr], P["params"][sig_tr], P["t"][sig_tr])
    z = bm.to_z(BASE_T + RES_SCALE * res).astype(np.float32)
    torch.manual_seed(seed)
    m = bm.BeadDiffusion(P["y"].shape[1])
    curve = bm.train(m, cond, z, epochs=epochs)
    pred = np.full((len(test), P["y"].shape[1]), BASE_T)
    amp = amp_predict(fits, P["params"][test], P["t_target"][test])
    sig_te = has_signal(P["params"][test], P["t"][test])
    if sig_te.any():
        tt = test[sig_te]
        c_te = bm.cond_vector(P["steady"][tt], P["params"][tt], P["t"][tt])
        d = bm.sample(m, c_te, z_clip=z_clip_for(z), n_steps=ddim, n_draws=draws, seed=seed).mean(0)
        rr = (d - BASE_T) / RES_SCALE
        for k, i in enumerate(tt):
            s = np.clip(base_shape(P, shp_all, sig_tr, i) + rr[k], -0.2, 1.5)
            s = s / max(s.max(), 1e-6)
            pred[np.flatnonzero(sig_te)[k]] = BASE_T + amp[np.flatnonzero(sig_te)[k]] * s
    return pred, fits, m, z, curve


def _init(pairs, epochs, draws, ddim, seed, threads):
    torch.set_num_threads(threads)
    _G.update(pairs=pairs, epochs=epochs, draws=draws, ddim=ddim, seed=seed)


def _fold(args):
    f, key, test, train = args
    P = _G["pairs"]
    t0 = time.time()
    pred, *_ = fit_predict(P, train, test, _G["epochs"], _G["draws"], _G["ddim"], _G["seed"] + 1 + f)
    rows = [{"held_out_setting": f"q={key[0]}_v={key[1]:g}", "run": P["case"][i], "P": P["params"][i][0], "v": P["params"][i][1],
             "hatch": P["params"][i][2], "t_target": int(P["t_target"][i]), "t_actual": float(P["t"][i]),
             "method": "hybrid v3: quadratic amplitude x (regression shape + diffusion residual)", **rmse_pair(pred[j], P["y"][i])}
            for j, i in enumerate(test)]
    return f, rows, time.time() - t0, pred, test


def main(epochs=1500, draws=8, ddim=50, seed=0, workers=3, threads=4):
    t_all = time.time()
    data = bd.load_cases()
    pairs = bd.build_pairs(data)
    mask = bd.active_node_mask(data)
    pairs["steady"], pairs["y"] = pairs["steady"][:, mask], pairs["y"][:, mask]
    keys = [group_key(p) for p in pairs["params"]]
    settings = sorted(set(keys))
    jobs = [(f, k, np.array([i for i, kk in enumerate(keys) if kk == k]), np.array([i for i, kk in enumerate(keys) if kk != k]))
            for f, k in enumerate(settings)]
    results, held = [], np.zeros_like(pairs["y"])
    t_cv = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(pairs, epochs, draws, ddim, seed, threads)) as ex:
        for f, rows, secs, pred, test in ex.map(_fold, jobs):
            results += rows
            held[test] = pred
            print(f"[v3 cv {f + 1:2d}/{len(jobs)}] {jobs[f][1]} {secs:.0f}s", flush=True)
    t_cv = time.time() - t_cv
    with open(OUT / "cv_hybrid_v3_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    np.savez_compressed(OUT / "cv_hybrid_v3_predictions.npz", pred=held, true=pairs["y"], t_target=pairs["t_target"],
                        params=pairs["params"], run=np.array(pairs["case"]))
    print(f"[v3 cv] wall {t_cv / 60:.1f} min; total {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
