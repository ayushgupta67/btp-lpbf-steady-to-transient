"""Variant: all physical inputs as numbers (P, v, conductivity alpha, cooling beta, spot radius r).

    python -u rod1d_allparams.py      (after rod1d_experiment.py; reuses its data)

Two models, N = 100 / 400 / 1,600 training runs, tested on the same 300 unseen runs:
  * steady field + all 5 numbers
  * all 5 numbers only (no steady field)
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch

import rod1d_sim as sim
from maxmap_model import Config, Model
from rod1d_experiment import BATCH, OUT, UPDATES, log_features, metrics

KEYS = ["P", "v", "alpha", "beta", "r"]
_D = {}


def _init(d):
    torch.set_num_threads(3)
    _D.update(d)


def _job(job):
    name, n, use_ss = job
    D = _D
    ss_tr = D["ss_train"][:n] if use_ss else np.full_like(D["ss_train"][:n], sim.T0)
    ss_te = D["ss_test"] if use_ss else np.full_like(D["ss_test"], sim.T0)
    act = np.arange(1, sim.N - 1)
    m = Model(Config(n_nodes=len(act), n_scalars=len(KEYS), hidden=256, depth=2, code=64), "cpu")
    m.scalar_fn = log_features
    t0 = time.time()
    m.fit(ss_tr, D["sc_train"][:n], D["tr_train"][:n], act, epochs=int(np.ceil(UPDATES / np.ceil(n / BATCH))), batch=BATCH, seed=0)
    pred, _ = m.predict(ss_te, D["sc_test"], seed=0)
    return {"name": name, "n_train": n, "steady_input": use_ss, "train_s": time.time() - t0, **metrics(pred, D["tr_test"])}


if __name__ == "__main__":
    z = np.load(OUT / "rod1d_data.npz")
    d = {"ss_train": z["ss_train"], "tr_train": z["tr_train"], "ss_test": z["ss_test"], "tr_test": z["tr_test"],
         "sc_train": np.column_stack([z[f"train_{k}"] for k in KEYS]), "sc_test": np.column_stack([z[f"test_{k}"] for k in KEYS])}
    jobs = [(nm, n, ss) for n in (100, 400, 1600) for nm, ss in (("all_params+ss", True), ("all_params_only", False))]
    res = []
    with ProcessPoolExecutor(4, mp_context=get_context("spawn"), initializer=_init, initargs=(d,)) as ex:
        for r in ex.map(_job, jobs):
            res.append(r)
            print(f"[{r['name']:<16} N={r['n_train']:5d}] RMSE {r['rmse_C']:7.2f} C  peak err mean {r['mean_abs_peak_err_pct']:5.2f}%  "
                  f"p95 {r['p95_abs_peak_err_pct']:5.2f}%", flush=True)
    (OUT / "rod1d_allparams.json").write_text(json.dumps(res, indent=1))
