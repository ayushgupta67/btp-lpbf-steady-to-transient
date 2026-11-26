"""Case 3 (geometry dataset): learn steady peak -> transient peak around each element.

    python -u tts_model.py

Sample = one element. Input: 5 x 5 patch of steady peaks centred on it (same layer; voids and the outside
are empty) + the layer number. Output: the 5 x 5 patch of transient peaks; the score uses the centre value.
Same diffusion code as Cases 1 and 2 (maxmap_model.py).
Split: train on layers 1-8, test on layers 9-10, where the hole widens into a round hole the model never saw.
Baselines on the same test elements: training mean, and a linear fit on the centre steady peak + layer.
Temperatures are shifted/scaled before the model (T' = 65 + (T - 90,000) / 100) because the raw values are ~1e5 C;
empty cells are T' = 65. Every reported number is back in C.
Outputs: tts_outputs/model_results.json, tts_outputs/model_predictions.npz
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from maxmap_model import Config, Model

OUT = Path(__file__).parent / "tts_outputs"
R, N_TEST_FROM = 2, 9
UPDATES, BATCH = 8000, 32


def to_t(T):
    return 65.0 + (T - 90000.0) / 100.0


def from_t(t):
    return (t - 65.0) * 100.0 + 90000.0


def load():
    rows = list(csv.DictReader(open(OUT / "elements.csv")))
    L = np.array([int(r["layer"]) for r in rows]); ix = np.array([int(r["ix"]) for r in rows]); iy = np.array([int(r["iy"]) for r in rows])
    ss = np.array([float(r["ss_peak"]) for r in rows]); tr = np.array([float(r["tr_peak"]) for r in rows])
    G = {k: np.full((11, 21 + 2 * R + 1, 21 + 2 * R + 1), 65.0) for k in ("ss", "tr")}
    for k, v in (("ss", ss), ("tr", tr)):
        G[k][L, iy + R, ix + R] = to_t(v)
    patch = lambda k: np.stack([G[k][l, y:y + 2 * R + 1, x:x + 2 * R + 1].ravel() for l, x, y in zip(L, ix, iy)])
    empty = np.array([sum(G["ss"][l, y + R + dy, x + R + dx] == 65.0 for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))) for l, x, y in zip(L, ix, iy)])
    return L, ix, iy, ss, tr, patch("ss"), patch("tr"), empty


def scores(pred, true, empty):
    e = np.abs(pred - true)
    return {"mae_C": float(e.mean()), "mae_next_to_void_or_wall_C": float(e[empty > 0].mean()),
            "mae_in_bulk_C": float(e[empty == 0].mean()), "corr": float(np.corrcoef(pred, true)[0, 1])}


def main():
    torch.set_num_threads(8)
    T = time.time()
    L, ix, iy, ss, tr, Pss, Ptr, empty = load()
    te = L >= N_TEST_FROM; trn = ~te
    c = (2 * R + 1) ** 2 // 2
    res = {"n_train": int(trn.sum()), "n_test": int(te.sum()), "test_layers": "9-10",
           "test_tr_peak_std_C": float(tr[te].std()), "test_tr_peak_range_C": [float(tr[te].min()), float(tr[te].max())]}
    res["mean_of_training"] = scores(np.full(te.sum(), tr[trn].mean()), tr[te], empty[te]) | {"corr": 0.0}
    X = np.column_stack([np.ones(len(ss)), ss, L])
    w = np.linalg.lstsq(X[trn], tr[trn], rcond=None)[0]
    lin = X[te] @ w
    res["linear_centre_steady"] = scores(lin, tr[te], empty[te])
    m = Model(Config(n_nodes=Pss.shape[1], n_scalars=1, hidden=512, depth=2, code=64), "cpu")
    m.scalar_fn = lambda p: np.asarray(p, dtype=np.float64).reshape(-1, 1)
    epochs = int(np.ceil(UPDATES / np.ceil(trn.sum() / BATCH)))
    t0 = time.time()
    m.fit(Pss[trn], L[trn], Ptr[trn], np.arange(Pss.shape[1]), epochs=epochs, batch=BATCH, seed=0)
    res["train_s"] = time.time() - t0
    full, spread = m.predict(Pss[te], L[te], ddim_steps=50, draws=8, seed=0)
    dif = from_t(full[:, c])
    res["diffusion"] = scores(dif, tr[te], empty[te])
    res["parameters"] = m.n_params(); res["epochs"] = epochs; res["total_s"] = time.time() - T
    (OUT / "model_results.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT / "model_predictions.npz", layer=L[te], ix=ix[te], iy=iy[te], true=tr[te], diffusion=dif, linear=lin, steady=ss[te])
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
