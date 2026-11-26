"""Train the steady -> transient diffusion model and evaluate it honestly.

    python -u train_bead_diffusion.py            # ~25 min on CPU at 1500 epochs

1. Train on every usable (case, whole-number time) row and SAVE the model to
   bead_outputs/model/bead_diffusion.pt. Nothing needs retraining afterwards;
   predict_bead.py loads that file.
2. Leave-one-case-out evaluation: retrain on 7 cases, predict the 8th. This is
   the number to quote, because the saved model has seen all 8 cases and
   scoring it on them would only measure memorisation.
3. Report against three baselines that use no learning at all.

Air nodes (constant 65.0 in every frame of every case; 4072 of 5166) are removed
first, so the model and every metric below cover the 1094 active nodes only.

Only 8 cases are usable (see bead_data.py): the steady-state folders cover
P = 60 only, so power is constant in the training set.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import bead_model as bm
from bead_data import BASE_T, EXCLUDED, active_node_mask, build_pairs, load_cases

OUT = Path(__file__).parent / "bead_outputs"
SIGNAL_MIN = 0.5  # C; below this the target field is effectively all-ambient


def make_xy(pairs, rows):
    cond = bm.cond_vector(pairs["steady"][rows], pairs["params"][rows], pairs["t"][rows])
    return cond, bm.to_z(pairs["y"][rows]).astype(np.float32)


def z_clip_for(target_z):
    return (float(target_z.min()) - 0.2, float(target_z.max()) + 0.2)


def baselines(pairs, train, test):
    """Predictions from methods that learn nothing, for the test rows."""
    y, tt, par = pairs["y"], pairs["t_target"], pairs["params"]
    out = {"ambient (65 everywhere)": np.full((len(test), y.shape[1]), BASE_T)}
    out["mean of training cases"] = np.stack([y[train][tt[train] == tt[i]].mean(0) for i in test])
    near = []
    for i in test:
        cand = train[tt[train] == tt[i]]
        d = np.abs(par[cand][:, 1:] / bm.PARAM_SCALE[1:] - par[i][1:] / bm.PARAM_SCALE[1:]).sum(1)
        near.append(y[cand[d.argmin()]])
    out["nearest case in (v, hatch)"] = np.stack(near)
    return out


def row_metrics(pred, true):
    excess = np.linalg.norm(true - BASE_T)
    return {
        "MAE": float(np.abs(pred - true).mean()),
        "max_abs_err": float(np.abs(pred - true).max()),
        "peak_true": float(true.max()),
        "peak_pred": float(pred.max()),
        "rel_L2": float(np.linalg.norm(pred - true) / excess) if excess > SIGNAL_MIN else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-loo", action="store_true")
    args = ap.parse_args()

    model_dir = OUT / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    data = load_cases()
    pairs = build_pairs(data)
    mask = active_node_mask(data)
    n_nodes = int(mask.sum())
    pairs["steady"], pairs["y"] = pairs["steady"][:, mask], pairs["y"][:, mask]
    cases = sorted(set(pairs["case"]))
    case_of = np.array(pairs["case"])
    n = len(case_of)
    print(f"{n} rows from {len(cases)} cases: {cases}\nexcluded: {sorted(EXCLUDED)}")
    print(f"air nodes removed: {mask.size - n_nodes}; modelling {n_nodes} active nodes\n")

    # ---- 1. final model on everything, saved first ----
    torch.manual_seed(args.seed)
    cond, z = make_xy(pairs, np.arange(n))
    clip = z_clip_for(z)
    model = bm.BeadDiffusion(n_nodes)
    print(f"[final] training on all {n} rows, {args.epochs} epochs")
    t0 = time.time()
    curve = bm.train(model, cond, z, epochs=args.epochs, log=250)
    meta = {
        "trained_on_cases": cases, "rows": n, "epochs": args.epochs, "seed": args.seed,
        "z_clip": clip, "ddim_steps": args.ddim_steps, "final_loss": curve[-1],
        "param_scale": bm.PARAM_SCALE.tolist(), "asinh_knee": bm.ASINH_KNEE, "z_div": bm.Z_DIV,
        "t_offset": bm.T_OFFSET, "t_span": bm.T_SPAN, "steps": bm.STEPS,
        "n_nodes": n_nodes, "active_node_indices": np.flatnonzero(mask).tolist(),
        "power_values_seen": sorted(set(pairs["params"][:, 0].tolist())),
        "note": "steady-state fields exist for P=60 only; do not trust other powers",
    }
    bm.save_checkpoint(model_dir / "bead_diffusion.pt", model, meta)
    with open(model_dir / "training_loss.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "loss"])
        w.writerows((i + 1, v) for i, v in enumerate(curve))
    (model_dir / "training_meta.json").write_text(json.dumps(
        {k: v for k, v in meta.items() if k != "active_node_indices"}, indent=2))
    print(f"[final] done in {time.time() - t0:.0f}s -> {model_dir / 'bead_diffusion.pt'}\n")
    if args.skip_loo:
        return

    # ---- 2. leave-one-case-out ----
    results = []  # one dict per (case, t_target, method)
    for f, held in enumerate(cases):
        test = np.flatnonzero(case_of == held)
        train = np.flatnonzero(case_of != held)
        torch.manual_seed(args.seed + 1 + f)
        c_tr, z_tr = make_xy(pairs, train)
        c_te, _ = make_xy(pairs, test)
        m = bm.BeadDiffusion(n_nodes)
        t0 = time.time()
        bm.train(m, c_tr, z_tr, epochs=args.epochs)
        draws = bm.sample(m, c_te, z_clip=z_clip_for(z_tr), n_steps=args.ddim_steps,
                          n_draws=args.draws, seed=args.seed + f)
        preds = {"diffusion (mean of draws)": draws.mean(0)}
        preds.update(baselines(pairs, train, test))
        spread = float(draws.std(0).mean())
        for name, pred in preds.items():
            for j, i in enumerate(test):
                row = row_metrics(pred[j], pairs["y"][i])
                results.append({"case": held, "P": pairs["params"][i][0], "v": pairs["params"][i][1],
                                "hatch": pairs["params"][i][2], "t_target": int(pairs["t_target"][i]),
                                "t_actual": float(pairs["t"][i]), "method": name, **row})
        pk = [r for r in results if r["case"] == held and r["t_target"] == 2]
        d, nb = (next(r for r in pk if r["method"].startswith(k)) for k in ("diffusion", "nearest"))
        print(f"[LOO {f + 1}/{len(cases)}] held out {held:<16} {time.time() - t0:4.0f}s  "
              f"t=2 peak: true {d['peak_true']:7.1f}  diffusion {d['peak_pred']:7.1f}  "
              f"nearest-case {nb['peak_pred']:7.1f}   draw spread {spread:.2f} C")

    with open(OUT / "loo_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    summarise(results)
    plot(results)


def summarise(results):
    methods = list(dict.fromkeys(r["method"] for r in results))
    print("\nLEAVE-ONE-CASE-OUT, mean over the 8 held-out cases")
    print(f"{'method':<28}{'t=2 relL2':>10}{'t=2 peak err %':>16}{'t=3 relL2':>11}{'t>=4 max|err| C':>17}")
    for m in methods:
        rows = [r for r in results if r["method"] == m]

        def avg(t, key):
            v = [r[key] for r in rows if r["t_target"] == t and not np.isnan(r[key])]
            return float(np.mean(v)) if v else float("nan")
        pk = np.mean([abs(r["peak_pred"] - r["peak_true"]) / (r["peak_true"] - BASE_T) * 100
                      for r in rows if r["t_target"] == 2])
        late = np.mean([r["max_abs_err"] for r in rows if r["t_target"] >= 4])
        print(f"{m:<28}{avg(2, 'rel_L2'):>10.3f}{pk:>16.1f}{avg(3, 'rel_L2'):>11.3f}{late:>17.3f}")
    n3 = sum(1 for r in results if r["method"] == methods[0] and r["t_target"] == 3 and not np.isnan(r["rel_L2"]))
    print(f"(t=3 relL2 uses only the {n3} cases whose truth rises >{SIGNAL_MIN} C above ambient; "
          f"relL2 = 1.0 means no better than predicting ambient)")


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = sorted({r["case"] for r in results})
    pick = lambda m: [next(r for r in results if r["case"] == c and r["t_target"] == 2
                           and r["method"].startswith(m)) for c in cases]
    truth, diff, near = pick("diffusion"), pick("diffusion"), pick("nearest")
    x, w = np.arange(len(cases)), 0.27
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - w, [r["peak_true"] for r in truth], w, label="truth", color="0.3")
    ax.bar(x, [r["peak_pred"] for r in diff], w, label="diffusion (held-out case)", color="tab:red")
    ax.bar(x + w, [r["peak_pred"] for r in near], w, label="nearest case in (v, hatch)", color="0.75")
    ax.set_xticks(x, cases, rotation=35, ha="right", fontsize=8)
    ax.set(ylabel="peak temperature at t = 2 (C)", title="Leave-one-case-out: peak temperature at t = 2")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "loo_peak_t2.png", dpi=140)
    print(f"wrote {OUT / 'loo_peak_t2.png'}")


if __name__ == "__main__":
    main()
