"""The same steady -> transient diffusion pipeline on the previous (wall) dataset.

    python -u train_wall_diffusion.py

Purpose: show the idea end to end on the first dataset. The wall data is ONE
build, so it has no (P, v, hatch) variation to learn from; the model gets the
steady field and the time only (the three parameter slots are held at 0).
Reuses bead_model.py unchanged, so the architecture, the temperature transform
and the sampler are identical to the single-bead experiment.

Split: the last 25% of the build is held out (whole unit-time blocks; a random
split leaks, see baselines.py). Scores are mean absolute error in C over the
1094 active cells, directly comparable with the baselines.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

import bead_model as bm
from baselines import fit_predictors
from wall_data import WallData, metrics

ROOT = Path(__file__).parent
OUT = ROOT / "wall_outputs"
T_START, T_SPAN = 2.1, 80.0  # wall times run 2.1 .. 82.0


def cond_for(steady, t):
    """steady (n, n_active) in C, t (n,) -> (n, n_active + 4); P, v, hatch slots are 0."""
    scalars = np.column_stack([np.zeros((len(t), 3)), (np.asarray(t) - T_START) / T_SPAN])
    return np.concatenate([bm.to_z(steady), scalars], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-only", action="store_true",
                    help="reload the saved model instead of training (redo scores and figure)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(args.seed)

    d = WallData(root=str(ROOT))
    steady, transient = d.Y, d.X  # steady is the input here, transient the target
    tr, te = d.split("extrapolate")
    n = d.n_active
    print(f"wall data: {len(d.t)} frames, {n} active cells; train {tr.sum()} / test {te.sum()} frames "
          f"(held out: time blocks {d.block[te].min()}..{d.block[te].max()})")

    c_tr, c_te = cond_for(steady[tr], d.t[tr]), cond_for(steady[te], d.t[te])
    z_tr = bm.to_z(transient[tr]).astype(np.float32)
    clip = (float(z_tr.min()) - 0.2, float(z_tr.max()) + 0.2)

    if args.plot_only:
        model, meta = bm.load_checkpoint(OUT / "wall_diffusion.pt")
        clip = tuple(meta["z_clip"])
        print("loaded", OUT / "wall_diffusion.pt")
    else:
        model = bm.BeadDiffusion(n)
        t0 = time.time()
        curve = bm.train(model, c_tr, z_tr, epochs=args.epochs, log=100)
        print(f"trained {args.epochs} epochs in {time.time() - t0:.0f}s, final loss {curve[-1]:.2e}")
        bm.save_checkpoint(OUT / "wall_diffusion.pt", model, {
            "epochs": args.epochs, "seed": args.seed, "z_clip": clip, "rows": int(tr.sum()),
            "ddim_steps": 50, "active_cell_mask": d.mask, "t_start": T_START, "t_span": T_SPAN,
            "split": "extrapolate (last 25% of time blocks held out)"})

    draws = bm.sample(model, c_te, z_clip=clip, n_draws=args.draws, seed=args.seed)
    pred = draws.mean(0)
    # In this direction the baseline called "identity" copies the steady field.
    results = {name.replace("identity (copy transient)", "copy the steady field"):
               metrics(fn(steady[te]), transient[te])["MAE"]
               for name, fn in fit_predictors(steady[tr], transient[tr]).items()}
    results["diffusion (mean of draws)"] = metrics(pred, transient[te])["MAE"]
    prev = transient[np.maximum(np.flatnonzero(te) - 1, 0)]
    oracle = float(np.abs(prev - transient[te]).mean())
    spread = float(draws.std(0).mean())

    print(f"\nsteady -> transient, held-out last 25% of the build, MAE in C (lower is better)")
    for name, mae in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"   {name:<34}{mae:8.4f}")
    print(f"   {'[oracle] previous frame':<34}{oracle:8.4f}   (needs the answer for the neighbouring frame)")
    best_base = min(v for k, v in results.items() if not k.startswith("diffusion")
                    and k != "copy the steady field")
    floor = results["per-cell mean"]
    print(f"\ndiffusion draw spread {spread:.3f} C;  no-skill (per-cell mean) {floor:.3f}; "
          f"best baseline {best_base:.3f} ({(floor - best_base) / floor * 100:.1f}% below no-skill)")

    with open(OUT / "wall_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "MAE_C"])
        w.writerows(sorted(results.items(), key=lambda kv: kv[1]))
    plot(d, te, pred, steady[tr], transient[tr], results)


def plot(d, te, pred, steady_tr, transient_tr, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(pred) // 2  # a frame in the middle of the held-out part
    cmean = transient_tr.mean(0)
    truth = d.X[te][k]

    def to_grid(vec):
        grid = np.full(d.mask.shape, np.nan)
        grid[d.mask] = vec
        return grid[:, 10, :].T  # slice through the wall: rows = z, cols = x

    panels = [("truth", truth), ("diffusion", pred[k]), ("no-skill (per-cell mean)", cmean)]
    lo, hi = min(p[1].min() for p in panels), max(p[1].max() for p in panels)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8), layout="constrained",
                           gridspec_kw={"width_ratios": [1, 1, 1, 1.6]})
    for a, (title, vec) in zip(ax, panels):
        im = a.imshow(to_grid(vec), origin="lower", cmap="inferno", vmin=lo, vmax=hi, aspect="auto")
        a.set(title=f"{title}\nMAE {np.abs(vec - truth).mean():.2f} C", xlabel="x (build direction)", ylabel="z")
    fig.colorbar(im, ax=ax[:3], label="temperature (C)", shrink=0.9, pad=0.02)
    names = sorted(results, key=results.get)
    ax[3].barh(range(len(names)), [results[n] for n in names],
               color=["tab:red" if "diffusion" in n else "0.7" for n in names])
    ax[3].set_yticks(range(len(names)), names, fontsize=7)
    ax[3].invert_yaxis()
    ax[3].set(xlabel="test MAE (C)", title="all methods, last 25% of build held out")
    ax[3].grid(alpha=0.3, axis="x")
    fig.suptitle(f"Wall data, steady -> transient. Slice through the wall (y=10), test frame at t={d.t[te][k]:.1f}")
    fig.savefig(OUT / "wall_demo.png", dpi=140)
    print("wrote", OUT / "wall_demo.png")


if __name__ == "__main__":
    main()
