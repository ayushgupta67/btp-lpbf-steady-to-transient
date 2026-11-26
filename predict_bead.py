"""Predict a transient temperature field with the saved diffusion model (no retraining).

    python predict_bead.py --P 60 --v 50 --hatch 0.15 --t 2

The steady-state field is looked up from Steady_state/ by (P, hatch) -- the
model's conditioning input -- so only combinations with a steady-state folder
work (currently P = 60 with hatch 0.10 / 0.15 / 0.18). If a transient run with
the same (P, v, hatch) exists it is used as ground truth for a comparison.

Writes bead_outputs/predictions/pred_P<P>_v<v>_h<hatch>_t<t>.csv for all 5166
nodes with columns node, active, T_pred, T_pred_std (spread over draws) and, if
available, T_true. Air nodes (active = 0) are exactly 65.0 by construction; the
model only predicts the active ones, and the error figures printed here cover
the active nodes only.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import bead_model as bm
from bead_data import BASE_T, EXCLUDED, expand, load_cases, steady_lookup

OUT = Path(__file__).parent / "bead_outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=float, required=True)
    ap.add_argument("--v", type=float, required=True)
    ap.add_argument("--hatch", type=float, required=True)
    ap.add_argument("--t", type=float, required=True, help="time instant, 2 .. 7")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--checkpoint", default=str(OUT / "model" / "bead_diffusion.pt"))
    args = ap.parse_args()

    model, meta = bm.load_checkpoint(args.checkpoint)
    data = load_cases()
    steady = steady_lookup(data["steady"])
    key = (args.P, args.hatch)
    if key not in steady:
        have = sorted(steady)
        raise SystemExit(f"no steady-state field for P={args.P}, hatch={args.hatch}; "
                         f"available (P, hatch): {have}")
    if args.P not in meta["power_values_seen"]:
        print(f"warning: the model was trained at P in {meta['power_values_seen']} only")

    mask = np.zeros(len(steady[key]), bool)
    mask[meta["active_node_indices"]] = True
    cond = bm.cond_vector(steady[key][mask][None], [[args.P, args.v, args.hatch]], [args.t])
    draws = bm.sample(model, cond, z_clip=tuple(meta["z_clip"]), n_steps=meta["ddim_steps"],
                      n_draws=args.draws)[:, 0]
    pred_a, spread_a = draws.mean(0), draws.std(0)
    pred = expand(pred_a, mask)
    spread = np.zeros(len(mask))
    spread[mask] = spread_a
    print(f"predicted peak {pred.max():.1f} C   (mean draw spread {spread_a.mean():.3f} C "
          f"over {mask.sum()} active nodes)")

    truth = None
    for c in data["transient"]:
        if (c.P, c.v, c.hatch) == (args.P, args.v, args.hatch) and c.folder not in EXCLUDED:
            k = int(np.abs(c.times - args.t).argmin())
            truth, t_used = c.fields[k], c.times[k]
            break
    if truth is not None:
        excess = np.linalg.norm(truth[mask] - BASE_T)
        print(f"ground truth: {c.folder}, frame {k} at t={t_used:.3f}\n"
              f"  true peak {truth.max():.1f} C   MAE {np.abs(pred_a - truth[mask]).mean():.3f} C   "
              + (f"relative L2 {np.linalg.norm(pred_a - truth[mask]) / excess:.3f}" if excess > 0.5
                 else "(truth is all-ambient at this time)"))
        print("  note: if this combination was in the training set, this is a memorisation check, "
              "not a test")

    out_dir = OUT / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pred_P{args.P:g}_v{args.v:g}_h{args.hatch:g}_t{args.t:g}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["node", "active", "T_pred", "T_pred_std"] + (["T_true"] if truth is not None else []))
        for i in range(len(pred)):
            w.writerow([i, int(mask[i]), f"{pred[i]:.4f}", f"{spread[i]:.4f}"]
                       + ([f"{truth[i]:.4f}"] if truth is not None else []))
    print("wrote", path)


if __name__ == "__main__":
    main()
