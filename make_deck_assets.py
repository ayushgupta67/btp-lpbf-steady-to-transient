"""Collect the numbers and draw the figures for the mid-semester deck.

    python make_deck_assets.py

Everything the deck quotes is read from files produced by the experiments
(bead_outputs/, wall_outputs/) or recomputed from the raw data here, and written
to deck_assets/deck_data.json. build_midsem_deck.py reads only that file, so no
number on a slide is typed by hand.

Figures (deck_assets/):
  fig_forward_process.png  the forward (noising) process applied to a real bead field
  fig_bead_fields.png      steady-state input and transient targets of one run
"""

import csv
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bead_model as bm
from bead_data import BASE_T, EXCLUDED, active_node_mask, load_cases, steady_lookup

ROOT = Path(__file__).parent
OUT = ROOT / "deck_assets"
NX, NY, NZ = 41, 42, 3  # x fastest: reshape(NZ, NY, NX)
RUN = "P60_v10_hpnt15"  # the run drawn in the figures


Y0, Y1 = 12, 30  # rows around the laser track (iy ~ 20.5); the rest of the plate is air
EXTENT = (-0.5, NX - 0.5, Y0 - 0.5, Y1 - 0.5)


def top_view(vec, y0=Y0, y1=Y1):
    """Maximum over depth of a flat node vector, cropped to a band of rows: (y1-y0, NX)."""
    return vec.reshape(NZ, NY, NX).max(0)[y0:y1]


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def collect(data):
    out = {}
    cases = {c.folder: c for c in data["transient"]}
    mask = active_node_mask(data)

    # Peak temperature of the whole run against P/hatch, per velocity (excluded run left out).
    peaks = {}
    for c in data["transient"]:
        if c.folder in EXCLUDED:
            continue
        peaks.setdefault(int(c.v), {})[round(c.P / c.hatch, 1)] = float(c.fields.max())
    ph = sorted({k for d in peaks.values() for k in d})
    out["peak_vs_ph"] = {"P_over_h": ph, **{f"v{v}": [peaks[v].get(k) for k in ph] for v in sorted(peaks)}}

    out["grid"] = {"nodes": 5166, "air": int((~mask).sum()), "active": int(mask.sum()),
                   "transient_runs": len(data["transient"]),
                   "frames_per_run": sorted({len(c.times) for c in data["transient"]}),
                   "steady_runs": len(data["steady"]),
                   "steady_distinct_fields": len(steady_lookup(data["steady"]))}

    # Signal left at each whole-number time, averaged over the 8 usable runs (peak rise above 65).
    rows = read_csv(ROOT / "bead_outputs" / "loo_results.csv")
    amb = [r for r in rows if r["method"].startswith("ambient")]
    out["peak_rise_by_t"] = {str(t): float(np.mean([float(r["peak_true"]) - BASE_T
                                                    for r in amb if int(r["t_target"]) == t]))
                             for t in range(2, 8)}
    out["peak_rise_max_by_t"] = {str(t): float(np.max([float(r["peak_true"]) - BASE_T
                                                       for r in amb if int(r["t_target"]) == t]))
                                 for t in range(2, 8)}

    # Leave-one-run-out results.
    methods = list(dict.fromkeys(r["method"] for r in rows))
    summary = {}
    for m in methods:
        sub = [r for r in rows if r["method"] == m]

        def avg(t, key):
            v = [float(r[key]) for r in sub if int(r["t_target"]) == t and r[key] != "nan"]
            return float(np.mean(v)) if v else None
        pk = [abs(float(r["peak_pred"]) - float(r["peak_true"])) / (float(r["peak_true"]) - BASE_T) * 100
              for r in sub if int(r["t_target"]) == 2]
        summary[m] = {"t2_relL2": avg(2, "rel_L2"), "t2_peak_err_pct": float(np.mean(pk)),
                      "t3_relL2": avg(3, "rel_L2"),
                      "late_max_abs_err": float(np.mean([float(r["max_abs_err"]) for r in sub
                                                         if int(r["t_target"]) >= 4]))}
    amp = read_csv(ROOT / "bead_outputs" / "loo_amplitude_shape.csv")
    summary["amplitude x shape"] = {
        "t2_relL2": float(np.mean([float(r["rel_L2"]) for r in amp])),
        "t2_peak_err_pct": float(np.mean([abs(float(r["peak_pred"]) - float(r["peak_true"]))
                                          / (float(r["peak_true"]) - BASE_T) * 100 for r in amp]))}
    out["loo_summary"] = summary

    per_case = []
    for r in amp:
        c = r["case"]
        pick = lambda key: next(x for x in rows if x["case"] == c and int(x["t_target"]) == 2
                                and x["method"].startswith(key))
        d, n = pick("diffusion"), pick("nearest")
        per_case.append({"case": c, "label": re.sub(r"P60_v(\d+)_h(?:pnt|ptn)(\d+)", lambda m: f"v{m.group(1)} h{ {'1': '0.10', '15': '0.15', '18': '0.18'}[m.group(2)] }", c),
                         "true": float(r["peak_true"]), "amp_shape": float(r["peak_pred"]),
                         "diffusion": float(d["peak_pred"]), "nearest": float(n["peak_pred"]),
                         "relL2_amp": float(r["rel_L2"]), "relL2_diff": float(d["rel_L2"]),
                         "relL2_near": float(n["rel_L2"])})
    out["per_case_t2"] = per_case

    # Which stored frame stands in for each whole-number time, for the example run.
    ex = cases[RUN]
    near = [(t, int(np.abs(ex.times - t).argmin())) for t in range(2, 8)]
    out["example_run"] = {"folder": RUN, "times": [float(ex.times[k]) for _, k in near],
                          "peak_rise": [float(ex.fields[k].max() - BASE_T) for _, k in near]}

    # Training record of the saved model.
    out["model"] = json.loads((ROOT / "bead_outputs" / "model" / "training_meta.json").read_text())
    loss = read_csv(ROOT / "bead_outputs" / "model" / "training_loss.csv")
    out["model"]["loss_epoch1"], out["model"]["loss_final"] = float(loss[0]["loss"]), float(loss[-1]["loss"])

    # Wall demo (steady -> transient), written by train_wall_diffusion.py.
    out["wall_demo"] = {r["method"]: float(r["MAE_C"]) for r in read_csv(ROOT / "wall_outputs" / "wall_results.csv")}

    # How much of the t=2 variation is one shape times an amplitude.
    pairs_t2 = [r for r in amp]
    ex = []
    from bead_data import build_pairs
    p = build_pairs(data)
    sel = p["t_target"] == 2
    E = p["y"][sel][:, mask] - BASE_T
    s = np.linalg.svd(E, compute_uv=False) ** 2
    out["first_component_pct"] = float(s[0] / s.sum() * 100)
    return out


def fig_forward(data):
    c = next(x for x in data["transient"] if x.folder == RUN)
    fy0, fy1 = 5, 36  # a wider band than the field figure, so the panels are taller
    z = bm.to_z(top_view(c.fields[0], fy0, fy1))
    ab = bm.alpha_bar_schedule().numpy()
    rng = np.random.default_rng(3)
    eps = rng.standard_normal(z.shape)
    taus = (0, 100, 200, 300, 399)
    fig, ax = plt.subplots(1, 5, figsize=(13, 2.7), layout="constrained")
    for a, t in zip(ax, taus):
        x = np.sqrt(ab[t]) * z + np.sqrt(1 - ab[t]) * eps
        a.imshow(x, cmap="inferno", vmin=-1.0, vmax=1.6, origin="lower", aspect="equal",
                 extent=(-0.5, NX - 0.5, fy0 - 0.5, fy1 - 0.5))
        a.set_title(f"step {t}\nsignal {np.sqrt(ab[t]) * 100:.0f}%, noise {np.sqrt(1 - ab[t]) * 100:.0f}%",
                    fontsize=15)
        a.set_xticks([]); a.set_yticks([])
    fig.savefig(OUT / "fig_forward_process.png", dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def fig_fields(data):
    cs = {x.folder: x for x in data["transient"]}
    ss = {x.folder: x for x in data["steady"]}
    c, s = cs[RUN], ss[RUN]
    k3 = int(np.abs(c.times - 3).argmin())
    panels = [("Steady-state field (input)", s.fields[-1], "final frame of the steady run"),
              ("Transient field at t = 2 (target)", c.fields[0], f"scan has just started, t = {c.times[0]:.3f}"),
              ("Transient field at t = 3 (target)", c.fields[k3], f"after the scan, t = {c.times[k3]:.2f}")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.3), layout="constrained")
    for a, (title, f, sub) in zip(ax, panels):
        im = a.imshow(top_view(f) - BASE_T, cmap="inferno", origin="lower", aspect="equal", extent=EXTENT)
        a.set_title(f"{title}\n{sub}", fontsize=14)
        a.set_xlabel("x: scan direction (node)", fontsize=13)
        a.set_ylabel("y: width (node)", fontsize=13)
        a.tick_params(labelsize=12)
        cb = fig.colorbar(im, ax=a, shrink=0.8, pad=0.02)
        cb.set_label("rise above 65 (C)", fontsize=13)
        cb.ax.tick_params(labelsize=12)
    fig.savefig(OUT / "fig_bead_fields.png", dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def fig_wall():
    """Truth / diffusion / no-skill slices of one held-out wall frame (steady -> transient)."""
    from train_wall_diffusion import cond_for
    from wall_data import WallData
    d = WallData(root=str(ROOT))
    steady, transient = d.Y, d.X
    tr, te = d.split("extrapolate")
    model, meta = bm.load_checkpoint(ROOT / "wall_outputs" / "wall_diffusion.pt")
    pred = bm.sample(model, cond_for(steady[te], d.t[te]), z_clip=tuple(meta["z_clip"]), n_draws=8, seed=0).mean(0)
    k = len(pred) // 2
    truth, cmean = transient[te][k], transient[tr].mean(0)

    def grid(vec):
        g = np.full(d.mask.shape, np.nan)
        g[d.mask] = vec
        return g[:, 10, :].T  # slice through the wall: rows = z, columns = x

    panels = [("truth", truth), ("diffusion", pred[k]), ("no skill", cmean)]
    lo, hi = min(v.min() for _, v in panels), max(v.max() for _, v in panels)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 5.0), layout="constrained")
    for a, (name, vec) in zip(ax, panels):
        im = a.imshow(grid(vec), origin="lower", cmap="inferno", vmin=lo, vmax=hi, aspect="auto")
        err = "" if name == "truth" else f"\nMAE {np.abs(vec - truth).mean():.2f}"
        a.set_title(name + err, fontsize=14)
        a.set_xlabel("x (build dir.)", fontsize=12)
        a.tick_params(labelsize=11)
    ax[0].set_ylabel("z (height)", fontsize=12)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cb.set_label("temperature (C)", fontsize=12)
    cb.ax.tick_params(labelsize=11)
    fig.savefig(OUT / "fig_wall_panels.png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return float(d.t[te][k])


def main():
    OUT.mkdir(exist_ok=True)
    data = load_cases()
    info = collect(data)
    fig_forward(data)
    fig_fields(data)
    info["wall_frame_t"] = fig_wall()
    (OUT / "deck_data.json").write_text(json.dumps(info, indent=2))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
