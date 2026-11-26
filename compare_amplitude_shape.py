"""Amplitude x shape predictor for the t = 2 field, scored like the diffusion model.

    python compare_amplitude_shape.py

At t = 2 the transient fields of all 8 usable runs are one fixed spatial shape
times an amplitude (the first singular component holds 99.7% of the energy).
So the field is predicted as

    amplitude(v, hatch) * shape

* amplitude: ln(peak rise above 65 C) = a + b*ln(v) + c*ln(hatch), fitted by
  least squares on the training runs (3 numbers),
* shape: the mean of the training fields, each divided by its own peak.

No epochs, no network. Evaluated leave-one-case-out exactly like the diffusion
model (same 8 folds, same 1094 active nodes, same metrics), then compared with
the diffusion and nearest-case numbers saved in bead_outputs/loo_results.csv.
Only t = 2 is covered: later times have little or no signal (the part is back at
65 C by t ~ 4), so a one-amplitude model is not meaningful there.
"""

import csv
from pathlib import Path

import numpy as np

from bead_data import BASE_T, active_node_mask, build_pairs, load_cases

OUT = Path(__file__).parent / "bead_outputs"


def main():
    data = load_cases()
    pairs = build_pairs(data)
    mask = active_node_mask(data)
    sel = pairs["t_target"] == 2
    excess = pairs["y"][sel][:, mask] - BASE_T  # (8, 1094)
    par, cases = pairs["params"][sel], list(np.array(pairs["case"])[sel])
    v, h, peak = par[:, 1], par[:, 2], excess.max(1)

    s = np.linalg.svd(excess, compute_uv=False) ** 2
    print(f"energy in first singular component of the t=2 fields: {s[0] / s.sum() * 100:.2f}%\n")

    rows = []
    for i, held in enumerate(cases):
        tr = np.arange(len(cases)) != i
        X = np.column_stack([np.ones(tr.sum()), np.log(v[tr]), np.log(h[tr])])
        coef = np.linalg.lstsq(X, np.log(peak[tr]), rcond=None)[0]
        amp = float(np.exp(coef @ [1.0, np.log(v[i]), np.log(h[i])]))
        shape = (excess[tr] / peak[tr][:, None]).mean(0)
        pred = amp * shape
        rows.append({"case": held, "peak_true": peak[i] + BASE_T, "peak_pred": pred.max() + BASE_T,
                     "rel_L2": float(np.linalg.norm(pred - excess[i]) / np.linalg.norm(excess[i]))})

    saved = {}
    for r in csv.DictReader(open(OUT / "loo_results.csv")):
        if r["t_target"] == "2":
            saved[(r["case"], r["method"].split(" ")[0])] = r

    def peak_err(true, pred):
        return abs(pred - true) / (true - BASE_T) * 100

    print(f"{'held-out case':<17}{'true peak':>10} | {'amp x shape':>11}{'err %':>7} | {'diffusion':>10}{'err %':>7} | {'nearest':>8}{'err %':>7}"
          f" | relL2: amp x shape / diffusion / nearest")
    tot = {"amp": [], "diff": [], "near": []}
    for r in rows:
        d, nb = saved[(r["case"], "diffusion")], saved[(r["case"], "nearest")]
        e_a = peak_err(r["peak_true"], r["peak_pred"])
        e_d = peak_err(r["peak_true"], float(d["peak_pred"]))
        e_n = peak_err(r["peak_true"], float(nb["peak_pred"]))
        tot["amp"] += [(e_a, r["rel_L2"])]
        tot["diff"] += [(e_d, float(d["rel_L2"]))]
        tot["near"] += [(e_n, float(nb["rel_L2"]))]
        print(f"{r['case']:<17}{r['peak_true']:>10.1f} | {r['peak_pred']:>11.1f}{e_a:>7.1f} | {float(d['peak_pred']):>10.1f}{e_d:>7.1f}"
              f" | {float(nb['peak_pred']):>8.1f}{e_n:>7.1f} | {r['rel_L2']:.3f} / {float(d['rel_L2']):.3f} / {float(nb['rel_L2']):.3f}")

    print("\nMEAN over the 8 held-out cases, t = 2")
    print(f"{'method':<28}{'peak err %':>11}{'whole-field relL2':>19}")
    for key, name in (("amp", "amplitude x shape"), ("diff", "diffusion"), ("near", "nearest case in (v, hatch)")):
        a = np.array(tot[key])
        print(f"{name:<28}{a[:, 0].mean():>11.1f}{a[:, 1].mean():>19.3f}")

    with open(OUT / "loo_amplitude_shape.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, saved)


def plot(rows, saved):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = [r["case"] for r in rows]
    x, w = np.arange(len(cases)), 0.2
    series = [("truth", [r["peak_true"] for r in rows], "0.3"),
              ("amplitude x shape", [r["peak_pred"] for r in rows], "tab:green"),
              ("diffusion", [float(saved[(c, "diffusion")]["peak_pred"]) for c in cases], "tab:red"),
              ("nearest case", [float(saved[(c, "nearest")]["peak_pred"]) for c in cases], "0.75")]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for k, (label, vals, color) in enumerate(series):
        ax.bar(x + (k - 1.5) * w, vals, w, label=label, color=color)
    ax.set_xticks(x, cases, rotation=35, ha="right", fontsize=8)
    ax.set(ylabel="peak temperature at t = 2 (C)", title="Leave-one-case-out: peak temperature at t = 2")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "loo_peak_t2_all_methods.png", dpi=140)
    print("wrote", OUT / "loo_peak_t2_all_methods.png")


if __name__ == "__main__":
    main()
