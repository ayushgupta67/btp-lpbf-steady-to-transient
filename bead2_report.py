"""Figures and RESULTS.md for the updated single-bead experiments (reads the cv_*.csv files)."""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import bead2_data as bd

OUT = Path(__file__).parent / "bead2_outputs"
files = ["cv_results.csv", "cv_hybrid_v2_results.csv", "cv_hybrid_v3_results.csv"]
rows = [r for f in files for r in csv.DictReader(open(OUT / f))]
methods = list(dict.fromkeys(r["method"] for r in rows))
agg = lambda rs, k: float(np.sqrt(np.mean([float(r[k]) ** 2 for r in rs])))
short = {"diffusion (mean of draws)": "Diffusion (plain)", "ambient (65 everywhere)": "Ambient 65",
         "mean of training runs": "Mean of runs", "nearest setting (ln q, ln v)": "Nearest setting",
         "ridge (ln q, ln v)": "Ridge", "amplitude x shape": "Amp x shape (mean)",
         "hybrid v2: quadratic amplitude x diffusion shape": "Hybrid: amp x diffusion shape",
         "amplitude x shape (same v)": "Amp x shape (same v)",
         "amplitude x shape (linear in ln q, same v)": "Amp x shape (ln q, same v)",
         "hybrid v3: quadratic amplitude x (regression shape + diffusion residual)": "Hybrid: regression + diffusion residual"}

# figure 1: RMSE by method
order = sorted(methods, key=lambda m: -agg([r for r in rows if r["method"] == m], "rmse_all5166"))
fig, ax = plt.subplots(figsize=(9, 5))
vals = [agg([r for r in rows if r["method"] == m], "rmse_all5166") for m in order]
cols = ["tab:red" if "Diffusion (plain)" == short[m] else "tab:blue" if "Hybrid" in short[m] else "0.65" for m in order]
ax.barh([short[m] for m in order], vals, color=cols)
for i, v in enumerate(vals):
    ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
ax.set(xscale="symlog", xlabel="cross-validated RMSE over all 5,166 nodes, all 6 times (C, log scale)",
       title="Unseen-setting error: 24-fold leave-one-setting-out")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout(); fig.savefig(OUT / "fig_rmse_by_method.png", dpi=140); plt.close(fig)

# figure 2: t=2 peak, truth vs plain diffusion vs best simple
def t2(m):
    rs = sorted([r for r in rows if r["method"] == m and int(r["t_target"]) == 2], key=lambda r: (float(r["P"]) / float(r["hatch"]), float(r["v"])))
    return rs
tr = t2("diffusion (mean of draws)"); bs = t2("amplitude x shape (linear in ln q, same v)")
x = np.arange(len(tr))
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.bar(x - 0.27, [float(r["peak_true"]) for r in tr], 0.27, label="truth", color="0.25")
ax.bar(x, [float(r["peak_pred"]) for r in tr], 0.27, label="plain diffusion (unseen setting)", color="tab:red")
ax.bar(x + 0.27, [float(r["peak_pred"]) for r in bs], 0.27, label="amplitude x shape (unseen setting)", color="tab:blue")
ax.set_xticks(x, [r["run"] for r in tr], rotation=60, ha="right", fontsize=7)
ax.set(ylabel="peak temperature at t = 2 (C)", title="Peak at t = 2 for each held-out run")
ax.legend(); ax.grid(alpha=0.3, axis="y"); fig.tight_layout(); fig.savefig(OUT / "fig_peak_t2.png", dpi=140); plt.close(fig)

# figure 3: example fields
data = bd.load_cases(); mask = bd.active_node_mask(data)
a = np.load(OUT / "cv_predictions.npz"); b = np.load(OUT / "cv_hybrid_v3_predictions.npz")
runs = a["run"]; i = next(k for k in range(len(runs)) if runs[k] == "P90_v10_hpnt1" and a["t_target"][k] == 2)
top = lambda f: bd.expand(f, mask).reshape(3, 42, 41).max(0)[12:30] - bd.BASE_T
panels = [("Truth", a["true"][i]), ("Plain diffusion (unseen)", a["pred"][i]), ("Amp x shape (unseen)", b["pred"][i])]
vmax = max(top(p).max() for _, p in panels)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.2), layout="constrained")
for ax, (n, f) in zip(axs, panels):
    im = ax.imshow(top(f), cmap="inferno", origin="lower", vmin=0, vmax=vmax)
    err = np.sqrt(((f - a["true"][i]) ** 2).mean() * len(f) / 5166)
    ax.set_title(f"{n}\nRMSE {err:.1f} C", fontsize=11); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axs, shrink=0.8, label="rise above 65 (C)")
fig.suptitle("P90, v10, hatch 0.10 at t = 2 (held out)", fontsize=11)
fig.savefig(OUT / "fig_example_fields.png", dpi=140); plt.close(fig)

# RESULTS.md
timing = json.loads((OUT / "timing.json").read_text())
ts = range(2, 8)
def table(key):
    s = "| method | " + " | ".join(f"t={t}" for t in ts) + " | ALL |\n|---|" + "---|" * 7 + "\n"
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        s += f"| {short[m]} | " + " | ".join(f"{agg([r for r in rs if int(r['t_target']) == t], key):.2f}" for t in ts) + f" | {agg(rs, key):.2f} |\n"
    return s
md = f"""# Updated single-bead dataset: results

Cross-validated RMSE in degrees C, 24-fold leave-one-setting-out (a setting is a (P/hatch, v) pair; identical twins held out together).

## All 5,166 nodes
{table('rmse_all5166')}
## 1,094 active nodes only
{table('rmse_active')}
## Computation (12th-gen i5-12500H laptop, CPU only)
- plain diffusion, final model: {timing['final_train_s']:.0f} s ({timing['params']:,} parameters, {timing['epochs']} epochs, {timing['rows']} rows)
- plain diffusion, 24-fold test: {timing['cv_wall_s'] / 60:.1f} min (6 workers; each fold {timing['cv_fold_train_s_mean'] / 60:.1f} min under contention)
- hybrid runs: about 11.5 min for 24 folds, 30 s for the final model
"""
(OUT / "RESULTS.md").write_text(md)
print("wrote figures and RESULTS.md")
