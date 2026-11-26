"""Figures and RESULTS.md for the max-temperature mapping (reads maxmap_outputs/)."""
import csv, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import maxmap_data as md

O = md.OUT
z = np.load(O / "cv_predictions.npz")
S = json.loads((O / "summary.json").read_text())
rows = list(csv.DictReader(open(O / "cv_results.csv")))
top = lambda f: f.reshape(3, 42, 41).max(0)[10:32] - md.BASE_T

# 1. example: input / truth / prediction for three runs
pick = [list(z["name"]).index(n) for n in ("P60_v100_hpnt18", "P90_v10_hpnt15", "P120_v10_hpnt1") if n in list(z["name"])]
fig, ax = plt.subplots(len(pick), 3, figsize=(13, 2.6 * len(pick)), layout="constrained")
for r, i in enumerate(pick):
    vmax = z["true"][i].max() - md.BASE_T
    for c, (lab, f) in enumerate((("SS_new max field (input)", z["ss"][i]), ("transient max field (truth)", z["true"][i]), ("diffusion, unseen setting", z["pred"][i]))):
        im = ax[r, c].imshow(top(f), cmap="inferno", origin="lower", vmin=0, vmax=max(vmax, 1))
        rm = np.sqrt(((f - z["true"][i]) ** 2).mean()) if c == 2 else None
        ax[r, c].set_title(f"{lab}\n{z['name'][i]}: peak {f.max():.0f} C" + (f", RMSE {rm:.0f} C" if rm is not None else ""), fontsize=9.5)
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    fig.colorbar(im, ax=ax[r, :], shrink=0.8, label="rise above 65 C")
fig.savefig(O / "fig_examples.png", dpi=130); plt.close(fig)

# 2. peak per held-out run
o = np.argsort(z["true"].max(1))
fig, ax = plt.subplots(figsize=(11, 4.2))
x = np.arange(len(o))
ax.bar(x - 0.2, z["true"].max(1)[o], 0.4, color="#15161B", label="transient max (truth)")
ax.bar(x + 0.2, z["pred"].max(1)[o], 0.4, color="#C8102E", label="diffusion, unseen setting")
ax.set_xticks(x, [z["name"][i] for i in o], rotation=65, ha="right", fontsize=7.5)
ax.set_ylabel("peak of the max field (C)"); ax.legend(frameon=False); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(O / "fig_peaks.png", dpi=140); plt.close(fig)

# 3. loss curve
L = np.loadtxt(O / "model" / "training_loss.csv", delimiter=",", skiprows=1)
fig, ax = plt.subplots(figsize=(6, 3.5)); ax.semilogy(L[:, 0], L[:, 1]); ax.set(xlabel="epoch", ylabel="training loss (z-space MSE)")
ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(O / "fig_loss.png", dpi=140); plt.close(fig)

worst = sorted(rows, key=lambda r: -abs(float(r["peak_err_pct"])))[:5]
md_txt = f"""# Max-temperature mapping: results

Task (sir's instruction): max steady-state field (SS_new) + P, v, hatch -> max transient field, one pair per run.

| quantity | value |
|---|---|
| pairs (runs) | {S['pairs']} (24 distinct settings; copied run excluded) |
| CV folds (leave one setting out) | {S['folds']} |
| RMSE, all 5,166 nodes | {S['rmse_all_C']:.1f} C |
| MAE, all nodes | {S['mae_all_C']:.1f} C |
| peak error, mean / median / worst | {S['mean_abs_peak_err_pct']:.1f} % / {S['median_abs_peak_err_pct']:.1f} % / {S['max_abs_peak_err_pct']:.1f} % |
| hot-spot overlap (Jaccard, >= half the peak rise) | {S['mean_hotspot_jaccard']:.2f} |
| relative L2 error (1.0 = predicting 65 C everywhere) | {S['mean_rel_l2']:.2f} |
| in-sample RMSE (seen data, not a test) | {S['in_sample_rmse_all_C']:.1f} C |
| parameters / epochs / batch | {S['parameters']:,} / {S['epochs']} / {S['batch']} |
| final model training time | {S['final_train_s']:.0f} s |
| cross-validation wall time | {S['cv_wall_s'] / 60:.1f} min ({S['cv_fold_train_s_mean']:.0f} s per fold, 4 in parallel) |

Worst held-out runs (peak error): """ + "; ".join(f"{r['run']} {float(r['peak_err_pct']):+.1f}%" for r in worst) + "\n"
(O / "RESULTS.md").write_text(md_txt)
print(md_txt)
