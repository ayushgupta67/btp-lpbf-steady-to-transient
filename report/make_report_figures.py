"""Figures for the BTP report (report/figures/). New figures are drawn from the saved results;
existing figures from the experiments and the presentation are copied in unchanged.

    python make_report_figures.py      (run from report/, with the project venv)
"""
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
DARK, RED, TEAL, GRAY, AMBER, MUTED = "#15161B", "#C8102E", "#1F7A8C", "#9AA0AA", "#C27C0E", "#5E6573"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8A909C", "axes.labelcolor": DARK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#E3E5E9", "grid.linewidth": 0.8, "axes.axisbelow": True})

COPY = {"fig_lpbf_frame.png": "final_deck_assets/lpbf_video_poster.png", "fig_overheat.png": "final_deck_assets/fig_overheat.png",
        "fig_bvp.png": "final_deck_assets/fig_bvp.png", "fig_forward.png": "deck_assets/fig_forward_process.png",
        "fig_wall.png": "deck_assets/fig_wall_panels.png", "fig_rod.png": "final_deck_assets/fig_rod.png",
        "fig_rod_max.png": "rod1d_outputs/fig5_max_explained.png", "fig_rod_chain.png": "rod1d_outputs/fig4_diffusion.png",
        "fig_rod_datasize.png": "rod1d_outputs/fig3_data.png", "fig_beads.png": "final_deck_assets/fig_beads.png",
        "fig_bead_rmse.png": "bead2_outputs/fig_rmse_by_method.png", "fig_bead_fields.png": "final_deck_assets/fig_bead_fields.png",
        "fig_ss_same.png": "final_deck_assets/fig_ss_same.png", "fig_weld.png": "final_deck_assets/fig_weld.png",
        "fig_ss_local.png": "final_deck_assets/fig_ss_local.png", "fig_lc.png": "final_deck_assets/fig_lc.png",
        "fig_same_geom.png": "final_deck_assets/fig_same_geom.png", "fig_diff_geom.png": "final_deck_assets/fig_diff_geom.png",
        "fig_part3d.png": "tts_outputs/fig3d_part.png", "fig_part_layers.png": "tts_outputs/fig_part_layers.png",
        "fig_geo_maps.png": "final_deck_assets/fig_geo_maps.png", "fig_bug_layers.png": "tts_outputs/fig_bug_layers.png",
        "fig_peaks_by_element.png": "tts_outputs/fig_peaks_by_element.png"}


def save(fig, name):
    fig.savefig(OUT / name, dpi=250, bbox_inches="tight", facecolor="white"); plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(10, 4.2)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 42)
    def b(x, y, w, h, title, sub, fc="#F2F3F5", tc=DARK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="none"))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=9, color=tc)
    def a(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.4, color=MUTED))
    b(0, 31, 21, 8, "Noisy target  $x_\\tau$", "n values", fc="#FBE9EC")
    b(0, 17, 21, 8, "Condition field  $c$", "steady field, n values", fc="#DDEEF1")
    b(0, 9, 21, 6, "Scalars  $s$", "process parameters", fc="#DDEEF1")
    b(0, 0, 21, 7, "Noise level  $\\tau$", "integer in [0, 399]")
    b(28, 14, 20, 10, "Condition encoder", "Linear(n+m, 256), SiLU,\nLinear(256, d$_c$)")
    b(28, 0, 20, 7, "Noise-level code", "sin/cos (64), Linear, SiLU")
    b(55, 0, 9, 39, "Join", "n + d$_c$\n+ m + 64")
    b(70, 0, 15, 39, "Denoiser MLP", "Linear(·, h), SiLU\nLinear(h, h), SiLU\nLinear(h, n)", fc=DARK, tc="white")
    b(89, 0, 11, 39, "Output", "$\\hat{x}_0$\nclean target\n(n values)", fc=RED, tc="white")
    a((21, 35), (55, 35)); a((21, 21), (28, 19.5)); a((21, 12), (28, 17)); a((21, 3.5), (28, 3.5)); a((48, 19), (55, 19)); a((48, 3.5), (55, 3.5))
    a((21, 11), (55, 11)); a((64, 19.5), (70, 19.5)); a((85, 19.5), (89, 19.5))
    save(fig, "fig_architecture.png")


def case1():
    d = np.load(ROOT / "rod1d_outputs/rod1d_data.npz"); pr = np.load(ROOT / "rod1d_outputs/pred_test_N1600.npy")
    pk = np.abs(pr.max(1) - d["tr_test"].max(1)) / (d["tr_test"].max(1) - 65) * 100
    i = int(np.argsort(pk)[150])
    res = list(csv.DictReader(open(ROOT / "rod1d_outputs/rod1d_results.csv")))
    ws = [r for r in res if r["name"] == "with_ss"]; no = [r for r in res if r["name"] == "no_ss" and r["n_train"] == "1600"][0]
    ap = {(r["name"], r["n_train"]): r for r in json.load(open(ROOT / "rod1d_outputs/rod1d_allparams.json"))}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
    a1.plot(d["x"], d["ss_test"][i], color=TEAL, lw=2, label="max steady field (input)")
    a1.plot(d["x"], d["tr_test"][i], color=DARK, lw=3.2, label="max transient field (truth)")
    a1.plot(d["x"], pr[i], color=RED, lw=1.8, ls="--", label="diffusion prediction")
    a1.set(xlabel="position along the rod (mm)", ylabel="temperature (°C)", title=f"(a) median-error test run: P = {d['test_P'][i]:.0f} W, v = {d['test_v'][i]:.0f} mm/s")
    a1.legend(frameon=False, fontsize=9.5); a1.title.set_fontsize(10.5)
    x = np.arange(2); w = 0.36
    wv = [float(ws[-1]["mean_abs_peak_err_pct"]), ap[("all_params+ss", 1600)]["mean_abs_peak_err_pct"]]
    nv = [float(no["mean_abs_peak_err_pct"]), ap[("all_params_only", 1600)]["mean_abs_peak_err_pct"]]
    b1 = a2.bar(x - w / 2 - 0.01, wv, w, color=TEAL, label="with steady field"); b2 = a2.bar(x + w / 2 + 0.01, nv, w, color=GRAY, label="without steady field")
    for bars in (b1, b2):
        for r in bars:
            a2.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.3, f"{r.get_height():.2f}", ha="center", fontsize=9.5)
    a2.set_xticks(x, ["material hidden\n(inputs: P, v)", "all inputs given\n(P, v, α, β, r)"]); a2.set_ylabel("mean peak error (%)")
    a2.set_title("(b) value of the steady field, 1,600 training runs", fontsize=10.5); a2.legend(frameon=False, fontsize=9.5); a2.grid(axis="x", visible=False)
    fig.tight_layout(); save(fig, "fig_case1_results.png")


def case2():
    z = np.load(ROOT / "maxmap_outputs/cv_predictions.npz"); tp, pp, sp = z["true"].max(1), z["pred"].max(1), z["ss"].max(1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))
    a1.plot([2000, 10000], [2000, 10000], color=GRAY, lw=1); a1.scatter(tp, pp, s=34, color=RED, edgecolor="white", lw=0.6, zorder=3)
    a1.set(xlabel="true peak of the max transient field (°C)", ylabel="predicted peak (°C)", xlim=(2000, 10000), ylim=(2000, 10000))
    a1.set_title("(a) held-out settings, 24-fold leave-one-setting-out", fontsize=10.5)
    o = np.argsort(tp)
    a2.plot(np.arange(1, 27), sp[o], color=TEAL, lw=2.5, marker="o", ms=4, label="max steady field (SS_new)")
    a2.plot(np.arange(1, 27), tp[o], color=RED, lw=2.5, marker="o", ms=4, label="max transient field")
    a2.set(xlabel="run (sorted by transient peak)", ylabel="peak temperature (°C)", ylim=(0, 10000)); a2.legend(frameon=False, fontsize=9.5)
    a2.set_title("(b) the steady peak is 6,666.6 °C in every run", fontsize=10.5)
    fig.tight_layout(); save(fig, "fig_case2_results.png")


def case3():
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    cats = ["0", "1", "2"]; s_ = [141.853, 165.602, 202.351]; t_ = [117.892, 120.189, 129.543]; x = np.arange(3); w = 0.36
    for xs, v, c, lab in ((x - w / 2 - 0.01, s_, TEAL, "steady peak"), (x + w / 2 + 0.01, t_, RED, "transient peak")):
        bars = ax.bar(xs, v, w, color=c, label=lab)
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 2, f"{r.get_height():.0f}", ha="center", fontsize=9.5)
    ax.set_xticks(x, cats); ax.set(xlabel="number of empty in-plane neighbours (void or outside)", ylabel="median peak (10³ °C)")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); save(fig, "fig_case3_neighbours.png")

    r0 = json.load(open(ROOT / "tts_outputs/model_results.json")); chk = json.load(open(ROOT / "tts_outputs/model_checks.json"))
    st = [r0["diffusion"]["mae_C"]] + [chk[f"steady_seed{k}"]["mae_C"] for k in (1, 2)]
    sh = [chk[f"shape_only_seed{k}"]["mae_C"] for k in (0, 1, 2)]
    pz = np.load(ROOT / "tts_outputs/model_predictions.npz")
    fig = plt.figure(figsize=(11, 4.0)); gs = fig.add_gridspec(1, 4, width_ratios=[1.55, 1, 1, 1], wspace=0.25)
    a = fig.add_subplot(gs[0])
    labs = ["guess the training\naverage", "straight line from\nsteady peak", "diffusion:\nsteady patch", "diffusion:\nshape only"]
    vals = [r0["mean_of_training"]["mae_C"], r0["linear_centre_steady"]["mae_C"], np.mean(st), np.mean(sh)]
    err = [[0, 0, np.mean(st) - min(st), np.mean(sh) - min(sh)], [0, 0, max(st) - np.mean(st), max(sh) - np.mean(sh)]]
    a.barh(labs[::-1], vals[::-1], color=[TEAL, RED, GRAY, GRAY])
    a.errorbar(vals[3:1:-1], [0, 1], xerr=[[e[3], e[2]] for e in err], fmt="none", ecolor=DARK, capsize=3)
    for k, v in enumerate(vals[::-1]):
        a.text(v + 60, k + 0.22, f"{v:,.0f}", fontsize=9.5)
    a.set_xlabel("mean absolute error, layers 9–10 (°C)"); a.set_title("(a) error on unseen layers", fontsize=10.5); a.grid(axis="y", visible=False)
    m = pz["layer"] == 10
    def grid(v):
        g = np.full((20, 20), np.nan); g[pz["iy"][m] - 1, pz["ix"][m] - 1] = v[m] / 1000; return g
    cm = matplotlib.colormaps["inferno"].copy(); cm.set_bad("white")
    lo, hi = np.nanpercentile(pz["true"][m] / 1000, 1), np.nanpercentile(pz["true"][m] / 1000, 99.5)
    for k, (key, t) in enumerate((("true", "(b) true transient peak"), ("diffusion", "(c) diffusion, steady patch"), ("linear", "(d) straight line"))):
        ax = fig.add_subplot(gs[k + 1]); im = ax.imshow(grid(pz[key]), cmap=cm, origin="lower", vmin=lo, vmax=hi)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_title(t + "\nlayer 10", fontsize=10)
    cb = fig.colorbar(im, ax=fig.axes[1:], shrink=0.75, pad=0.02); cb.set_label("peak (10³ °C)")
    save(fig, "fig_case3_results.png")


if __name__ == "__main__":
    for k, v in COPY.items():
        shutil.copy(ROOT / v, OUT / k)
    for f in (architecture, case1, case2, case3):
        f(); print("done", f.__name__)
