"""Figures for the revised final deck (sir's comments, Nov 2026): the Case 1 flux rod and three
small schematics of what local overheating does.      python make_final_deck_figs.py"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle
from scipy.spatial import Voronoi

import rod_flux_sim as sim

OUT = "final_deck_assets/"
DARK, RED, TEAL, GRAY, SOLID, POWDER, MUTED = "#15161B", "#C8102E", "#1F7A8C", "#B9BEC7", "#8D94A1", "#D9DCE2", "#5E6573"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 13})


def rod_setup():
    """Left: the vertical rod with flux on top, fixed bottom and the hidden patch. Right: steady vs
    transient maximum along the height for one run, showing the patch is felt only near the top."""
    fig = plt.figure(figsize=(11.2, 4.08))
    ax = fig.add_axes([0.0, 0.0, 0.40, 1.0]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-1.2, 12.6)
    ax.add_patch(Rectangle((2.6, 0), 1.6, 10, color=SOLID))
    ax.add_patch(Rectangle((2.6, 7.1), 1.6, 1.0, color=TEAL))
    ax.add_patch(Rectangle((1.6, -0.9), 3.6, 0.9, color="#4A4F5A"))
    for x in (2.9, 3.4, 3.9):
        ax.add_patch(FancyArrowPatch((x, 12.2), (x, 10.15), arrowstyle="-|>", mutation_scale=16, lw=2.4, color=RED))
    ax.text(4.7, 11.2, "heat flux q\nfor time t$_h$", color=RED, fontsize=14, fontweight="bold", va="center")
    ax.text(4.7, 7.6, "low-k patch\n(hidden depth)", color=TEAL, fontsize=13, fontweight="bold", va="center")
    ax.text(5.6, 0.9, "bottom fixed at 65 °C\n(Dirichlet)", color=DARK, fontsize=13, va="center")
    ax.text(0.2, 5.0, "10 mm", rotation=90, va="center", color=MUTED, fontsize=12)
    ax.add_patch(FancyArrowPatch((1.1, 0), (1.1, 10), arrowstyle="<->", mutation_scale=12, lw=1.2, color=MUTED))

    p = {"q": 0.3, "t_h": 1.0, "k0": 0.012, "c": 7.6, "f": 0.12}
    ss = sim.steady(p["q"], p["k0"], p["c"], p["f"])
    tr = sim.transient_max(p["q"], p["t_h"], p["k0"], p["c"], p["f"])
    a = fig.add_axes([0.50, 0.17, 0.48, 0.78])
    a.plot(ss - 65, sim.X, color=TEAL, lw=2.6, label="steady (t = ∞)")
    a.plot(tr - 65, sim.X, color=RED, lw=2.6, label="transient max")
    a.axhspan(p["c"] - 0.5, p["c"] + 0.5, color=TEAL, alpha=0.15)
    a.set_xlim(0, 480)
    a.set_xlabel("temperature rise (°C)"); a.set_ylabel("height (mm)")
    a.spines[["top", "right"]].set_visible(False); a.legend(frameon=False, fontsize=12, loc="lower right")
    lc = np.sqrt(p["k0"] / sim.RHO_C * p["t_h"])
    a.annotate("", xy=(150, 10), xytext=(150, 10 - lc), arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.2))
    a.text(160, 10 - lc / 2, f"L$_c$ = {lc:.1f} mm", fontsize=12, va="center")
    fig.savefig(OUT + "fig_rodflux_setup.png", dpi=200, facecolor="white"); plt.close(fig)


def thumbs():
    # 1. dross / melt balls under an overhang
    fig, ax = plt.subplots(figsize=(3.2, 2.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7.5)
    ax.add_patch(Rectangle((0, 0), 10, 7.5, color="#EEF0F3"))
    ax.add_patch(Polygon([[0.5, 0.6], [3.5, 0.6], [3.5, 4.2], [9.5, 4.2], [9.5, 6.4], [0.5, 6.4]], color=SOLID))
    rng = np.random.default_rng(2)
    for x in np.linspace(4.0, 9.2, 9):
        h = rng.uniform(0.5, 1.6)
        ax.add_patch(Ellipse((x, 4.2 - h / 2), rng.uniform(0.4, 0.7), h, color="#B6461F"))
    for x, y, r in ((5.0, 1.5, 0.35), (7.4, 1.9, 0.45), (8.6, 1.2, 0.3)):
        ax.add_patch(Ellipse((x, y), 2 * r, 2 * r, color="#B6461F"))
    ax.text(6.6, 0.4, "dross and melt balls", ha="center", fontsize=12, fontweight="bold", color=DARK)
    fig.subplots_adjust(0, 0, 1, 1); fig.savefig(OUT + "thumb_dross.png", dpi=160); plt.close(fig)

    # 2. fine vs coarse grains
    fig, axs = plt.subplots(1, 2, figsize=(3.2, 2.4))
    for ax, n, lab in ((axs[0], 90, "normal"), (axs[1], 14, "overheated")):
        pts = np.random.default_rng(n).uniform(0, 1, (n, 2))
        vor = Voronoi(np.vstack([pts, [[-5, -5], [-5, 5], [5, -5], [5, 5]]]))
        for reg in vor.point_region[:n]:
            v = vor.regions[reg]
            if -1 in v or not v:
                continue
            ax.add_patch(Polygon(vor.vertices[v], fc=plt.cm.Greys(np.random.default_rng(reg).uniform(0.25, 0.6)), ec="white", lw=0.8))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(lab, fontsize=12, fontweight="bold", color=RED if n == 14 else DARK)
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.86, wspace=0.08); fig.savefig(OUT + "thumb_grains.png", dpi=160); plt.close(fig)

    # 3. curled part hitting the recoater
    fig, ax = plt.subplots(figsize=(3.2, 2.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7.5)
    ax.add_patch(Rectangle((0, 0), 10, 7.5, color="#EEF0F3"))
    ax.add_patch(Rectangle((0.5, 0.5), 9, 0.8, color="#4A4F5A"))
    xs = np.linspace(1.5, 8.5, 40); top = 3.2 + 1.6 * np.clip((xs - 5.5) / 3, 0, 1) ** 2
    ax.add_patch(Polygon(np.vstack([np.c_[xs, top], [[8.5, 1.3], [1.5, 1.3]]]), color=SOLID))
    ax.add_patch(Polygon([[7.4, 5.2], [9.6, 5.2], [9.6, 6.8], [7.4, 6.1]], color=TEAL))
    ax.annotate("", xy=(6.6, 5.6), xytext=(4.0, 5.6), arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2))
    ax.text(8.1, 4.5, "X", color=RED, fontsize=20, fontweight="bold", ha="center", va="center")
    ax.text(3.2, 6.6, "curl jams the recoater", ha="center", fontsize=12, fontweight="bold", color=DARK)
    fig.subplots_adjust(0, 0, 1, 1); fig.savefig(OUT + "thumb_recoater.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    rod_setup(); thumbs(); print("ok")
