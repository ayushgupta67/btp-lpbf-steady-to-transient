"""Figures for the final presentation (schematics after Ayush's handwritten notes, plus data figures).

    python make_final_assets.py && python build_final_deck.py

Outputs in final_deck_assets/.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle, Wedge
from PIL import Image
from scipy.linalg import solve_banded
from scipy.special import erfc

import rod1d_sim as sim

ROOT = Path(__file__).parent
OUT = ROOT / "final_deck_assets"
OUT.mkdir(exist_ok=True)
DARK, INK, MUTED = "#15161B", "#1F2128", "#5E6573"
RED, TEAL, AMBER, GRAY, SOLID, POWDER = "#C8102E", "#1F7A8C", "#C27C0E", "#B9BEC7", "#8D94A1", "#D9DCE2"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 15, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRAY})
HEAT = LinearSegmentedColormap.from_list("heat", ["#F2F3F5", "#F6C453", "#E8590C", "#C8102E"])


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def save(fig, name, dpi=200):
    fig.savefig(OUT / name, dpi=dpi, facecolor="white"); plt.close(fig)


def arrow(ax, a, b, color=RED, lw=2.4, style="-|>", rad=0.0, ms=16):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=8))


TEXT_SCALE = [1.0]  # set per figure: labels must stay readable when the figure is shown small


def note(ax, x, y, s, color=INK, size=15, ha="center", va="center", bold=False):
    ax.text(x, y, s, color=color, fontsize=size * TEXT_SCALE[0], ha=ha, va=va, fontweight="bold" if bold else "normal", zorder=10)


# ---- LPBF animation ---------------------------------------------------------------------------
def lpbf_gif():
    rng = np.random.default_rng(3)
    px = rng.uniform(1.0, 15.0, 900); py = rng.uniform(1.75, 4.55, 900)
    x0, x1, lh, base = 5.0, 11.0, 0.42, 1.7
    n_scan, n_coat, layers = 26, 12, 3
    frames = []
    for L in range(layers):
        for k in range(n_scan):
            frames.append((L, "scan", k / (n_scan - 1)))
        for k in range(n_coat):
            frames.append((L, "coat", k / (n_coat - 1)))
    start = n_scan // 2
    frames = frames[start:] + frames[:start]
    imgs = []
    for L, mode, u in frames:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=110)
        ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis("off"); fig.subplots_adjust(0, 0, 1, 1)
        top = base + 3 * lh + L * lh + lh  # surface of the layer being scanned
        ax.add_patch(Rectangle((0.6, 1.0), 14.8, 0.7, color="#4A4F5A"))
        ax.scatter(px[py < top], py[py < top], s=16, color=POWDER, linewidths=0)
        ax.add_patch(Rectangle((x0, base), x1 - x0, top - lh - base, color=SOLID))
        y = top - lh
        if mode == "scan":
            xl = x0 + u * (x1 - x0)
            n = 40
            xs = np.linspace(x0, xl, n)
            for a, b in zip(xs[:-1], xs[1:]):
                age = (xl - a) / (x1 - x0)
                c = HEAT(max(0.0, 1 - 2.6 * age))
                c = tuple(np.array(c[:3]) * (1 - min(age * 2.2, 1)) + np.array(matplotlib.colors.to_rgb(SOLID)) * min(age * 2.2, 1))
                ax.add_patch(Rectangle((a, y), b - a + 0.02, lh, color=c))
            ax.add_patch(Polygon([[7.8, 9.0], [8.2, 9.0], [xl + 0.05, top], [xl - 0.05, top]], color=RED, alpha=0.75))
            ax.add_patch(Ellipse((xl, top - 0.2), 1.0, 0.75, color="#F6C453", zorder=5))
            ax.add_patch(Ellipse((xl, top - 0.2), 0.5, 0.42, color="#FFF4C2", zorder=6))
            note(ax, xl + 2.0, top + 1.25, "Melt pool", size=14, bold=True)
            arrow(ax, (xl + 1.3, top + 1.0), (xl + 0.35, top + 0.2), color=INK, lw=1.5, ms=12)
            rx = 1.3
        else:
            ax.add_patch(Rectangle((x0, y), x1 - x0, lh, color=SOLID))
            rx = 1.3 + u * 13.0
            m = (px < rx) & (py >= top) & (py < top + lh)
            ax.scatter(px[m], py[m], s=16, color=POWDER, linewidths=0)
        ax.add_patch(Polygon([[rx - 0.25, top + 1.2 + (0 if mode == "scan" else 0)], [rx + 0.25, top + 1.2], [rx + 0.1, top + 0.05], [rx - 0.1, top + 0.05]], color=TEAL))
        ax.add_patch(Rectangle((7.3, 9.0), 1.4, 0.6, color=DARK))
        note(ax, 9.0, 9.3, "Laser scanner", size=14, ha="left", bold=True)
        note(ax, 2.6, 2.3, "Powder", size=14, bold=True)
        note(ax, 8.0, 2.3, "Solid part", size=14, color="white", bold=True)
        note(ax, 8.0, 0.55, "Baseplate (heat sink)", size=14, bold=True)
        note(ax, max(rx, 1.5), top + 1.75, "Recoater", size=14, color=TEAL, bold=True)
        fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(48, dither=Image.Dither.NONE))
        plt.close(fig)
    imgs[0].save(OUT / "lpbf_anim.gif", save_all=True, append_images=imgs[1:], duration=90, loop=0, optimize=True)


# ---- motivation ---------------------------------------------------------------------------------
def overheat():
    fig, ax = canvas(6.4, 4.6, (0, 16), (0, 11.5))
    ax.add_patch(Rectangle((0.5, 0.6), 15, 1.0, color="#4A4F5A"))
    ax.add_patch(Rectangle((0.5, 1.6), 15, 8.2, color="#ECEEF1"))
    body = Polygon([[3, 1.6], [7, 1.6], [7, 6.2], [13.5, 8.0], [13.5, 9.8], [3, 9.8]], color=SOLID)
    ax.add_patch(body)
    hot = Ellipse((11.3, 8.8), 5.2, 2.6, color=RED, alpha=0.8); ax.add_patch(hot); hot.set_clip_path(body)
    for x in (4.2, 5.8):
        arrow(ax, (x, 8.6), (x, 2.2), color=TEAL, lw=3)
    arrow(ax, (11.2, 8.4), (11.2, 5.4), color=RED, lw=3, style="-[", ms=10)
    note(ax, 5.0, 10.6, "Heat escapes to baseplate", color=TEAL, bold=True)
    note(ax, 11.3, 10.6, "Heat is trapped", color=RED, bold=True)
    note(ax, 11.0, 3.5, "Loose powder\n(about 1% of bulk k)", color=MUTED)
    note(ax, 8, 0.15, "Baseplate", color=MUTED, size=13)
    save(fig, "fig_overheat.png")


def bvp():
    fig, ax = canvas(6.6, 4.6, (0, 15), (0, 10.4))
    ax.add_patch(Rectangle((1, 0.9), 13, 1.0, color="#4A4F5A"))
    ax.add_patch(Rectangle((4, 1.9), 7, 4.2, color=SOLID))
    ax.add_patch(Rectangle((4, 6.1), 7, 0.8, color=RED))
    for x in (5, 6.5, 8, 9.5):
        arrow(ax, (x, 8.6), (x, 7.0), color=RED)
    note(ax, 5.6, 9.5, "Heat flux Q on the new layer", color=RED, bold=True)
    arrow(ax, (10.6, 7.0), (11.9, 8.2), color=AMBER)
    note(ax, 12.9, 9.0, "Convection\n+ radiation", color=AMBER, bold=True)
    note(ax, 2.1, 4.1, "Insulated\nsides\n(powder)", color=TEAL, bold=True)
    note(ax, 12.9, 4.1, "Insulated\nsides\n(powder)", color=TEAL, bold=True)
    note(ax, 7.5, 0.4, "Baseplate: fixed T (heat sink)", color=MUTED, bold=True)
    note(ax, 7.5, 4.0, "Heat equation\nin the part", color="white", bold=True)
    note(ax, 7.5, 6.5, "new layer", color="white", size=13)
    save(fig, "fig_bvp.png")


def time_curves():
    P, v, r, a, b = 100.0, 20.0, 0.25, 6.0, 2.0
    n = sim.N - 2
    ab = sim._operator(a, b)
    u = np.zeros(sim.N); u[1:-1] = solve_banded((1, 1), ab, sim.source(5.0, P, r)[1:-1])
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(sim.X, sim.T0 + u, color=TEAL, lw=3)
    ax.set(xlabel="position x (mm)", ylabel="T (°C)", yticks=[]); ax.set_xticks([0, 5, 10])
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); save(fig, "fig_ss_curve.png")

    dt = 0.002
    A = sim._operator(a, b) * dt; A[1, :] += 1.0
    u = np.zeros(sim.N); ts, rec = [], []
    idx = [np.argmin(abs(sim.X - x)) for x in (3, 5, 7)]
    t_scan = 8 / v
    for k in range(int(0.9 / dt)):
        t = (k + 1) * dt
        s = sim.source(1 + v * t, P, r)[1:-1] if t <= t_scan else 0.0
        u[1:-1] = solve_banded((1, 1), A, u[1:-1] + dt * s)
        ts.append(t); rec.append(u[idx] + sim.T0)
    rec = np.array(rec)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for c, col in enumerate(("#E8590C", RED, "#7A0A1E")):
        ax.plot(ts, rec[:, c], color=col, lw=3)
    ax.set(xlabel="time t (s)", ylabel="T (°C)", yticks=[])
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); save(fig, "fig_tr_curves.png")


# ---- case 1 / 2 -----------------------------------------------------------------------------------
def rod():
    fig, ax = canvas(7.4, 2.7, (0, 15), (0, 5.5))
    grad = np.exp(-0.5 * ((np.linspace(0, 1, 300) - 0.62) / 0.05) ** 2) + 0.75 * (np.linspace(0, 1, 300) < 0.62) * np.exp(-(0.62 - np.linspace(0, 1, 300)) / 0.14)
    ax.imshow(grad[None], extent=(1.5, 13.5, 1.2, 2.2), cmap=HEAT, aspect="auto", vmin=0, vmax=1.2, zorder=2)
    ax.add_patch(Rectangle((1.5, 1.2), 12, 1.0, fill=False, ec=MUTED, lw=1.5, zorder=3))
    xl = 1.5 + 12 * 0.62
    ax.add_patch(Polygon([[xl - .2, 4.8], [xl + .2, 4.8], [xl + .03, 2.2], [xl - .03, 2.2]], color=RED))
    note(ax, xl, 5.15, "Laser: power P", color=RED, bold=True)
    arrow(ax, (xl + 0.5, 3.4), (xl + 2.6, 3.4), color=INK); note(ax, xl + 3.3, 3.4, "speed v", bold=True)
    note(ax, 0.7, 1.7, "65 °C", color=TEAL, bold=True); note(ax, 14.3, 1.7, "65 °C", color=TEAL, bold=True)
    note(ax, 7.5, 0.6, "10 mm rod, 101 points", color=MUTED)
    save(fig, "fig_rod.png")


def beads():
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.1))
    for ax in axs:
        ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((0.5, 0.7), 11, 1.5, color=SOLID))
    a, b = axs
    note(a, 6, 6.0, "Transient: moving laser, solved in time", color=RED, bold=True, size=12.5)
    n = 60
    for x0, x1 in zip(np.linspace(1.6, 8, n)[:-1], np.linspace(1.6, 8, n)[1:]):
        a.add_patch(Rectangle((x0, 2.2), x1 - x0 + .02, 0.9, color=HEAT(0.15 + 0.85 * ((x0 - 1.6) / 6.4) ** 3)))
    a.add_patch(Polygon([[7.8, 5.2], [8.2, 5.2], [8.03, 3.1], [7.97, 3.1]], color=RED))
    arrow(a, (8.5, 4.2), (10.4, 4.2), color=INK); note(a, 6, 0.25, "History matters", color=MUTED)
    note(b, 6, 6.0, "Steady: laser held at each position", color=TEAL, bold=True, size=12.5)
    b.add_patch(Rectangle((1.6, 2.2), 8.8, 0.9, color=SOLID))
    for x in (3, 6, 9):
        b.add_patch(Polygon([[x - .18, 5.2], [x + .18, 5.2], [x + .03, 3.1], [x - .03, 3.1]], color=TEAL, alpha=0.9))
        b.add_patch(Rectangle((x - .45, 2.2), .9, .9, color=RED))
    note(b, 6, 0.25, "No time: one solve per position", color=MUTED)
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.05); save(fig, "fig_beads.png")


def bead_fields():
    z = np.load(ROOT / "maxmap_outputs/cv_predictions.npz")
    names = list(z["name"])
    top = lambda f: f.reshape(3, 42, 41).max(0)[10:32] - 65.0
    i = names.index("P90_v10_hpnt15")
    vmax = z["true"][i].max() - 65
    fig, axs = plt.subplots(3, 1, figsize=(4.6, 5.0), layout="constrained")
    for ax, (lab, f) in zip(axs, (("Steady input", z["ss"][i]), ("True transient", z["true"][i]), ("Diffusion", z["pred"][i]))):
        im = ax.imshow(top(f), cmap="inferno", origin="lower", vmin=0, vmax=vmax); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{lab}: peak {f.max():,.0f} °C", fontsize=14)
    fig.colorbar(im, ax=axs, shrink=0.75, label="rise above 65 °C")
    save(fig, "fig_bead_fields.png")

    pair = [names.index("P60_v100_hpnt18"), names.index("P120_v10_hpnt1")]
    vmax = max(z["true"][p].max() for p in pair) - 65
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.2), layout="constrained")
    labs = ("P 60 W, v 100 mm/s, h 0.18", "P 120 W, v 10 mm/s, h 0.10")
    for c, p in enumerate(pair):
        for r, (k, lab) in enumerate((("ss", "Steady"), ("true", "Transient"))):
            im = axs[r, c].imshow(top(z[k][p]), cmap="inferno", origin="lower", vmin=0, vmax=vmax)
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            axs[r, c].set_title(f"{lab}: peak {z[k][p].max():,.0f} °C", fontsize=14)
            if r == 0:
                axs[r, c].set_title(f"{labs[c]}\n{lab}: peak {z[k][p].max():,.0f} °C", fontsize=14)
    fig.colorbar(im, ax=axs, shrink=0.75, label="rise above 65 °C")
    save(fig, "fig_ss_same.png")


# ---- physics --------------------------------------------------------------------------------------
def weld():
    fig, ax = canvas(8.0, 4.3, (0, 16), (0, 8.6))
    plate = Rectangle((0.6, 0.8), 14.8, 3.2, color=SOLID); ax.add_patch(plate)
    ax.add_patch(Rectangle((0.6, 4.0), 6.8, 0.8, color="#A4AAB5"))  # older bead
    ax.add_patch(Rectangle((7.4, 4.0), 1.5, 0.8, color=RED))  # new bead
    for r, c in zip((3.0, 2.3, 1.6), ("#F6C453", "#E8590C", RED)):
        w = Wedge((8.2, 4.0), r, 180, 360, color=c, alpha=0.75); ax.add_patch(w); w.set_clip_path(plate)
    ax.add_patch(Polygon([[9.0, 4.85], [9.45, 4.85], [12.0, 8.2], [11.5, 8.2]], color=DARK))
    note(ax, 14.0, 7.6, "Heat source", bold=True)
    note(ax, 11.8, 5.6, "Melt pool", color=RED, bold=True, ha="left")
    arrow(ax, (11.7, 5.5), (9.1, 3.7), color=RED, lw=1.8, ms=13)
    note(ax, 5.0, 6.7, "New bead heats older beads", color=RED, bold=True)
    arrow(ax, (7.5, 5.3), (4.0, 5.2), color=RED, rad=0.3)
    note(ax, 3.3, 2.3, "Isothermal\nlines", color="white", bold=True)
    arrow(ax, (4.5, 2.3), (5.45, 2.8), color="white", lw=1.8, ms=13)
    save(fig, "fig_weld.png")


def ss_local():
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.5))
    for ax in axs:
        ax.set_xlim(0, 14); ax.set_ylim(0, 7.2); ax.set_aspect("equal"); ax.axis("off")
    for ax, loc in zip(axs, (False, True)):
        plate = Rectangle((1, 0.7), 12, 3.8, color=POWDER if loc else "#F4C7C3"); ax.add_patch(plate)
        ax.add_patch(Rectangle((3.4, 4.5), 1.3, 0.9, color=SOLID)); ax.add_patch(Rectangle((4.7, 4.5), 1.3, 0.9, color=RED))
        if loc:
            w = Wedge((5.35, 4.5), 2.5, 180, 360, color=TEAL, alpha=0.75); ax.add_patch(w); w.set_clip_path(plate)
            note(ax, 7, 6.6, "Closer to reality: local domain only", color=TEAL, bold=True)
            note(ax, 10.3, 2.5, "Heat has not\nreached here", color=MUTED)
        else:
            for p in ((1.5, 1.1), (5.3, 1.0), (12.4, 1.1), (9.5, 1.1)):
                arrow(ax, (5.35, 4.3), p, color=RED, lw=2)
            note(ax, 7, 6.6, "Steady state: wait for infinite time", color=RED, bold=True)
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.04); save(fig, "fig_ss_local.png")


def lc():
    alpha, depth = 5.18, 8.0
    x = np.linspace(0, depth, 300)
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 4.5), sharey=True)
    for ax, t in zip(axs, (0.1, 1.0, 4.0)):
        L = np.sqrt(alpha * t)
        T = 20 + 80 * erfc(x / (2 * L))
        ax.imshow(T[:, None], extent=(0, 1, depth, 0), aspect="auto", cmap=HEAT, vmin=20, vmax=100)
        ax.axhline(L, color=INK, lw=2.2, ls="--")
        ax.text(0.5, L + 0.12, f"Lc = {L:.1f} mm", ha="center", va="top", fontsize=14, fontweight="bold")
        ax.set_title(f"t = {t:g} s", fontsize=15, fontweight="bold"); ax.set_xticks([])
        ax.tick_params(left=False)
    axs[0].set_ylabel("distance from the 100 °C end (mm)", fontsize=14)
    fig.tight_layout(); save(fig, "fig_lc.png")


def geometry():
    fig, ax = canvas(9.4, 3.9, (0, 18), (0, 7.5))
    block = Rectangle((0.6, 0.7), 16.8, 4.2, color=POWDER); ax.add_patch(block)
    R = 2.1
    for x in (3.4, 8.2, 13.8):
        ax.add_patch(Rectangle((x - .6, 4.9), 1.2, 0.8, color=RED))
        w = Wedge((x, 4.9), R, 180, 360, color=TEAL, alpha=0.8); ax.add_patch(w); w.set_clip_path(block)
    note(ax, 8.2, 6.7, "Every bead sees the same hemisphere", color=TEAL, bold=True, size=21)
    ax.plot([17.4, 17.4], [0.7, 4.9], color=INK, lw=3)
    note(ax, 13.8, 1.7, "End wall is outside\nthe domain", size=18)
    note(ax, 3.4, 1.7, "Same domain", size=18); note(ax, 8.2, 1.7, "Same domain", size=18)
    save(fig, "fig_same_geom.png")

    fig, ax = canvas(7.4, 5.0, (0, 16), (0, 10.8))
    block = Rectangle((1, 0.6), 14, 8.6, color=POWDER); ax.add_patch(block)
    R = 2.0
    pts = [(8.0, 6.9, "Above a hole:\ndomain is cut"), (2.0, 2.9, "At the wall:\ndomain is cut"), (12.2, 3.9, "In the bulk:\nfull domain")]
    for x, y, _ in pts:
        w = Wedge((x, y), R, 180, 360, color=TEAL, alpha=0.8); ax.add_patch(w); w.set_clip_path(block)
        ax.add_patch(Rectangle((x - .55, y), 1.1, .75, color=RED, zorder=6))
    ax.add_patch(Circle((8, 4.0), 1.9, color="white", ec=MUTED, lw=1.5, zorder=5))
    note(ax, 8, 4.0, "Hole", color=MUTED, bold=True, size=19)
    note(ax, 8.0, 9.9, "Above a hole: domain is cut", bold=True, size=19); note(ax, 3.6, 5.1, "At the wall:\ndomain is cut", bold=True, size=19)
    note(ax, 12.4, 5.9, "In the bulk:\nfull domain", bold=True, size=19)
    arrow(ax, (8, 9.55), (8, 7.75), color=INK, lw=1.6, ms=12)
    save(fig, "fig_diff_geom.png")


# ---- case 3 ---------------------------------------------------------------------------------------
def geo_data():
    im = Image.open(ROOT / "tts_outputs/fig3d_part.png"); w, h = im.size
    im.crop((int(.04 * w), int(.17 * h), int(.43 * w), int(.90 * h))).save(OUT / "fig_part3d.png")
    rows = list(csv.DictReader(open(ROOT / "tts_outputs/elements.csv")))
    def grid(layer, key):
        g = np.full((24, 24), np.nan)
        for r in rows:
            if int(r["layer"]) == layer:
                g[int(r["iy"]), int(r["ix"])] = float(r[key]) / 1000
        return g
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.2), layout="constrained")
    cm = matplotlib.colormaps["inferno"].copy(); cm.set_bad("white")
    for r, (key, lab) in enumerate((("ss_peak", "Steady"), ("tr_peak", "Transient"))):
        gs = [grid(l, key) for l in (6, 10)]
        lo = min(np.nanpercentile(g, 1) for g in gs); hi = max(np.nanpercentile(g, 99) for g in gs)
        for c, (g, l) in enumerate(zip(gs, (6, 10))):
            m = axs[r, c].imshow(g, cmap=cm, origin="lower", vmin=lo, vmax=hi); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            axs[r, c].set_title(f"{lab}, layer {l}", fontsize=19)
        cb = fig.colorbar(m, ax=axs[r, :], shrink=0.85); cb.set_label("peak (10³ °C)", fontsize=17); cb.ax.tick_params(labelsize=15)
    save(fig, "fig_geo_maps.png")


def title_art():
    rows = list(csv.DictReader(open(ROOT / "tts_outputs/elements.csv")))
    g = np.full((22, 22), np.nan)
    for r in rows:
        if int(r["layer"]) == 10:
            g[int(r["iy"]) - 1 if int(r["iy"]) > 0 else 0, int(r["ix"]) - 1 if int(r["ix"]) > 0 else 0] = float(r["ss_peak"])
    cm = matplotlib.colormaps["inferno"].copy(); cm.set_bad(DARK)
    fig, ax = plt.subplots(figsize=(5, 5)); ax.imshow(g, cmap=cm, origin="lower", vmin=np.nanpercentile(g, 2), vmax=np.nanpercentile(g, 98))
    ax.axis("off"); fig.subplots_adjust(0, 0, 1, 1); fig.savefig(OUT / "fig_title.png", dpi=200, facecolor=DARK); plt.close(fig)


if __name__ == "__main__":
    for f in (title_art, bvp, time_curves, rod, beads, bead_fields, weld, ss_local, lc, geometry, geo_data):
        TEXT_SCALE[0] = {"weld": 1.5, "ss_local": 1.45, "beads": 1.3}.get(f.__name__, 1.0)
        f(); print("done", f.__name__, flush=True)
