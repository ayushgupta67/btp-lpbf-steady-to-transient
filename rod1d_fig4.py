"""Figure for the meet-9 diffusion slide: training (forward + one-jump guess) and prediction (50-step backward chain).
Uses the saved 1D model on test run 12.   python rod1d_fig4.py"""
import matplotlib
import numpy as np
import torch
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rod1d_sim as sim
from maxmap_model import Model, alpha_bar, from_z, to_z

m, _ = Model.load("rod1d_outputs/rod1d_diffusion.pt", "cpu")
m.scalar_fn = lambda p: np.log(np.asarray(p, dtype=np.float64))
z = np.load("rod1d_outputs/rod1d_data.npz"); i = 12
ss, tr, sc = z["ss_test"][i:i + 1], z["tr_test"][i], np.array([[z["test_P"][i], z["test_v"][i]]])
xn = sim.X[1:-1]; y0 = to_z(tr[1:-1], m.cfg); ab = alpha_bar(m.cfg, "cpu")
fz, s = m._inputs(ss, sc)
eps = np.random.default_rng(0).standard_normal(y0.shape)
t = 200; noisy = np.sqrt(ab[t].item()) * y0 + np.sqrt(1 - ab[t].item()) * eps
with torch.no_grad():
    guess = m.net(torch.tensor(noisy[None], dtype=torch.float32), torch.tensor([t]), fz, s).numpy()[0]
ts = torch.linspace(399, 0, 50).long()
x = torch.randn(1, len(m.active), generator=torch.Generator().manual_seed(1)); snaps = {0: x.numpy()[0].copy()}
with torch.no_grad():
    for k, tt in enumerate(ts):
        x0 = m.net(x, tt.repeat(1), fz, s).clamp(*m.z_clip)
        if k < 49:
            e = (x - ab[tt].sqrt() * x0) / (1 - ab[tt]).sqrt(); x = ab[ts[k + 1]].sqrt() * x0 + (1 - ab[ts[k + 1]]).sqrt() * e
        else:
            x = x0
        if k + 1 in (30, 45, 50): snaps[k + 1] = x.numpy()[0].copy()
PINK, DARK, TEAL, RED = "#E8A0AE", "#15161B", "#1F7A8C", "#C8102E"
fig = plt.figure(figsize=(13, 5.8))
W, H, y_top, y_bot = 0.17, 0.28, 0.57, 0.08


def axes(x, y, title):
    ax = fig.add_axes([x, y, W, H]); ax.set_ylim(-3.6, 2.8); ax.set_xticks([0, 5, 10]); ax.tick_params(labelsize=8.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=10.5); return ax


def arrow(x0, x1, y, label, col):
    fig.patches.append(matplotlib.patches.FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, lw=2, color=col))
    if label:
        fig.text((x0 + x1) / 2, y + 0.04, label, ha="center", va="bottom", fontsize=9.5, color=col, fontweight="bold")


xt = [0.05, 0.40, 0.75]
a = axes(xt[0], y_top, "known transient curve"); a.plot(xn, y0, color=RED, lw=2)
a = axes(xt[1], y_top, "noisy curve (level 200 of 400)"); a.plot(xn, y0, color=PINK, lw=2); a.plot(xn, noisy, color=DARK, lw=1)
a = axes(xt[2], y_top, "network's guess vs truth"); a.plot(xn, y0, color=PINK, lw=4); a.plot(xn, guess, color=TEAL, lw=1.4)
arrow(xt[0] + W + 0.01, xt[1] - 0.01, y_top + H / 2, "forward:\nadd noise", RED)
arrow(xt[1] + W + 0.01, xt[2] - 0.01, y_top + H / 2, "backward, one jump:\nguess the clean curve", TEAL)
xb = [0.05, 0.29, 0.53, 0.77]
for k, (x_, key, ttl) in enumerate(zip(xb, (0, 30, 45, 50), ["start: pure noise", "after 30 of the 50 steps", "after 45 of the 50 steps", "after 50 steps = prediction"])):
    ax = axes(x_, y_bot, ttl); ax.plot(xn, y0, color=PINK, lw=2.5); ax.plot(xn, snaps[key], color=TEAL, lw=1.3); ax.set_xlabel("position (mm)", fontsize=9)
    if k < 3:
        arrow(x_ + W + 0.005, xb[k + 1] - 0.005, y_bot + H / 2, "", TEAL)
fig.text(0.01, 0.93, "TRAINING  (true curve known)", fontsize=12, fontweight="bold", color=RED)
fig.text(0.01, 0.445, "PREDICTION  (true curve unknown): the same one-jump guess, repeated 50 times, keeping a little less noise each time", fontsize=12, fontweight="bold", color=TEAL)
fig.savefig("rod1d_outputs/fig4_diffusion.png", dpi=200)
print("one-jump guess peak", round(float(from_z(guess, m.cfg).max())), "C; true", round(float(tr.max())), "C")
