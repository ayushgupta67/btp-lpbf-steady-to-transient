"""Transient -> steady field mapping: diffusion vs regression vs baselines.


1. The task is the real one. We map the transient field to the steady field of
   the wall dataset, instead of denoising a field while being handed a close
   relative of the answer as conditioning.

2. Splits keep whole unit-time blocks together. The old script never split at
   all; a random per-frame split leaks, because each frame's neighbours differ
   by an MAE of ~0.75 on a field whose sd is 6.7. baselines.py shows a ridge
   regression scoring 0.97 under a random split and 2.02 under a block split --
   the same model, a 2x difference, entirely from leakage.

3. Every number is reported next to the baselines, on the same cells and split.
   The old plot compared two diffusion variants to each other, so both could
   have been worse than "predict the per-cell mean" without it showing.

4. Metrics cover only the 1094 active cells. 2602 of 3696 cells are exactly
   65.0 in every frame; including them inflates any score for free.

5. The physics term is the steady-state residual, not a mangled heat equation.
   A steady field satisfies the Laplace equation away from sources, so the
   residual is just the 3D Laplacian of the prediction -- no time derivative,
   no diffusivity, consistent units. The old term summed dT/dt over the batch
   and all nodes, then compared it to a spatial mean, so it did not correspond
   to the heat equation in any form.

6. A plain MLP regressor is trained on the identical data and reported beside
   the diffusion model. Diffusion is only the right tool if it wins here.

    python training/diffusion_compare.py --epochs 300
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / d) for d in ("model", "simulators", "data_loaders", "training")]

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from baselines import fit_predictors
from wall_data import WallData, metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)


# --------------------------------------------------------------------------
# Normalisation. Statistics come from the training split only -- computing them
# over the whole dataset leaks the test distribution into the model.
# --------------------------------------------------------------------------
class Scaler:
    def __init__(self, arr):
        self.mean = arr.mean(0, keepdims=True)
        self.sd = arr.std(0, keepdims=True)
        self.sd[self.sd < 1e-6] = 1.0  # ~31% of active cells are near-constant

    def fwd(self, a):
        return (a - self.mean) / self.sd

    def inv(self, a):
        return a * self.sd + self.mean


# --------------------------------------------------------------------------
# Physics: steady state means the Laplacian vanishes away from sources.
# We evaluate it on the grid, so predictions have to be scattered back from the
# flat active-cell vector into the 8 x 22 x 21 volume.
# --------------------------------------------------------------------------
class LaplacianResidual(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.idx = torch.tensor(np.flatnonzero(mask.ravel()), dtype=torch.long)
        self.shape = mask.shape

    def forward(self, flat_field):
        """flat_field: (B, n_active) in physical units. Returns scalar residual."""
        b = flat_field.shape[0]
        vol = torch.zeros(b, int(np.prod(self.shape)), device=flat_field.device)
        vol[:, self.idx.to(flat_field.device)] = flat_field
        vol = vol.view(b, *self.shape)

        # Second difference along each axis, interior points only.
        lap = (
            (vol[:, 2:, 1:-1, 1:-1] - 2 * vol[:, 1:-1, 1:-1, 1:-1] + vol[:, :-2, 1:-1, 1:-1])
            + (vol[:, 1:-1, 2:, 1:-1] - 2 * vol[:, 1:-1, 1:-1, 1:-1] + vol[:, 1:-1, :-2, 1:-1])
            + (vol[:, 1:-1, 1:-1, 2:] - 2 * vol[:, 1:-1, 1:-1, 1:-1] + vol[:, 1:-1, 1:-1, :-2])
        )
        # Only penalise cells that are interior to the active region, so the
        # geometry boundary and the heat source are not forced to be harmonic.
        m = self.mask[1:-1, 1:-1, 1:-1]
        return ((lap * m) ** 2).sum() / (m.sum() * b)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, tau):
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, device=tau.device).float() / half
        )
        args = tau.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Regressor(nn.Module):
    """Deterministic transient -> steady map. The fair comparison for diffusion."""

    def __init__(self, d, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, cond, t_phys):
        return self.net(torch.cat([cond, t_phys], 1))


class DiffusionNet(nn.Module):
    """Predicts the noise added to the steady field, given transient + time."""

    def __init__(self, d, hidden=512, temb=64):
        super().__init__()
        self.embed = SinusoidalEmbedding(temb)
        self.emb_mlp = nn.Sequential(nn.Linear(temb, temb), nn.SiLU())
        self.net = nn.Sequential(
            nn.Linear(2 * d + 1 + temb, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, x_tau, tau, cond, t_phys):
        e = self.emb_mlp(self.embed(tau))
        return self.net(torch.cat([x_tau, cond, t_phys, e], 1))


# --------------------------------------------------------------------------
# Diffusion schedule
# --------------------------------------------------------------------------
STEPS = 400


def make_schedule(device):
    betas = torch.linspace(1e-4, 0.02, STEPS, device=device)
    return torch.cumprod(1.0 - betas, 0)


@torch.no_grad()
def ddim_sample(model, cond, t_phys, alpha_bar, n_steps=50, n_draws=8):
    """DDIM sampling. Deterministic per draw, so n_draws only varies the seed.

    We report both the mean over draws and the spread: a collapsed model gives a
    near-zero spread, which the old script's 10-sample average hid completely.
    """
    model.eval()
    b, d = cond.shape
    ts = torch.linspace(STEPS - 1, 0, n_steps, device=cond.device).long()
    draws = []
    for _ in range(n_draws):
        x = torch.randn(b, d, device=cond.device)
        for i, t_cur in enumerate(ts):
            tau = t_cur.repeat(b)
            eps = model(x, tau, cond, t_phys)
            ab = alpha_bar[t_cur]
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            if i < len(ts) - 1:
                ab_next = alpha_bar[ts[i + 1]]
                x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
            else:
                x = x0
        draws.append(x)
    return torch.stack(draws)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(model, X, Y, t, *, mode, epochs, phys_weight, lap, y_scaler, batch=32, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    alpha_bar = make_schedule(X.device)
    n = X.shape[0]
    t0 = time.time()

    for ep in range(epochs):
        perm = torch.randperm(n, device=X.device)
        tot = 0.0
        for k in range(0, n, batch):
            sel = perm[k:k + batch]
            cond, target, tp = X[sel], Y[sel], t[sel]

            if mode == "diffusion":
                tau = torch.randint(0, STEPS, (len(sel),), device=X.device)
                noise = torch.randn_like(target)
                ab = alpha_bar[tau][:, None]
                x_tau = ab.sqrt() * target + (1 - ab).sqrt() * noise
                eps = model(x_tau, tau, cond, tp)
                loss = nn.functional.mse_loss(eps, noise)
                # Reconstruct x0 for the physics term. Only meaningful at low
                # tau: at high tau, dividing by sqrt(alpha_bar) ~ 1e-3 amplifies
                # any error by ~300x, which is what made the old loss explode.
                if phys_weight > 0:
                    keep = tau < STEPS // 8
                    if keep.any():
                        x0 = (x_tau[keep] - (1 - ab[keep]).sqrt() * eps[keep]) / ab[keep].sqrt()
                        loss = loss + phys_weight * lap(y_scaler.inv_t(x0))
            else:
                pred = model(cond, tp)
                loss = nn.functional.mse_loss(pred, target)
                if phys_weight > 0:
                    loss = loss + phys_weight * lap(y_scaler.inv_t(pred))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(sel)
        sched.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"    epoch {ep+1:4d}/{epochs}  loss {tot/n:.5f}  ({time.time()-t0:.0f}s)")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--split", default="extrapolate", choices=["extrapolate", "interpolate", "random"])
    ap.add_argument("--phys-weight", type=float, default=1e-4)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--plot", default="comparison.png")
    args = ap.parse_args()

    data = WallData()
    tr, te = data.split(args.split)
    print(f"device {DEVICE}   split {args.split}   train {tr.sum()} / test {te.sum()} frames")
    print(f"active cells {data.n_active}   steady sd {data.Y.std():.2f}\n")

    xs, ys = Scaler(data.X[tr]), Scaler(data.Y[tr])
    tmax = data.t[tr].max()

    def to_t(a):
        return torch.tensor(a, dtype=torch.float32, device=DEVICE)

    Xtr, Ytr = to_t(xs.fwd(data.X[tr])), to_t(ys.fwd(data.Y[tr]))
    Xte = to_t(xs.fwd(data.X[te]))
    ttr, tte = to_t(data.t[tr] / tmax)[:, None], to_t(data.t[te] / tmax)[:, None]

    # Physics term needs physical units, so give the scaler a torch inverse.
    ys.inv_t = lambda a, m=to_t(ys.mean), s=to_t(ys.sd): a * s + m
    lap = LaplacianResidual(data.mask).to(DEVICE)

    results = {}

    # Baselines, from the identical split.
    for name, fn in fit_predictors(data.X[tr], data.Y[tr]).items():
        results[name] = metrics(fn(data.X[te]), data.Y[te])

    # Regressor, with and without the physics term.
    for tag, pw in [("regressor MLP", 0.0), ("regressor MLP + physics", args.phys_weight)]:
        print(f"  training {tag}")
        m = Regressor(data.n_active).to(DEVICE)
        train(m, Xtr, Ytr, ttr, mode="reg", epochs=args.epochs,
              phys_weight=pw, lap=lap, y_scaler=ys)
        with torch.no_grad():
            pred = ys.inv(m(Xte, tte).cpu().numpy())
        results[tag] = metrics(pred, data.Y[te])

    # Diffusion, with and without the physics term.
    diff_preds = {}
    alpha_bar = make_schedule(DEVICE)
    for tag, pw in [("diffusion", 0.0), ("diffusion + physics", args.phys_weight)]:
        print(f"  training {tag}")
        m = DiffusionNet(data.n_active).to(DEVICE)
        train(m, Xtr, Ytr, ttr, mode="diffusion", epochs=args.epochs,
              phys_weight=pw, lap=lap, y_scaler=ys)
        draws = ddim_sample(m, Xte, tte, alpha_bar, n_draws=args.draws).cpu().numpy()
        pred = ys.inv(draws.mean(0))
        results[tag] = metrics(pred, data.Y[te])
        # Spread across draws, in physical units. Near zero means the model
        # ignores its noise input and is really a regressor in disguise.
        results[tag]["draw_spread"] = float((draws.std(0) * ys.sd).mean())
        diff_preds[tag] = pred

    print(f"\n=== test metrics, {args.split} split, {data.n_active} active cells ===")
    print(f"{'predictor':<34}{'MAE':>9}{'RMSE':>9}{'max err':>10}{'spread':>9}")
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["MAE"]):
        sp = f"{m['draw_spread']:>9.3f}" if "draw_spread" in m else " " * 9
        print(f"{name:<34}{m['MAE']:>9.4f}{m['RMSE']:>9.4f}{m['max_err']:>10.2f}{sp}")

    best_base = min(m["MAE"] for n, m in results.items()
                    if n in ("global mean", "per-cell mean", "identity (copy transient)",
                             "per-cell affine") or n.startswith("ridge"))
    print(f"\nbest baseline MAE {best_base:.4f} -- a learned model is only "
          f"interesting below this.")

    plot(data, te, diff_preds, results, args)


def plot(data, te, diff_preds, results, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A vertical line up the middle of the wall: x=5, y=10, z=2..18.
    xi, yi, zi = np.where(data.mask)
    line = np.flatnonzero((xi == 5) & (yi == 10))
    order = np.argsort(zi[line])
    line, zvals = line[order], zi[line][order]
    frame = len(np.flatnonzero(te)) // 2  # a mid-test-set frame

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(zvals, data.Y[te][frame, line], "k-", lw=2.5, label="steady (truth)")
    ax[0].plot(zvals, data.X[te][frame, line], color="0.6", ls=":", lw=2,
               label=f"transient (input), MAE {results['identity (copy transient)']['MAE']:.2f}")
    cmean = data.Y[data.split(args.split)[0]].mean(0)
    ax[0].plot(zvals, cmean[line], ls="--", color="tab:orange",
               label=f"per-cell mean, MAE {results['per-cell mean']['MAE']:.2f}")
    for tag, style in [("diffusion", "tab:red"), ("diffusion + physics", "tab:blue")]:
        ax[0].plot(zvals, diff_preds[tag][frame, line], "-o", ms=4, color=style,
                   alpha=0.8, label=f"{tag}, MAE {results[tag]['MAE']:.2f}")
    ax[0].set(xlabel="z (cell index up the wall)", ylabel="temperature",
              title=f"Profile at x=5, y=10  ({args.split} split, test frame {frame})")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    names = [n for n, _ in sorted(results.items(), key=lambda kv: kv[1]["MAE"])]
    vals = [results[n]["MAE"] for n in names]
    colors = ["tab:red" if "diffusion" in n else "tab:green" if "regressor" in n else "0.7"
              for n in names]
    ax[1].barh(range(len(names)), vals, color=colors)
    ax[1].set_yticks(range(len(names)), names, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set(xlabel="test MAE (lower is better)", title="All predictors, same split and cells")
    ax[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=140)
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
