"""Conditional diffusion model: steady-state field + (P, v, hatch, t) -> transient field.

Shared by train_bead_diffusion.py and predict_bead.py so that a saved
checkpoint is always run through the same code that trained it.

Design notes
------------
* Time is an input. The task is "transient field at that time instant", so the
  same steady field and parameters map to a different field at t=2 and at t=3;
  without t the mapping would be one-to-many. We feed the actual time of the
  stored frame, not the rounded whole number.
* Temperatures are modelled as z = asinh((T - 65) / 10) / 4. Excess over the
  65 base temperature spans 0 .. ~6000 C and the fields are mostly exactly 65,
  so a plain linear scaling would squash everything after t=2 into noise.
  asinh is linear below ~10 C, logarithmic above, and handles the small
  negative undershoots the solver produces.
* The network predicts the clean field (x0) rather than the noise. With fields
  that are mostly exactly 65 and a strong conditioning input, noise prediction
  has to learn a time-dependent identity pass-through and sat at loss ~0.96
  (1.0 = no information) after 600 epochs; sampling then blew up. Predicting x0
  makes the conditioning the direct path to the answer.
* Air nodes (constant 65.0 in every frame of every case) are removed before the
  model sees anything: n_nodes is the number of active nodes, not 5166.
* DDPM noising with 400 steps, DDIM sampling in 50 steps.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / d) for d in ("model", "simulators", "data_loaders", "training")]

import math

import numpy as np
import torch
import torch.nn as nn

from bead_data import BASE_T

STEPS = 400
ASINH_KNEE = 10.0  # C
Z_DIV = 4.0
PARAM_SCALE = np.array([120.0, 100.0, 0.18])  # P, v, hatch -> roughly [0, 1]
T_OFFSET, T_SPAN = 2.0, 5.0  # time runs 2 .. 7


def to_z(temp):
    return np.arcsinh((np.asarray(temp) - BASE_T) / ASINH_KNEE) / Z_DIV


def from_z(z):
    return BASE_T + ASINH_KNEE * np.sinh(np.clip(np.asarray(z) * Z_DIV, -8.0, 8.0))


def cond_vector(steady, params, t):
    """steady (n, n_active) in C, params (n, 3) = P, v, hatch, t (n,) -> (n, n_active + 4)."""
    scaled = np.column_stack([np.asarray(params) / PARAM_SCALE,
                              (np.asarray(t) - T_OFFSET) / T_SPAN])
    return np.concatenate([to_z(steady), scaled], axis=1).astype(np.float32)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, tau):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=tau.device).float() / half)
        args = tau.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class BeadDiffusion(nn.Module):
    """Predicts the clean transient field (in z space) from a noisy one plus conditioning."""

    def __init__(self, n_nodes, hidden=512, code=128, temb=64):
        super().__init__()
        self.n_nodes = n_nodes
        self.config = dict(n_nodes=n_nodes, hidden=hidden, code=code, temb=temb)
        self.embed = SinusoidalEmbedding(temb)
        self.emb_mlp = nn.Sequential(nn.Linear(temb, temb), nn.SiLU())
        # The steady field is n_nodes numbers but P, v, hatch, t are 4; compress
        # the field first, then also pass the 4 scalars straight to the main
        # network so they cannot be drowned out.
        self.cond_enc = nn.Sequential(nn.Linear(n_nodes + 4, 256), nn.SiLU(), nn.Linear(256, code))
        self.net = nn.Sequential(
            nn.Linear(n_nodes + code + 4 + temb, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_nodes),
        )

    def forward(self, x_tau, tau, cond):
        feats = torch.cat([x_tau, self.cond_enc(cond), cond[:, -4:],
                           self.emb_mlp(self.embed(tau))], dim=1)
        return self.net(feats)


def alpha_bar_schedule(device="cpu"):
    betas = torch.linspace(1e-4, 0.02, STEPS, device=device)
    return torch.cumprod(1.0 - betas, 0)


def train(model, cond, target_z, *, epochs, batch=16, lr=1e-3, device="cpu", log=None):
    """cond (n, n_nodes + 4) and target_z (n, n_nodes) are float32 numpy. Returns the loss curve."""
    cond = torch.tensor(cond, device=device)
    target = torch.tensor(target_z, dtype=torch.float32, device=device)
    ab = alpha_bar_schedule(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n, curve = len(cond), []
    model.train()
    for ep in range(epochs):
        perm, total = torch.randperm(n, device=device), 0.0
        for k in range(0, n, batch):
            sel = perm[k:k + batch]
            tau = torch.randint(0, STEPS, (len(sel),), device=device)
            noise = torch.randn_like(target[sel])
            a = ab[tau][:, None]
            x_tau = a.sqrt() * target[sel] + (1 - a).sqrt() * noise
            loss = nn.functional.mse_loss(model(x_tau, tau, cond[sel]), target[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(sel)
        sched.step()
        curve.append(total / n)
        if log and ((ep + 1) % log == 0 or ep == 0):
            print(f"      epoch {ep + 1:4d}/{epochs}  loss {curve[-1]:.5f}")
    return curve


@torch.no_grad()
def sample(model, cond, *, z_clip, n_steps=50, n_draws=8, device="cpu", seed=0):
    """DDIM sampling. Returns temperatures in C, shape (n_draws, n, n_nodes).

    z_clip = (lo, hi): range of z seen in training, padded. The predicted clean
    field is clamped to it at every step so an off-distribution prediction
    cannot run away through sinh().
    """
    model.eval()
    cond = torch.tensor(cond, device=device)
    ab = alpha_bar_schedule(device)
    ts = torch.linspace(STEPS - 1, 0, n_steps, device=device).long()
    gen = torch.Generator(device=device).manual_seed(seed)
    draws = []
    for _ in range(n_draws):
        x = torch.randn(len(cond), model.n_nodes, device=device, generator=gen)
        for i, t_cur in enumerate(ts):
            x0 = model(x, t_cur.repeat(len(cond)), cond).clamp(*z_clip)
            if i < len(ts) - 1:
                eps = (x - ab[t_cur].sqrt() * x0) / (1 - ab[t_cur]).sqrt()
                a_next = ab[ts[i + 1]]
                x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
            else:
                x = x0
        draws.append(from_z(x.cpu().numpy()))
    return np.stack(draws)


def save_checkpoint(path, model, meta):
    torch.save({"state_dict": model.state_dict(), "config": model.config, "meta": meta}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = BeadDiffusion(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["meta"]
