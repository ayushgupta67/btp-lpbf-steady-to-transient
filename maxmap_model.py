"""Conditional diffusion model for field -> field mapping with scalar conditions.

Generic on purpose, so more data or bigger meshes need no code change:
  * n_nodes, n_scalars and layer widths come from the data / Config, not constants;
  * temperatures use a fixed asinh transform, scalars are standardised with statistics
    computed on the training set and saved inside the checkpoint;
  * training streams minibatches through a torch DataLoader and uses the GPU when present;
  * the checkpoint stores config + scalers + active-node indices, so predict needs nothing else.

Used here as: max steady-state field (+ P, v, hatch) -> max transient field.
Network: x0-prediction denoiser (MLP). Condition = encoder(field) + scalars; noise level via a
sinusoidal code. DDPM noising in `steps` steps, DDIM sampling in `ddim_steps`.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

BASE_T = 65.0


@dataclass
class Config:
    n_nodes: int
    n_scalars: int
    hidden: int = 512
    depth: int = 2  # hidden layers in the denoiser
    code: int = 128  # size of the encoded condition field
    temb: int = 64  # noise-level code
    steps: int = 400
    knee: float = 10.0  # asinh knee in C
    zdiv: float = 4.0


def to_z(T, cfg):
    return np.arcsinh((np.asarray(T, dtype=np.float64) - BASE_T) / cfg.knee) / cfg.zdiv


def from_z(z, cfg):
    return BASE_T + cfg.knee * np.sinh(np.clip(np.asarray(z) * cfg.zdiv, -9.0, 9.0))


def scalar_features(params):
    """P, v, hatch -> ln P, ln v, ln hatch, ln(P/hatch). Log because effects are multiplicative."""
    p = np.asarray(params, dtype=np.float64)
    return np.column_stack([np.log(p[:, 0]), np.log(p[:, 1]), np.log(p[:, 2]), np.log(p[:, 0] / p[:, 2])])


class Embed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, tau):
        half = self.dim // 2
        f = torch.exp(-math.log(10000.0) * torch.arange(half, device=tau.device).float() / half)
        a = tau.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], 1)


class Denoiser(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Sequential(Embed(cfg.temb), nn.Linear(cfg.temb, cfg.temb), nn.SiLU())
        self.cond = nn.Sequential(nn.Linear(cfg.n_nodes + cfg.n_scalars, 256), nn.SiLU(), nn.Linear(256, cfg.code))
        layers, w = [], cfg.n_nodes + cfg.code + cfg.n_scalars + cfg.temb
        for _ in range(cfg.depth):
            layers += [nn.Linear(w, cfg.hidden), nn.SiLU()]
            w = cfg.hidden
        layers.append(nn.Linear(w, cfg.n_nodes))
        self.net = nn.Sequential(*layers)

    def forward(self, x, tau, field_c, scal_c):
        c = torch.cat([field_c, scal_c], 1)
        return self.net(torch.cat([x, self.cond(c), scal_c, self.emb(tau)], 1))


def alpha_bar(cfg, device):
    return torch.cumprod(1 - torch.linspace(1e-4, 0.02, cfg.steps, device=device), 0)


class Model:
    """Wraps the network with its scalers so callers work in deg C and raw (P, v, hatch)."""

    def __init__(self, cfg: Config, device=None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Denoiser(cfg).to(self.device)
        self.s_mean = np.zeros(cfg.n_scalars)
        self.s_std = np.ones(cfg.n_scalars)
        self.active = None  # indices of the modelled nodes in the full field
        self.n_full = None
        self.z_clip = (-3.0, 3.0)
        self.scalar_fn = scalar_features  # raw scalar parameters -> features; replace for other datasets

    # ---- data -> tensors ----------------------------------------------------------------
    def _inputs(self, field_in, params):
        fz = to_z(np.asarray(field_in)[:, self.active], self.cfg)
        s = (self.scalar_fn(params) - self.s_mean) / self.s_std
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)
        return t(fz), t(s)

    def fit(self, field_in, params, field_out, active, *, epochs=2500, batch=8, lr=1e-3, seed=0, log_every=0):
        """field_in / field_out: (n, n_full) in C. active: indices of nodes to model."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.active, self.n_full = np.asarray(active), field_in.shape[1]
        f = self.scalar_fn(params)
        self.s_mean, self.s_std = f.mean(0), f.std(0) + 1e-8
        fz, s = self._inputs(field_in, params)
        y = torch.tensor(to_z(np.asarray(field_out)[:, self.active], self.cfg), dtype=torch.float32, device=self.device)
        self.z_clip = (float(y.min()) - 0.2, float(y.max()) + 0.2)
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(fz, s, y), batch_size=batch, shuffle=True)
        ab = alpha_bar(self.cfg, self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        curve = []
        self.net.train()
        for ep in range(epochs):
            tot = 0.0
            for fb, sb, yb in dl:
                tau = torch.randint(0, self.cfg.steps, (len(yb),), device=self.device)
                a = ab[tau][:, None]
                x = a.sqrt() * yb + (1 - a).sqrt() * torch.randn_like(yb)
                loss = nn.functional.mse_loss(self.net(x, tau, fb, sb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(yb)
            sched.step()
            curve.append(tot / len(y))
            if log_every and ((ep + 1) % log_every == 0 or ep == 0):
                print(f"    epoch {ep + 1:5d}/{epochs}  loss {curve[-1]:.6f}", flush=True)
        return curve

    @torch.no_grad()
    def predict(self, field_in, params, *, ddim_steps=50, draws=8, seed=0):
        """Mean of `draws` DDIM samples, full field in C, shape (n, n_full). Also returns the draw spread."""
        self.net.eval()
        fz, s = self._inputs(field_in, params)
        ab = alpha_bar(self.cfg, self.device)
        ts = torch.linspace(self.cfg.steps - 1, 0, ddim_steps, device=self.device).long()
        g = torch.Generator(device=self.device).manual_seed(seed)
        outs = []
        for _ in range(draws):
            x = torch.randn(len(fz), len(self.active), device=self.device, generator=g)
            for i, t in enumerate(ts):
                x0 = self.net(x, t.repeat(len(fz)), fz, s).clamp(*self.z_clip)
                if i < len(ts) - 1:
                    eps = (x - ab[t].sqrt() * x0) / (1 - ab[t]).sqrt()
                    x = ab[ts[i + 1]].sqrt() * x0 + (1 - ab[ts[i + 1]]).sqrt() * eps
                else:
                    x = x0
            outs.append(from_z(x.cpu().numpy(), self.cfg))
        d = np.stack(outs)
        full = np.full((d.shape[1], self.n_full), BASE_T)
        full[:, self.active] = d.mean(0)
        return full, float(d.std(0).mean())

    def save(self, path, meta=None):
        torch.save({"config": asdict(self.cfg), "state": self.net.state_dict(), "s_mean": self.s_mean, "s_std": self.s_std,
                    "active": self.active, "n_full": self.n_full, "z_clip": self.z_clip, "meta": meta or {}}, path)

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(Config(**ck["config"]), device)
        m.net.load_state_dict(ck["state"])
        m.s_mean, m.s_std, m.active, m.n_full, m.z_clip = ck["s_mean"], ck["s_std"], ck["active"], ck["n_full"], tuple(ck["z_clip"])
        return m, ck["meta"]

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())
