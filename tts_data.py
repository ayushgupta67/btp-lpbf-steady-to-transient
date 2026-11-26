"""Loader for the geometry dataset (Trans_to_SS_dataset): one part built element by element.

Layout:  Trans_to_SS_dataset/<part>/SS/my_temp_<k>.txt     steady state, one file per step
         Trans_to_SS_dataset/<part>/Trans/my_temp_<k>.txt  transient, several files per step
Each file: line 1 " total time is <t>", line 2: N temperatures (C), N = (e_x+1)(e_y+1)(e_z+snelz+1),
x fastest: reshape(nz, ny, nx).

Steps (from the generating scripts): remove all, add substrate, then for every layer each active
element is added and heated for t_on = layer_thickness / v (1 s here), followed by a cooling step of
recoater_time (5 s). SS: one steady solve per step, sink at 65 C on active nodes outside a hemisphere
of radius 1.5*sqrt(5.18 * t_on) around the new element. Transient: time-stepped, base held at 180 C.
Both runs end each element step on the same total time, which is how they are paired here.
Parsed parts are cached in tts_outputs/cache/.
"""

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE / "Trans_to_SS_dataset"
OUT = HERE / "tts_outputs"


def _read(path):
    head, data = Path(path).read_text().split("\n")[:2]
    return float(head.split()[-1]), np.array(data.split(), dtype=np.float32)


def _param(d, name, cast=float):
    p = Path(d) / f"{name}.txt"
    return cast(p.read_text().split()[0]) if p.exists() else None


def read_folder(d, workers=12):
    d = Path(d)
    files = {int(re.search(r"my_temp_(\d+)", f.name).group(1)): f for f in d.glob("my_temp_*.txt")}
    ids = sorted(files)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_read, [files[k] for k in ids], chunksize=64))
    return np.array([t for t, _ in res]), np.stack([v for _, v in res])


def geometry(d):
    d = Path(d)
    ex, ey, ez = (_param(d, k, int) for k in ("e_x", "e_y", "e_z"))
    sn = _param(d, "snelz", int) or 3
    vox = np.loadtxt(d / "voxels.txt").reshape(ez, ey, ex) > 0.5  # part elements, layer by layer (x fastest)
    return dict(ex=ex, ey=ey, ez=ez, snelz=sn, nx=ex + 1, ny=ey + 1, nz=ez + sn + 1, vox=vox,
                P=_param(d, "power"), v=_param(d, "velocity"), hatch=_param(d, "hatch"),
                layers=_param(d, "no_of_layers", int), height=_param(d, "part_height_in_mm"),
                recoater=_param(d, "recoater_time"))


def load_part(part="part_1", use_cache=True):
    OUT.joinpath("cache").mkdir(parents=True, exist_ok=True)
    c = OUT / "cache" / f"{part}.npz"
    pdir = ROOT / part
    if use_cache and c.exists():
        z = np.load(c)
        ss_t, ss_f, tr_t, tr_f = z["ss_t"], z["ss_f"], z["tr_t"], z["tr_f"]
    else:
        ss_t, ss_f = read_folder(pdir / "SS")
        tr_t, tr_f = read_folder(pdir / "Trans")
        np.savez(c, ss_t=ss_t, ss_f=ss_f, tr_t=tr_t, tr_f=tr_f)
    return geometry(pdir / "SS"), ss_t, ss_f, tr_t, tr_f
