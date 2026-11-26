"""Data layer for the max-temperature mapping (sir's instruction, Oct 2026).

Task: for each run (one bead, one setting P, v, hatch)
    input  = maximum steady-state temperature at every node   (SS_new, max over its frames)
    output = maximum transient temperature at every node      (max over all transient frames)
so each run gives ONE training pair (not 6 whole-number frames).

Designed to grow: point TRANSIENT_ROOT / SS_ROOT (or the --transient-root / --ss-root flags of
the scripts) at any folder tree. Every directory that contains my_temp_<k>.txt files is one run;
its settings come from power.txt, velocity.txt and hatch.txt inside it (never from the name).
Runs are matched transient <-> steady by (P, v, hatch). Each parsed run is cached on its own in
maxmap_outputs/cache/, keyed by the folder's file count and newest modification time, so adding
new runs only parses the new folders.

File format of my_temp_<k>.txt: line 1 " total time is <t>", line 2: N_NODES temperatures (C).
"""

import hashlib
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
TRANSIENT_ROOT = HERE / "Updated_complete_dataset_singlebead" / "Single_bead_Data_set" / "Transient"
SS_ROOT = HERE / "SS_new" / "dataset"
OUT = HERE / "maxmap_outputs"
CACHE = OUT / "cache"

N_NODES = 5166
NX, NY, NZ = 41, 42, 3  # x fastest: reshape(NZ, NY, NX)
BASE_T = 65.0

# Transient runs that must never be trained or tested on (checked, see maxmap_analyze.py).
EXCLUDED_TRANSIENT = {
    "P60_v50_hptn1": "bit-for-bit copy of P60_v10_hpnt1 (v=10 time stamps) although its files say v=50",
}


@dataclass
class Run:
    kind: str  # "transient" or "steady"
    name: str  # folder path relative to the root
    P: float
    v: float
    hatch: float
    times: np.ndarray
    fields: np.ndarray  # (n_frames, N_NODES), float32
    frame_ids: np.ndarray = field(default=None)

    @property
    def key(self):
        return (self.P, self.v, round(self.hatch, 4))

    @property
    def q(self):  # line energy proxy: the heat source scales with P / hatch
        return self.P / self.hatch


def _read_frame(path):
    header, data = Path(path).read_text().split("\n")[:2]
    vals = np.array(data.split(), dtype=np.float32)
    return float(header.split()[-1]), vals


def _param(d, name):
    for p in (d / f"{name}.txt", d.parent / f"{name}.txt"):
        if p.exists():
            return float(p.read_text().split()[0])
    return float("nan")


def _parse_dir(args):
    d, kind, root = args
    d = Path(d)
    files = {int(re.search(r"my_temp_(\d+)", f.name).group(1)): f for f in d.glob("my_temp_*.txt")}
    ids = np.array(sorted(files))
    tf = [_read_frame(files[k]) for k in ids]
    times = np.array([t for t, _ in tf])
    fields = np.stack([v for _, v in tf])
    if fields.shape[1] != N_NODES:
        raise ValueError(f"{d}: {fields.shape[1]} values per frame, expected {N_NODES}")
    return dict(kind=kind, name=str(d.relative_to(root)), P=_param(d, "power"), v=_param(d, "velocity"),
                hatch=_param(d, "hatch"), times=times, fields=fields, frame_ids=ids)


def _signature(d):
    files = list(d.glob("my_temp_*.txt"))
    mt = max(f.stat().st_mtime for f in files)
    return hashlib.md5(f"{d}|{len(files)}|{mt:.0f}".encode()).hexdigest()[:16]


def find_run_dirs(root):
    return sorted({p.parent for p in Path(root).rglob("my_temp_*.txt")})


def load_runs(root, kind, workers=8, use_cache=True):
    """All runs under `root`. Parsed in parallel; each run cached separately."""
    root = Path(root)
    CACHE.mkdir(parents=True, exist_ok=True)
    dirs = find_run_dirs(root)
    out, todo = {}, []
    for d in dirs:
        c = CACHE / f"{kind}_{_signature(d)}.npz"
        if use_cache and c.exists():
            z = np.load(c, allow_pickle=False)
            out[d] = dict(kind=kind, name=str(z["name"]), P=float(z["P"]), v=float(z["v"]), hatch=float(z["hatch"]),
                          times=z["times"], fields=z["fields"], frame_ids=z["frame_ids"])
        else:
            todo.append(d)
    if todo:
        with ProcessPoolExecutor(workers) as ex:
            for d, r in zip(todo, ex.map(_parse_dir, [(d, kind, root) for d in todo])):
                out[d] = r
                np.savez(CACHE / f"{kind}_{_signature(d)}.npz", **{k: np.asarray(v) for k, v in r.items() if k != "kind"})
    return [Run(**out[d]) for d in dirs]


def max_field(run):
    """Per-node maximum over all frames of the run, and the frame time at which it occurs."""
    k = run.fields.argmax(0)
    return run.fields.max(0), run.times[k]


def build_pairs(transient, steady, exclude=EXCLUDED_TRANSIENT):
    """One pair per transient run that has a steady run with the same (P, v, hatch)."""
    ss = {}
    for r in steady:
        ss.setdefault(r.key, r)  # first run per setting (SS_new has exactly one)
    rows = dict(name=[], params=[], ss_max=[], tr_max=[], ss_name=[])
    skipped = []
    for r in transient:
        top = r.name.split("/")[0]  # some runs keep their files in an AM_wall/ subfolder
        if top in exclude:
            skipped.append((top, exclude[top]))
            continue
        s = ss.get(r.key)
        if s is None:
            skipped.append((r.name, "no steady run with the same P, v, hatch"))
            continue
        rows["name"].append(top)
        rows["ss_name"].append(s.name)
        rows["params"].append(r.key)
        rows["ss_max"].append(max_field(s)[0])
        rows["tr_max"].append(max_field(r)[0])
    pairs = {k: (np.array(v, dtype=np.float32) if k in ("params", "ss_max", "tr_max") else v) for k, v in rows.items()}
    return pairs, skipped


def group_of(params):
    """Runs with the same P/hatch and v are physically identical (checked); keep them together in CV."""
    P, v, h = params
    return (round(float(P) / float(h), 1), float(v))
