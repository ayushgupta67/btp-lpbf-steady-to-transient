"""Loader for the UPDATED single-bead dataset (Updated_complete_dataset_singlebead).

Layout on disk
--------------
Single_bead_Data_set/<Transient|Steady_state>/<case>/   (a case may keep its files
one level deeper, in AM_wall/, or even in both places)
    my_temp_<k>.txt   line 0: " total time is  <t>",  line 1: 5166 floats
    power.txt, velocity.txt, hatch.txt   the run's parameters (units W, mm/s, mm)

5166 = 41 x 42 x 3 nodes (flat vector is x-fastest: reshape(3, 42, 41)).

Every directory that holds my_temp_*.txt files is one "group". A case folder normally
has one group; if it has two (as Steady_state/P60_v10_hpnt15 does) both are loaded and
the later one is named "<case>/<subdir>" so nothing is silently dropped. Folders with
no frames at all (Steady_state/P60_v100_hpnt18 at the time of writing) are reported in
`MISSING`, not loaded.

`load_cases()` caches the parsed frames in results/case2_snapshot_task/cache/frames.npz.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / d) for d in ("model", "simulators", "data_loaders", "training")]

import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

ROOT = REPO / "data" / "Updated_complete_dataset_singlebead" / "Single_bead_Data_set"
CACHE = REPO / "results" / "case2_snapshot_task" / "cache" / "frames.npz"

N_NODES = 5166
NX, NY, NZ = 41, 42, 3
BASE_T = 65.0  # every untouched node reads exactly this
WHOLE_TIMES = (2, 3, 4, 5, 6, 7)

MISSING = []  # (kind, folder, reason), filled by load_cases


@dataclass
class Case:
    kind: str  # "transient" or "steady"
    name: str  # folder name, plus "/<subdir>" for a second group
    folder: str  # top-level folder name
    P: float
    v: float
    hatch: float
    times: np.ndarray  # (n_frames,)
    fields: np.ndarray  # (n_frames, N_NODES)
    frame_ids: np.ndarray  # my_temp_<k> numbers, in time order

    @property
    def key(self):
        return (self.P, self.v, self.hatch)

    @property
    def label_matches_params(self):
        m = re.match(r"P(\d+)_v(\d+)_h(?:pnt|ptn)(\d+)", self.folder)
        want = (float(m.group(1)), float(m.group(2)), float("0." + m.group(3)))
        return want == self.key


def _param(group_dir, case_dir, name):
    for d in (group_dir, case_dir):
        p = d / f"{name}.txt"
        if p.exists():
            return float(p.read_text().split()[0])
    hits = list(case_dir.rglob(f"{name}.txt"))
    return float(hits[0].read_text().split()[0]) if hits else float("nan")


def _read_one(path):
    header, data = Path(path).read_text().split("\n")[:2]
    vals = np.array(data.split(), dtype=np.float64)
    return float(header.split()[-1]), vals


def _read_group(group_dir):
    files = {int(re.search(r"my_temp_(\d+)", f.name).group(1)): f for f in group_dir.glob("my_temp_*.txt")}
    ids = sorted(files)
    times, fields = [], []
    for k in ids:
        t, v = _read_one(files[k])
        assert v.size == N_NODES, f"{files[k]}: {v.size} values"
        times.append(t)
        fields.append(v)
    return np.array(ids), np.array(times), np.array(fields)


def _groups(case_dir):
    """Directories under case_dir (including itself) that hold my_temp_*.txt, shallowest first."""
    dirs = {case_dir} | {p.parent for p in case_dir.rglob("my_temp_*.txt")}
    return sorted((d for d in dirs if any(d.glob("my_temp_*.txt"))), key=lambda d: (len(d.parts), str(d)))


def load_cases(root=ROOT, use_cache=True):
    """Return {"transient": [Case...], "steady": [Case...]}, sorted by name."""
    MISSING.clear()
    cached = np.load(CACHE) if use_cache and CACHE.exists() else None
    out, new = {}, {}
    for kind, sub in (("transient", "Transient"), ("steady", "Steady_state")):
        out[kind] = []
        for case_dir in sorted(p for p in (root / sub).iterdir() if p.is_dir()):
            groups = _groups(case_dir)
            if not groups:
                MISSING.append((kind, case_dir.name, "no my_temp files"))
                continue
            for gi, g in enumerate(groups):
                name = case_dir.name if gi == 0 else f"{case_dir.name}/{g.relative_to(case_dir)}"
                key = f"{kind}/{name}"
                if cached is not None and key + "/F" in cached:
                    ids, times, fields = cached[key + "/I"], cached[key + "/T"], cached[key + "/F"].astype(np.float64)
                else:
                    ids, times, fields = _read_group(g)
                    new[key] = (ids, times, fields)
                out[kind].append(Case(kind, name, case_dir.name, _param(g, case_dir, "power"),
                                      _param(g, case_dir, "velocity"), _param(g, case_dir, "hatch"),
                                      times, fields, ids))
    if use_cache and (cached is None or new):
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **{
            f"{c.kind}/{c.name}/{part}": arr
            for cs in out.values() for c in cs
            for part, arr in (("T", c.times), ("F", c.fields.astype(np.float32)), ("I", c.frame_ids))})
    return out


def active_node_mask(cases, tol=1e-6):
    """Boolean (N_NODES,): False for air nodes that never leave their start temperature."""
    lo = np.full(N_NODES, np.inf)
    hi = np.full(N_NODES, -np.inf)
    for cs in cases.values():
        for c in cs:
            lo = np.minimum(lo, c.fields.min(0))
            hi = np.maximum(hi, c.fields.max(0))
    return (hi - lo) > tol


def expand(active_values, mask):
    full = np.full(active_values.shape[:-1] + (len(mask),), BASE_T)
    full[..., mask] = active_values
    return full


def whole_number_frames(case, targets=WHOLE_TIMES):
    """For each target time, the nearest stored frame: [(target, frame, t_actual)]."""
    near = np.abs(case.times[:, None] - np.asarray(targets)[None, :]).argmin(0)
    return [(t, int(k), float(case.times[k])) for t, k in zip(targets, near)]


# ---- what is usable --------------------------------------------------------------------
# Checked in bead2_analyze.py: this transient folder is a bit-for-bit copy of P60_v10_hpnt1
# (same v=10 time stamps) although its parameter files say v=50. Keep it in the CSVs,
# flagged, but never train or test on it. P90_v50_hpnt15 is the genuine q=600, v=50 run.
EXCLUDED = {
    "P60_v50_hptn1": "fields are an exact copy of P60_v10_hpnt1 (v=10), not a v=50 run",
}


def steady_lookup(steady_cases):
    """{(P, hatch): final steady frame}. Velocity does not matter: every group with the same
    (P, hatch) is bit-identical (checked), so the assertion below guards the assumption."""
    found = {}
    for c in steady_cases:
        field = c.fields[-1]
        key = (c.P, c.hatch)
        if key in found:
            assert np.allclose(found[key], field, rtol=0, atol=1e-6), f"steady fields differ for {key}"
        found[key] = field
    return found


def build_pairs(cases, targets=WHOLE_TIMES):
    """Training table: one row per (usable transient run, whole-number time).

    A transient run is usable if it is not in EXCLUDED and a steady field exists for its
    (P, hatch). Returns a dict of arrays plus the run names.
    """
    steady = steady_lookup(cases["steady"])
    rows = {"steady": [], "params": [], "t": [], "t_target": [], "y": [], "case": []}
    for c in cases["transient"]:
        if c.folder in EXCLUDED or (c.P, c.hatch) not in steady:
            continue
        for t_target, k, t_actual in whole_number_frames(c, targets):
            rows["steady"].append(steady[(c.P, c.hatch)])
            rows["params"].append((c.P, c.v, c.hatch))
            rows["t"].append(t_actual)
            rows["t_target"].append(t_target)
            rows["y"].append(c.fields[k])
            rows["case"].append(c.folder)
    return {k: (np.array(v) if k != "case" else v) for k, v in rows.items()}
