"""Loader for the Single_bead_Data_set (Transient / Steady_state folders).

Layout on disk
--------------
Single_bead_Data_set/<Transient|Steady_state>/<case>/   (some cases sit one
level deeper, under an AM_wall/ folder -- we search recursively)
    my_temp_<k>.txt   line 0: " total time is  <t>",  line 1: 5166 floats
    power.txt, velocity.txt, hatch.txt, ...   the run's parameters

5166 = 41 x 42 x 3 nodes, matching e_x=40, e_y=41, e_z=2 elements. The flat
vector is x-fastest: reshape(3, 42, 41) gives T[iz, iy, ix]. Inferred, not
documented: only this ordering makes the hot spot of frame 0 a compact block
(6 x 6 x 2 nodes; any other ordering gives a box at least 1.8x larger), and
under it the hot spot then travels along x at constant speed on the mid-width
line iy ~ 20.5, i.e. the laser track of the single bead.

What the data turned out to contain (all checked, see build_bead_csv.py)
-----------------------------------------------------------------------
* Parameters are read from the txt files, never from folder names. The
  Steady_state folder "P60_v50_hpnt1" really has hatch = 0.18.
* Transient: 27 cases = 3 powers x 3 velocities x 3 hatch values, ~326 frames
  each. The scan finishes within 0.32 time units and the part is back at the
  65.0 base temperature everywhere by t ~ 4.
* Steady_state: only 4 folders (all P = 60) and two of them (v = 10 and
  v = 50 at hatch 0.18) agree to round-off, 1e-6 C over all 100 frames, while
  hatch 0.15 vs 0.18 differs by 191 C. So steady state does not depend on
  velocity, which leaves 3 distinct steady fields, for hatch 0.10/0.15/0.18.
* Peak temperature depends on P and hatch only through P/hatch (the paper's
  volumetric source Q_v = gamma*P/(A*l)), which is why P60/h0.10 and
  P90/h0.15 are bit-identical runs.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent / "Single_bead_Data_set"
CACHE = Path(__file__).parent / "bead_outputs" / "cache" / "frames.npz"

N_NODES = 5166
BASE_T = 65.0  # every untouched node reads exactly this (Amb_Temp.txt says 25.0)
WHOLE_TIMES = (2, 3, 4, 5, 6, 7)

# Transient/P60_v50_hptn1 is bit-identical to P60_v10_hpnt1 (a v=10 run, with the
# v=10 time stamps) although its parameter files say v=50. By the P/hatch rule
# its fields should match P90_v50_hpnt15 (peak 4973), not 5444. Keep it in the
# CSVs, flagged, but never train on it.
EXCLUDED = {
    "P60_v50_hptn1": "fields are a copy of the v=10 run P60_v10_hpnt1, not a v=50 run",
}


@dataclass
class Case:
    kind: str  # "transient" or "steady"
    folder: str
    P: float
    v: float
    hatch: float
    times: np.ndarray  # (n_frames,)
    fields: np.ndarray  # (n_frames, N_NODES)

    @property
    def label_matches_params(self):
        """Does the folder name agree with the parameter files?"""
        m = re.match(r"P(\d+)_v(\d+)_h(?:pnt|ptn)(\d+)", self.folder)
        want = (float(m.group(1)), float(m.group(2)), float("0." + m.group(3)))
        return want == (self.P, self.v, self.hatch)

    @property
    def status(self):
        if self.kind == "transient" and self.folder in EXCLUDED:
            return "excluded: " + EXCLUDED[self.folder]
        if not self.label_matches_params:
            return "folder name disagrees with parameter files (files used)"
        return "ok"


def _read_param(case_dir, name):
    path = next(case_dir.rglob(f"{name}.txt"))
    return float(path.read_text().split()[0])


def _read_frames(case_dir):
    files = {int(re.search(r"my_temp_(\d+)", f.name).group(1)): f
             for f in case_dir.rglob("my_temp_*.txt")}
    times, fields = [], []
    for k in sorted(files):
        header, data = files[k].read_text().split("\n")[:2]
        times.append(float(header.split()[-1]))
        values = np.array(data.split(), dtype=np.float64)
        assert values.size == N_NODES, f"{files[k]}: {values.size} values"
        fields.append(values)
    return np.array(times), np.array(fields)


def load_cases(root=ROOT, use_cache=True):
    """Return {"transient": [Case...], "steady": [Case...]}, sorted by folder name."""
    cached = np.load(CACHE) if use_cache and CACHE.exists() else None
    out = {}
    for kind, sub in (("transient", "Transient"), ("steady", "Steady_state")):
        out[kind] = []
        for case_dir in sorted((root / sub).iterdir()):
            key = f"{kind}/{case_dir.name}"
            if cached is not None and key + "/F" in cached:
                times, fields = cached[key + "/T"], cached[key + "/F"].astype(np.float64)
            else:
                times, fields = _read_frames(case_dir)
            out[kind].append(Case(kind, case_dir.name, _read_param(case_dir, "power"),
                                  _read_param(case_dir, "velocity"),
                                  _read_param(case_dir, "hatch"), times, fields))
    if cached is None and use_cache:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **{
            f"{c.kind}/{c.folder}/{part}": arr
            for cs in out.values() for c in cs
            for part, arr in (("T", c.times), ("F", c.fields.astype(np.float32)))})
    return out


def active_node_mask(cases, tol=1e-6):
    """Boolean (N_NODES,): False for air nodes, True for the rest.

    Air = a node that never moves from its starting temperature in any frame of
    any case, transient or steady (4072 of 5166 nodes, all exactly 65.0). The
    model only sees and predicts the remaining 1094; air is filled back in with
    BASE_T afterwards. Defined from the mesh response, so it carries no
    information about which case is held out.
    """
    lo = np.full(N_NODES, np.inf)
    hi = np.full(N_NODES, -np.inf)
    for cs in cases.values():
        for c in cs:
            lo = np.minimum(lo, c.fields.min(0))
            hi = np.maximum(hi, c.fields.max(0))
    return (hi - lo) > tol


def expand(active_values, mask):
    """Put (..., n_active) values back into full (..., N_NODES) fields; air = BASE_T."""
    full = np.full(active_values.shape[:-1] + (len(mask),), BASE_T)
    full[..., mask] = active_values
    return full


def whole_number_frames(case, targets=WHOLE_TIMES):
    """For each target time, the nearest stored frame: [(target, frame, t_actual)].

    No case stores an exact whole-number time, and linear interpolation is
    worse than it looks right after the scan (it moves the field by as much as
    the signal itself at t=3), so we use real frames and record the offset.
    """
    near = np.abs(case.times[:, None] - np.asarray(targets)[None, :]).argmin(0)
    return [(t, int(k), float(case.times[k])) for t, k in zip(targets, near)]


def steady_lookup(steady_cases):
    """{(P, hatch): final steady frame}. Velocity does not matter (checked)."""
    found = {}
    for c in steady_cases:
        field = c.fields[-1]
        key = (c.P, c.hatch)
        if key in found:
            # Same (P, hatch) at different velocity agrees to round-off (4.5e-13
            # in the final frame), so a tolerance, not array_equal.
            assert np.allclose(found[key], field, rtol=0, atol=1e-6), \
                f"steady fields differ for {key}"
        found[key] = field
    return found


def build_pairs(cases, targets=WHOLE_TIMES):
    """Training table: one row per (usable transient case, whole-number time).

    A transient case is usable if it is not excluded and a steady field exists
    for its exact (P, hatch). Returns a dict of arrays plus the case names.
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


if __name__ == "__main__":
    data = load_cases()
    print(f"{'kind':<10}{'folder':<18}{'P':>5}{'v':>5}{'hatch':>7}{'frames':>8}  status")
    for cs in data.values():
        for c in cs:
            print(f"{c.kind:<10}{c.folder:<18}{c.P:>5.0f}{c.v:>5.0f}{c.hatch:>7.2f}"
                  f"{len(c.times):>8}  {c.status}")
    pairs = build_pairs(data)
    print(f"\ntraining table: {len(pairs['case'])} rows from {len(set(pairs['case']))} cases")
