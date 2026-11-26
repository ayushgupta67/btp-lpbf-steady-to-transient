"""Critical checks of the new steady-state data (SS_new) and of the max-temperature pairs.

    python analysis/maxmap_analyze.py        # prints the report, writes results/case2_single_bead/analysis.json
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / d) for d in ("model", "simulators", "data_loaders", "training")]

import json
from itertools import combinations

import numpy as np

import maxmap_data as md

B = md.BASE_T


def hot_x(field):
    """x index (along the bead) of the hottest node."""
    return int(np.unravel_index(int(field.argmax()), (md.NZ, md.NY, md.NX))[2])


def main():
    st = md.load_runs(md.SS_ROOT, "steady")
    tr = md.load_runs(md.TRANSIENT_ROOT, "transient")
    rep = {}
    print("== 1. SS_new on disk ==")
    print(f"runs {len(st)}, frames {sum(len(r.times) for r in st)}, frames per run {sorted({len(r.times) for r in st})}")
    grid = {(P, v, h) for P in (60.0, 90.0, 120.0) for v in (10.0, 50.0, 100.0) for h in (0.1, 0.15, 0.18)}
    keys = [r.key for r in st]
    print("settings covered:", len(set(keys)), "of 27; missing:", sorted(grid - set(keys)),
          "; duplicated:", [k for k in set(keys) if keys.count(k) > 1])
    bad_name = []
    for r in st:
        P, v, h = (float(x) for x in __import__("re").match(r"P(\d+)_V(\d+)_H([\d.]+)", r.name).groups())
        if (P, v, round(h, 4)) != r.key:
            bad_name.append(r.name)
    print("folder name disagrees with parameter files:", bad_name)
    rep.update(n_runs=len(st), missing=sorted(grid - set(keys)), name_mismatch=bad_name)

    print("\n== 2. Values ==")
    rows = []
    for r in st:
        f = r.fields
        rows.append(dict(name=r.name, P=r.P, v=r.v, hatch=r.hatch, frames=len(r.times), t0=float(r.times[0]), t1=float(r.times[-1]),
                         nan=bool(np.isnan(f).any()), tmin=float(f.min()), peak=float(f.max()),
                         below_64=int((f.min(1) < 64).sum())))
    for x in rows:
        print(f"{x['name']:<18} P={x['P']:<5} v={x['v']:<5} h={x['hatch']:<5} frames={x['frames']} t=[{x['t0']:.4f},{x['t1']:.4f}] "
              f"peak={x['peak']:7.1f} min={x['tmin']:7.2f} frames_with_T<64={x['below_64']} nan={x['nan']}")
    rep["steady_rows"] = rows

    print("\n== 3. Do the steady frames follow the laser like the transient frames? (same frame index) ==")
    trk = {r.key: r for r in tr}
    for r in st[:: max(1, len(st) // 6)]:
        t = trk.get(r.key)
        if t is None:
            continue
        idx = [0, 80, 160, 240, len(r.times) - 1]
        print(f"{r.name:<18} SS hot x: {[hot_x(r.fields[i]) for i in idx]}  transient hot x: {[hot_x(t.fields[i]) for i in idx]}  "
              f"time match: {np.allclose(r.times, t.times[:len(r.times)], atol=1e-6)}")

    print("\n== 4. Exact copies between steady runs ==")
    copies = []
    for a, b in combinations(st, 2):
        if a.fields.shape == b.fields.shape and np.abs(a.fields - b.fields).max() < 1e-3:
            copies.append((a.name, b.name))
    by = {}
    for a, b in copies:
        by.setdefault(a, []).append(b)
    print(f"{len(copies)} identical pairs")
    for a, bs in by.items():
        print(f"  {a} == {bs}")
    rep["steady_copies"] = copies

    print("\n== 5. Does the steady field depend on v? (same P, hatch; compare fields frame by frame) ==")
    for (P, h) in sorted({(r.P, round(r.hatch, 4)) for r in st}):
        g = sorted([r for r in st if (r.P, round(r.hatch, 4)) == (P, h)], key=lambda r: r.v)
        if len(g) > 1:
            d = max(np.abs(a.fields - b.fields).max() for a, b in combinations(g, 2))
            print(f"  P={P:<5} h={h:<5} speeds {[r.v for r in g]}: max |difference| {d:.3g} C")

    print("\n== 6. Max-over-frames fields (the new input / output) ==")
    pairs, skipped = md.build_pairs(tr, st)
    print("pairs:", len(pairs["name"]), "| skipped:", skipped)
    ssm, trm = pairs["ss_max"], pairs["tr_max"]
    act = (np.maximum(ssm.max(0), trm.max(0)) - B > 1e-3) | (np.minimum(ssm.min(0), trm.min(0)) < B - 1e-3)
    print(f"nodes whose max differs from 65 in any pair: {int(act.sum())} of {md.N_NODES}")
    for i in np.argsort(pairs["params"][:, 0] / pairs["params"][:, 2]):
        n, p = pairs["name"][i], pairs["params"][i]
        print(f"  {n:<18} P/h={p[0] / p[2]:7.1f} v={p[1]:<5} SS max {ssm[i].max():8.1f}  transient max {trm[i].max():8.1f}  "
              f"ratio {(trm[i].max() - B) / (ssm[i].max() - B):.3f}  hot nodes (>100C rise) SS {int((ssm[i] - B > 100).sum()):4d} tr {int((trm[i] - B > 100).sum()):4d}")
    g = [md.group_of(p) for p in pairs["params"]]
    print("distinct (P/hatch, v) groups:", len(set(g)))
    tw = []
    for i, j in combinations(range(len(g)), 2):
        if np.abs(trm[i] - trm[j]).max() < 1e-3:
            tw.append((pairs["name"][i], pairs["name"][j]))
    print("identical transient max fields:", tw)
    rep.update(n_pairs=len(pairs["name"]), active_nodes=int(act.sum()), groups=len(set(g)), skipped=skipped, transient_twins=tw)
    md.OUT.mkdir(exist_ok=True)
    (md.OUT / "analysis.json").write_text(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
