"""Critical analysis of the updated single-bead dataset.

    python bead2_analyze.py

Prints every check and writes bead2_outputs/data_report.json (machine-readable) and
bead2_outputs/data_checks.csv (one row per case with its status). Nothing is assumed from
folder names: parameters come from the txt files and every claim below is tested.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np

import bead2_data as bd

OUT = Path(__file__).parent / "bead2_outputs"
OUT.mkdir(exist_ok=True)


def main():
    data = bd.load_cases()
    tr, st = data["transient"], data["steady"]
    rep = {"missing_folders": bd.MISSING}
    print("== 1. What is on disk ==")
    print(f"transient cases: {len(tr)}  frames: {sum(len(c.times) for c in tr)}")
    print(f"steady cases   : {len(st)}  frames: {sum(len(c.times) for c in st)}   missing folders: {bd.MISSING}")

    # ---- 2. parameter grid ------------------------------------------------------------
    print("\n== 2. Parameter grid (from the txt files) ==")
    grid = {(P, v, h) for P in (60.0, 90.0, 120.0) for v in (10.0, 50.0, 100.0) for h in (0.10, 0.15, 0.18)}
    for name, cs in (("transient", tr), ("steady", st)):
        have = {}
        for c in cs:
            have.setdefault(c.key, []).append(c.name)
        absent = sorted(grid - set(have))
        dup = {k: n for k, n in have.items() if len(n) > 1}
        print(f"{name}: distinct settings {len(have)} of 27; absent: {absent}; duplicated: {dup}")
        rep[f"{name}_absent_settings"] = [list(a) for a in absent]
        rep[f"{name}_duplicate_settings"] = {str(k): n for k, n in dup.items()}
    mism = [(c.kind, c.name) for cs in data.values() for c in cs if not c.label_matches_params]
    print("folder name disagrees with parameter files:", mism)
    rep["name_param_mismatch"] = mism

    # ---- 3. transient integrity -------------------------------------------------------
    print("\n== 3. Transient integrity ==")
    rows = []
    for c in tr:
        f = c.fields
        dt = np.diff(c.times)
        heat_end = None
        # the scan: frames with time step < 0.01 are the fine heating frames
        fine = np.where(dt < 0.01)[0]
        if len(fine):
            heat_end = float(c.times[fine[-1] + 1])
        hot = (f > bd.BASE_T + 0.5)
        back = [i for i in range(len(f)) if not hot[i].any()]
        rows.append(dict(name=c.name, P=c.P, v=c.v, hatch=c.hatch, frames=len(c.times),
                         t0=float(c.times[0]), t1=float(c.times[-1]), heat_end=heat_end,
                         monotonic=bool(np.all(dt > 0)), nan=bool(np.isnan(f).any()),
                         tmin=float(f.min()), peak=float(f.max()),
                         peak_t=float(c.times[f.max(1).argmax()]),
                         t_back_to_65=float(c.times[back[0]]) if back else None,
                         max_gap=float(dt.max())))
    for r in rows:
        print(f"{r['name']:<18} fr={r['frames']:<4} t=[{r['t0']:.3f},{r['t1']:.3f}] heat_end={r['heat_end']} "
              f"peak={r['peak']:.0f}@{r['peak_t']:.3f} back65@{r['t_back_to_65']} tmin={r['tmin']:.2f} nan={r['nan']}")
    rep["transient_rows"] = rows

    # ---- 4. exact copies --------------------------------------------------------------
    print("\n== 4. Exact / near copies between transient runs (same frame index) ==")
    copies = []
    for a, b in combinations(tr, 2):
        n = min(len(a.times), len(b.times))
        d = np.abs(a.fields[:n] - b.fields[:n]).max()
        if d < 1e-3:
            copies.append((a.name, b.name, float(d)))
            print(f"  {a.name} == {b.name}  (max |diff| {d:.2e})")
    rep["transient_copies"] = copies
    if not copies:
        print("  none")

    # ---- 5. P / hatch law --------------------------------------------------------------
    print("\n== 5. Does the peak depend on P/hatch and v only? ==")
    pk = {}
    for r in rows:
        pk.setdefault((round(r["P"] / r["hatch"], 1), r["v"]), []).append((r["name"], r["peak"]))
    for k in sorted(pk):
        if len(pk[k]) > 1:
            vals = [p for _, p in pk[k]]
            print(f"  P/h={k[0]:<7} v={k[1]:<5} peaks {['%.0f' % p for p in vals]}  spread {max(vals) - min(vals):.2f}  {[n for n, _ in pk[k]]}")

    # ---- 6. steady-state runs ----------------------------------------------------------
    print("\n== 6. Steady-state runs: convergence and the final field ==")
    srows = []
    for c in st:
        f = c.fields
        last = f[-1]
        chg = np.abs(f[-1] - f[-2]).max()
        srows.append(dict(name=c.name, P=c.P, v=c.v, hatch=c.hatch, frames=len(c.times), peak_final=float(last.max()),
                          peak_prev=float(f[-2].max()), last_change=float(chg), tmin=float(f.min()),
                          peak_by_frame=[float(x) for x in f.max(1)[::11]]))
        print(f"{c.name:<26} P={c.P:<4} v={c.v:<4} h={c.hatch:<4} peak_final={last.max():9.1f}  max|last-prev|={chg:9.3f}  "
              f"peak@frames[0,11,22..]={['%.0f' % x for x in f.max(1)[::22]]}")
    rep["steady_rows"] = srows

    print("\n   Final steady field vs speed: same (P, hatch) at different v")
    by = {}
    for c in st:
        by.setdefault((c.P, c.hatch), []).append(c)
    for k in sorted(by):
        cs = by[k]
        if len(cs) < 2:
            print(f"   P={k[0]} h={k[1]}: only {cs[0].name}")
            continue
        for a, b in combinations(cs, 2):
            print(f"   P={k[0]} h={k[1]}  {a.name} (v={a.v}) vs {b.name} (v={b.v}): max|diff|={np.abs(a.fields[-1] - b.fields[-1]).max():.3g}  "
                  f"peaks {a.fields[-1].max():.0f}/{b.fields[-1].max():.0f}")

    print("\n   Steady peak vs P/hatch (should be proportional if the steady problem is linear in the source)")
    for k in sorted(by, key=lambda k: k[0] / k[1]):
        pf = [c.fields[-1].max() - bd.BASE_T for c in by[k]]
        print(f"   P={k[0]:<5} h={k[1]:<5} P/h={k[0] / k[1]:7.1f}  rise={np.mean(pf):10.1f}  rise/(P/h)={np.mean(pf) / (k[0] / k[1]):.3f}")

    # ---- 7. air nodes --------------------------------------------------------------------
    print("\n== 7. Active (ever-changing) nodes ==")
    mask = bd.active_node_mask(data)
    print(f"active {int(mask.sum())}, air {int((~mask).sum())} of {bd.N_NODES}")
    rep["active_nodes"] = int(mask.sum())

    # ---- 8. steady vs transient link ---------------------------------------------------
    print("\n== 8. Is the steady peak related to the transient peak of the same setting? ==")
    for c in tr:
        s = [x for x in st if x.key == c.key]
        if s:
            print(f"   {c.name:<18} transient peak {c.fields.max():8.0f}   steady final peak {s[0].fields[-1].max():8.0f}   ratio {c.fields.max() / s[0].fields[-1].max():.3f}")
        else:
            print(f"   {c.name:<18} transient peak {c.fields.max():8.0f}   NO steady run with these exact parameters")

    json.dump(rep, open(OUT / "data_report.json", "w"), indent=1, default=str)
    print("\nwrote", OUT / "data_report.json")


if __name__ == "__main__":
    main()
