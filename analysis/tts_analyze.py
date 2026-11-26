"""Critical analysis of the geometry dataset (part_1): pairing, value ranges, and whether the
steady-state temperature now changes from element to element (the point of the local-domain model).

    python analysis/tts_analyze.py      -> prints the report, writes results/case3_geometry/elements.csv and figures
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / d) for d in ("model", "simulators", "data_loaders", "training")]

import csv
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tts_data as td

OUT = td.OUT


def main():
    g, ss_t, ss_f, tr_t, tr_f = td.load_part()
    nx, ny, nz, sn = g["nx"], g["ny"], g["nz"], g["snelz"]
    vox = g["vox"]
    rep = {}

    # ---- element deposition order (as in the scripts: layer by layer, y rows, x fastest) ----------
    order = [(k, j, i) for k in range(g["ez"]) for j in range(g["ey"]) for i in range(g["ex"]) if vox[k, j, i]]
    n_el = len(order)
    per_layer = [int(vox[k].sum()) for k in range(g["ez"])]
    print(f"part elements {n_el}; per layer {per_layer}")

    # ---- steady frames: element steps (dt = 1) and cooling steps (dt = 5) ---------------------------
    dt = np.diff(np.concatenate([[ss_t[0] - 1], ss_t]))
    ss_el = np.flatnonzero(np.isclose(dt, 1.0))
    ss_cool = np.flatnonzero(np.isclose(dt, 5.0))
    print(f"SS frames {len(ss_t)}: element steps {len(ss_el)}, cooling steps {len(ss_cool)} (expected {n_el} and {g['ez']})")
    rep.update(ss_frames=len(ss_t), ss_element_steps=len(ss_el), ss_cooling_steps=len(ss_cool), part_elements=n_el)

    # ---- transient element steps: a heating step is the 1 s ending on a whole-number time T and has a
    # frame at T - 0.9 (sub-steps 0.1, 0.2, 0.3, 0.45, 0.675, 1.0). Cooling steps (5 s) are skipped.
    # A steady solve has no memory and the SS run puts its cooling steps elsewhere, so steps are paired
    # BY ELEMENT NUMBER (n-th heating step <-> n-th steady element step), not by time.
    tr_r = np.round(tr_t, 4)
    have = set(tr_r.tolist())
    ends = [float(t) for t in tr_r if abs(t - round(t)) < 1e-6 and round(t - 0.9, 4) in have]
    print(f"transient heating steps found: {len(ends)} (expected {n_el})")
    rep["transient_heating_steps"] = len(ends)
    m = min(n_el, len(ends), len(ss_el))
    rows = []
    for e in range(m):
        k, j, i = order[e]
        T_end = ends[e]
        sel = (tr_t > T_end - 1.0 + 1e-6) & (tr_t <= T_end + 1e-6)
        ssf = ss_f[ss_el[e]]
        trs = tr_f[sel]
        node = (k + sn + 1) * nx * ny + (j + 1) * nx + (i + 1)  # a top corner node of the new element
        rows.append(dict(element=e, layer=k + 1, ix=i, iy=j, ss_time=float(ss_t[ss_el[e]]), tr_end_time=T_end, n_tr_frames=int(sel.sum()),
                         ss_peak=float(ssf.max()), ss_min=float(ssf.min()), tr_peak=float(trs.max()), tr_min=float(trs.min()),
                         tr_end_peak=float(tr_f[sel][-1].max()), ss_at_new=float(ssf[node]), tr_at_new=float(trs[:, node].max())))
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    with open(OUT / "elements.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def stats(x, name):
        x = x[np.isfinite(x)]
        q = np.percentile(x, [0, 1, 25, 50, 75, 99, 100])
        print(f"{name:<28} min {q[0]:10.1f}  p1 {q[1]:9.1f}  p25 {q[2]:9.1f}  median {q[3]:9.1f}  p75 {q[4]:9.1f}  p99 {q[5]:10.1f}  max {q[6]:10.1f}")
        return dict(zip(["min", "p1", "p25", "median", "p75", "p99", "max"], map(float, q)))
    print("\n== per-element peaks (C) ==")
    rep["ss_peak"] = stats(R["ss_peak"], "steady peak")
    rep["tr_peak"] = stats(R["tr_peak"], "transient peak in the step")
    rep["tr_min"] = stats(R["tr_min"], "transient minimum in the step")
    print("transient frames per element step:", dict(zip(*[a.tolist() for a in np.unique(R["n_tr_frames"], return_counts=True)])))

    print("\n== does the steady peak change from element to element? ==")
    u = np.unique(np.round(R["ss_peak"], 1))
    print(f"distinct steady peaks (0.1 C): {len(u)} of {len(R['ss_peak'])}; spread {R['ss_peak'].max() - R['ss_peak'].min():.1f} C; "
          f"coefficient of variation {R['ss_peak'].std() / R['ss_peak'].mean() * 100:.1f}%")
    rep["distinct_ss_peaks"] = int(len(u))
    for L in range(1, g["ez"] + 1):
        m = R["layer"] == L
        if m.any():
            print(f"  layer {L:2d}: {int(m.sum()):3d} elements | steady peak median {np.median(R['ss_peak'][m]):9.1f} "
                  f"[{R['ss_peak'][m].min():9.1f}, {R['ss_peak'][m].max():9.1f}] | transient peak median {np.nanmedian(R['tr_peak'][m]):9.1f} "
                  f"[{np.nanmin(R['tr_peak'][m]):9.1f}, {np.nanmax(R['tr_peak'][m]):9.1f}]")

    ok = np.isfinite(R["tr_peak"])
    cc = float(np.corrcoef(R["ss_peak"][ok], R["tr_peak"][ok])[0, 1])
    print(f"\ncorrelation between steady peak and transient peak over elements: {cc:.3f}")
    rep["corr_ss_tr_peak"] = cc
    (OUT / "analysis.json").write_text(json.dumps(rep, indent=1))

    # ---- figures ------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(R["ss_peak"], lw=0.6, color="#15161B", label="steady peak")
    ax[0].plot(R["tr_peak"], lw=0.6, color="#C8102E", alpha=0.7, label="transient peak (during its 1 s step)")
    ax[0].set(yscale="log", xlabel="element (deposition order)", ylabel="peak temperature (C)")
    ax[0].legend(frameon=False)
    ax[1].scatter(R["ss_peak"][ok], R["tr_peak"][ok], s=4, c=R["layer"][ok], cmap="viridis")
    ax[1].set(xscale="log", yscale="log", xlabel="steady peak (C)", ylabel="transient peak (C)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_peaks_by_element.png", dpi=140)
    plt.close(fig)
    fig, axs = plt.subplots(2, 5, figsize=(14, 6), layout="constrained")
    for L, a in zip(range(1, 11), axs.flat):
        img = np.full((g["ey"], g["ex"]), np.nan)
        m = R["layer"] == L
        img[R["iy"][m], R["ix"][m]] = R["ss_peak"][m]
        im = a.imshow(img, cmap="inferno", origin="lower")
        a.set_title(f"layer {L}", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, shrink=0.75)
    fig.suptitle("steady peak when each element is added (white = no element: void or outside the part)")
    fig.savefig(OUT / "fig_ss_peak_maps.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
