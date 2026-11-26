"""Write every number of the updated single-bead dataset to CSV (bead2_outputs/csv/).

    python bead2_build_csv.py

Files (one column per node, node_0 .. node_5165, in the solver's output order, 4 decimals):
  runs.csv                          one row per run/group: parameters (from the txt files), frame
                                    count, time range, peak, minimum, status flags. START HERE.
  nodes.csv                         node -> (ix, iy, iz) grid position and whether it is an active
                                    node (the other 4072 never leave 65 C)
  frame_index.csv                   every frame of every run: time, min / max / mean (no fields)
  transient_whole_number_times.csv  every transient run at t = 2..7 (nearest stored frame), with
                                    a `usable` column; this is the model's target table
  steady_state_fields.csv           the final frame of every steady-state group
  steady_fields_by_P_hatch.csv      the distinct steady inputs, one per (P, hatch)
  transient_all_frames.csv.gz       every transient frame (8,801 rows)
  steady_state_all_frames.csv.gz    every steady-state frame
Parameters come from the txt files, never from folder names. Nothing is dropped: suspect runs
are kept and described in `status`.
"""

import csv
import gzip
import io
from pathlib import Path

import numpy as np

import bead2_data as bd

OUT = Path(__file__).parent / "bead2_outputs" / "csv"
NODE_COLS = [f"node_{i}" for i in range(bd.N_NODES)]


def fmt(row):
    return ",".join(f"{x:.4f}" for x in row)


def write_wide(path, meta_cols, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    n = 0
    buf = io.StringIO()
    mw = csv.writer(buf, lineterminator="")
    with opener(path, "wt", newline="") as fh:
        fh.write(",".join(meta_cols + NODE_COLS) + "\n")
        for meta, field in rows:
            buf.seek(0)
            buf.truncate()
            mw.writerow(meta)
            fh.write(buf.getvalue() + "," + fmt(field) + "\n")
            n += 1
    print(f"  {path.name:<36}{n:>6} rows  {path.stat().st_size / 1e6:>7.1f} MB")


def status_of(c, data):
    flags = []
    if c.kind == "transient":
        if c.folder in bd.EXCLUDED:
            flags.append("EXCLUDED: " + bd.EXCLUDED[c.folder])
        if not c.label_matches_params:
            flags.append("folder name disagrees with parameter files")
        if "hptn" in c.name:
            flags.append("folder name typo 'hptn' (means hpnt)")
        if len(c.times) != 326:
            flags.append(f"{len(c.times)} frames instead of 326 (fine-scan frames missing)")
        if c.fields.min() < bd.BASE_T - 1:
            flags.append(f"numerical undershoot during the scan: min T = {c.fields.min():.1f} C (below the 65 C base)")
        same = [o.name for o in data["transient"] if o is not c and o.folder not in bd.EXCLUDED
                and c.folder not in bd.EXCLUDED and len(o.times) == len(c.times)
                and np.abs(o.fields - c.fields).max() < 1e-3]
        if same:
            flags.append("same physics as " + "/".join(same) + " (equal P/hatch and v)")
    else:
        if not c.label_matches_params:
            flags.append("folder name disagrees with parameter files")
        if "/" in c.name:
            flags.append("second copy of the files inside the folder (AM_wall)")
        pk = c.fields.max(1)
        flags.append(f"not converged: peak still rising {(pk[-1] - pk[-11]) / 10:.1f} C per frame at the last frame")
    return "; ".join(flags) if flags else "ok"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = bd.load_cases()
    mask = bd.active_node_mask(data)
    print("writing to", OUT)

    # runs.csv
    with open(OUT / "runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "folder", "P", "v", "hatch", "P_over_hatch", "frames", "t_first", "t_last",
                    "T_max", "T_min", "usable_for_training", "status"])
        for kind, cs in data.items():
            for c in cs:
                st = status_of(c, data)
                use = (kind == "transient" and c.folder not in bd.EXCLUDED)
                w.writerow([kind, c.name, c.P, c.v, c.hatch, round(c.P / c.hatch, 1), len(c.times),
                            f"{c.times[0]:.4f}", f"{c.times[-1]:.4f}", f"{c.fields.max():.2f}",
                            f"{c.fields.min():.2f}", "yes" if use else ("input only" if kind == "steady" else "no"), st])
        for kind, folder, why in bd.MISSING:
            w.writerow([kind, folder, "", "", "", "", 0, "", "", "", "", "no", f"MISSING: {why}"])
    print(f"  {'runs.csv':<36}{sum(len(v) for v in data.values()) + len(bd.MISSING):>6} rows")

    with open(OUT / "nodes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["node", "ix", "iy", "iz", "active"])
        for i in range(bd.N_NODES):
            w.writerow([i, i % bd.NX, (i // bd.NX) % bd.NY, i // (bd.NX * bd.NY), int(mask[i])])

    with open(OUT / "frame_index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "folder", "P", "v", "hatch", "frame", "time", "T_min", "T_max", "T_mean", "nodes_above_base"])
        n = 0
        for kind, cs in data.items():
            for c in cs:
                for k, (t, f) in enumerate(zip(c.times, c.fields)):
                    w.writerow([kind, c.name, c.P, c.v, c.hatch, k, f"{t:.4f}", f"{f.min():.4f}",
                                f"{f.max():.4f}", f"{f.mean():.4f}", int((f > bd.BASE_T + 0.1).sum())])
                    n += 1
    print(f"  {'frame_index.csv':<36}{n:>6} rows")

    meta_t = ["folder", "P", "v", "hatch", "t_target", "t_actual", "t_offset", "frame", "usable"]

    def whole():
        for c in data["transient"]:
            for t_target, k, t_actual in bd.whole_number_frames(c):
                yield ([c.name, c.P, c.v, c.hatch, t_target, f"{t_actual:.4f}", f"{t_actual - t_target:+.4f}", k,
                        "no" if c.folder in bd.EXCLUDED else "yes"], c.fields[k])
    write_wide(OUT / "transient_whole_number_times.csv", meta_t, whole())

    meta_s = ["folder", "P", "v", "hatch", "frame", "time", "status"]
    write_wide(OUT / "steady_state_fields.csv", meta_s,
               (([c.name, c.P, c.v, c.hatch, len(c.times) - 1, f"{c.times[-1]:.4f}", status_of(c, data)], c.fields[-1])
                for c in data["steady"]))
    lookup = bd.steady_lookup(data["steady"])
    write_wide(OUT / "steady_fields_by_P_hatch.csv", ["P", "hatch", "P_over_hatch"],
               (([P, h, round(P / h, 1)], f) for (P, h), f in sorted(lookup.items())))

    meta_all = ["folder", "P", "v", "hatch", "frame", "time"]
    for kind, name in (("transient", "transient_all_frames.csv.gz"), ("steady", "steady_state_all_frames.csv.gz")):
        rows = (([c.name, c.P, c.v, c.hatch, k, f"{t:.4f}"], f)
                for c in data[kind] for k, (t, f) in enumerate(zip(c.times, c.fields)))
        write_wide(OUT / name, meta_all, rows)


if __name__ == "__main__":
    main()
