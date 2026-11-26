"""CSV export of the new steady-state data (SS_new) and of the max-temperature pairs.

    python maxmap_build_csv.py

maxmap_outputs/csv/
  ss_new_runs.csv              one row per SS_new run: settings (from the txt files), frames, times, peak, min, status
  ss_new_all_frames.csv.gz     every SS_new frame: run, P, v, hatch, frame, time, node_0 .. node_5165
  ss_new_max_fields.csv        per run: the per-node maximum over its 320 frames (the model INPUT)
  transient_max_fields.csv     per transient run: the per-node maximum over all its frames (the model TARGET)
  pairs.csv                    which transient run is paired with which SS_new run, and whether it is used
Temperatures in deg C, 4 decimals; node order is the solver's (x fastest, 41 x 42 x 3).
"""
import csv
import gzip
import io

import numpy as np

import maxmap_data as md

OUT = md.OUT / "csv"
NODE_COLS = [f"node_{i}" for i in range(md.N_NODES)]


def write_wide(path, meta_cols, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    n, buf = 0, io.StringIO()
    mw = csv.writer(buf, lineterminator="")
    with opener(path, "wt", newline="") as fh:
        fh.write(",".join(meta_cols + NODE_COLS) + "\n")
        for meta, field in rows:
            buf.seek(0)
            buf.truncate()
            mw.writerow(meta)
            fh.write(buf.getvalue() + "," + ",".join(f"{x:.4f}" for x in field) + "\n")
            n += 1
    print(f"  {path.name:<28}{n:>6} rows  {path.stat().st_size / 1e6:7.1f} MB")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = md.load_runs(md.SS_ROOT, "steady")
    tr = md.load_runs(md.TRANSIENT_ROOT, "transient")
    ref = np.stack([r.fields for r in st])
    spread = float((ref.max(0) - ref.min(0)).max())
    with open(OUT / "ss_new_runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "P", "v", "hatch", "P_over_hatch", "frames", "t_first", "t_last", "T_max", "T_min", "status"])
        for r in st:
            other = max(float(np.abs(r.fields - o.fields).max()) for o in st if o is not r)
            w.writerow([r.name, r.P, r.v, r.hatch, round(r.q, 1), len(r.times), f"{r.times[0]:.4f}", f"{r.times[-1]:.4f}",
                        f"{r.fields.max():.2f}", f"{r.fields.min():.2f}",
                        f"fields identical to every other SS_new run within {other:.3f} C (no dependence on P, v, hatch)"])
    print(f"  ss_new_runs.csv {len(st):>17} rows   (max spread between runs: {spread:.3f} C)")
    write_wide(OUT / "ss_new_all_frames.csv.gz", ["run", "P", "v", "hatch", "frame", "time"],
               (([r.name, r.P, r.v, r.hatch, int(k), f"{t:.4f}"], f) for r in st for k, t, f in zip(r.frame_ids, r.times, r.fields)))
    write_wide(OUT / "ss_new_max_fields.csv", ["run", "P", "v", "hatch"], (([r.name, r.P, r.v, r.hatch], md.max_field(r)[0]) for r in st))
    write_wide(OUT / "transient_max_fields.csv", ["run", "P", "v", "hatch", "used"],
               (([r.name.split("/")[0], r.P, r.v, r.hatch, "no" if r.name.split("/")[0] in md.EXCLUDED_TRANSIENT else "yes"], md.max_field(r)[0]) for r in tr))
    ss_by = {r.key: r.name for r in st}
    with open(OUT / "pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["transient_run", "ss_new_run", "P", "v", "hatch", "used", "note"])
        for r in tr:
            top = r.name.split("/")[0]
            w.writerow([top, ss_by.get(r.key, ""), r.P, r.v, r.hatch, "no" if top in md.EXCLUDED_TRANSIENT else "yes",
                        md.EXCLUDED_TRANSIENT.get(top, "")])
    print("  pairs.csv", len(tr), "rows")


if __name__ == "__main__":
    main()
