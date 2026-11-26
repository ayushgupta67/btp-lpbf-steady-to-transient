"""Write the temperature CSVs for the single-bead dataset.

    python build_bead_csv.py

Outputs, in bead_outputs/csv/ (one column per node, node_0 .. node_5165, in the
solver's output order; temperatures to 4 decimals):

  frame_index.csv                  every frame of every case: parameters, time,
                                   min / max / mean, hot-node count (no fields)
  transient_whole_number_times.csv the 27 transient cases at t = 2..7, using the
                                   stored frame nearest each whole number
  steady_state_fields.csv          the final frame of each steady-state case
  transient_all_frames.csv.gz      every transient frame (8,801 rows)
  steady_state_all_frames.csv.gz   every steady-state frame (400 rows)

Parameters (P, v, hatch) come from the txt files inside each folder, not from
the folder names. Nothing is dropped: suspect cases are kept and described in
the `status` column.
"""

import csv
import gzip
import io
from pathlib import Path

import numpy as np

from bead_data import BASE_T, N_NODES, WHOLE_TIMES, load_cases, whole_number_frames

OUT = Path(__file__).parent / "bead_outputs" / "csv"
NODE_COLS = [f"node_{i}" for i in range(N_NODES)]


def fmt(row):
    return ",".join(f"{x:.4f}" for x in row)


def write_wide(path, meta_cols, rows):
    """rows: iterable of (meta_values, field). Plain CSV, or gzip if path ends in .gz."""
    opener = gzip.open if path.suffix == ".gz" else open
    n = 0
    buf = io.StringIO()
    meta_writer = csv.writer(buf, lineterminator="")  # quotes fields such as status
    with opener(path, "wt", newline="") as fh:
        fh.write(",".join(meta_cols + NODE_COLS) + "\n")
        for meta, field in rows:
            buf.seek(0)
            buf.truncate()
            meta_writer.writerow(meta)
            fh.write(buf.getvalue() + "," + fmt(field) + "\n")
            n += 1
    print(f"  {path.name:<34}{n:>6} rows  {path.stat().st_size / 1e6:>7.1f} MB")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_cases()
    print("writing to", OUT)

    # 1. Index of every frame.
    with open(OUT / "frame_index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "folder", "P", "v", "hatch", "P_over_hatch", "frame", "time",
                    "T_min", "T_max", "T_mean", "nodes_above_base", "status"])
        n = 0
        for kind, cases in data.items():
            for c in cases:
                for k, (t, f) in enumerate(zip(c.times, c.fields)):
                    w.writerow([kind, c.folder, c.P, c.v, c.hatch, round(c.P / c.hatch, 1), k,
                                f"{t:.4f}", f"{f.min():.4f}", f"{f.max():.4f}", f"{f.mean():.4f}",
                                int((f > BASE_T + 0.1).sum()), c.status])
                    n += 1
    print(f"  {'frame_index.csv':<34}{n:>6} rows")

    meta_t = ["folder", "P", "v", "hatch", "t_target", "t_actual", "t_offset", "frame", "status"]
    meta_all = ["folder", "P", "v", "hatch", "frame", "time", "status"]

    # 2. Transient fields at whole-number times.
    def whole():
        for c in data["transient"]:
            for t_target, k, t_actual in whole_number_frames(c, WHOLE_TIMES):
                yield ([c.folder, c.P, c.v, c.hatch, t_target, f"{t_actual:.4f}",
                        f"{t_actual - t_target:+.4f}", k, c.status], c.fields[k])
    write_wide(OUT / "transient_whole_number_times.csv", meta_t, whole())

    # 3. Steady-state field per case (final frame of the run).
    def steady_final():
        for c in data["steady"]:
            yield ([c.folder, c.P, c.v, c.hatch, len(c.times) - 1, f"{c.times[-1]:.4f}",
                    c.status], c.fields[-1])
    write_wide(OUT / "steady_state_fields.csv", meta_all, steady_final())

    # 4. Full dumps.
    for kind, name in (("transient", "transient_all_frames.csv.gz"),
                       ("steady", "steady_state_all_frames.csv.gz")):
        rows = (([c.folder, c.P, c.v, c.hatch, k, f"{t:.4f}", c.status], f)
                for c in data[kind] for k, (t, f) in enumerate(zip(c.times, c.fields)))
        write_wide(OUT / name, meta_all, rows)


if __name__ == "__main__":
    main()
