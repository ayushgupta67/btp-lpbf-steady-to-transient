"""Dataset for the Transient_wall -> Steady_state_wall mapping.

Layout on disk
--------------
Each folder holds 480 matched frames from ONE simulation trajectory:
    my_temp_<k>.txt   line 0: " total time is  <t>",  line 1: 3696 floats
    eos_temp_<k>.vtr  the same field as VTK RectilinearGrid (redundant)

The .txt payload is z-fastest, so reshape(8, 22, 21) gives T[ix, iy, iz]
on the 8 x 22 x 21 cell grid. We read only the .txt.

Structure that matters for modelling
------------------------------------
* Times run 2.1 -> 82.0 as 80 unit-time blocks x 6 fixed sub-offsets
  (0.0, 0.1, 0.2, 0.35, 0.575, 0.9125). The block is the natural unit for
  splitting: frames inside one block are near-duplicates.
* 2602 of 3696 cells sit at exactly 65.0 in every frame. Only 1094 carry
  signal, and ~31% of even those are near-constant. Metrics must be taken
  on the active cells or they are dominated by free constants.
* Geometry is a substrate slab (all of x=1,2) plus a 2-cell-thick wall at
  y=10,11, z=2..18 running through x=3..7.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / d) for d in ("model", "simulators", "data_loaders", "training")]

import numpy as np

NX, NY, NZ = 8, 22, 21
NCELLS = NX * NY * NZ  # 3696
N_FRAMES = 480
AMBIENT = 65.0
SUB_OFFSETS = 6  # frames per unit-time block


def load_frame(path):
    """Return (time, field) with field shaped (NX, NY, NZ)."""
    with open(path) as fh:
        header = fh.readline()
        values = np.array(fh.readline().split(), dtype=np.float64)
    assert values.size == NCELLS, f"{path}: got {values.size} values"
    return float(header.split()[-1]), values.reshape(NX, NY, NZ)


def load_folder(folder, n=N_FRAMES):
    """Return (times (n,), fields (n, NX, NY, NZ)) ordered by frame index."""
    times = np.empty(n)
    fields = np.empty((n, NX, NY, NZ))
    for k in range(n):
        times[k], fields[k] = load_frame(f"{folder}/my_temp_{k}.txt")
    return times, fields


class WallData:
    """Loads both folders, flattens to the active cells, and splits by time block."""

    def __init__(self, root=str(REPO / "data"), n=N_FRAMES):
        self.t, transient = load_folder(f"{root}/Transient_wall", n)
        self.t_steady, steady = load_folder(f"{root}/Steady_state_wall", n)

        # Cells never exactly ambient in either field. The two folders agree on
        # this set, so either one defines it.
        self.mask = ~np.all(steady == AMBIENT, axis=0)
        self.transient_grid, self.steady_grid = transient, steady
        self.X = transient[:, self.mask]  # (n, 1094)
        self.Y = steady[:, self.mask]

        # Unit-time block index: frames within a block are near-duplicates, so
        # a split has to keep whole blocks on one side.
        self.block = np.floor(self.t).astype(int)

    @property
    def n_active(self):
        return int(self.mask.sum())

    def split(self, kind="extrapolate", frac=0.25, seed=0):
        """Return (train_idx, test_idx) as boolean masks over frames.

        extrapolate -- hold out the LAST blocks. Tests whether the model
                       generalises forward in the build, which is the use case.
        interpolate -- hold out random whole blocks. Easier; use it to show how
                       much of any score is just temporal interpolation.
        random      -- the naive per-frame split. Included only to demonstrate
                       that it leaks: every test frame's neighbours are in train.
        """
        blocks = np.unique(self.block)
        if kind == "extrapolate":
            cut = blocks[int(len(blocks) * (1 - frac))]
            test = self.block >= cut
        elif kind == "interpolate":
            rng = np.random.default_rng(seed)
            held = rng.choice(blocks, int(len(blocks) * frac), replace=False)
            test = np.isin(self.block, held)
        elif kind == "random":
            rng = np.random.default_rng(seed)
            test = np.zeros(len(self.t), bool)
            test[rng.choice(len(self.t), int(len(self.t) * frac), replace=False)] = True
        else:
            raise ValueError(kind)
        return ~test, test


def metrics(pred, true):
    """MAE, RMSE and skill vs the train-mean predictor, on whatever cells are passed."""
    err = pred - true
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "max_err": float(np.abs(err).max()),
    }
