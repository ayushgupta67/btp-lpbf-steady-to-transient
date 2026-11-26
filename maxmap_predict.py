"""Predict the max transient temperature field with the saved model.

    python maxmap_predict.py --ss-dir SS_new/dataset/P90_V10_H0.15            # settings read from its txt files
    python maxmap_predict.py --ss-dir <folder> --P 75 --v 30 --hatch 0.12      # override settings
Writes maxmap_outputs/prediction_<name>.csv (node, ix, iy, iz, T_max_pred) and prints the peak.
"""
import argparse
from pathlib import Path

import numpy as np

import maxmap_data as md
from maxmap_model import Model

ap = argparse.ArgumentParser()
ap.add_argument("--ss-dir", required=True)
ap.add_argument("--model", default=str(md.OUT / "model" / "maxmap_diffusion.pt"))
ap.add_argument("--P", type=float)
ap.add_argument("--v", type=float)
ap.add_argument("--hatch", type=float)
ap.add_argument("--draws", type=int, default=8)
a = ap.parse_args()
d = Path(a.ss_dir)
r = md.Run(**md._parse_dir((d, "steady", d.parent)))
P, v, h = a.P or r.P, a.v or r.v, a.hatch or r.hatch
m, meta = Model.load(a.model)
pred, spread = m.predict(md.max_field(r)[0][None], np.array([[P, v, h]]), draws=a.draws)
out = md.OUT / f"prediction_{d.name}.csv"
ix, iy, iz = np.arange(md.N_NODES) % md.NX, (np.arange(md.N_NODES) // md.NX) % md.NY, np.arange(md.N_NODES) // (md.NX * md.NY)
np.savetxt(out, np.column_stack([np.arange(md.N_NODES), ix, iy, iz, pred[0]]), delimiter=",", fmt=["%d", "%d", "%d", "%d", "%.3f"],
           header="node,ix,iy,iz,T_max_pred", comments="")
print(f"P={P} v={v} hatch={h}: predicted peak {pred[0].max():.1f} C (draw spread {spread:.2f} C) -> {out}")
