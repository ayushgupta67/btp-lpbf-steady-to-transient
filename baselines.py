"""Baselines every learned model on this dataset has to beat.

Run this before trusting any diffusion result. It reports the same metric on
the same cells and the same split for a ladder of predictors, from "ignore the
input entirely" up to "linear map of the whole field". If a neural model does
not clear the top of this ladder, it has not learned the mapping.

    python baselines.py
"""

import numpy as np

from wall_data import WallData, metrics

SPLITS = ["extrapolate", "interpolate", "random"]


def fit_predictors(X_tr, Y_tr):
    """Return {name: predict_fn}. Each fn maps transient (n, d) -> steady (n, d)."""
    d = X_tr.shape[1]
    preds = {}

    # 0. Ignore the input: one number for the whole field.
    gmean = Y_tr.mean()
    preds["global mean"] = lambda X: np.full_like(X, gmean)

    # 1. Ignore the input, but per cell. This is the real floor -- it captures
    #    everything that is constant in time, which is most of the field.
    cmean = Y_tr.mean(0)
    preds["per-cell mean"] = lambda X: np.broadcast_to(cmean, X.shape).copy()

    # 2. Copy the input.
    preds["identity (copy transient)"] = lambda X: X.copy()

    # 3. Per-cell affine fit: steady_j = a_j * transient_j + b_j. No spatial
    #    coupling, 2 parameters per cell.
    a = np.empty(d)
    b = np.empty(d)
    for j in range(d):
        v = X_tr[:, j]
        if v.std() < 1e-12:
            a[j], b[j] = 0.0, Y_tr[:, j].mean()
        else:
            a[j], b[j] = np.polyfit(v, Y_tr[:, j], 1)
    preds["per-cell affine"] = lambda X: X * a + b

    # 4. Ridge on the whole field: the full linear operator, d x d parameters.
    #    This is the strongest baseline and the one that matters -- a diffusion
    #    model is only interesting if it beats a linear map.
    Xc, Yc = X_tr - X_tr.mean(0), Y_tr - Y_tr.mean(0)
    for lam in (1e-2, 1e2):
        W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ Yc)
        xm, ym = X_tr.mean(0), Y_tr.mean(0)
        preds[f"ridge field->field (lam={lam:g})"] = (
            lambda X, W=W, xm=xm, ym=ym: (X - xm) @ W + ym
        )
    return preds


def main():
    data = WallData()
    print(f"frames {len(data.t)}   active cells {data.n_active}   "
          f"time {data.t.min():.1f}..{data.t.max():.1f}   blocks {len(np.unique(data.block))}")
    print(f"steady field on active cells: mean {data.Y.mean():.2f}  sd {data.Y.std():.2f}\n")

    for kind in SPLITS:
        tr, te = data.split(kind)
        X_tr, Y_tr, X_te, Y_te = data.X[tr], data.Y[tr], data.X[te], data.Y[te]
        print(f"--- split: {kind}  (train {tr.sum()} / test {te.sum()} frames) ---")
        print(f"{'predictor':<34}{'MAE':>9}{'RMSE':>9}{'max err':>10}")
        for name, fn in fit_predictors(X_tr, Y_tr).items():
            m = metrics(fn(X_te), Y_te)
            print(f"{name:<34}{m['MAE']:>9.4f}{m['RMSE']:>9.4f}{m['max_err']:>10.2f}")

        # Not a real predictor -- it needs the answer for the neighbouring frame.
        # Shown because under a random split it is what a model can learn to
        # imitate, which is how a leaky split flatters a weak model.
        prev = data.Y[np.maximum(np.where(te)[0] - 1, 0)]
        print(f"{'[oracle] previous frame':<34}{np.abs(prev - Y_te).mean():>9.4f}")
        print()


if __name__ == "__main__":
    main()
