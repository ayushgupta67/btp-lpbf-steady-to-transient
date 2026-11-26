# Updated single-bead dataset: results

Cross-validated RMSE in degrees C, 24-fold leave-one-setting-out (a setting is a (P/hatch, v) pair; identical twins held out together).

## All 5,166 nodes
| method | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | ALL |
|---|---|---|---|---|---|---|---|
| Diffusion (plain) | 16.77 | 0.11 | 0.08 | 0.08 | 0.08 | 0.08 | 6.85 |
| Ambient 65 | 69.21 | 0.36 | 0.00 | 0.00 | 0.00 | 0.00 | 28.26 |
| Mean of runs | 43.23 | 0.31 | 0.00 | 0.00 | 0.00 | 0.00 | 17.65 |
| Nearest setting | 10.93 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 4.46 |
| Ridge | 16.04 | 0.13 | 0.00 | 0.00 | 0.00 | 0.00 | 6.55 |
| Amp x shape (mean) | 6.14 | 0.31 | 0.00 | 0.00 | 0.00 | 0.00 | 2.51 |
| Hybrid: amp x diffusion shape | 1.77 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.72 |
| Amp x shape (same v) | 1.73 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.71 |
| Amp x shape (ln q, same v) | 0.55 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.22 |
| Hybrid: regression + diffusion residual | 0.55 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.22 |

## 1,094 active nodes only
| method | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | ALL |
|---|---|---|---|---|---|---|---|
| Diffusion (plain) | 36.44 | 0.24 | 0.18 | 0.18 | 0.18 | 0.18 | 14.88 |
| Ambient 65 | 150.41 | 0.79 | 0.01 | 0.01 | 0.01 | 0.01 | 61.40 |
| Mean of runs | 93.94 | 0.68 | 0.00 | 0.00 | 0.00 | 0.00 | 38.35 |
| Nearest setting | 23.76 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 9.70 |
| Ridge | 34.86 | 0.29 | 0.00 | 0.00 | 0.00 | 0.00 | 14.23 |
| Amp x shape (mean) | 13.35 | 0.68 | 0.00 | 0.00 | 0.00 | 0.00 | 5.46 |
| Hybrid: amp x diffusion shape | 3.84 | 0.04 | 0.01 | 0.01 | 0.01 | 0.01 | 1.57 |
| Amp x shape (same v) | 3.76 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 1.53 |
| Amp x shape (ln q, same v) | 1.19 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.48 |
| Hybrid: regression + diffusion residual | 1.19 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.48 |

## Computation (12th-gen i5-12500H laptop, CPU only)
- plain diffusion, final model: 92 s (1,803,270 parameters, 1500 epochs, 156 rows)
- plain diffusion, 24-fold test: 45.7 min (6 workers; each fold 10.7 min under contention)
- hybrid runs: about 11.5 min for 24 folds, 30 s for the final model
