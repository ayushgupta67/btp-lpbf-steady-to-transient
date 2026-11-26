# Max-temperature mapping: results

Task (sir's instruction): max steady-state field (SS_new) + P, v, hatch -> max transient field, one pair per run.

| quantity | value |
|---|---|
| pairs (runs) | 26 (24 distinct settings; copied run excluded) |
| CV folds (leave one setting out) | 24 |
| RMSE, all 5,166 nodes | 9.8 C |
| MAE, all nodes | 0.9 C |
| peak error, mean / median / worst | 1.3 % / 1.1 % / 5.1 % |
| hot-spot overlap (Jaccard, >= half the peak rise) | 1.00 |
| relative L2 error (1.0 = predicting 65 C everywhere) | 0.01 |
| in-sample RMSE (seen data, not a test) | 2.6 C |
| parameters / epochs / batch | 2,005,668 / 1500 / 8 |
| final model training time | 41 s |
| cross-validation wall time | 17.2 min (168 s per fold, 4 in parallel) |

Worst held-out runs (peak error): P60_v100_hpnt18 +5.1%; P60_v10_hpnt18 +3.6%; P60_v100_hpnt15 +2.2%; P120_v10_hpnt1 -2.0%; P120_v100_hpnt18 +1.8%

## Shortcomings found in SS_new (maxmap_analyze.py, analysis_log.txt)
1. **All 27 SS_new runs are the same field.** Over every node and every frame, the 27 runs differ by at most 0.12 C, even between P60/hatch 0.18 and P120/hatch 0.10 (3.6x the heat input). The power, speed and hatch files differ, but the temperatures do not respond to them. The steady input therefore carries no information about the setting; the model's skill comes entirely from (P, v, hatch). Most likely the steady-state runs used the same heat flux for every case. Ask Rajit / sir to check the flux input of the SS runs.
2. Peak 6666.6 C in every run, reached at the start of the bead (frame 9); 5,892 C at the end; about 3,206 C along the middle. The start/end spikes are edge effects.
3. The steady hot zone is about 9x wider than the transient one (280 vs 32 nodes above 100 C rise at the same frame).
4. All SS_new runs use the v = 10 time stamps (2.001 to 2.320) even for v = 50 and 100; frame k is the laser position, and the steady hot spot sits about 2 nodes behind the transient one at the same frame index.
5. Good: 27 of 27 settings present, 320 frames each, folder names match the parameter files, no NaN, no values below 65 C (unlike the transient decimal frames).

## What can be improved
- Fix the SS_new runs so each setting gets its own heat input (most important; with the current data the steady input is a constant).
- More settings (P, v, hatch) – the max mapping has only 26 pairs / 24 distinct settings.
- Per-frame pairing is possible: SS_new frame k and transient frame k share the laser position, which would give ~320 pairs per run (about 8,300 in total) instead of 1.
- Smooth the sampled field (the predictions show small speckle along the track); more DDIM draws or a light smoothing would reduce it.
- With more data, a GPU would cut training time; the code already uses one automatically.
