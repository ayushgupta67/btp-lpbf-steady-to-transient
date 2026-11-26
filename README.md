# Mapping steady-state temperature to transient temperature in laser powder bed fusion

B.Tech project (BTP), Department of Mechanical Engineering, IIT Jodhpur, 2026
Ayush Gupta (B23ME1014) · supervisor: Dr. Rajit Ranjan

> **Can a computationally fast steady-state thermal response help us predict the hotspots of the slow
> transient analysis?** We train a conditional diffusion model to map steady-state temperature fields to
> transient ones, on three cases of increasing realism, and find that the mapping makes sense
> **as a geometry mapping, not as a mapping of the process parameters** (laser power *P*, scan speed *v*,
> hatch spacing *h*).

| Document | |
|---|---|
| Final report (32 pages) | [`report/BTP_report.pdf`](report/BTP_report.pdf) |
| Final presentation (17 slides) | [`presentation/BTP_final_presentation.pdf`](presentation/BTP_final_presentation.pdf) |

---

## Contents

1. [Motivation](#1-motivation)
2. [Method: conditional diffusion model](#2-method-conditional-diffusion-model)
3. [Case 1: 1D rod](#3-case-1-1d-rod)
4. [Case 2: single-bead deposition](#4-case-2-single-bead-deposition)
5. [Why Case 2 fails: the local domain](#5-why-case-2-fails-the-local-domain)
6. [Case 3: a part with voids](#6-case-3-a-part-with-voids)
7. [Earlier study: wall dataset](#7-earlier-study-wall-dataset)
8. [Repository contents](#8-repository-contents)
9. [Running the code](#9-running-the-code)
10. [Limitations](#10-limitations)

---

## 1. Motivation

In laser powder bed fusion (LPBF) a laser melts metal powder layer by layer, and the heat must flow through the
part to the baseplate. Where it cannot escape (above overhangs, near voids) the part overheats: dross and melt
balls, coarse microstructure, distortion that can jam the recoater. The layer-by-layer thermal model of
[Ranjan et al. (2020)](https://doi.org/10.3390/ma13204576) predicts these hotspots, but the transient analysis of
their test part took **21 h 28 min** on 20 cores, while the steady-state simplification took **2 min 9 s**
(about 600× faster) with a peak-temperature error of 65 %, i.e. only qualitatively right.

* **Steady state** is analytical in nature: setting ∂T/∂t = 0 (t → ∞) removes time, and one solve gives the field.
* **Transient** is a numerical experiment: the build is replayed step by step in time.

The question of the project is whether a data-driven model can learn the mapping between the two. Even then the
match is expected to be qualitative: the aim is to locate the hotspots.

## 2. Method: conditional diffusion model

[`maxmap_model.py`](maxmap_model.py) is the model used in every case (a few hundred lines, PyTorch, CPU or GPU).

* **Forward (add noise):** x<sub>τ</sub> = √ᾱ<sub>τ</sub> x<sub>0</sub> + √(1 − ᾱ<sub>τ</sub>) ε, with a linear
  schedule β from 10⁻⁴ to 0.02 over 400 noise levels.
* **Train:** a fully connected denoiser receives the noisy target x<sub>τ</sub>, the noise level τ (64 sine/cosine
  features), and the condition: the steady-state field, compressed by a two-layer encoder, plus scalar inputs
  (log-transformed and standardised on the training set). It predicts the clean target x<sub>0</sub> directly;
  loss ‖f<sub>θ</sub> − x<sub>0</sub>‖². AdamW, learning rate 10⁻³ with cosine decay, gradient clipping 1.0.
* **Generate:** DDIM, 50 deterministic steps from pure noise, the estimate clipped to the training range at every
  step; 8 samples are averaged and their spread is recorded.
* **Temperatures** are modelled as z = asinh((T − 65 °C)/10 °C)/4: linear for small rises, logarithmic for large ones.

Predicting x<sub>0</sub> instead of the noise was necessary: most nodes of an LPBF field sit at a constant
temperature and a noise-prediction network did not train. Diffusion learns a probability distribution while this
problem is deterministic (one input, one answer); averaging the samples makes it act like a regressor, and the
report discusses where it does not fit.

**Evaluation.** Every number is measured on simulations the model never saw (separate test runs,
leave-one-setting-out cross-validation, or held-out layers) and compared with simple baselines (constant,
nearest setting, linear fit, the steady field itself). The main metric is the peak-temperature error relative to
the true peak rise.

## 3. Case 1: 1D rod

### Final version: heat flux at the top, bottom fixed
Code: [`rod_flux_sim.py`](rod_flux_sim.py), [`rod_flux_experiment.py`](rod_flux_experiment.py) ·
results: [`rodflux_outputs/summary.json`](rodflux_outputs/summary.json),
[`rodflux_outputs/patch_effect.csv`](rodflux_outputs/patch_effect.csv)

A 10 mm rod on 101 nodes (the 1D model of Ranjan et al., Sec. 3.2). A heat flux *q* enters at the top for a heating
time *t*<sub>h</sub>, then the rod cools for 2 s; the bottom is held at 65 °C. A 1 mm patch of lower conductivity
sits at a random, hidden depth (a 1D stand-in for powder or a void). There is no scan speed.
The model receives the max steady field, ln *q* and ln *t*<sub>h</sub>; the conductivity and the patch are
hidden, so only the steady field reveals them. 1,600 training runs and 300 unseen test runs, 8,000 updates.

| Model, 300 unseen runs | Mean peak error |
|---|---|
| Steady field used directly as the answer | 932 % overestimate |
| Diffusion, steady field + ln *q*, ln *t*<sub>h</sub> | **4.3 %** (median 2.4 %) |
| Same, steady field removed | 25.8 % |
| All five properties given as numbers, with / without steady field | 1.9 % / 2.1 % |
| Training runs: 50 / 200 / 1,600 (with steady field) | 10.8 % / 5.6 % / 4.3 % |

**The steady field helps exactly when it carries information that the scalar inputs do not.**

The patch also shows the local domain in 1D ([`patch_effect.csv`](rodflux_outputs/patch_effect.csv), medians over
the 300 test runs; *L*<sub>c</sub> = √(α *t*<sub>h</sub>)):

| Patch depth below the top | Steady peak rise | Transient peak rise |
|---|---|---|
| < 1 *L*<sub>c</sub> | +43 % | +42 % |
| 1–2 *L*<sub>c</sub> | +39 % | +4 % |
| 2–4 *L*<sub>c</sub> | +53 % | 0 % |
| > 4 *L*<sub>c</sub> | +51 % | 0 % |

The steady field sees the patch at any depth; the transient only within about one characteristic length.

### Earlier version: laser moving along the rod
Code: [`rod1d_sim.py`](rod1d_sim.py), [`rod1d_experiment.py`](rod1d_experiment.py),
[`rod1d_allparams.py`](rod1d_allparams.py) · results: [`rod1d_outputs/rod1d_summary.json`](rod1d_outputs/rod1d_summary.json),
[`rod1d_outputs/rod1d_allparams.json`](rod1d_outputs/rod1d_allparams.json)

A Gaussian source of power *P* moves at speed *v* along a rod with both ends at 65 °C; the material is hidden.
This is the version described in the report (Chapter 4). Mean peak error on 300 unseen runs: **1.0 %** with
1,600 training runs (3.0 % with 25), against 174 % for the steady field taken as the answer and 18 % without the
steady field; with all properties given as numbers, 0.76 % with and 0.72 % without the steady field.

## 4. Case 2: single-bead deposition

Code: [`maxmap_data.py`](maxmap_data.py), [`maxmap_analyze.py`](maxmap_analyze.py), [`maxmap_train.py`](maxmap_train.py) ·
results: [`maxmap_outputs/RESULTS.md`](maxmap_outputs/RESULTS.md), [`maxmap_outputs/summary.json`](maxmap_outputs/summary.json)

One bead on a plate (mesh 41 × 42 × 3 = 5,166 nodes) simulated with the thermal model of Ranjan et al. for 27
settings: *P* ∈ {60, 90, 120} W, *v* ∈ {10, 50, 100} mm/s, *h* ∈ {0.10, 0.15, 0.18} mm. The localised steady
state (SS_new) solves the steady problem once for each of 320 laser positions. Task: per-node maximum of the
steady fields + ln *P*, ln *v*, ln *h*, ln(*P*/*h*) → per-node maximum of the transient run; 26 usable runs
(one mislabelled copy excluded), 24 distinct settings, 1,252 modelled nodes, 2.0 M parameters, 1,500 epochs.

| Leave-one-setting-out (24 folds) | |
|---|---|
| Mean / median / worst peak error | **1.3 % / 1.1 % / 5.1 %** |
| RMSE / MAE over all nodes | 9.8 °C / 0.9 °C |
| Hot-spot overlap (Jaccard) | 1.00 |

**But** the steady fields of all 27 runs are identical to within 0.12 °C (peak 6,666.6 °C in every run) while the
transient peak varies from 2,652 to 9,311 °C ([`maxmap_analyze.py`](maxmap_analyze.py)). The steady input carries
no information about the setting, and all of the accuracy comes from *P*, *v*, *h*.

**Earlier formulation (snapshots).** [`bead2_data.py`](bead2_data.py), [`bead2_train.py`](bead2_train.py),
[`bead2_hybrid.py`](bead2_hybrid.py), [`bead2_best_model.py`](bead2_best_model.py) (with
[`bead_model.py`](bead_model.py), [`bead_data.py`](bead_data.py)) predicted the transient field at whole-number
times t = 2…7 s from an earlier steady field. Cross-validated RMSE over all nodes and times
([`bead2_outputs/RESULTS.md`](bead2_outputs/RESULTS.md)): plain diffusion 6.85 °C, nearest training setting
4.46 °C, amplitude × shape model 0.22 °C, regression + diffusion residual 0.22 °C. With 26 runs, structure had to
be supplied rather than learned.

## 5. Why Case 2 fails: the local domain

A steady state means depositing and waiting for infinite time, so heat would reach the whole part; in a build the
next bead arrives long before that. Heat penetrates only about a characteristic length

*L*<sub>c</sub> = √(α t), α = k / (ρ c<sub>p</sub>)

(four times the time gives only twice the distance), so the steady analysis is restricted to a local domain
around the new material. **With one standard geometry, every bead sees the same local domain, so the steady
field is the same for every bead and every run** (the end wall lies outside the domain). With walls and voids
nearby, the local domains differ from bead to bead, and the steady field carries the geometry.

## 6. Case 3: a part with voids

Code: [`tts_data.py`](tts_data.py), [`tts_analyze.py`](tts_analyze.py), [`tts_model.py`](tts_model.py) ·
results: [`tts_outputs/DATA_REPORT.md`](tts_outputs/DATA_REPORT.md), [`tts_outputs/model_results.json`](tts_outputs/model_results.json),
[`tts_outputs/model_checks.json`](tts_outputs/model_checks.json)

A 20 × 20 mm block, 10 layers of 1 mm, with a hole that widens near the top and a slot, built element by element
(3,844 elements). The steady run uses a 65 °C sink outside a hemisphere of radius 3.41 mm around each new element
(diameter 3 *L*<sub>c</sub> with α = 5.18 mm²/s, t = 1 s). Steady and transient peaks are paired by element number.

* The steady peak now varies between elements (3,033 distinct values). Both peaks rise next to walls and voids,
  and they correlate at 0.88 from element 1,365 onwards (0.65 over all elements).
* Model: input the 5 × 5 patch of steady peaks around each element plus the layer number, target the patch of
  transient peaks; train on layers 1–8 (3,132 elements), test on layers 9–10 (712 elements), where the round hole
  is new.

| Method, layers 9–10 | Mean absolute error (dataset units) |
|---|---|
| Guess the training mean | 2,634 |
| Straight line from the steady peak (+ layer) | 1,490 |
| Diffusion, steady patch, seeds 0–2 | 1,075 (903–1,231) |
| Diffusion, shape only (solid cells set to one constant), seeds 0–2 | 893 (771–955) |

About 2.5× less error than a constant guess, and the shape alone is as informative as the steady values: in this
dataset the signal is the geometry. The dataset has known problems, listed in
[`DATA_REPORT.md`](tts_outputs/DATA_REPORT.md): temperatures of about 10⁵ °C (not physical, so only patterns are
read), different reference temperatures in the two runs, a misplaced local domain in the steady script for most
elements, and only one part.

## 7. Earlier study: wall dataset

[`wall_data.py`](wall_data.py), [`baselines.py`](baselines.py), [`diffusion_compare.py`](diffusion_compare.py):
one simulated build of a thin wall (8 × 22 × 21 cells, 1,094 of them material, 480 frames that form about 80
independent time blocks). As reported in the report (Appendix A.2), ridge regression (MAE 2.16) beat diffusion
(5.42), and a random train/test split would have scored 0.97 instead of 2.16 for the same model. This is why every
later experiment holds out whole runs, settings or layers and compares against baselines.

## 8. Repository contents

| File | What it does | Writes |
|---|---|---|
| [`maxmap_model.py`](maxmap_model.py) | conditional diffusion model (`Config`, `Model.fit`, `Model.predict`, save/load) | – |
| [`rod_flux_sim.py`](rod_flux_sim.py) | Case 1 simulator: flux at the top, fixed bottom, hidden patch | – |
| [`rod_flux_experiment.py`](rod_flux_experiment.py) | Case 1 experiment, ablations, patch-depth study | `rodflux_outputs/` |
| [`rod1d_sim.py`](rod1d_sim.py) | earlier 1D simulator: moving laser | – |
| [`rod1d_experiment.py`](rod1d_experiment.py) | earlier 1D experiment and data-size study | `rod1d_outputs/` |
| [`rod1d_allparams.py`](rod1d_allparams.py) | earlier 1D: all properties as inputs | `rod1d_outputs/rod1d_allparams.json` |
| [`maxmap_data.py`](maxmap_data.py) | Case 2 loader: transient and SS_new runs, max fields, matched by (P, v, h) | `maxmap_outputs/cache/` |
| [`maxmap_analyze.py`](maxmap_analyze.py) | Case 2 data checks (identical SS_new runs, copies, ranges) | `maxmap_outputs/analysis.json` |
| [`maxmap_train.py`](maxmap_train.py) | Case 2 model + 24-fold cross-validation | `maxmap_outputs/` |
| [`maxmap_report.py`](maxmap_report.py) | Case 2 figures and `RESULTS.md` from the cross-validation | `maxmap_outputs/RESULTS.md` |
| [`bead2_data.py`](bead2_data.py) | loader for the updated single-bead dataset | `bead2_outputs/cache/` |
| [`bead2_train.py`](bead2_train.py) | snapshot task: diffusion + baselines, 24 folds | `bead2_outputs/` |
| [`bead2_hybrid.py`](bead2_hybrid.py) | snapshot task: amplitude × diffusion shape | `bead2_outputs/` |
| [`bead2_best_model.py`](bead2_best_model.py) | snapshot task: amplitude × shape model, fit and predict | `bead2_outputs/model/` |
| [`bead2_report.py`](bead2_report.py) | snapshot task: figures and `RESULTS.md` | `bead2_outputs/RESULTS.md` |
| [`bead_model.py`](bead_model.py), [`bead_data.py`](bead_data.py) | network and constants used by the snapshot scripts | – |
| [`tts_data.py`](tts_data.py) | Case 3 loader: geometry, steady and transient steps | `tts_outputs/cache/` |
| [`tts_analyze.py`](tts_analyze.py) | Case 3 analysis: pairing, ranges, effect of walls and voids | `tts_outputs/` |
| [`tts_model.py`](tts_model.py) | Case 3 model and baselines; `--checks` for seeds and shape only | `tts_outputs/model_*.json` |
| [`wall_data.py`](wall_data.py), [`baselines.py`](baselines.py), [`diffusion_compare.py`](diffusion_compare.py) | wall study | `comparison.png` |
| `*_outputs/` | the results summaries quoted above | |
| `report/`, `presentation/` | final report and presentation | |

Simulation data, caches, trained models and figures are not part of the repository.

## 9. Running the code

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Case 1 generates its own data (about 1.5 min on a laptop CPU, 6 models in parallel)
.venv/bin/python rod_flux_sim.py                 # quick check of a few runs
.venv/bin/python -u rod_flux_experiment.py
.venv/bin/python -u rod1d_experiment.py && .venv/bin/python -u rod1d_allparams.py

# Cases 2 and 3 need the simulation datasets (not included), in the folders named in the loaders:
#   Updated_complete_dataset_singlebead/Single_bead_Data_set/Transient, SS_new/dataset, Trans_to_SS_dataset/part_1
.venv/bin/python maxmap_analyze.py
.venv/bin/python -u maxmap_train.py --epochs 1500      # the reported run used 1,500 epochs (script default 2,500)
.venv/bin/python maxmap_report.py
.venv/bin/python tts_analyze.py
.venv/bin/python -u tts_model.py && .venv/bin/python -u tts_model.py --checks
```

All results were produced on a laptop CPU (Intel Core i5-12500H). Training on a multithreaded CPU is not
bit-for-bit repeatable: re-running `tts_model.py --checks` reproduced four of the five saved runs exactly, while
the steady-patch run with seed 1 gave 785 instead of 903.

## 10. Limitations

* Little data: 26 runs and 24 settings in Case 2, one part in Case 3.
* Case 3 temperatures are not physical and its steady script misplaces the local domain for most elements, so
  Case 3 shows patterns, not temperatures; it tests new layers of the same part, not a new part.
* The localised steady fields of Case 2 do not respond to *P*, *v* or *h*, so the effect of process parameters has
  to enter as scalars.
* Diffusion is a probabilistic model applied to a deterministic problem; a deterministic field-to-field model could
  serve the same purpose at lower cost.

## Reference

R. Ranjan, C. Ayas, M. Langelaar, F. van Keulen. Fast detection of heat accumulation in powder bed fusion using
computationally efficient thermal models. *Materials* 13(20), 4576 (2020).
