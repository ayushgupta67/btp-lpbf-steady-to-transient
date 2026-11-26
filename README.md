# Mapping steady-state temperature to transient temperature in laser powder bed fusion

B.Tech project (BTP), Department of Mechanical Engineering, IIT Jodhpur, 2026.
Ayush Gupta (B23ME1014), supervised by Dr. Rajit Ranjan.

**Question.** A transient thermal simulation of an LPBF build predicts overheating accurately but is slow
(21 h 28 min for one part in [Ranjan et al. 2020](https://doi.org/10.3390/ma13204576)); a steady-state
analysis takes about 2 minutes but is only qualitative. Can a data-driven model turn the fast steady-state
field into the transient one?

**Key message.** The steady → transient mapping makes sense **as a geometry mapping, not as a mapping of the
process parameters** (power *P*, speed *v*, hatch *h*). A steady state means waiting for infinite time, while
in a build heat only travels a characteristic length *L*<sub>c</sub> = √(αt) before the next bead arrives, so
the steady analysis is local, and a local steady field can only change when the geometry around the heat
source changes.

| | |
|---|---|
| Final report | [`report/BTP_report.pdf`](report/BTP_report.pdf) (32 pages) |
| Final presentation | [`presentation/BTP_final_presentation.pdf`](presentation/BTP_final_presentation.pdf) (17 slides) |

The simulation datasets (generated with the thermal model of Ranjan et al. in CalculiX by the research group)
are **not** included in this repository; the code expects them in the folders named in each loader.

---

## Method

A conditional denoising diffusion model ([`maxmap_model.py`](maxmap_model.py)), used in every case:

* **add noise:** x<sub>τ</sub> = √ᾱ<sub>τ</sub> x<sub>0</sub> + √(1 − ᾱ<sub>τ</sub>) ε, 400 noise levels;
* **train:** a fully connected network sees the noisy target, the noise level, the steady-state field and
  the scalar inputs, and predicts the clean target x<sub>0</sub> (loss ‖f<sub>θ</sub> − x<sub>0</sub>‖²);
* **generate:** 50 DDIM steps from pure noise, average of 8 samples.

Temperatures are mapped with asinh((T − 65)/10)/4. Every result is measured on simulations the model never saw
(held-out runs, leave-one-setting-out cross-validation, or held-out layers) and compared with simple baselines.
Diffusion learns a distribution while this problem is deterministic; averaging the samples makes it behave
like a regressor, and the report discusses where it does not fit.

## Results

### Case 1: 1D rod (final version: heat flux at the top, bottom fixed)
[`rod_flux_sim.py`](rod_flux_sim.py), [`rod_flux_experiment.py`](rod_flux_experiment.py) → [`rodflux_outputs/summary.json`](rodflux_outputs/summary.json)

A 10 mm rod, heat flux *q* at the top for a heating time *t*<sub>h</sub>, bottom fixed at 65 °C, and a
low-conductivity patch at a hidden depth. 1,600 training runs, 300 unseen test runs.

| Model (300 unseen runs) | Mean peak error |
|---|---|
| Steady field used directly as the answer | +932 % |
| Diffusion, steady field + ln *q*, ln *t*<sub>h</sub> | **4.3 %** (median 2.4 %) |
| Same, without the steady field | 25.8 % |
| All properties given as numbers, with / without steady field | 1.9 % / 2.1 % |

The steady field helps only when it carries information the scalar inputs do not. The hidden patch changes the
steady peak by 40–50 % at any depth, but the transient peak only when it lies within about one *L*<sub>c</sub>
of the top (42 % within 1 *L*<sub>c</sub>, 0 % deeper than 2 *L*<sub>c</sub>): the local domain, in 1D.

An earlier 1D variant with a laser moving along the rod ([`rod1d_sim.py`](rod1d_sim.py),
[`rod1d_experiment.py`](rod1d_experiment.py), [`rod1d_allparams.py`](rod1d_allparams.py)) reached 1.0 % mean peak error
and is the version described in the report.

### Case 2: single-bead deposition (27 settings of P, v, h)
[`maxmap_data.py`](maxmap_data.py), [`maxmap_analyze.py`](maxmap_analyze.py), [`maxmap_train.py`](maxmap_train.py) → [`maxmap_outputs/RESULTS.md`](maxmap_outputs/RESULTS.md)

Max steady field (SS_new) + *P*, *v*, *h* → max transient field, 24-fold leave-one-setting-out:
**1.3 %** mean peak error (worst 5.1 %). But the steady fields of all 27 runs are identical to within 0.12 °C:
every run uses one standard geometry, so the local steady model sees the same domain each time and the
accuracy comes only from *P*, *v*, *h*.

An earlier snapshot task (transient fields at whole-number times, [`bead2_train.py`](bead2_train.py),
[`bead2_best_model.py`](bead2_best_model.py) → [`bead2_outputs/RESULTS.md`](bead2_outputs/RESULTS.md)) showed that plain diffusion (6.85 °C RMSE) lost to a nearest-setting copy
(4.46 °C) and that an amplitude × shape model with a diffusion residual reached 0.22 °C.

### Case 3: geometry (a block with a hole and a slot, 3,844 elements)
[`tts_data.py`](tts_data.py), [`tts_analyze.py`](tts_analyze.py), [`tts_model.py`](tts_model.py) → [`tts_outputs/`](tts_outputs/DATA_REPORT.md)

Steady and transient peaks both rise next to walls and voids (correlation 0.88 after element 1,364). Trained on
layers 1–8 and tested on the unseen layers 9–10:

| Method | Mean absolute error (dataset units) |
|---|---|
| Guess the training average | 2,634 |
| Straight line from the steady peak | 1,490 |
| Diffusion, 5 × 5 steady patch (3 seeds) | 1,075 (903–1,231) |
| Diffusion, shape only | 893 (771–955) |

The shape alone is as informative as the steady values: the signal is the geometry. The dataset's temperatures
(~10⁵ °C) are not physical, so only patterns are read ([`tts_outputs/DATA_REPORT.md`](tts_outputs/DATA_REPORT.md)).

### Earlier study: wall dataset
One simulated build of a thin wall ([`wall_data.py`](wall_data.py), [`baselines.py`](baselines.py),
[`diffusion_compare.py`](diffusion_compare.py)). With about 80 independent samples, ridge regression (MAE 2.16)
beat diffusion (5.42), and a random train/test split would have scored 0.97 instead of 2.16: honest held-out
tests and simple baselines are used everywhere since.

## Repository map

| Files | Purpose |
|---|---|
| `maxmap_model.py` | conditional diffusion model shared by all cases |
| `rod_flux_sim.py`, `rod_flux_experiment.py` | Case 1, final 1D rod (heat flux at the top, bottom fixed) |
| `rod1d_sim.py`, `rod1d_experiment.py`, `rod1d_allparams.py` | Case 1, earlier 1D rod (moving laser) |
| `maxmap_data.py`, `maxmap_analyze.py`, `maxmap_train.py` | Case 2, max-temperature mapping and the SS_new checks |
| `bead2_*.py`, `bead_model.py`, `bead_data.py` | Case 2, earlier snapshot task and baselines |
| `tts_data.py`, `tts_analyze.py`, `tts_model.py` | Case 3, geometry dataset |
| `wall_data.py`, `baselines.py`, `diffusion_compare.py` | earlier wall study |
| `*_outputs/` | results summaries |
| `report/`, `presentation/` | final report and presentation (PDF) |

## Running

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python rod_flux_experiment.py      # Case 1, generates its own data (about 1.5 min on a laptop CPU)
.venv/bin/python rod1d_experiment.py         # earlier 1D variant
```

Cases 2 and 3 need the simulation datasets (not included). Everything ran on a laptop CPU (Intel i5-12500H).

## Reference

R. Ranjan, C. Ayas, M. Langelaar, F. van Keulen. Fast detection of heat accumulation in powder bed fusion
using computationally efficient thermal models. *Materials* 13(20), 4576 (2020).
