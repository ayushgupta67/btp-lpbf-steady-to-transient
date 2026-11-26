# Slide content — LPBF wall dataset + diffusion mapping

Handoff notes for whoever builds the deck: every number below is measured from
the files, not estimated. Keep them exact. Audience is a research advisor and
lab group who know heat transfer but have not seen this dataset. Suggested
length 12 slides, ~15 minutes.

---

## Slide 1 — Title

**Transient → Steady-State Temperature Mapping for an LPBF Wall Build**

Dataset characterisation and a diffusion-model feasibility check

Ayush · BTP progress review

---

## Slide 2 — What the data is

Simulation of a **laser powder bed fusion** build — a thin metal wall printed
layer by layer on a base plate.

Two folders, supplied as matched pairs:

| Folder | Contents |
|---|---|
| `Transient_wall` | Temperature field **during** the build, heat still redistributing |
| `Steady_state_wall` | Temperature field the part would **settle to** |

- 480 `.txt` files per folder (plus 480 `.vtr`, which hold identical values in ParaView format — redundant)
- File index matches across folders: `my_temp_57.txt` in each is the same instant

*Speaker note: the `eos_` filename prefix indicates an EOS machine.*

---

## Slide 3 — The computational grid

The domain is discretised into a rectilinear cell grid:

| Axis | Cells | Role |
|---|---|---|
| **x** | **8** | build direction — the wall grows along this axis |
| **y** | **22** | width |
| **z** | **21** | height |

**8 × 22 × 21 = 3696 cells**

Each file holds exactly 3696 temperatures — one scalar per cell — plus a
timestamp on line 1.

*Suggested visual: a simple 3D box diagram with the three axis counts labelled.*

---

## Slide 4 — Only a third of the domain is material

Of the 3696 cells, **2602 sit at exactly 65.0 °C in every one of the 480
frames.** That is the ambient chamber temperature — those cells are gas, not part.

```
3696 total cells
− 2602 ambient (never change)
─────────
  1094 material cells
```

The 1094 decompose as:

| Cells | Component | Extent |
|---|---|---|
| **924** | base plate | x = 1–2, full y and z → 2 × 22 × 21 |
| **170** | the wall | x = 3–7, y = 10–11, z = 2–18 → 5 × 2 × 17 |

The wall is **2 cells thick and 17 cells tall**, standing on the plate.

---

## Slide 5 — Why the mask matters (methodological point)

**All metrics in this work are computed on the 1094 material cells only.**

If scored across all 3696 cells, a model gets 2602 of them correct for free by
outputting the constant 65.0. Reported accuracy becomes dominated by cells that
carry no information.

*This is a small implementation detail with a large effect on any published
number — worth stating explicitly.*

---

## Slide 6 — The geometry is not static

The wall is **still being printed** across the 480 frames. Material cells are
activated progressively:

| Frame | Time | Active cells | Wall front at |
|---|---|---|---|
| 0 | 2.1 | 882 | x = 3 |
| 160 | 28.9 | 982 | x = 4 |
| 320 | 55.4 | 1040 | x = 6 |
| 479 | 82.0 | 1094 | x = 7 |

The wall grows from **4 cells to 170 cells**. Each completed x-layer adds
2 × 17 = **34 cells**.

**Implication:** the 1094-cell mask is the *final* geometry. At early times many
of those cells are still unfused powder, so the model is being asked about
regions that do not yet exist as solid.

---

## Slide 7 — Time structure

Timestamps run **2.1 → 82.0**. They are not arbitrary — they repeat on a fixed
pattern:

```
2.1   2.2   2.35   2.575   2.9125   → 3.0
3.1   3.2   3.35   3.575   3.9125   → 4.0
4.1   4.2   4.35   4.575   4.9125   → 5.0
```

Six fixed sub-offsets (0.0, 0.1, 0.2, 0.35, 0.575, 0.9125) inside every unit
interval, repeated **exactly 80 times**.

**80 blocks × 6 samples = 480 frames**

---

## Slide 8 — The core limitation

The 480 frames are **one simulation run**, not 480 independent cases.

- Consecutive frames differ by an MAE of only **0.76**, against a field standard deviation of **6.73**
- Frames within a block are near-duplicates
- Effective independent sample count is closer to **80 than 480**

**This single fact governs every result that follows.**

*Suggested visual: a timeline showing 80 blocks, one expanded to show its 6
near-identical sub-samples.*

---

## Slide 9 — The mapping

```
   INPUT                              OUTPUT
   transient field (1094 values)  →   steady field (1094 values)
   + timestamp (1 value)
```

**Motivation:** the transient solve is expensive. A learned map would jump
directly to the settled field.

**Why it is not trivial:** correlation between a cell's *own* transient value and
its *own* steady value is **−0.13** — effectively zero, with no cell exceeding
|r| = 0.8.

The mapping cannot be pointwise. Heat arrives from neighbours, so the model must
read the whole spatial field. Confirming this: **copying the transient field
straight through is the worst predictor tested (MAE 4.38).**

---

## Slide 10 — Train/test splitting is the decisive choice

Same ridge regression, three splitting strategies:

| Split strategy | Test MAE |
|---|---|
| Random per-frame shuffle | **0.97** |
| Hold out whole time blocks | 2.02 |
| Hold out the final 20 blocks | **2.16** |

The random split leaks: every held-out frame has a near-twin in training, so the
model is recognising rather than predicting.

**All results here use the block-wise split.** The 0.97 figure is an artefact.

*Speaker note: this is the most transferable finding — it applies to any
time-series simulation dataset, not just this one.*

---

## Slide 11 — Results

Test MAE on the 1094 material cells, held-out final 20 blocks.
Field standard deviation = 6.73.

| Method | MAE |
|---|---|
| **Ridge regression (linear)** | **2.16** |
| Neural network + physics term | 2.36 |
| Neural network | 2.48 |
| *Per-cell mean (ignores input)* | *2.63* |
| Copy transient field | 4.38 |
| **Diffusion model** | **5.31** |
| Diffusion + physics term | 5.31 |

**Two observations:**

1. Diffusion performs **worse than ignoring the input entirely**.
2. The genuine headroom is narrow: 2.63 → 2.16 is about **18%** of achievable
   improvement over a no-skill predictor.

Diffusion's spread across stochastic draws was **12.4** — nearly twice the
field's own standard deviation. It is generating noise, not structure.

*Suggested visual: horizontal bar chart, baselines in grey, learned models in
colour, with a dashed line at the 2.63 no-skill level.*

---

## Slide 12 — Why diffusion underperforms

This is a **data limitation, not an implementation defect**.

- Diffusion models learn a distribution over outputs; that needs many independent samples
- Output dimension here is 1094; effective sample count is ~80
- The task wants **one correct answer**, not diverse plausible ones — which is what diffusion is designed to provide

**The decisive question for this project:**

> Are additional simulation runs available — varying laser power, scan speed,
> wall thickness, or material?

- **If yes:** sample diversity rises and diffusion becomes worth revisiting
- **If no:** a deterministic regressor is the appropriate method, and it already
  outperforms diffusion here

---

## Slide 13 — Code

Three modules, each with one responsibility:

| File | Role |
|---|---|
| `wall_data.py` | Parses the `.txt` files, builds the material-cell mask, provides the three split strategies |
| `baselines.py` | Baseline ladder — global mean, per-cell mean, identity, per-cell affine, ridge |
| `diffusion_compare.py` | Trains regressor and diffusion variants, reports all results against the baselines, plots |

**Design decisions worth noting:**

- Normalisation statistics computed from the **training split only**
- Physics term is the **steady-state residual** (∇²T = 0 away from sources), which is the constraint the target actually satisfies — not a time-dependent heat equation
- Sampling uses **DDIM, 50 steps** rather than full 400-step DDPM
- Draw spread is reported alongside MAE, to detect a collapsed model that has stopped using its noise input

---

## Slide 14 — Open questions

**1. Unphysical values in the steady field.**
300 of 480 steady frames contain cells **below 0 °C**, against a 65 °C ambient;
395 contain cells below 20 °C. These cluster in 168 cells at the wall–plate
junction. Per-frame maxima reach 269.9 °C.

*Is the steady field the quantity we assume, or is the steady solve
extrapolating where geometry is not yet fused?*

**2. Data availability.** Are further runs obtainable? (See slide 12.)

**3. Direction of the mapping.** Current work predicts steady from transient.
The reverse is also defensible and changes the appropriate physics constraint.
Which serves the project goal?

---

## Slide 15 — Summary

- **3696 cells**, of which **1094 are material** — metrics must mask the rest
- Geometry is **dynamic**: 882 → 1094 active cells as the wall prints
- **480 frames = 80 blocks × 6**, from a **single run** — ~80 effective samples
- Transient and steady are **pointwise uncorrelated** (r = −0.13); the map must be spatial
- **Block-wise splitting is mandatory**; random splitting inflates scores 2×
- **Ridge regression (2.16) currently beats diffusion (5.31)**
- Next step depends on whether **more simulation runs** can be obtained
