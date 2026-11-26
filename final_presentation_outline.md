# BTP final presentation: rough outline (from Ayush's handwritten notes, 26 Oct 2026)

Draft only; Ayush will explain each part later. Transcribed as written, with question marks where the handwriting was unclear.

## 1. Motivation
- LPBF process (show a video).
- Why thermal LPBF; motivations.
- How the data is generated (read the Ranjan paper).
- Main motivation: **time**.
  - Steady state (SS): analytical, **fast**.
  - Transient: temperature vs time curves from numerical experiments, which **take time**.
  - Question: **SS → transient?**

## 2. Method: generative AI
- Diffusion is the choice we have made. It can be used to learn a probabilistic distribution, but this is a deterministic problem.
- "It might not be the best; we are trying."
- Others have used / are using FNO (Fourier neural operator), possibly the best choice for physics.

## 3. Three cases
- **Case 1: 1D.** Network architecture (look up), show some results, what we learnt.
- **Case 2: single bead deposition.** Transient vs SS.
  - Problem: the local SS model gives the same SS temperature value for every run.
- **Case 3: geometry.**

## 4. Physics behind it (why SS is not physical, and why local)
- Welding/LPBF picture: melt pool, isothermal lines; a new bead heats the older beads.
- SS temperature = deposit and wait for infinite time, which is not physical.
- In infinite time heat only needs to reach a **local domain**, not the whole part. Localising the domain is closer to reality.
- Heat travels like a wave. Characteristic length **Lc = √(α t)**, with α = k / (ρ c) in m²/s.
  - Example: a bar at 100 °C at the top. After time t, how far has the heat (the "20 °C wave") gone? That distance is Lc.

## 5. Geometry is what makes SS → transient meaningful
- **Same geometry:** same local domain for every bead, so the same result for every bead. The end wall doesn't matter. This explains the **Case 2 failure** (the identical SS_new runs).
- **Different geometry:** different local domains for different beads, so the effect of the geometry is captured. Geometry comes in as an input.
- **Key message:** SS → transient makes sense **not as a P, v, h mapping but as a geometry mapping**.
- Sketch: a box with a hole, showing how geometry affects the local domain.
