# Geometry dataset (Trans_to_SS_dataset, part_1): data analysis, 26 Oct 2026

Scripts: tts_data.py (loader), tts_analyze.py (analysis). Outputs: tts_outputs/elements.csv (one row per element),
analysis_log.txt, fig_peak_maps_ss_vs_tr.png, fig_ss_peak_maps.png, fig_peaks_by_element.png.

## What the data is
- One part, 22 x 22 elements in plan, 10 layers of 1 mm on a 3-layer substrate; Inconel 718, P 260, v 1, hatch 0.11.
- The part is a 20 x 20 block with voids: a 2 x 2 hole (layers 1-8) that widens into a round hole (layers 9-10), and a 6 x 2 slot (from layer 6).
- Built element by element: 3,844 elements; each is heated for 1 s, then 5 s cooling after each layer.
- SS: 3,853 frames (3,844 element steps + 9 cooling). Steady solve per element; sink 65 C outside a hemisphere of radius 3.41 mm around the new element.
- Trans: 23,134 frames; each element step has 6 sub-steps (0.1 ... 1.0 s); base held at 180 C; film + radiation.
- 7,406 nodes per frame (23 x 23 x 14).

## Pairing
By element number (n-th SS element step <-> n-th transient heating step): 3,844 pairs, all complete.
Pairing by time fails for 27 elements because the two runs place their cooling steps differently (see 3).

## Good
- The steady peak now changes with the element (3,033 distinct values, 16 % spread), unlike SS_new.
- Both steady and transient peaks rise next to walls and voids (median steady 141,853 -> 165,602 -> 202,351 C for 0 / 1 / 2 empty neighbours; transient 117,892 -> 120,189 -> 129,543 C).
- Steady and transient peaks correlate 0.65 over all elements, 0.88 from element 1,365 on.

## Problems
1. Temperatures are not physical: steady peaks 95,000-318,000 C, transient peaks ~118,000 C. Likely the heat input
   (energy density uses a 0.1 mm nominal layer while elements are 1 mm, v = 1 mm/s). Relative patterns may still be usable.
2. Different reference temperatures: transient starts at and holds the base at 180 C; the steady sink is 65 C. The transient has
   film + radiation, the steady run has none.
3. SS script, local domain in the wrong place (most important): when the sink sets are built, the hemisphere centre uses
   new_id = sub_ele_no + ele + 1 and misses the running count ele_number, while the heated element is sub_ele_no + ele_number + ele + 1.
   So after the first group of 484 elements the hemisphere is centred 9-25 mm from the heated block (correct for only 484 of 3,844).
   node_active is marked with the same wrong ids, so only the substrate, layer 1 and the first 88 blocks of layer 2 can ever be held
   at 65 C: a "cold floor". Layer 2 sits directly on it (median steady peak 102,694 C, the dark band); higher layers are farther from
   it, so the steady peak rises with height. (The script also groups elements by the substrate layer size, so its cooling steps fall
   after elements 484, 968 ...; for a steady solve this alone should not matter.)
4. Transient values below the 180 C base in 333 element steps (151 below 0 C, minimum -1,145 C), most in layer 1.
5. Diffusivity: the SS hemisphere uses a hard-coded 5.18 mm^2/s; the files give k/(rho c) = 39.3 mm^2/s (the Inconel 718 library
   file actually used is not included, so the true value is unknown).
6. Only one geometry: 3,844 element pairs, but no second part to test generalisation to a new geometry.

## Questions for Rajit / sir
- Is the heat input (and so the 10^5 C range) intended, or should the energy density use the real 1 mm layer?
- Can the SS script use the part layers (396, 396, ... 356) instead of the substrate layer counts, and the same 180 C reference?
- Which diffusivity should define the hemisphere?
- How many parts / geometries will there be?
