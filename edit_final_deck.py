"""Apply the supervisor's comments (16-17 Nov 2026) to his copy of the final deck and write
10-minute speaker notes.

    python edit_final_deck.py  IN.pptx  OUT.pptx

Slide 2: references for the three overheating effects + schematics of each effect.
Slide 3: the heat equation and boundary conditions.   Slide 4: what "analytical in nature" and
"numerical experiment" mean; the match is qualitative.   Slide 5: FNO removed, diffusion maths added.
Slides 7-8 and 16: Case 1 is the supervisor's rod (heat flux at the top, bottom fixed, no scan speed),
results from rod_flux_experiment.py (rodflux_outputs/).
"""
import copy
import json
import sys

import numpy as np
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

import rod_flux_sim as sim

A = "final_deck_assets/"
DARK, WHITE, LIGHT = "15161B", "FFFFFF", "D5D8DE"
R = json.load(open("rodflux_outputs/summary.json"))
RUN = {(r["name"], r["n_train"]): r for r in R["runs"]}
W1600, NO, ALLS, ALLO = RUN[("with_ss", 1600)], RUN[("no_ss", 1600)], RUN[("all_params+ss", 1600)], RUN[("all_params_only", 1600)]
GAP = R["steady_as_answer"]


def shp(slide, sid):
    for s in slide.shapes:
        if s.shape_id == sid:
            return s
    raise KeyError(sid)


def inch(s, x=None, y=None, w=None, h=None):
    if x is not None: s.left = Inches(x)
    if y is not None: s.top = Inches(y)
    if w is not None: s.width = Inches(w)
    if h is not None: s.height = Inches(h)


def replace(shape, old, new):
    n = 0
    for p in shape.text_frame.paragraphs:
        for r in p.runs:
            if old in r.text:
                r.text = r.text.replace(old, new); n += 1
    assert n, (shape.shape_id, old)


def set_paras(shape, paras):
    """paras: list of lists of (text, opts) segments. Formatting of the shape's first run is kept;
    opts: bold, size, color, sub, sup, italic, font."""
    tf = shape.text_frame
    p0 = tf.paragraphs[0]
    pPr = copy.deepcopy(p0._p.pPr) if p0._p.pPr is not None else None
    rPr = copy.deepcopy(p0.runs[0]._r.rPr) if p0.runs and p0.runs[0]._r.rPr is not None else None
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    for r in list(p0.runs):
        r._r.getparent().remove(r._r)
    for i, segs in enumerate(paras):
        p = p0 if i == 0 else tf.add_paragraph()
        if i and pPr is not None:
            old = p._p.pPr
            if old is not None:
                p._p.remove(old)
            p._p.insert(0, copy.deepcopy(pPr))
        for text, o in segs:
            r = p.add_run()
            if rPr is not None:
                r._r.insert(0, copy.deepcopy(rPr))
            r.text = text
            f = r.font
            if "bold" in o: f.bold = o["bold"]
            if "italic" in o: f.italic = o["italic"]
            if "size" in o: f.size = Pt(o["size"])
            if "color" in o: f.color.rgb = RGBColor.from_string(o["color"])
            if "font" in o: f.name = o["font"]
            if o.get("sub"): r._r.get_or_add_rPr().set("baseline", "-25000")
            if o.get("sup"): r._r.get_or_add_rPr().set("baseline", "30000")


def S(t, **o):
    return (t, o)


def box(slide, x, y, w, h, fill=DARK):
    b = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    b.adjustments[0] = min(0.5, 0.08 / min(w, h))
    b.fill.solid(); b.fill.fore_color.rgb = RGBColor.from_string(fill); b.line.fill.background(); b.shadow.inherit = False
    return b


def text(slide, x, y, w, h, paras, size=14, color=DARK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Arial"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, segs in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        for t, o in segs:
            r = p.add_run(); r.text = t
            r.font.size = Pt(o.get("size", size)); r.font.name = o.get("font", font)
            r.font.bold = o.get("bold", False); r.font.italic = o.get("italic", False)
            r.font.color.rgb = RGBColor.from_string(o.get("color", color))
            if o.get("sub"): r._r.get_or_add_rPr().set("baseline", "-25000")
            if o.get("sup"): r._r.get_or_add_rPr().set("baseline", "30000")
    return tb


def picture(slide, path, x, y, w, alt):
    p = slide.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))
    p._element.nvPicPr.cNvPr.set("descr", alt)
    return p


def axis_title(axis, t):
    axis.axis_title.text_frame.paragraphs[0].runs[0].text = t


def main(src, dst):
    prs = Presentation(src)
    sl = prs.slides

    # ---- slide 2: references and pictures of the damage -----------------------------------------
    s = sl[1]
    inch(shp(s, 4), w=6.4, h=3.6)
    inch(shp(s, 5), w=6.4, y=5.25); inch(shp(s, 6), w=6.4, y=5.95)
    refs = {9: "Mertens 2014; Sames 2016", 12: "Mohr 2020", 15: "Kastsian 2017"}
    for oval, title, body, y in ((7, 8, 9, 1.55), (10, 11, 12, 2.87), (13, 14, 15, 4.19)):
        inch(shp(s, oval), x=7.3, y=y + 0.02)
        inch(shp(s, title), x=7.9, y=y, w=3.15)
        b = shp(s, body); inch(b, x=7.9, y=y + 0.36, w=3.15, h=0.9)
        first = b.text_frame.text.replace("[Add Reference]", "").strip()
        set_paras(b, [[S(first + " "), S(f"[{refs[body]}]", size=11, italic=True)]])
    for f, y, alt in (("thumb_dross.png", 1.55, "Sketch: dross and melt balls under an overhang"),
                      ("thumb_grains.png", 2.87, "Sketch: fine grains in normal material, coarse grains where it overheated"),
                      ("thumb_recoater.png", 4.19, "Sketch: a curled part edge hit by the recoater blade")):
        picture(s, A + f, 11.15, y, 1.58, alt)
    inch(shp(s, 16), x=7.3, w=5.43); inch(shp(s, 17), x=7.47, w=5.08)

    # ---- slide 3: the heat equation ----------------------------------------------------------------
    s = sl[2]
    pic = shp(s, 4); ratio = pic.width / pic.height
    inch(pic, y=1.45, h=3.45); pic.width = int(pic.height * ratio); pic.left = Inches(0.6) + (Inches(6.5) - pic.width) // 2
    box(s, 0.6, 5.0, 6.5, 1.25)
    text(s, 0.8, 5.05, 6.1, 0.62, [[S("ρc", font="Cambria Math"), S("p", font="Cambria Math", sub=True),
                                      S(" ∂T/∂t = ∇·(k ∇T) + Q", font="Cambria Math")]], size=22, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.8, 5.67, 6.1, 0.5, [[S("baseplate: T = T"), S("0", sub=True), S("   ·   sides: −k ∂T/∂n = 0   ·   top: h(T − T"), S("a", sub=True),
                                     S(") + σε(T"), S("4", sup=True), S(" − T"), S("a", sub=True), S("4", sup=True), S(")")]], size=12, color=LIGHT, align=PP_ALIGN.CENTER)
    inch(shp(s, 17), y=6.35)

    # ---- slide 4: analytical vs numerical; qualitative --------------------------------------------
    s = sl[3]
    set_paras(shp(s, 8), [[S("Analytical in nature: ", bold=True), S("set ∂T/∂t = 0 (t = ∞), so time drops out. For a rod the answer is a formula: T = T"),
                           S("0", sub=True), S(" + qx/k. One solve.")]])
    set_paras(shp(s, 14), [[S("Numerical experiment: ", bold=True), S("the build is replayed step by step in time, like a virtual experiment. Thousands of solves.")]])
    band = shp(s, 17).text_frame
    p = band.add_paragraph(); r = p.add_run()
    r.text = "Even with data-driven methods the match will be qualitative: the aim is to locate the hotspots."
    r.font.size = Pt(15); r.font.bold = False; r.font.color.rgb = RGBColor.from_string(LIGHT); r.font.name = "Arial"
    p.space_before = Pt(6)

    # ---- slide 5: no FNO; the maths of diffusion ---------------------------------------------------
    s = sl[4]
    set_paras(shp(s, 7), [[S("Diffusion: the maths")]])
    el = shp(s, 8)._element; el.getparent().remove(el)            # subtitle not needed
    inch(shp(s, 16), y=3.8, h=1.8)
    for sid in range(9, 16):                                         # the one-to-many icon moves onto the right card
        sh = shp(s, sid); sh.left += Inches(6.18)
        tree = sh._element.getparent(); tree.remove(sh._element); tree.append(sh._element)   # draw it above the card
    set_paras(shp(s, 16), [
        [S("Add noise: ", bold=True), S("x"), S("τ", sub=True), S(" = √ᾱ"), S("τ", sub=True), S(" x"), S("0", sub=True), S(" + √(1 − ᾱ"), S("τ", sub=True), S(") ε,  τ = 1 … 400")],
        [S("Train: ", bold=True), S("network f"), S("θ", sub=True), S("(x"), S("τ", sub=True), S(", τ, steady field, P, v, h) predicts x"), S("0", sub=True),
         S(";  loss ‖f"), S("θ", sub=True), S(" − x"), S("0", sub=True), S("‖"), S("2", sup=True)],
        [S("Generate: ", bold=True), S("start from pure noise, 50 DDIM steps back to x"), S("0", sub=True), S("; average 8 samples")]])
    for r in (p for para in shp(s, 16).text_frame.paragraphs for p in para.runs):
        r.font.size = Pt(14)
    set_paras(shp(s, 18), [[S("Why not the best?")]])
    set_paras(shp(s, 19), [[S("One input, many answers")]])
    for sid in (20, 21, 22):
        el = shp(s, sid)._element; el.getparent().remove(el)
    set_paras(shp(s, 23), [[S("It learns a probability distribution")], [S("Our problem is deterministic: one input, one answer")],
                           [S("We average 8 samples, so it acts like a regressor")]])

    # ---- slide 7: Case 1 is the supervisor's rod -----------------------------------------------------
    s = sl[6]
    old = shp(s, 4); x, y, w = old.left, old.top, old.width
    old._element.getparent().remove(old._element)
    p = s.shapes.add_picture(A + "fig_rodflux_setup.png", x, y, width=w)
    p._element.nvPicPr.cNvPr.set("descr", "Case 1 rod: heat flux at the top for a time t_h, bottom fixed at 65 C, a hidden low-conductivity patch; steady versus transient maximum along the height")
    set_paras(shp(s, 13), [[S("k and patch: only steady shows them")]])
    replace(shp(s, 14), "99", "100")
    set_paras(shp(s, 15), [[S("Steady + ln q, ln t", bold=True), S("h", bold=True, sub=True), S(" → encoder: 64 + 2")]])
    replace(shp(s, 17), "229", "230")
    replace(shp(s, 18), "229 → 256 → 256 → 99", "230 → 256 → 256 → 100")
    replace(shp(s, 19), "99", "100")
    replace(shp(s, 24), "197 k parameters", f"{W1600['params'] / 1000:.0f} k parameters")

    # ---- slide 8: Case 1 results -------------------------------------------------------------------
    s = sl[7]
    replace(s.shapes.title, "1.0%", f"{W1600['mean_abs_peak_err_pct']:.1f}%")
    z = np.load("rodflux_outputs/rodflux_data.npz"); pr = np.load("rodflux_outputs/pred_test.npy")
    pk = np.abs(pr.max(1) - z["tr_test"].max(1)) / (z["tr_test"].max(1) - sim.T0) * 100
    i = int(np.argsort(pk)[150])
    set_paras(shp(s, 4), [[S(f"A typical unseen run (q = {z['test_q'][i]:.2f} W/mm², t"), S("h", sub=True), S(f" = {z['test_t_h'][i]:.2f} s)")]])
    ch = shp(s, 5).chart
    cd = XyChartData()
    for name, ys in (("Steady field (input)", z["ss_test"][i]), ("True transient", z["tr_test"][i]), ("Diffusion prediction", pr[i])):
        ser = cd.add_series(name)
        for a, b in zip(z["x"], ys):
            ser.add_data_point(float(a), float(b))
    ch.replace_data(cd)
    axis_title(ch.category_axis, "Height above the fixed bottom (mm)")
    ch2 = shp(s, 7).chart
    cd2 = CategoryChartData(); cd2.categories = ["Conductivity & patch hidden", "All numbers given"]
    cd2.add_series("With steady field", [W1600["mean_abs_peak_err_pct"], ALLS["mean_abs_peak_err_pct"]])
    cd2.add_series("Without steady field", [NO["mean_abs_peak_err_pct"], ALLO["mean_abs_peak_err_pct"]])
    ch2.replace_data(cd2)
    b = shp(s, 9)
    set_paras(b, [[S(f"Using steady as the answer overshoots the peak by {GAP['mean_peak_overestimate_pct']:.0f}% on average; the model is within "
                     f"{W1600['mean_abs_peak_err_pct']:.1f}% (median {W1600['median_abs_peak_err_pct']:.1f}%).", size=16, color=LIGHT, bold=False)],
                  [S("Lesson: the steady field is valuable only when it carries something q and t", size=19, bold=True, color=WHITE),
                   S("h", size=19, bold=True, color=WHITE, sub=True), S(" do not.", size=19, bold=True, color=WHITE)]])

    # ---- slide 16: key message card ------------------------------------------------------------------
    s = sl[15]
    set_paras(shp(s, 7), [[S(f"{W1600['mean_abs_peak_err_pct']:.1f}% peak error, when the steady field holds information the numbers do not.")]])

    # ---- notes ---------------------------------------------------------------------------------------
    for k, note in enumerate(NOTES):
        sl[k].notes_slide.notes_text_frame.text = note
    prs.save(dst)


W = W1600
NOTES = [
    "[0:15]  Good morning. My BTP is on mapping steady-state temperature to transient temperature in laser powder bed fusion, under Dr. Rajit Ranjan. The question: can a fast steady-state thermal response help us predict hotspots accurately?",

    "[0:45]  (Play the video.) This is a real LPBF machine: the recoater spreads a thin powder layer, then the laser melts the shape; the sparks are spatter. All heat must flow down through the part to the baseplate. Where it cannot escape, for example above an overhang, the part overheats. On the right are the three effects, with references: dross and melt balls on overhangs, coarse grains where cooling is slow, and curled edges that can jam the recoater and stop the build. So the key message: we need to predict the hotspots before printing.",

    "[0:35]  Our training data comes from the layer-by-layer thermal model of Ranjan et al. This is the heat equation it solves: rho c_p dT/dt equals the divergence of k grad T plus the heat source Q. Boundary conditions: the baseplate is held at T0 and acts as the heat sink; the sides touching powder are insulated, because powder conducts only about one percent as well as metal; and the top loses heat by convection and radiation. Material is added layer by layer, the new layer gets the heat flux, and CalculiX solves this with finite elements, giving the temperature at every node and step.",

    "[0:40]  The main motivation is time. A full transient simulation of the paper's test part took 21 hours 28 minutes. The steady-state version took 2 minutes 9 seconds. Steady state is analytical in nature: we set dT/dt to zero, which is time going to infinity, so time drops out; for a rod the answer is a simple formula, T0 plus q x over k. The transient is a numerical experiment: the whole build is replayed step by step in time, like running an experiment on the computer. So the question: can we learn the mapping from steady to transient with data-driven methods? Even then, the match will be qualitative: the aim is to locate the hotspots.",

    "[0:45]  We use a diffusion model, a generative AI method. Three steps. First, add noise: the true transient field x0 is mixed with Gaussian noise; alpha-bar is the fraction of signal left at noise level tau. Second, train: a network sees the noisy field, the noise level, and the steady field with P, v, h, and predicts the clean field x0; the loss is the squared difference. Third, generate: start from pure noise and clean it in 50 steps, and average 8 samples. Why it might not be the best: it learns a probability distribution, but our problem is deterministic, one input, one answer. Averaging makes it behave like a regressor. We are trying it, and we say where it does not fit.",

    "[0:15]  Three cases, each closer to a real part: a 1D rod, a single bead from the real thermal model, and a part with holes, where the geometry changes.",

    "[0:40]  Case 1 is a 1D rod, 10 mm long. A heat flux q enters at the top for a heating time t_h; the bottom is fixed at 65 degrees; there is no scan speed. Inside the rod there is a low-conductivity patch at a hidden depth. The conductivity and the patch are not given to the model, so only the steady field can show them. The plot shows one run: the steady field rises all along the rod, the transient only near the top. 1,600 training runs, 300 unseen test runs; the same small network as before.",

    f"[0:45]  On runs it never saw, the model's mean peak error is {W['mean_abs_peak_err_pct']:.1f} percent, median {W['median_abs_peak_err_pct']:.1f}. Using the steady field directly would overshoot the peak by about {GAP['mean_peak_overestimate_pct']:.0f} percent, so the two are very different. The lesson is on the right: with the conductivity and patch hidden, removing the steady field raises the error from {W['mean_abs_peak_err_pct']:.1f} to {NO['mean_abs_peak_err_pct']:.0f} percent. When all numbers are given, the steady field adds almost nothing, {ALLS['mean_abs_peak_err_pct']:.1f} against {ALLO['mean_abs_peak_err_pct']:.1f} percent. So the steady field helps only when it carries information the inputs do not.",

    "[0:30]  Case 2 is one bead on a plate from the real thermal model, with 27 settings of power, velocity and hatch. The transient is solved in time; the steady case holds the laser at each position. On unseen settings the mean peak error is 1.3 percent, worst 5.1. It looks very accurate. But where does the accuracy come from?",

    "[0:45]  Look at 60 watts and 120 watts: very different settings, but the steady fields are identical, 6,667 degrees peak in every run. All 27 steady runs differ by at most 0.12 degrees, while the transient peak goes from about 2,650 to 9,300. Why? All runs use one standard geometry, the same bead on the same plate, so the local steady model sees the same domain every time. The steady field carries no information, and the accuracy came only from P, v and h. To see why, a little physics.",

    "[0:35]  Think of welding: a melt pool under the source, isothermal lines around it, and the new bead heating the older beads. Steady state means depositing and waiting for infinite time, so heat would reach the whole part. That is not physical, because the next bead arrives much sooner. Closer to reality: solve the steady problem only on the local domain that heat can reach in the time available.",

    "[0:40]  How far does heat go? Take a bar at 20 degrees with one end at 100. After time t, heat has gone about L_c equal to the square root of alpha t, with alpha equal to k over rho c. It moves like a wave that slows down: four times the time, only twice the distance. Our Case 1 rod shows this directly: a low-conductivity patch within one L_c of the top raised the transient peak by 42 percent, but a patch deeper than two L_c changed it by 0 percent, while the steady peak rose 40 to 50 percent wherever the patch was. In the Case 3 script alpha is 5.18 and t is 1 second, so L_c is 2.28 mm.",

    "[0:35]  Now Case 2 makes sense. On a plain plate every bead sees the same hemisphere, and even the end wall lies outside it: same domain, same steady result. That is the Case 2 failure. With a hole or a wall nearby, part of the domain is missing, so the domains differ from bead to bead. Geometry becomes the input.",

    "[0:35]  Case 3 is a 20 by 20 mm block, 10 layers, with a hole and a slot, built one element at a time: 3,844 pairs. Now the steady field changes with position. Both steady and transient peaks rise next to the walls, the hole and the slot, and they correlate at 0.88 after element 1,364.",

    "[0:45]  We trained the same diffusion model on layers 1 to 8 and tested on layers 9 and 10, where the round hole is new. The error is about 1,075, against 2,634 for guessing the average: about 2.5 times better. And given only the shape, where solid and holes are, it does as well, about 890. So here the information is the geometry itself. Read these with care: the temperatures are around 100,000 degrees, which is not physical, the two runs use different reference temperatures, and it is one part.",

    f"[0:30]  To conclude. Case 1: the idea works, {W['mean_abs_peak_err_pct']:.1f} percent, when the steady field holds information the numbers do not. Case 2: one standard geometry gives the same local domain and the same steady field, so the accuracy came from P, v and h. Case 3: geometry changes both fields and the model learns it. So steady to transient makes sense not as a P, v, h mapping, but as a geometry mapping. Thank you.",

    "(Leave on screen during questions.)",
]

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
