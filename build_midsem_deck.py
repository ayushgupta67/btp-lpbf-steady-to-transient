"""Build the mid-semester presentation.

    python make_deck_assets.py && python build_midsem_deck.py

Structure: background (one slide per earlier meet), the wall dataset (meet 6),
then the single-bead work (the main part), conclusions, references.
Every number comes from deck_assets/deck_data.json, which make_deck_assets.py
writes from the saved experiment results.
"""

import json
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent
ASSETS = ROOT / "deck_assets"
D = json.loads((ASSETS / "deck_data.json").read_text())
OUT_FILE = ROOT / "BTP_midsem_presentation.pptx"

# ---- design system -------------------------------------------------------
DARK, INK, MUTED, FAINT = "15161B", "1F2128", "5E6573", "8A909C"
LINE, PANEL, WHITE = "DADDE3", "F2F3F5", "FFFFFF"
RED, RED_DK, TEAL, GRAY, AMBER = "C8102E", "8E0B21", "1F7A8C", "B9BEC7", "C27C0E"
HEAD, BODY = "Cambria", "Calibri"
LM, CW, TOP = 0.6, 12.133, 1.62
SW, SH = 13.333, 7.5


def rgb(h):
    return RGBColor.from_string(h)


def R(text, **kw):
    return (text, kw)


def P(*runs, **kw):
    return {"runs": runs, **kw}


# ---- text ------------------------------------------------------------------
def _style(run, base, kw):
    f = run.font
    f.name = kw.get("font", base["font"])
    f.size = Pt(kw.get("size", base["size"]))
    f.bold = kw.get("bold", base["bold"])
    f.italic = kw.get("italic", base["italic"])
    f.color.rgb = rgb(kw.get("color", base["color"]))
    if "spc" in kw:
        run._r.get_or_add_rPr().set("spc", str(kw["spc"]))


def _bullet(p, color, indent=0.24):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent))))
    pPr.set("indent", str(-int(Inches(indent))))
    clr = etree.SubElement(pPr, qn("a:buClr"))
    etree.SubElement(clr, qn("a:srgbClr")).set("val", color)
    etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
    etree.SubElement(pPr, qn("a:buChar")).set("char", "•")


def fill_text(tf, paras, size=14, color=INK, font=BODY, bold=False, italic=False,
              align="left", anchor="top", line_spacing=None, space_after=0):
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}[anchor]
    base = dict(size=size, color=color, font=font, bold=bold, italic=italic)
    if isinstance(paras, (str, dict)):
        paras = [paras]
    for i, spec in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        spec = {"runs": (spec,)} if isinstance(spec, (str, tuple)) else spec
        pb = dict(base)
        for key in ("size", "color", "font", "bold", "italic"):
            if key in spec:
                pb[key] = spec[key]
        p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[spec.get("align", align)]
        ls = spec.get("line_spacing", line_spacing)
        if ls:
            p.line_spacing = ls
        p.space_after = Pt(spec.get("space_after", space_after))
        for r in spec["runs"]:
            text, kw = (r, {}) if isinstance(r, str) else r
            run = p.add_run()
            run.text = text
            _style(run, pb, kw)
        if spec.get("bullet"):
            _bullet(p, spec.get("bullet_color", RED))


def text(slide, x, y, w, h, paras, **kw):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    fill_text(tb.text_frame, paras, **kw)
    return tb


def label(slide, x, y, w, s, color=RED, size=11, align="left"):
    return text(slide, x, y, w, 0.25, P(R(s.upper(), bold=True, spc=120)), size=size, color=color, align=align)


# ---- shapes ----------------------------------------------------------------
def box(slide, x, y, w, h, fill=PANEL, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, lw=1.0):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        shp.adjustments[0] = min(0.5, radius / min(w, h))
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(fill)
    else:
        shp.fill.background()
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(lw)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def stat(slide, x, y, w, h, number, caption, fill=PANEL, num_color=RED, cap_color=MUTED, num_size=34, cap_size=13):
    box(slide, x, y, w, h, fill=fill)
    text(slide, x + 0.2, y + 0.12, w - 0.4, h * 0.5, P(R(number, bold=True)), size=num_size, color=num_color,
         font=HEAD, anchor="middle")
    caption = caption.split("\n") if isinstance(caption, str) else caption
    text(slide, x + 0.2, y + h * 0.55, w - 0.4, h * 0.42, caption, size=cap_size, color=cap_color)


def numdot(slide, x, y, n, d=0.38, fill=RED, color=WHITE, size=14):
    c = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    fill_text(c.text_frame, P(R(str(n), bold=True)), size=size, color=color, align="center", anchor="middle")
    return c


def arrow(slide, x, y, w, h, fill=GRAY, direction="right"):
    shape = {"right": MSO_SHAPE.RIGHT_ARROW, "down": MSO_SHAPE.DOWN_ARROW}[direction]
    return box(slide, x, y, w, h, fill=fill, shape=shape)


def picture(slide, path, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def table(slide, x, y, col_w, rows, row_h=0.4, size=13, head_fill=DARK, bold_col0=True,
          highlight=(), align=None, head_size=None):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                                   Inches(sum(col_w)), Inches(row_h * len(rows)))
    tbl = shape.table
    tbl._tbl.tblPr.set("firstRow", "0")
    tbl._tbl.tblPr.set("bandRow", "0")
    for j, w in enumerate(col_w):
        tbl.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            head = i == 0
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(head_fill if head else ("FBE9EC" if i in highlight else (PANEL if i % 2 == 0 else WHITE)))
            cell.margin_left = cell.margin_right = Inches(0.1)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            a = (align[j] if align else ("left" if j == 0 else "center"))
            fill_text(cell.text_frame, P(R(str(val), bold=head or (j == 0 and bold_col0) or (i in highlight))),
                      size=(head_size or size) if head else size, color=WHITE if head else INK, align=a, anchor="middle")
            cell.text_frame.margin_left = cell.text_frame.margin_right = 0
    return shape


# ---- charts ----------------------------------------------------------------
def style_chart(chart, size=12, legend=True):
    chart.font.size = Pt(size)
    chart.font.name = BODY
    chart.font.color.rgb = rgb(INK)
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)


def style_axes(chart, vtitle=None, htitle=None, size=12, vmax=None, vmin=None, grid=True, vfmt=None):
    va, ca = chart.value_axis, chart.category_axis
    va.has_major_gridlines = grid
    if grid:
        va.major_gridlines.format.line.color.rgb = rgb(LINE)
        va.major_gridlines.format.line.width = Pt(0.75)
    va.format.line.fill.background()
    ca.format.line.color.rgb = rgb(GRAY)
    va.tick_labels.font.size = ca.tick_labels.font.size = Pt(size)
    if vfmt:
        va.tick_labels.number_format = vfmt
        va.tick_labels.number_format_is_linked = False
    if vmax is not None:
        va.maximum_scale = vmax
    if vmin is not None:
        va.minimum_scale = vmin
    for ax, t in ((va, vtitle), (ca, htitle)):
        if t:
            ax.has_title = True
            ax.axis_title.text_frame.text = t
            r = ax.axis_title.text_frame.paragraphs[0].runs[0]
            r.font.size, r.font.bold, r.font.name = Pt(size), False, BODY
            r.font.color.rgb = rgb(MUTED)


def color_series(ser, color):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = rgb(color)


def bar_chart(slide, x, y, w, h, cats, series, colors, kind=XL_CHART_TYPE.COLUMN_CLUSTERED, labels=True,
              fmt="0.00", legend=None, gap=60, overlap=None, size=12, **axis):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series:
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    chart = gf.chart
    style_chart(chart, size, legend if legend is not None else len(series) > 1)
    plot = chart.plots[0]
    plot.gap_width = gap
    if overlap is not None:
        plot.overlap = overlap
    for ser, c in zip(plot.series, colors):
        color_series(ser, c)
    if labels:
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.font.size = Pt(size)
        dl.number_format = fmt
        dl.number_format_is_linked = False
        if kind in (XL_CHART_TYPE.COLUMN_CLUSTERED, XL_CHART_TYPE.BAR_CLUSTERED):
            dl.position = XL_LABEL_POSITION.OUTSIDE_END
    style_axes(chart, size=size, **axis)
    return chart


# ---- slide scaffolding -------------------------------------------------------
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
prs.core_properties.title = "Learning Transient Temperature Fields from Steady-State Fields in LPBF"
prs.core_properties.author = "Ayush"
KINDS = []  # per slide: "title", "dark", "light"


def new_slide(title, section=None, dark=False, kind=None):
    s = prs.slides.add_slide(prs.slide_layouts[5])  # Title Only
    KINDS.append(kind or ("dark" if dark else "light"))
    if dark:
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(DARK)
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(LM), Inches(0.62), Inches(CW), Inches(0.8)
    fill_text(t.text_frame, P(R(title, bold=True)), size=28, color=WHITE if dark else INK, font=HEAD, anchor="top")
    if section:
        label(s, LM, 0.34, 9, section, color="E8344E" if dark else RED)
    return s


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def foot(slide, n, total, dark):
    c = FAINT
    text(slide, LM, 7.05, 8, 0.25, "BTP mid-semester presentation  ·  steady → transient mapping in LPBF",
         size=10, color=c)
    text(slide, SW - LM - 1.2, 7.05, 1.2, 0.25, f"{n} / {total}", size=10, color=c, align="right")


def pct(x):
    return f"{x:.1f}%"


# =============================================================================
# 1  Title
# =============================================================================
s = new_slide("Learning Transient Temperature Fields from Steady-State Fields in Laser Powder Bed Fusion",
              dark=True, kind="title")
s.shapes.title.left, s.shapes.title.top, s.shapes.title.width, s.shapes.title.height = (
    Inches(LM), Inches(2.05), Inches(7.6), Inches(2.3))
fill_text(s.shapes.title.text_frame, P(R("Learning Transient Temperature Fields from Steady-State Fields in "
                                          "Laser Powder Bed Fusion", bold=True)), size=36, color=WHITE, font=HEAD)
label(s, LM, 1.55, 7, "BTP mid-semester presentation", color="E8344E", size=12)
text(s, LM, 4.55, 7.4, 1.0, "Conditional diffusion models: literature, prototypes and first results on "
     "simulated LPBF data", size=18, color="C9CCD3")
text(s, LM, 6.35, 7, 0.4, P(R("Ayush", bold=True)), size=18, color=WHITE)
# decorative heat map: a laser track with a cooling tail
cols, rows_, cell, gap = 11, 7, 0.36, 0.07
x0, y0 = 8.55, 1.9
base = (0x23, 0x25, 0x2C)
hot = (0xE8, 0x34, 0x4E)
for r in range(rows_):
    for c in range(cols):
        d = abs(r - 3)
        k = 0.0
        if 0 < c <= 8:
            k = max(0.0, 1.0 - 0.13 * (8 - c)) * (1.0 if d == 0 else 0.35 if d == 1 else 0.12 if d == 2 else 0.0)
        col = "".join(f"{int(base[i] + (hot[i] - base[i]) * k):02X}" for i in range(3))
        box(s, x0 + c * (cell + gap), y0 + r * (cell + gap), cell, cell, fill=col, radius=0.05)
text(s, x0, y0 + rows_ * (cell + gap) + 0.1, 4.6, 0.3, "laser track and its cooling tail (schematic)", size=11, color=FAINT)
notes(s, "Introduce the goal in one sentence: learn the map from a cheap steady-state temperature field to the "
         "expensive transient field in laser powder bed fusion, using a conditional diffusion model. "
         "The talk goes: why, what we learned from the literature (meets 1-5), the wall dataset (meet 6), "
         "and the new single-bead parametric dataset, which is the main new work.")

# =============================================================================
# 3  Journey
# =============================================================================
s = new_slide("What we tried and what we learnt", "Overview")
text(s, LM, 1.1, CW, 0.35, P(R("Project goal: ", bold=True, color=RED),
     R("use diffusion models to turn a cheap steady-state field into the accurate transient field in LPBF.")),
     size=14, color=MUTED)
steps = [("Meet 1", "Diffusion basics", "Forward noising, reverse denoising; uses in AM",
          "A diffusion model learns to remove noise one step at a time.", TEAL),
         ("Meet 3", "Bridging scales", "Coarse-to-fine super-resolution with diffusion",
          "Our task is fidelity enhancement: cheap input, accurate output.", TEAL),
         ("Meet 4", "Physics-informed", "A physics penalty inside the diffusion loss",
          "Physics can be added to training; be loose while the field is noisy.", TEAL),
         ("Meet 5", "1D rod prototype", "Baseline vs physics-informed on 1,000 synthetic rods",
          "Results were poor: sampling errors snowball and we had no fair test.", TEAL),
         ("Meet 6", "LPBF wall data", "First real LPBF fields: steady vs transient",
          "One build is about 80 samples; random splits leak; ridge beat diffusion.", RED),
         ("Now", "Single-bead data", "27 runs from the thermal model of [1]",
          "Temperatures follow P/hatch and speed; the limit is data volume.", RED)]
cw, gap = 1.9, 0.146
text(s, LM, 1.55, 4 * (cw + gap) - gap, 0.3, P(R("LITERATURE AND PROTOTYPE", bold=True, spc=120)), size=12, color=TEAL)
text(s, LM + 4 * (cw + gap), 1.55, 2 * cw + gap, 0.3, P(R("REAL LPBF SIMULATION DATA", bold=True, spc=120)), size=12, color=RED)
box(s, LM + cw / 2, 2.2, 5 * (cw + gap), 0.04, fill=GRAY, shape=MSO_SHAPE.RECTANGLE)
for i, (m, t, d, l, c) in enumerate(steps):
    x = LM + i * (cw + gap)
    box(s, x + cw / 2 - 0.16, 2.06, 0.32, 0.32, fill=c, shape=MSO_SHAPE.OVAL)
    box(s, x, 2.6, cw, 4.2, fill=PANEL)
    text(s, x + 0.15, 2.75, cw - 0.3, 0.28, P(R(m.upper(), bold=True, spc=100)), size=12, color=c)
    text(s, x + 0.15, 3.07, cw - 0.3, 0.7, P(R(t, bold=True)), size=17, font=HEAD)
    text(s, x + 0.15, 3.85, cw - 0.3, 0.25, P(R("WHAT WE DID", bold=True, spc=80)), size=10, color=FAINT)
    text(s, x + 0.15, 4.1, cw - 0.3, 1.0, d, size=12, color=MUTED)
    box(s, x + 0.15, 5.12, cw - 0.3, 0.02, fill=LINE, shape=MSO_SHAPE.RECTANGLE)
    text(s, x + 0.15, 5.22, cw - 0.3, 0.25, P(R("WHAT WE LEARNT", bold=True, spc=80)), size=10, color=RED)
    text(s, x + 0.15, 5.47, cw - 0.3, 1.25, l, size=12.5, color=INK)
notes(s, "Walk left to right. The first five meets built the method and a prototype whose results were poor; each failure produced a lesson "
         "that shaped the real-data work from meet 6 on. Today's work is the first dataset with process parameters.")

# =============================================================================
# 4  Meet 1: diffusion in one slide
# =============================================================================
s = new_slide("Diffusion models in one slide", "Meet 1  \u00b7  Background")
blocks = [("FORWARD", "Add Gaussian noise step by step (a Markov chain) until the field is pure noise.",
           "x\u209c = \u221a\u03b1\u0304\u209c x\u2080 + \u221a(1\u2212\u03b1\u0304\u209c) \u03b5"),
          ("TRAINING", "A network learns to name the noise that was added, at every noise level.",
           "loss = \u2016\u03b5 \u2212 \u03b5\u03b8(x\u209c, t)\u2016\u00b2"),
          ("REVERSE", "Start from pure noise and repeatedly remove the predicted noise. A different start gives a different sample.",
           "x\u209c\u208b\u2081 from x\u209c and \u03b5\u03b8")]
for i, (name, desc, eq) in enumerate(blocks):
    x = LM + i * 4.07
    box(s, x, 1.55, 3.9, 1.95, fill=PANEL)
    label(s, x + 0.22, 1.7, 3.4, name, size=11)
    text(s, x + 0.22, 1.98, 3.5, 0.95, desc, size=13.5, color=MUTED)
    text(s, x + 0.22, 3.0, 3.5, 0.4, P(R(eq, italic=True)), size=16, font=HEAD)
pic = picture(s, ASSETS / "fig_forward_process.png", LM + 0.9, 3.65, w=CW - 1.8)
bottom = (pic.top + pic.height) / 914400
text(s, LM, bottom + 0.04, CW, 0.3, "Forward process on one of our own bead fields: the hot spot dissolves into noise, and the network "
     "is trained to undo it. Equations: Ho et al. [2].", size=11.5, color=MUTED, align="center")
box(s, LM, 6.12, CW, 0.78, fill=DARK)
text(s, LM + 0.25, 6.12, CW - 0.5, 0.78, [P(R("Used in additive manufacturing  ", bold=True, color="E8344E"),
     R("layer-wise defect monitoring from a model of perfect layers [5];  high-fidelity melt-pool prediction at low cost [4].", color=WHITE))],
     size=14, anchor="middle")
notes(s, "Meet 1 explained diffusion with analogies (ink in water, TV static, the Markov chain 'telephone game'). "
         "The picture is a real field from our own data run through the forward process at the same noise "
         "levels our model uses. [4] and [5] are the two additive-manufacturing papers from meet 1: "
         "(5) learns only what a perfect printed layer looks like and flags defects as the gap to a flawed scan; "
         "(4) is the melt-pool paper that meet 3 builds on.")

# =============================================================================
# 5  Meet 3: bridging scales
# =============================================================================
s = new_slide("Diffusion as fidelity enhancement: coarse to fine", "Meet 3  ·  Bridging scales")
box(s, LM, 1.62, 3.85, 3.5, fill=PANEL)
label(s, LM + 0.22, 1.78, 3.4, "The paper (meso-scale)")
text(s, LM + 0.22, 2.12, 3.4, 0.8, P(R("20 μm → 5 μm", bold=True)), size=30, font=HEAD, color=RED)
text(s, LM + 0.22, 2.95, 3.4, 2.1, "A diffusion model upscales a melt-pool simulation computed at 20 μm resolution to a 5 μm "
     "prediction, at a fraction of the cost [4].", size=14)
box(s, 4.7, 1.62, 3.95, 3.5, fill=PANEL)
label(s, 4.92, 1.78, 3.5, "Our reading (macro-scale)")
text(s, 4.92, 2.12, 3.55, 0.8, P(R("coarse mesh → fine mesh", bold=True)), size=19, font=HEAD, color=RED)
text(s, 4.92, 2.85, 3.55, 2.2, [
    P(R("Predict the fine-mesh result directly from a fast coarse solve."), space_after=6),
    P(R("Illustrative examples: ", bold=True), R("a coarse FEA mesh averages a peak stress of 500 down to 420; a coarse CFD mesh cannot resolve turbulent eddies."))],
     size=13.5)
box(s, 8.8, 1.62, 3.93, 3.5, fill=PANEL)
label(s, 9.02, 1.78, 3.5, "Related work scanned")
text(s, 9.02, 2.12, 3.55, 2.95, [
    P(R("Physics-informed diffusion for flow-field reconstruction (Shu et al., 2023)"), bullet=True, space_after=4),
    P(R("“Stressed”: diffusion for 2D stress estimation (Jadhav et al., 2023)"), bullet=True, space_after=4),
    P(R("SRDiff, SR3 image super-resolution (2022)"), bullet=True, space_after=4),
    P(R("GAN-based super-resolution of flows (2019 to 2022)"), bullet=True)], size=13)
box(s, LM, 5.4, CW, 1.2, fill=DARK)
text(s, LM + 0.3, 5.5, CW - 0.6, 1.0, [
    P(R("Link to this project  ", bold=True, color="E8344E"),
      R("steady state is the cheap, low-fidelity input and the transient field the expensive, high-fidelity "
        "output; process parameters (P, v, hatch) are extra conditions.", color=WHITE))],
     size=16, anchor="middle")
notes(s, "Meet 3 also covered the forward and reverse equations and an FDM experimental setup; those are in the "
         "meet 3 deck. The stress and eddy numbers are illustrative examples used in the meet, not measured data. "
         "The related-work list comes from the references of the melt-pool paper [4].")

# =============================================================================
# 6  Meet 4: PIDM
# =============================================================================
s = new_slide("Physics-informed diffusion models (PIDM)", "Meet 4  ·  Bastek, Sun & Kochmann, ICLR 2025")
box(s, LM, 1.62, 5.6, 3.55, fill=PANEL)
label(s, LM + 0.22, 1.78, 5, "The idea")
text(s, LM + 0.22, 2.12, 5.2, 3.0, [
    P(R("Diffusion models fit data but never see the governing equations, so a sample can look right and still break the physics."), bullet=True, space_after=7),
    P(R("PIDM adds a physics-residual penalty to the training loss, as PINNs do."), bullet=True, space_after=7),
    P(R("Difficulty: physics is only meaningful on a clean field, but training sees noisy ones. Rule: loose penalty at noisy steps, strict near clean."), bullet=True)],
     size=14)
text(s, 6.45, 1.62, 6.28, 0.3, P(R("TWO WAYS TO ESTIMATE THE CLEAN FIELD", bold=True, spc=120)), size=11, color=RED)
table(s, 6.45, 1.95, [1.9, 2.19, 2.19], [
    ["", "Mean estimation", "Sample estimation"],
    ["How", "reuse the network's own guess", "DDIM jump to a real sample"],
    ["Training cost", "about +23%", "about +69%"],
    ["Won on", "Darcy flow", "topology optimization"]], row_h=0.5, size=12.5)
stat(s, 6.45, 4.15, 2.0, 1.02, "~100×", "lower physics violation", num_size=26, cap_size=11.5)
stat(s, 8.57, 4.15, 2.0, 1.02, "33–35%", "lower residual, unseen loads", num_size=26, cap_size=11.5)
stat(s, 10.69, 4.15, 2.04, 1.02, "0", "extra cost at generation", num_size=26, cap_size=11.5)
box(s, LM, 5.42, CW, 1.15, fill="FBE9EC")
text(s, LM + 0.3, 5.5, CW - 0.6, 1.0, [
    P(R("For us  ", bold=True, color=RED_DK),
      R("a physics term can be added to the training loss without changing generation. "
        "Limitation noted by the authors: pixel grids only; graph networks would be needed for arbitrary 3D meshes, like ours."))],
     size=15, anchor="middle")
notes(s, "Mean estimation reuses the network's current prediction of the clean field (the Jensen gap makes it slightly "
         "inaccurate for non-linear physics). Sample estimation uses DDIM to jump to a real sample, which is more "
         "accurate and costs more. The headline results (about 100x less physics violation, 33-35% residual reduction "
         "even on unseen boundary conditions) are as reported in the paper.")

# =============================================================================
# 7  Meet 5: poor results
# =============================================================================
s = new_slide("1D rod prototype: the results were poor", "Meet 5  ·  First implementation")
stat(s, LM, 1.62, 1.95, 1.2, "1,000", "random rods", num_size=28)
stat(s, LM + 2.07, 1.62, 1.95, 1.2, "11", "nodes per rod", num_size=28)
stat(s, LM + 4.14, 1.62, 1.95, 1.2, "8,000", "training pairs", num_size=28)
label(s, LM, 3.05, 6, "Network input: 55 numbers")
segs = [("11", "noisy transient", TEAL, 1.75), ("11", "steady-state hint", DARK, 1.75), ("1", "time", RED, 0.85), ("32", "step code", MUTED, 1.75)]
x = LM
for n, cap, c, ww in segs:
    box(s, x, 3.4, ww - 0.05, 0.6, fill=c, radius=0.05)
    text(s, x, 3.4, ww - 0.05, 0.6, P(R(n, bold=True)), size=15, color=WHITE, align="center", anchor="middle")
    text(s, x - 0.05, 4.05, ww + 0.1, 0.5, cap, size=11, color=MUTED, align="center")
    x += ww
text(s, LM, 4.7, 6.1, 1.4, [
    P(R("ResNet MLP, 3 × 256, outputs the 11-node noise."), bullet=True, space_after=6),
    P(R("Physics-informed variant: penalty on (∂T/∂t − α ∂²T/∂x²)², only for noise steps τ < 200, weight 0.01."), bullet=True)],
     size=14)
box(s, 7.0, 1.62, 5.73, 4.5, fill=PANEL)
label(s, 7.25, 1.78, 5, "What we saw")
for i, (lab, d) in enumerate([("999", 0.26), ("750", 0.34), ("500", 0.42), ("250", 0.52), ("0", 0.62)]):
    cx = 7.55 + i * 1.07
    box(s, cx, 2.5 - d / 2, d, d, fill=RED if i == 4 else "D9808F", shape=MSO_SHAPE.OVAL)
    text(s, cx - 0.3, 2.9, d + 0.6, 0.3, f"step {lab}", size=10.5, color=MUTED, align="center")
obs = ["The physics loss reached the millions until it was masked to low-noise steps.",
       "After masking, MAE improved 100×, yet predictions still zigzagged between −400 and +300 (the rod sits near 180).",
       "A 1% error at the first reverse step compounds over 1,000 steps: the snowball effect."]
for i, t in enumerate(obs, 1):
    numdot(s, 7.25, 3.4 + (i - 1) * 0.88, i, d=0.34, size=13)
    text(s, 7.75, 3.4 + (i - 1) * 0.88, 4.75, 0.85, t, size=13.5)
box(s, LM, 6.3, CW, 0.55, fill=DARK)
text(s, LM + 0.3, 6.3, CW - 0.6, 0.55, [P(R("Verdict  ", bold=True, color="E8344E"),
     R("not usable yet. The next slide shows why, and what we changed because of it.", color=WHITE))], size=15, anchor="middle")
notes(s, "This was the Colab prototype on synthetic 1D rods with random piecewise conductivity. Be direct: it did not work. "
         "The value of the exercise is in the lessons on the next slide.")

# =============================================================================
# 8  Meet 5: lessons
# =============================================================================
s = new_slide("Why it failed, and what we changed", "Meet 5  ·  Lessons")
rows_ = [["What went wrong", "What we learnt", "What we changed later"],
         ["A 1% error compounded over 1,000 reverse steps", "Long sampling chains amplify small errors",
          "DDIM with 50 steps; the network predicts the clean field"],
         ["The physics loss reached the millions at high noise", "Dividing by √ᾱ ≈ 0.003 magnifies errors 100 to 300×",
          "Physics term only at low noise (τ < 200), small weight"],
         ["The final plot used a training sample; no held-out data", "A score on data the model has seen proves nothing",
          "Held-out time blocks (wall); leave-one-run-out (bead)"],
         ["Synthetic 1D rods with 8 snapshots each", "A toy dataset is too simple and too small",
          "Real LPBF simulation data from the thermal model of [1]"]]
table(s, LM, 1.7, [4.1, 3.9, 4.13], rows_, row_h=0.9, size=14, align=["left", "left", "left"])
box(s, LM, 6.3, CW, 0.55, fill=DARK)
text(s, LM + 0.3, 6.3, CW - 0.6, 0.55, [P(R("Result  ", bold=True, color="E8344E"),
     R("these four changes are what the wall and single-bead experiments are built on.", color=WHITE))], size=15, anchor="middle")
notes(s, "Each row ties a failure in the 1D rod to something concrete we do now. The 100-300x amplification is the factor quoted in the meet 5 deck "
         "(dividing by the square root of alpha-bar when reconstructing the clean field at high noise). The final plot of the Colab code drew a sample from the training set.")

# =============================================================================
# 9  Meet 6: wall dataset
# =============================================================================
s = new_slide("The wall dataset: one build, 480 frames", "Meet 6  ·  First real LPBF data")
cd = CategoryChartData()
cd.categories = ["3,696 cells"]
for n, v in (("ambient 2,602", 2602), ("plate 924", 924), ("wall 170", 170)):
    cd.add_series(n, [v])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED, Inches(LM), Inches(1.6), Inches(7.2), Inches(1.9), cd)
ch = gf.chart
style_chart(ch, 12, True)
ch.plots[0].gap_width = 35
ch.plots[0].overlap = 100
for ser, c in zip(ch.plots[0].series, (GRAY, TEAL, RED)):
    color_series(ser, c)
for ser, show, lc in zip(ch.plots[0].series, (True, True, False), (INK, WHITE, WHITE)):
    dl = ser.data_labels
    dl.show_value = show
    dl.font.size = Pt(14)
    dl.font.bold = True
    dl.font.color.rgb = rgb(lc)
    dl.number_format = "#,##0"
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.CENTER
ch.value_axis.minimum_scale = 0
ch.value_axis.maximum_scale = 3696
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
ch.category_axis.visible = False
stat(s, LM, 3.7, 2.3, 1.3, "8×22×21", "cells = 3,696", num_size=26)
stat(s, LM + 2.45, 3.7, 2.3, 1.3, "1,094", "material cells", num_size=26)
stat(s, LM + 4.9, 3.7, 2.3, 1.3, "80 × 6", "= 480 frames", num_size=26)
text(s, LM, 5.25, 7.2, 1.5, [
    P(R("Transient and steady-state folders hold matched pairs; the wall grows from 4 to 170 cells as it is printed."), bullet=True, space_after=6),
    P(R("The 2,602 ambient cells never leave 65.0, so every score is computed on the 1,094 material cells only."), bullet=True)],
     size=14)
box(s, 8.2, 1.62, 4.53, 5.0, fill=DARK)
label(s, 8.5, 1.85, 4, "The core limitation", color="E8344E")
text(s, 8.5, 2.2, 3.95, 1.5, P(R("One simulation run, not 480 independent cases.", bold=True)), size=22, font=HEAD, color=WHITE)
text(s, 8.5, 3.85, 3.95, 2.6, [
    P(R("Consecutive frames differ by a mean error of 0.76 against a field spread of 6.73.", color="C9CCD3"), space_after=8),
    P(R("Six frames per time block are near-copies, so there are about 80 effective samples.", color="C9CCD3"), space_after=8),
    P(R("This governs every result that follows.", color=WHITE, bold=True))], size=14.5)
notes(s, "The wall is 2 cells thick and 17 cells tall on a base plate; x is the build direction. "
         "Timestamps repeat in six fixed offsets inside each whole second, 80 times. "
         "The wall grows from 4 to 170 cells as it is printed (882 to 1,094 active cells).")

# =============================================================================
# 10  Meet 6: results and leakage
# =============================================================================
s = new_slide("Wall results: diffusion lost to a linear model", "Meet 6  ·  Results and lessons")
text(s, LM, 1.1, CW, 0.35, P(R("Task here: ", bold=True, color=RED),
     R("input = transient field, output = steady field (the reverse of the direction we want later, shown on slide 11).")),
     size=14, color=MUTED)
meth = [("Ridge regression", 2.16), ("Network + physics", 2.36), ("Network", 2.48),
        ("Diffusion + physics", 5.31), ("Diffusion", 5.42)]
ch = bar_chart(s, LM, 1.55, 5.3, 4.35, [m for m, _ in reversed(meth)],
               [("Average error (MAE), lower is better", [v for _, v in reversed(meth)])], [GRAY],
               kind=XL_CHART_TYPE.BAR_CLUSTERED, legend=False, fmt="0.00", vmin=0, vmax=6.5, gap=45, grid=True, size=11)
ser = ch.plots[0].series[0]
for i, (m, v) in enumerate(reversed(meth)):
    pt = ser.points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = rgb(RED if m.startswith("Diffusion") else TEAL if m.startswith("Ridge") else GRAY)
how = [("Ridge regression", "One fixed formula: every output cell is a weighted sum of all the input cells."),
       ("Network + physics", "A small neural network, plus a penalty when its output breaks the physics."),
       ("Network", "A small neural network that maps the input field to the output field."),
       ("Diffusion + physics", "Starts from noise and cleans it step by step, guided by the input, with a physics penalty."),
       ("Diffusion", "The same step-by-step cleaning from noise, without the physics penalty.")]
table(s, 6.1, 1.55, [1.85, 4.78], [["Method", "How it gets its answer"]] + [[a, b] for a, b in how],
      row_h=0.72, size=12.5, align=["left", "left"])
chips = [("Diffusion is the worst of the five.", " 5.3 to 5.4, against 2.16 for ridge regression."),
         ("Physics helps only a little.", " Network 2.48 to 2.36; diffusion 5.42 to 5.31."),
         ("A random split would have hidden this.", " Ridge scores 0.97 on random frames but 2.16 on held-out blocks.")]
for i, (b, r) in enumerate(chips):
    x = LM + i * 4.1
    box(s, x, 6.02, 3.93, 0.88, fill=PANEL)
    numdot(s, x + 0.15, 6.02 + 0.27, i + 1, d=0.34, size=13)
    text(s, x + 0.65, 6.02, 3.2, 0.88, P(R(b, bold=True), R(r)), size=12, anchor="middle")
notes(s, "In this meet 6 experiment the model was given the transient field and had to predict the steady field; slide 11 repeats it in the opposite, useful direction (steady in, transient out). "
         "Setup: test on the last 20 time blocks, 1,094 material cells. Cause of the diffusion result: about 80 effective independent samples, an output of 1,094 values, "
         "and a task that wants one correct answer rather than diverse samples. A data limitation, not a bug. Diffusion's repeated draws spread 12.4 to 12.9, nearly twice the field's own spread (6.73): noise, not structure. "
         "The leakage lesson: ridge scores 0.97 on a random split but 2.16 on held-out blocks, with the same model. Note: plain diffusion scored 5.42 and diffusion with the physics term 5.31 (the meet 6 slides quote 5.31 for both).")

# =============================================================================
# 2  Motivation
# =============================================================================
s = new_slide("Why map the steady state to the transient field?", "Motivation  \u00b7  the idea we test next")
pts = [("Heat accumulation is a design problem.",
        " In LPBF, heat that cannot drain to the baseplate causes local overheating; finding those zones early guides the design [1]."),
       ("The accurate analysis is slow.",
        " The layer-by-layer transient reference model of [1] took 21 h 28 min on 20 cores for one part."),
       ("The cheap analysis is only qualitative.",
        " The steady-state model needs 2 min 9 s (599× faster) but overestimates the peak temperature by 65% and matches the hot-spot map only 75% [1].")]
y = 1.7
for i, (b, rest) in enumerate(pts, 1):
    numdot(s, LM, y + 0.02, i)
    text(s, LM + 0.6, y, 5.95, 1.35, P(R(b, bold=True), R(rest)), size=15, line_spacing=1.03)
    y += 1.4
bx = 7.75
box(s, bx, 1.7, 4.98, 1.05, fill=PANEL)
text(s, bx + 0.25, 1.7, 4.5, 1.05, [P(R("Steady-state model", bold=True), size=18),
                                     P(R("2 min 9 s · qualitative [1]"), size=14, color=MUTED)], anchor="middle")
arrow(s, bx + 2.3, 2.8, 0.4, 0.34, fill=RED, direction="down")
box(s, bx, 3.2, 4.98, 1.05, fill=DARK)
text(s, bx + 0.25, 3.2, 4.5, 1.05, [P(R("Conditional diffusion model", bold=True, color=WHITE), size=18),
                                     P(R("inputs: steady field, power P, speed v, hatch, time t"), size=14, color="C9CCD3")], anchor="middle")
arrow(s, bx + 2.3, 4.3, 0.4, 0.34, fill=RED, direction="down")
box(s, bx, 4.7, 4.98, 1.05, fill="FBE9EC")
text(s, bx + 0.25, 4.7, 4.5, 1.05, [P(R("Transient reference model", bold=True), size=18),
                                     P(R("21 h 28 min · accurate [1]"), size=14, color=MUTED)], anchor="middle")
box(s, LM, 6.05, CW, 0.8, fill=DARK)
text(s, LM + 0.3, 6.05, CW - 0.6, 0.8, [P(R("Our question  ", bold=True, color="E8344E"),
     R("Can a conditional diffusion model turn the cheap steady-state field, plus the process parameters, into the accurate transient field?", color=WHITE))],
     size=16, anchor="middle")
notes(s, "All three timings and percentages are from Ranjan et al. [1], Table 2: reference model 21 h 28 min 32 s on a 20-core HPC node; "
         "steady-state model (S6) 2 min 9 s, computational gain 599, peak-temperature error 65.3%, Jaccard similarity 74.8%; "
         "the paper calls S6 a qualitative indication only. Our idea is a learned bridge between the two.")

# =============================================================================
# 11  Wall demo: steady -> transient
# =============================================================================
s = new_slide("Same idea in the useful direction: steady → transient", "Meet 7  ·  new work on the earlier wall data")
text(s, LM, 1.1, CW, 0.35, P(R("Task: ", bold=True, color=RED),
     R("input = steady-state field + time, output = transient field (one print, so no power, speed or thickness to feed in).")),
     size=14, color=MUTED)
pic = picture(s, ASSETS / "fig_wall_panels.png", LM, 1.5, h=4.05)
text(s, LM, 1.5 + 4.05 + 0.04, pic.width / 914400, 0.45, f"Slice through the wall (y = 10) of one held-out frame, t = {D['wall_frame_t']:.0f}. "
     "MAE in the dataset's temperature units.", size=11, color=MUTED)
box(s, LM, 6.06, 6.6, 0.9, fill=PANEL)
text(s, LM + 0.2, 6.06, 6.25, 0.9, [P(R("Model: ", bold=True), R("the single-bead network (next slide), power/speed/hatch set to 0.")),
                                      P(R("Training: ", bold=True), R("400 epochs (9,200 updates, batch 16), 71 s on CPU.")),
                                      P(R("Data: ", bold=True), R("first 359 frames to train, last 121 to test.")),
                                      P(R("Sampling: ", bold=True), R("50 DDIM steps, average of 8 draws."))], size=11, anchor="middle")
wd = D["wall_demo"]
names = [("Diffusion", wd["diffusion (mean of draws)"]), ("Ridge regression", wd["ridge field->field (lam=0.01)"]),
         ("Per-cell affine", wd["per-cell affine"]), ("No skill (per-cell mean)", wd["per-cell mean"]),
         ("Global mean", wd["global mean"]), ("Copy the steady field", wd["copy the steady field"])]
ch = bar_chart(s, 7.45, 1.45, 5.3, 3.5, [n for n, _ in reversed(names)], [("MAE", [v for _, v in reversed(names)])], [GRAY],
               kind=XL_CHART_TYPE.BAR_CLUSTERED, legend=False, fmt="0.00", vmin=0, vmax=5.2, gap=45, size=12)
ser = ch.plots[0].series[0]
for i, (n, v) in enumerate(reversed(names)):
    ser.points[i].format.fill.solid()
    ser.points[i].format.fill.fore_color.rgb = rgb(RED if n == "Diffusion" else TEAL if n.startswith("Ridge") else GRAY)
text(s, 7.45, 5.1, 5.3, 1.7, [
    P(R("The pipeline runs end to end and finds the laser hot spot, but one cell short and too cool: the wall front at x = 7 occurs only in the held-out part of the build."), space_after=6),
    P(R("Diffusion is 9% below no skill and level with the best linear model: one build cannot show more.", bold=True))], size=13.5)
notes(s, "As advised, the wall data is used to show the idea in the direction we actually want (steady in, transient "
         "out), with the same code as the single-bead model. Scores are mean absolute error on the last 25% of the "
         "build. The difference between diffusion (1.48) and ridge (1.51) is within noise: one seed, 121 neighbouring "
         "test frames. Say 'level with the best linear model', not 'beats'. "
         "Inputs: the steady-state field (1,094 values) and the time. The three process-parameter slots (power, speed, hatch) of the single-bead model are set to 0, "
         "because the wall data is one print with nothing to vary. Output: the transient field (1,094 values). "
         "Same network as the single-bead model (1.80 M parameters): 400 epochs, 23 batches of 16 per epoch (359 training frames), AdamW with learning rate 1e-3 and cosine decay, "
         "71 s on CPU, final training loss 6e-6; the test set is time blocks 62 to 82 (121 frames).")

# =============================================================================
# 12  Single-bead model: architecture
# =============================================================================
s = new_slide("Architecture of the single-bead model", "Meet 7  ·  Model")
text(s, LM, 1.1, CW, 0.35, P(R("Task: ", bold=True, color=RED),
     R("input = steady-state field + power, speed, hatch + time; output = transient field at that time.")),
     size=14, color=MUTED)
# ---- zone headings ----
for x_, w_, t_ in ((LM, 3.55, "INPUTS"), (4.55, 4.75, "LAYERS"), (9.65, 3.08, "OUTPUT")):
    text(s, x_, 1.5, w_, 0.28, P(R(t_, bold=True, spc=140)), size=12, color=RED)
# ---- inputs ----
text(s, LM, 1.82, 3.55, 0.25, P(R("WHAT WE KNOW", bold=True, spc=80)), size=10, color=FAINT)
known = [("Steady-state field", "1,094 values → condition encoder"),
         ("Power P, speed v, hatch", "3 values, rescaled to about 0 to 1 → encoder and join"),
         ("Time t", "1 value (time of the frame) → encoder and join")]
for k, (a, b) in enumerate(known):
    y = 2.08 + k * 0.88
    box(s, LM, y, 3.55, 0.78, fill=PANEL)
    text(s, LM + 0.2, y, 3.2, 0.78, [P(R(a, bold=True), size=14.5), P(R(b), size=12, color=MUTED)], anchor="middle")
text(s, LM, 4.74, 3.55, 0.25, P(R("DIFFUSION'S WORKING PARTS", bold=True, spc=80)), size=10, color=FAINT)
work = [("Noisy transient field", "1,094 values → join", "pure random noise when predicting", 0.88),
        ("Noise level τ", "1 value (0 to 399) → noise-level code", None, 0.78)]
y = 5.0
for a, b, c, h_ in work:
    box(s, LM, y, 3.55, h_, fill=PANEL)
    lines = [P(R(a, bold=True), size=14.5), P(R(b), size=12, color=MUTED)]
    if c:
        lines.append(P(R(c), size=12, color=MUTED))
    text(s, LM + 0.2, y, 3.2, h_, lines, anchor="middle")
    y += h_ + 0.08
arrow(s, 4.2, 3.85, 0.3, 0.4, fill=RED)
# ---- layers ----
def layer_box(x_, y_, w_, h_, fill_, head, lines_, lc="C9CCD3"):
    box(s, x_, y_, w_, h_, fill=fill_)
    text(s, x_ + 0.15, y_, w_ - 0.3, h_, [P(R(head, bold=True, color=WHITE), size=14, space_after=3)] +
         [P(R(t_, color=lc), size=11.5) for t_ in lines_], anchor="middle")
layer_box(4.55, 2.08, 2.3, 1.7, DARK, "Condition encoder",
          ["in: steady field 1,094 + 4 scalars (P, v, hatch, t) = 1,098", "Linear 1,098 → 256, SiLU", "Linear 256 → 128", "out: 128 values"])
layer_box(7.0, 2.08, 2.3, 1.7, DARK, "Noise-level code",
          ["τ → 64 sine and cosine values", "Linear 64 → 64, SiLU", "out: 64 values"])
arrow(s, 5.55, 3.82, 0.3, 0.2, fill=GRAY, direction="down")
arrow(s, 8.0, 3.82, 0.3, 0.2, fill=GRAY, direction="down")
layer_box(4.55, 4.05, 4.75, 1.0, TEAL, "Join (concatenate)",
          ["noisy field 1,094 + condition 128 + 4 scalars + noise code 64 = 1,290 values"], lc=WHITE)
arrow(s, 6.77, 5.08, 0.3, 0.2, fill=GRAY, direction="down")
layer_box(4.55, 5.3, 4.75, 1.45, DARK, "Main network",
          ["Linear 1,290 → 512, SiLU", "Linear 512 → 512, SiLU", "Linear 512 → 1,094"])
arrow(s, 9.35, 3.85, 0.3, 0.4, fill=RED)
# ---- output ----
box(s, 9.65, 2.08, 3.08, 2.0, fill="FBE9EC")
text(s, 9.85, 2.08, 2.7, 2.0, [P(R("Predicted clean transient field", bold=True), size=15, space_after=4),
                                P(R("1,094 values"), size=22, font=HEAD, color=RED, bold=True),
                                P(R("one per active node, at time t"), size=12, color=MUTED)], anchor="middle")
box(s, 9.65, 4.25, 3.08, 1.3, fill=PANEL)
text(s, 9.85, 4.25, 2.7, 1.3, [P(R("Then converted back to temperature", bold=True), size=14, space_after=3),
                                P(R("the 4,072 air nodes are put back at 65: 5,166 values"), size=12, color=MUTED)], anchor="middle")
stat(s, 9.65, 5.7, 3.08, 1.05, "1.80 M", "parameters in total", num_size=24, cap_size=12)
notes(s, "Read left to right. Inputs: the steady-state field (1,094 values), power, speed and hatch (one value each) and the time are real information about the situation. "
         "The noisy transient field and the noise level are diffusion's working parts. During training the noisy field is the true transient field with noise mixed in; "
         "when predicting it starts as pure random noise, so the true transient field is never given to the model. "
         "Layers: the condition encoder turns the steady field plus the four scalars (1,098 values) into 128 values; the noise level tau becomes a 64-number sine and cosine code and goes through one layer; "
         "these are joined with the noisy field and the four scalars into 1,290 values, then pass through 512, 512 and 1,094 units. "
         "Output: the predicted clean transient field, 1,094 values, converted back to temperature; the 4,072 air nodes are put back at 65 to give the full 5,166 values. "
         "The wall demo on slide 11 uses this same network with the power, speed and hatch inputs set to 0, because the wall is a single print. "
         "Training: mean squared error against the true transient field; AdamW, learning rate 1e-3 with cosine decay, gradient clipping 1.0, 400 noise steps, DDIM sampling in 50 steps with 8 draws averaged. "
         "Temperatures are rescaled with asinh((T - 65) / 10) / 4 and rescaled back at the end. Total parameters 1,803,270.")

# =============================================================================
# 12  Data source: thermal model
# =============================================================================
s = new_slide("Where the data comes from: a thermal model", "Data source  ·  Ranjan et al. [1]")
text(s, LM, 1.62, 6.3, 0.5, P(R("ρ(T) cₚ(T) ∂T/∂t = ∇·(k(T) ∇T) + Qᵥ", italic=True)), size=20, font=HEAD)
text(s, LM, 2.15, 6.3, 0.3, "heat equation with temperature-dependent Ti-6Al-4V properties (Eq. 1)", size=12, color=MUTED)
text(s, LM, 2.65, 6.3, 3.0, [
    P(R("Powder is ignored: ", bold=True), R("its conductivity is about 1% of the bulk, so the lateral sides are insulated."), bullet=True, space_after=7),
    P(R("Baseplate: ", bold=True), R("held at the preheat temperature T₀, acting as the heat sink."), bullet=True, space_after=7),
    P(R("Top surface: ", bold=True), R("loses heat by convection and radiation."), bullet=True, space_after=7),
    P(R("Layers: ", bold=True), R("each new lumped layer starts at T₀, is heated by a uniform volumetric source, then cools."), bullet=True, space_after=7),
    P(R("Solver: ", bold=True), R("finite elements with element birth-and-death (CalculiX)."), bullet=True)], size=14.5)
stat(s, LM, 5.7, 2.95, 1.1, "2.16 M", "elements in the reference model", num_size=26, cap_size=12)
stat(s, LM + 3.15, 5.7, 3.15, 1.1, "21 h 28 min", "on 20 cores, one part", num_size=26, cap_size=12)
x0 = 7.45
box(s, x0, 5.3, 5.0, 0.55, fill="9AA3AF", radius=0.04)
text(s, x0, 5.3, 5.0, 0.55, P(R("baseplate (heat sink)", bold=True)), size=13, color=WHITE, align="center", anchor="middle")
for k in range(3):
    box(s, x0 + 1.0, 4.78 - k * 0.5, 3.0, 0.45, fill=TEAL, radius=0.04)
text(s, x0 + 1.0, 4.28, 3.0, 0.45, P(R("part built so far", bold=True)), size=13, color=WHITE, align="center", anchor="middle")
box(s, x0 + 1.0, 3.28, 3.0, 0.45, fill=RED, radius=0.04)
text(s, x0 + 1.0, 3.28, 3.0, 0.45, P(R("new layer", bold=True)), size=13, color=WHITE, align="center", anchor="middle")
for ax in (1.45, 2.4, 3.35):
    arrow(s, x0 + ax, 2.55, 0.3, 0.65, fill=RED, direction="down")
text(s, x0 + 1.0, 2.1, 3.0, 0.35, P(R("heat source Qᵥ", bold=True)), size=14, color=RED, align="center")
text(s, x0, 3.8, 0.95, 0.6, "insulated (powder)", size=11, color=MUTED, align="right")
text(s, x0 + 4.05, 3.8, 0.95, 0.6, "insulated (powder)", size=11, color=MUTED)
for tx in (0.7, 2.3, 3.9):
    box(s, x0 + tx, 5.9, 0.4, 0.34, fill=DARK, shape=MSO_SHAPE.ISOSCELES_TRIANGLE)
text(s, x0, 6.3, 5.0, 0.3, "T = T₀ (fixed)", size=12, color=MUTED, align="center")
notes(s, "Equation numbers refer to the paper: governing equation (1), insulated lateral sides (2), fixed baseplate temperature (3), "
         "convection (4) and radiation (5) on the top. The reference run in the paper used 2.16 million elements and took 21 h 28 min 32 s on a 20-core node. "
         "In our folders run_it.sh points to a CalculiX-PW build in the paper author's directory, so the data comes from the same solver family.")

# =============================================================================
# 13  Data source: setting up a run
# =============================================================================
s = new_slide("From the physics to a simulation run", "Data source  ·  Ranjan et al. [1]")
box(s, LM, 1.62, 5.9, 5.15, fill=PANEL)
label(s, LM + 0.25, 1.78, 5, "Two formulas set up each layer")
text(s, LM + 0.25, 2.15, 5.4, 0.45, P(R("tₕ = A / (h · v)", italic=True)), size=21, font=HEAD)
text(s, LM + 0.25, 2.62, 5.4, 0.6, "heating time: scan the layer area A with hatch spacing h at speed v (Eq. 6)", size=13, color=MUTED)
text(s, LM + 0.25, 3.3, 5.4, 0.45, P(R("Qᵥ = γ P / (A · l)", italic=True)), size=21, font=HEAD)
text(s, LM + 0.25, 3.77, 5.4, 0.6, "heat source: absorbed power γP spread over the new layer, area A and thickness l (Eq. 7)", size=13, color=MUTED)
text(s, LM + 0.25, 4.5, 5.4, 0.3, "then cool for the inter-layer time", size=13, color=MUTED)
label(s, LM + 0.25, 5.0, 5, "Reference parameters (Table 1)")
text(s, LM + 0.25, 5.3, 5.4, 1.4, "P = 200 W,  γ = 0.45,  v = 1 m/s,  h = 0.14 mm,  l = 0.05 mm,  superlayer S = 0.5 mm,  "
     "T₀ = 180 °C,  Tₐ = 25 °C,  h_conv = 10 W/m²K,  ε = 0.35", size=13.5)
x = 6.85
text(s, x, 1.62, 5.9, 0.3, P(R("HOW OUR RUNS ARE PRODUCED", bold=True, spc=120)), size=11, color=RED)
steps_ = [("Parameter files", "power, velocity, hatch, material, ambient temperature, layers, part height, inter-layer time, mesh"),
          ("Thermal solve", "CalculiX-PW (ccx_2.12_MT, 16 threads), started by run_it.sh"),
          ("Stored frames", "one my_temp_k.txt per stored time step: 5,166 nodal temperatures"),
          ("This project", "27 transient runs and 4 steady-state runs")]
for i, (t, b) in enumerate(steps_):
    y = 2.0 + i * 1.15
    box(s, x, y, 5.88, 1.0, fill=DARK if i == 3 else PANEL)
    numdot(s, x + 0.2, y + 0.3, i + 1, d=0.38, fill=RED)
    text(s, x + 0.8, y, 4.9, 1.0, [P(R(t, bold=True, color=WHITE if i == 3 else INK), size=15),
                                    P(R(b, color="C9CCD3" if i == 3 else MUTED), size=12.5)], anchor="middle")
    if i < 3:
        arrow(s, x + 2.85, y + 1.0, 0.3, 0.15, fill=GRAY, direction="down")
notes(s, "t_h and Q_v are Eqs. (6) and (7) of the paper. In our folders the process inputs are plain text files "
         "(power.txt, velocity.txt, hatch.txt, material.txt, Amb_Temp.txt, no_of_layers.txt, part_height_in_mm.txt, recoater_time.txt) "
         "next to the mesh files; the solver is the CalculiX-PW build in the author's directory, run by run_it.sh. "
         "Each stored time step is written as my_temp_k.txt (and a ParaView .vtr copy). Our files record: Ti-6Al-4V, ambient 25, 10 layers, part height 1 mm, inter-layer time 5. "
         "Units for P, v and hatch are not stated in the files.")

# =============================================================================
# 14  Single bead: the dataset
# =============================================================================
s = new_slide("The dataset: 27 runs, generation in progress", "Single bead  ·  Data")
usable = {(60, 10, .10), (60, 10, .15), (60, 10, .18), (60, 50, .15), (60, 50, .18),
          (60, 100, .10), (60, 100, .15), (60, 100, .18)}
excluded = {(60, 50, .10)}
hv = [.10, .15, .18]
vv = [10, 50, 100]
for bi, P_ in enumerate((60, 90, 120)):
    bx = LM + bi * 3.15
    text(s, bx + 0.7, 1.62, 2.2, 0.35, P(R(f"P = {P_}", bold=True)), size=17, font=HEAD)
    for j, h in enumerate(hv):
        text(s, bx + 0.7 + j * 0.68, 2.02, 0.62, 0.25, f"{h:.2f}", size=11, color=MUTED, align="center")
    for i, v in enumerate(vv):
        text(s, bx, 2.38 + i * 0.68, 0.62, 0.55, f"v = {v}", size=11, color=MUTED, align="right", anchor="middle")
        for j, h in enumerate(hv):
            key = (P_, v, h)
            c = TEAL if key in usable else AMBER if key in excluded else GRAY
            box(s, bx + 0.7 + j * 0.68, 2.3 + i * 0.68, 0.62, 0.62, fill=c, radius=0.06)
text(s, LM, 4.45, 9.2, 0.3, "columns: hatch spacing   ·   rows: scan speed v   ·   units as in the simulation files", size=11, color=MUTED)
for k, (c, t) in enumerate(((TEAL, "usable, both fields (8 runs)"),
                            (AMBER, "excluded: copy of another run (1)"),
                            (GRAY, "steady state pending (18)"))):
    lx = LM + k * 3.15
    box(s, lx, 4.88, 0.22, 0.22, fill=c, radius=0.04)
    text(s, lx + 0.32, 4.84, 2.8, 0.32, t, size=12)
dec = [("4,072", "of 5,166 nodes are air (65.0 in every frame): removed"),
       ("1e−6", "steady state ignores speed: 4 folders serve 8 runs"),
       ("2", "labelling errors found: 1 run excluded")]
for i, (big, cap) in enumerate(dec):
    stat(s, LM + i * 3.1, 5.4, 2.95, 1.35, big, cap, num_size=24, cap_size=12)
box(s, 9.95, 1.62, 2.78, 5.13, fill=DARK)
label(s, 10.15, 1.82, 2.4, "Source", color="E8344E")
text(s, 10.15, 2.15, 2.4, 4.5, [
    P(R("Thermal model of Ranjan et al. [1], run with CalculiX-PW.", color="C9CCD3"), space_after=7),
    P(R("326 frames per run; 5,166 nodes (41 × 42 × 3).", color="C9CCD3"), space_after=7),
    P(R("Power, speed and hatch are read from the parameter files, not the folder names.", color="C9CCD3"), space_after=7),
    P(R("In progress: steady-state runs for P = 90 and 120.", bold=True, color=WHITE))], size=12.5)
notes(s, "27 = 3 powers x 3 speeds x 3 hatch values (you may also call hatch the bead 'thickness'; the files call it hatch). Only P = 60 has steady-state folders so far. "
         "Three data decisions, each checked against the files: (1) 4,072 nodes never leave 65.0 in any frame of any run, so they are removed and put back at 65 afterwards; "
         "(2) steady fields at v = 10 and v = 50 with the same hatch agree to 1e-6 (hatch 0.15 vs 0.18 differs by 191), so the 4 steady folders cover 8 transient runs; "
         "(3) a steady folder named hatch 0.10 is really 0.18, and the transient run labelled P 60, v 50, hatch 0.10 is a copy of the v = 10 run (peak 5,444 instead of about 4,973), so it is excluded. "
         "The remaining 18 runs wait for steady-state fields, which are being generated.")

# =============================================================================
# 15  What one run looks like
# =============================================================================
s = new_slide("One run: the signal is all at t = 2 and t = 3", "Single bead  \u00b7  Data")
pic = picture(s, ASSETS / "fig_bead_fields.png", LM, 1.5, w=CW)
ty = (pic.top + pic.height) / 914400 + 0.2
ex = D["example_run"]
ts = [f"{t}" for t in range(2, 8)]
mean_rise = D["peak_rise_by_t"]
rows_ = [["whole-number time t"] + ts,
         ["stored frame used (example run)"] + [f"{x:.3f}" for x in ex["times"]],
         ["peak rise above 65, mean of 8 runs"] + [f"{mean_rise[t]:,.0f}" if mean_rise[t] > 100 else f"{mean_rise[t]:.1f}" for t in ts]]
table(s, LM, ty, [4.3] + [1.3] * 6, rows_, row_h=0.44, size=13)
text(s, LM, ty + 0.44 * 3 + 0.2, CW, 0.9, [
    P(R("No stored frame sits exactly on a whole number, so the nearest frame is used. ", bold=True),
      R("By t = 4 the part is back at 65.0 to within 0.03 everywhere: four of the six requested times carry no signal, so scores are reported separately for t = 2, t = 3 and t \u2265 4."))],
     size=14)
notes(s, "Example run: P = 60, v = 10, hatch 0.15 (top view, maximum over depth). The scan runs along x on the mid-width line. "
         "The steady-state input has its hot spot at the start of the track, like the t = 2 frame. By t = 3 only a faint "
         "tail (about 13 above ambient) remains. Node layout (x fastest) was inferred from the compact hot spot in "
         "frame 0 and its straight, constant-speed motion along x.")

# =============================================================================
# 16  Task and model
# =============================================================================
s = new_slide("The task and the model", "Single bead  ·  Method")
from bead_model import BeadDiffusion  # noqa: E402
n_params = sum(p.numel() for p in BeadDiffusion(D["grid"]["active"]).parameters())
ins = [("Steady-state field", "1,094 values"), ("P, v, hatch", "3 values"), ("Time t", "1 value (frame time)"),
       ("Noisy field + noise level", "1,094 + 64 values")]
for i, (a, b) in enumerate(ins):
    y = 1.62 + i * 0.88
    box(s, LM, y, 3.7, 0.78, fill=PANEL)
    text(s, LM + 0.2, y, 3.3, 0.78, [P(R(a, bold=True), size=14), P(R(b), size=12, color=MUTED)], anchor="middle")
arrow(s, 4.45, 3.1, 0.5, 0.5, fill=RED)
box(s, 5.1, 1.62, 3.75, 3.4, fill=DARK)
text(s, 5.35, 1.8, 3.3, 3.1, [
    P(R("Conditional diffusion network", bold=True, color=WHITE), size=17, space_after=10),
    P(R("condition encoder", color="E8344E", bold=True), size=12),
    P(R("1,098 → 256 → 128", color="C9CCD3"), size=14, space_after=8),
    P(R("main network", color="E8344E", bold=True), size=12),
    P(R("1,290 → 512 → 512 → 1,094", color="C9CCD3"), size=14, space_after=8),
    P(R(f"{n_params / 1e6:.2f} M parameters", color=WHITE, bold=True), size=14)], size=14)
arrow(s, 9.0, 3.1, 0.5, 0.5, fill=RED)
box(s, 9.65, 1.62, 3.08, 1.7, fill="FBE9EC")
text(s, 9.85, 1.62, 2.7, 1.7, [P(R("Transient field at t", bold=True), size=15),
                                P(R("1,094 active nodes"), size=12, color=MUTED)], anchor="middle")
box(s, 9.65, 3.45, 3.08, 1.57, fill=PANEL)
text(s, 9.85, 3.45, 2.7, 1.57, [P(R("+ 4,072 air nodes", bold=True), size=15),
                                 P(R("set to 65 → 5,166 values"), size=12, color=MUTED)], anchor="middle")
m = D["model"]
tiles = [(str(m["rows"]), "training examples\n8 runs × 6 times"),
         (f"{m['epochs']:,}", "epochs, 4,500 updates\nAdamW, batch 16"),
         ("36 s", "training time on CPU\nloss 0.004 → 0.000003"),
         ("50", "DDIM steps; 8 draws\naveraged per prediction")]
for i, (big, cap) in enumerate(tiles):
    stat(s, LM + i * 3.06, 5.3, 2.95, 1.35, big, cap, num_size=28, cap_size=12)
notes(s, "Input to the main network: noisy field 1,094 + compressed condition 128 + the four scalars (P, v, hatch, t) "
         "passed straight through + 64-number noise-level code = 1,290. The model predicts the clean field (x0) rather "
         "than the noise: the standard noise-prediction version never trained here (loss stuck at 0.96 where 1.0 means "
         "no information), because the fields are mostly exactly 65. Training loss only shows it fits the 48 examples; "
         "the next slides test it on runs it has not seen.")

# =============================================================================
# 17  Results
# =============================================================================
s = new_slide("Results: diffusion matches the nearest-run copy", "Single bead  ·  Results")
L = D["loo_summary"]
dif, near, mean, amb = (L["diffusion (mean of draws)"], L["nearest case in (v, hatch)"],
                        L["mean of training cases"], L["ambient (65 everywhere)"])


def f2(x):
    return "n/a" if x is None else f"{x:.2f}"


text(s, LM, 1.5, 6.65, 0.62, P(R("Test: ", bold=True, color=RED),
     R("leave one run out. Train on 7 runs, predict the 8th, repeat for all 8. Error = size of the error ÷ size of the true rise above 65 (1.0 = no skill).")),
     size=12.5, color=MUTED)
rows_ = [["Method", "t = 2 error", "t = 2 peak", "t = 3 error", "t ≥ 4 max (°C)"],
         ["Diffusion", f2(dif["t2_relL2"]), pct(dif["t2_peak_err_pct"]), f2(dif["t3_relL2"]), f2(dif["late_max_abs_err"])],
         ["Nearest run", f2(near["t2_relL2"]), pct(near["t2_peak_err_pct"]), f2(near["t3_relL2"]), "0.00"],
         ["Mean of runs", f2(mean["t2_relL2"]), pct(mean["t2_peak_err_pct"]), f2(mean["t3_relL2"]), "0.00"],
         ["No skill (65)", "1.00", "100.0%", "1.00", f2(amb["late_max_abs_err"])]]
table(s, LM, 2.2, [1.9, 1.15, 1.15, 1.15, 1.35], rows_, row_h=0.44, size=13, highlight=(1,), head_size=12)
pc = D["per_case_t2"]
bar_chart(s, 7.4, 1.5, 5.35, 3.55, [c["label"] for c in pc],
          [("truth", [c["true"] for c in pc]), ("diffusion", [c["diffusion"] for c in pc]), ("nearest run", [c["nearest"] for c in pc])],
          [DARK, RED, GRAY], labels=False, legend=True, size=11, gap=40, overlap=-5, vmin=0, vfmt="#,##0")
text(s, 7.4, 5.05, 5.35, 0.3, "peak temperature at t = 2 for each held-out run", size=11.5, color=MUTED, align="center")
items = [("t = 2: ", "error 0.22 against 1.0 for no skill, so the model learned real structure, but it only ties the nearest-run copy (0.19)."),
         ("t = 3: ", "clearly worse (0.59 vs 0.17); it blurs a 5 to 17 degree bump into nearly the same shape for every run."),
         ("t ≥ 4: ", "nothing to predict; the model's noise floor is 0.5 °C.")]
for i, (b, rest) in enumerate(items):
    text(s, LM, 4.55 + i * 0.78, 6.65, 0.75, P(R(b, bold=True, color=RED), R(rest)), size=13)
text(s, 7.4, 5.45, 5.35, 1.4, [
    P(R("Draws barely differ (spread at most 0.12 °C), so the sampler behaves like a regressor: there is one right answer."), space_after=5),
    P(R("Worst case: v = 50, hatch 0.18, overshoots the peak by 46%.", bold=True))], size=13)
notes(s, "Mean over the 8 held-out runs. The saved model has seen all 8 runs, so scoring it on them would only measure memorisation: that is why each round hides one run. "
         "Diffusion beats the nearest-run copy on peak error in 5 of 8 runs but on whole-field error in only 2 of 8. "
         "t = 3 only has signal for 3 of the 8 runs (the three v = 10 runs).")

# =============================================================================
# 18  Structure: amplitude x shape
# =============================================================================
s = new_slide("The t = 2 fields are one shape times an amplitude", "Single bead  ·  Insight")
stat(s, LM, 1.62, 3.4, 1.7, f"{D['first_component_pct']:.1f}%", "of the variation across the 8 runs is one fixed spatial shape",
     num_size=40, cap_size=13)
text(s, LM, 3.55, 5.6, 3.1, [
    P(R("Prediction rule", bold=True, color=RED), space_after=3),
    P(R("field = amplitude(v, hatch) × shape", italic=True), size=17, font=HEAD, space_after=8),
    P(R("amplitude: ln(peak) = a + b ln v + c ln hatch, fitted on 7 runs (3 numbers)"), bullet=True, space_after=4),
    P(R("shape: mean of the training fields, each divided by its own peak"), bullet=True, space_after=4),
    P(R("same leave-one-run-out test; no network, no epochs"), bullet=True)], size=14)
a, n_, d_ = L["amplitude x shape"], near, dif
bar_chart(s, 6.5, 1.55, 6.25, 4.2,
          ["Amplitude × shape", "Nearest run", "Diffusion"],
          [("peak error at t = 2 (%)", [a["t2_peak_err_pct"], n_["t2_peak_err_pct"], d_["t2_peak_err_pct"]])],
          [TEAL], legend=False, fmt='0.0"%"', vmin=0, vmax=25, gap=70, vtitle="mean peak error (%)", size=13)
ser = s.shapes[-1].chart.plots[0].series[0]
for i, c in enumerate((TEAL, GRAY, RED)):
    ser.points[i].format.fill.solid()
    ser.points[i].format.fill.fore_color.rgb = rgb(c)
text(s, 6.5, 5.85, 6.25, 0.9, f"Whole-field relative error: {a['t2_relL2']:.2f} vs {n_['t2_relL2']:.2f} (nearest run) and "
     f"{d_['t2_relL2']:.2f} (diffusion). t = 2 only: later times have too little signal for a one-amplitude model.",
     size=12.5, color=MUTED)
notes(s, "This is the main insight of the new dataset. The first singular component of the eight t = 2 fields holds "
         "99.73% of their energy. A 3-parameter model therefore reaches 5.2% peak error and 0.11 relative error, "
         "against 19% and 0.22 for the diffusion network, which has to discover this structure from 7 examples. "
         "It does not win everywhere: diffusion was better on the v = 10, hatch 0.15 run (2.0% vs 4.8%). "
         "Implication: a hybrid (physics-style amplitude + learned shape/residual) is the natural next model.")

# =============================================================================
# 19  What we learnt
# =============================================================================
s = new_slide("What we learnt", "Lessons")
lessons = [("Test fairly first.", "A random split scored 0.97 where honest splits gave 2.16 (wall). Now whole runs are held out."),
           ("Beat a simple baseline.", "Ridge beat diffusion on the wall (2.16 vs 5.4); a nearest-run copy ties it on the bead data."),
           ("Data volume decides.", "One build is about 80 independent samples; the bead set has 8 usable runs. Diffusion needs far more."),
           ("Sampling errors snowball.", "A 1% error compounded over 1,000 steps (1D rod). Use fewer DDIM steps and predict the clean field."),
           ("Add physics carefully.", "The physics loss exploded at high noise. Apply it at low-noise steps only, with a small weight."),
           ("Look for structure.", "The t = 2 fields are one shape times an amplitude (99.7%); 3 parameters reach 5% peak error.")]
for i, (h_, b) in enumerate(lessons):
    x = LM + (i % 3) * 4.1
    y = 1.65 + (i // 3) * 2.62
    box(s, x, y, 3.93, 2.45, fill=PANEL)
    numdot(s, x + 0.25, y + 0.25, i + 1, d=0.4)
    text(s, x + 0.8, y + 0.2, 2.95, 0.55, P(R(h_, bold=True)), size=17, font=HEAD, anchor="middle")
    text(s, x + 0.25, y + 0.95, 3.45, 1.45, b, size=14, color=MUTED)
notes(s, "Each lesson has evidence from a specific meet: 1 and 2 from the wall data (meet 6), 3 from both datasets, 4 and 5 from the 1D rod (meet 5), 6 from today's single-bead work.")

# =============================================================================
# 20  Where we go next
# =============================================================================
s = new_slide("Where we go next", "Plan")
box(s, LM, 1.62, 5.6, 5.15, fill=PANEL)
label(s, LM + 0.25, 1.78, 5, "Dataset generation (in progress)")
gen = [("DONE", "27 transient runs", "P = 60, 90, 120  ×  v = 10, 50, 100  ×  hatch = 0.10, 0.15, 0.18"),
       ("DONE", "4 steady-state runs", "all at P = 60; they cover 8 usable transient runs"),
       ("PENDING", "Steady-state runs for P = 90 and 120", "unlocks 18 more transient runs and makes power a real input"),
       ("PENDING", "Re-run P 60, v 50, hatch 0.10", "the current transient result is a copy of the v = 10 run")]
for i, (tag, t, b) in enumerate(gen):
    y = 2.3 + i * 1.07
    chip = box(s, LM + 0.25, y + 0.1, 0.95, 0.32, fill=TEAL if tag == "DONE" else AMBER, radius=0.1)
    fill_text(chip.text_frame, P(R(tag, bold=True, spc=40)), size=10, color=WHITE, align="center", anchor="middle")
    text(s, LM + 1.4, y, 4.0, 1.0, [P(R(t, bold=True), size=14.5), P(R(b), size=12.5, color=MUTED)])
rx = 6.4
box(s, rx, 1.62, 6.33, 5.15, fill=PANEL)
label(s, rx + 0.25, 1.78, 5, "Modelling plan")
plan = [("Train on ~320 frames per run", "heating-phase frames instead of 6 whole-number times", "now"),
        ("Hybrid model", "physics-style amplitude plus diffusion for shape and residual", "now"),
        ("Physics-informed loss", "heat-equation or steady-state residual, masked by noise level", "now"),
        ("Score as in the paper [1]", "peak error δ and Jaccard index J on the hot-spot map", "now"),
        ("Test on an unseen power", "hold out P = 90 or 120 once those runs exist", "data")]
for i, (t, b, tag) in enumerate(plan):
    y = 2.3 + i * 0.86
    numdot(s, rx + 0.25, y + 0.15, i + 1, d=0.36, fill=TEAL if tag == "now" else RED, size=13)
    text(s, rx + 0.8, y, 3.5, 0.82, [P(R(t, bold=True), size=14.5), P(R(b), size=12), ])
    chip = box(s, rx + 4.5, y + 0.18, 1.6, 0.34, fill=TEAL if tag == "now" else RED, radius=0.1)
    fill_text(chip.text_frame, P(R("CAN START NOW" if tag == "now" else "NEEDS DATA", bold=True, spc=40)), size=10,
              color=WHITE, align="center", anchor="middle")
notes(s, "Left: the dataset work, which your supervisor confirmed is still being generated. Right: modelling steps; four can start with existing data. "
         "Using the heating-phase frames is the cheapest way to get more training examples, though neighbouring frames are correlated, so leave-one-run-out must stay. "
         "The paper's metrics are delta (percentage error of the maximum temperature) and the Jaccard index J of the normalised fields, Eqs. (22) and (23).")

# =============================================================================
# 21  Summary
# =============================================================================
s = new_slide("Summary", "Conclusions", dark=True)
pts = ["Literature (meets 1 to 5): diffusion models can bridge fidelity, and physics can be added to their loss.",
       "The 1D rod results were poor. They taught us to test fairly, cap sampling errors, and move to real data.",
       "Wall data: one build is about 80 samples; diffusion lost to a linear model, and a random split would have hidden it.",
       "Single-bead data (thermal model of [1]): the model ties a nearest-run copy at t = 2, and a 3-parameter amplitude × shape fit reaches 5% peak error.",
       "The limit is data: steady-state runs for the remaining powers are being generated, and the plan is built around them."]
y = 1.65
for i, t in enumerate(pts, 1):
    numdot(s, LM, y + 0.03, i, d=0.4, fill=RED)
    text(s, LM + 0.65, y, CW - 0.7, 0.95, t, size=16.5, color=WHITE, anchor="middle")
    y += 1.02
notes(s, "Close on the last point: the method works as far as 8 runs allow, the structure in the data is clear, and the next jump in accuracy depends on the steady-state fields now being generated.")

# =============================================================================
# 22  References
# =============================================================================
s = new_slide("References", "Sources")
refs = ["R. Ranjan, C. Ayas, M. Langelaar, F. van Keulen. Fast detection of heat accumulation in powder bed fusion using computationally efficient thermal models. Materials 13(20), 4576 (2020).",
        "J. Ho, A. Jain, P. Abbeel. Denoising diffusion probabilistic models. NeurIPS (2020).",
        "J. Song, C. Meng, S. Ermon. Denoising diffusion implicit models. ICLR (2021).",
        "Inexpensive high-fidelity melt pool models in additive manufacturing using generative deep diffusion. arXiv:2311.16168 (2023).",
        "Diffusion generative model-based learning for smart layer-wise monitoring of additive manufacturing. ASME J. Comput. Inf. Sci. Eng. (2024).",
        "J.-H. Bastek, W. Sun, D. M. Kochmann. Physics-informed diffusion models. ICLR (2025).",
        "M. Raissi, P. Perdikaris, G. E. Karniadakis. Physics-informed neural networks. J. Comput. Phys. 378 (2019).",
        "Related work from the references of [4]: Shu, Li & Farimani (2023); Jadhav et al. (2023); Li et al., SRDiff (2022); Saharia et al., SR3 (2022); Deng et al. (2019); Liu et al. (2020); Li & McComb (2022); Fathi et al. (2020); Wang et al., ESRGAN (2018)."]
y = 1.6
for i, r in enumerate(refs, 1):
    h = 0.62 if len(r) > 135 else 0.4
    text(s, LM, y, 0.5, h, P(R(f"[{i}]", bold=True)), size=13, color=RED)
    text(s, LM + 0.55, y, CW - 0.6, h, r, size=13)
    y += h + 0.12
notes(s, "References [2], [3] and [7] are standard citations for methods used (DDPM, DDIM, PINNs). "
         "Author lists for [4] and [5] are not recorded in the earlier decks and should be added from the papers.")

# ---- footers and save ------------------------------------------------------------
TIMES = [0.5, 1.5, 1, 1, 1, 1.5, 1.5, 1, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 1.5, 1.5, 2, 1.5, 1.5, 1.5, 1, 0]  # minutes per main slide; sums to about 29.5
for i, sl in enumerate(prs.slides):
    t = TIMES[i] if i < len(TIMES) else None
    note = sl.notes_slide.notes_text_frame
    tag = f"[about {t:g} min] " if t else ("[backup slide] " if i >= len(TIMES) else "")
    note.text = tag + note.text
total = len(prs.slides)
for i, (sl, kind) in enumerate(zip(prs.slides, KINDS), 1):
    if kind != "title":
        foot(sl, i, total, kind == "dark")
prs.save(OUT_FILE)
print(f"saved {OUT_FILE.name}: {total} slides")
