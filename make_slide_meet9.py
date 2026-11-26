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

ROOT = Path("/home/ayush/Desktop/Ayush_btp")
ASSETS = ROOT / "deck_assets"
D = json.loads((ASSETS / "deck_data.json").read_text())
OUT_FILE = ROOT / "slide_meet9.pptx"

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


F = ROOT / "rod1d_outputs"
TOTAL = 7
def pn(s, n):
    text(s, SW - LM - 1.2, 7.05, 1.2, 0.25, f"{n} / {TOTAL}", size=10, color=FAINT, align="right")

# ---- 1. why 1D again ------------------------------------------------------------------------
s = new_slide("Back to 1D, on the same idea as the bead", "Meet 9  ·  1D demonstration")
for i, (title, colr, fill, pts) in enumerate((
        ("Meet 5: the first 1D attempt", GRAY, PANEL, [
            ("Poor results: ", "predictions zigzagged between −400 and +300 °C on a rod near 180 °C."),
            ("Errors snowballed ", "over 1,000 denoising steps."),
            ("No fair test: ", "the final plot used a training sample."),
            ("Target unlike our task: ", "transient snapshots of a toy rod.")]),
        ("Now: the same task as the bead", RED, "FBE9EC", [
            ("Max steady → max transient: ", "a laser moving along a rod, exactly sir's mapping."),
            ("Enough data: ", "1,600 training runs and 300 unseen test runs, generated in 8 s."),
            ("Fair test: ", "every number is on runs the model never saw."),
            ("Fixed method: ", "predict the clean field, 50 DDIM steps.")]))):
    x = LM + i * 6.183
    box(s, x, 1.4, 5.95, 5.4, fill=fill)
    text(s, x + 0.3, 1.6, 5.4, 0.45, P(R(title, bold=True)), size=20, font=HEAD, color=INK if i == 0 else RED_DK)
    text(s, x + 0.3, 2.4, 5.4, 4.3, [P(R(a, bold=True), R(b, color=MUTED), bullet=True, bullet_color=colr, space_after=14) for a, b in pts], size=16)
pn(s, 1)

# ---- 2. setup -------------------------------------------------------------------------------
s = new_slide("The 1D model: a laser moving along a rod", "Meet 9  ·  1D demonstration")
picture(s, F / "fig1_one_run.png", LM, 1.35, w=7.0)
text(s, LM, 4.85, 7.0, 0.4, "One run (P = 100 W, v = 20 mm/s): the steady maximum is 1,862 °C, the transient maximum only 690 °C.", size=11.5, color=MUTED)
text(s, 7.85, 1.35, 4.9, 4.0, [
    P(R("Transient run: ", bold=True, color=RED), R("the laser moves from 1 to 9 mm at speed v; we record the hottest temperature each point ever reaches."), space_after=8),
    P(R("Steady run: ", bold=True, color=RED), R("the laser held still at 81 positions, each solved without time (as SS_new); the hottest value at each point is kept."), space_after=8),
    P(R("Input: ", bold=True), R("max steady field (101 points) + P and v.")),
    P(R("Output: ", bold=True), R("max transient field (101 points)."), space_after=8),
    P(R("Varied per run: ", bold=True), R("power 20–200 W, speed 2–50 mm/s, conductivity, cooling and spot size.", color=MUTED))], size=13.5)
rows = [["Data", "Runs", "Time"], ["Training", "1,600", "8 s to generate all"], ["Test (unseen)", "300", "same"]]
table(s, LM, 5.45, [2.6, 1.6, 2.8], rows, row_h=0.42, size=12.5, head_size=12)
text(s, 7.85, 5.5, 4.9, 1.3, [P(R("Model: ", bold=True), R("the same diffusion code as the bead task, 197 k parameters, 8,000 updates per training.", color=MUTED))], size=12.5)
pn(s, 2)

# ---- 3. how diffusion works here ---------------------------------------------------------------------
s = new_slide("How the diffusion model learns and predicts", "Meet 9  ·  1D demonstration")
IW = 10.1
pic = picture(s, F / "fig4_diffusion.png", LM + (CW - IW) / 2, 1.22, w=IW)
yb = 1.22 + pic.height / 914400 + 0.1
for k, (lab, body, col) in enumerate((
        ("Training (known: steady curve, P, v and the true transient): ",
         "add noise at a random level from 1 to 400; the network guesses the clean curve in one jump; compare with the truth and adjust. Repeated 8,000 times.", RED),
        ("Prediction (known: steady curve, P, v only): ",
         "“30 of 50” means 30 one-jump steps are done. Each step guesses the clean curve, then keeps a little less noise. After 50 steps the noise is gone; 8 predictions are averaged.", TEAL))):
    x = LM + k * 6.183
    box(s, x, yb, 5.95, 6.92 - yb, fill=PANEL)
    text(s, x + 0.2, yb, 5.6, 6.92 - yb, P(R(lab, bold=True, color=col), R(body)), size=11, anchor="middle")
pn(s, 3)

# ---- 3. results --------------------------------------------------------------------------------
s = new_slide("Steady alone is far off; diffusion closes the gap", "Meet 9  ·  Results on 300 unseen runs")
picture(s, F / "fig2_examples.png", LM, 1.35, w=CW)
yb = 1.35 + CW * 720 / 2600 + 0.25
for i, (n, c, col) in enumerate((("+174%", "average peak overestimate if the steady field is used as the answer (median +84%, worst +1,180%)", GRAY),
                                 ("1.0%", "average peak error of the diffusion model on unseen runs (95% of runs within 2.4%)", RED),
                                 ("12 °C", "RMSE over the rod, for peaks of about 140 to 5,100 °C", RED))):
    stat(s, LM + i * 4.1, yb, 3.93, 6.85 - yb, n, c, num_color=col, num_size=30, cap_size=12)
pn(s, 4)

# ---- 4. data -----------------------------------------------------------------------------------
s = new_slide("More training runs, lower error", "Meet 9  ·  How much data is needed")
picture(s, F / "fig3_data.png", LM, 1.35, w=6.9)
text(s, 7.75, 1.4, 5.0, 5.3, [
    P(R("25 runs: ", bold=True, color=RED), R("3.0% peak error, RMSE 34 °C.")),
    P(R("200 runs: ", bold=True, color=RED), R("1.2%, RMSE 15 °C.")),
    P(R("1,600 runs: ", bold=True, color=RED), R("1.0%, RMSE 12 °C."), space_after=12),
    P(R("The error falls fast up to a few hundred runs, then levels off.", color=MUTED), space_after=12),
    P(R("Our bead data has 26 runs: ", bold=True), R("the left end of this curve. More settings are the most direct improvement."), space_after=12),
    P(R("Without the steady field ", bold=True), R("the error stays near 18% however much data is added (grey).", color=MUTED))], size=15)
pn(s, 5)

# ---- 5. when does steady help ------------------------------------------------------------------------
s = new_slide("When does the steady field help?", "Meet 9  ·  1,600 training runs, 300 unseen")
rows = [["Numbers given to the model", "With steady field", "Without steady field"],
        ["P and v only\n(conductivity, cooling, spot size hidden)", "1.0%  ·  12 °C", "18%  ·  417 °C"],
        ["P, v, conductivity, cooling, spot size", "0.8%  ·  7 °C", "0.7%  ·  8 °C"]]
table(s, LM, 1.45, [5.2, 3.45, 3.483], rows, row_h=0.85, size=14, head_size=13, align=["left", "center", "center"])
text(s, LM, 4.15, CW, 0.3, "Mean peak error · RMSE on the 300 unseen runs.", size=11.5, color=MUTED)
box(s, LM, 4.6, CW, 2.2, fill=DARK)
text(s, LM + 0.35, 4.6, CW - 0.7, 2.2, [
    P(R("The steady field is valuable when it carries information the numbers do not. ", bold=True, color="E8344E"),
      R("Here, the material hidden from the model; in real parts, the geometry.", color=WHITE), space_after=10),
    P(R("Single bead: ", bold=True, color=WHITE), R("everything is in P, v and hatch, and the SS_new runs are identical, so little gain is expected there.", color="C9CCD3"), space_after=6),
    P(R("Geometry dataset: ", bold=True, color=WHITE), R("this is where the steady field should make the difference.", color="C9CCD3"))], size=15, anchor="middle")
pn(s, 6)

# ---- 6. next ------------------------------------------------------------------------------------
s = new_slide("What 1D tells us, and next steps", "Meet 9  ·  Conclusions")
box(s, LM, 1.4, CW, 2.35, fill=PANEL)
text(s, LM + 0.35, 1.55, CW - 0.7, 2.1, [
    P(R("The idea works: ", bold=True), R("max steady + P, v → max transient within 1% peak error on unseen runs, where the steady field alone is off by 174%."), bullet=True, space_after=7),
    P(R("Data is the lever: ", bold=True), R("error drops from 3.0% to 1.0% between 25 and 1,600 runs."), bullet=True, space_after=7),
    P(R("Use of the steady field: ", bold=True), R("it pays off when it holds information the inputs do not (material here, geometry next)."), bullet=True)], size=15)
text(s, LM, 4.0, 6, 0.3, P(R("NEXT STEPS", bold=True, spc=120)), size=11, color=RED)
nxt = [("Geometry dataset", "train and test the same pipeline as soon as it arrives."),
       ("Single bead", "re-run once the SS heat input is fixed; add settings."),
       ("1D extensions", "temperature-dependent conductivity; layer-by-layer heating."),
       ("Final presentation", "build the 1D → bead → geometry story with sir.")]
for i, (a, b) in enumerate(nxt):
    x = LM + (i % 2) * 6.183
    y = 4.4 + (i // 2) * 1.2
    numdot(s, x, y + 0.03, i + 1)
    text(s, x + 0.6, y, 5.3, 0.4, P(R(a, bold=True)), size=16)
    text(s, x + 0.6, y + 0.43, 5.3, 0.7, b, size=13, color=MUTED)
pn(s, 7)
prs.save(OUT_FILE)
print("saved", OUT_FILE)
