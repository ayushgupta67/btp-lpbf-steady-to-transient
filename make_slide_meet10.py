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
OUT_FILE = ROOT / "slide_meet10.pptx"

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


F = ROOT / "tts_outputs"
TOTAL = 6
def pn(s, n):
    text(s, SW - LM - 1.2, 7.05, 1.2, 0.25, f"{n} / {TOTAL}", size=10, color=FAINT, align="right")

def cards(s, items, y0, h, cols=2, tsize=15, bsize=12):
    w = (CW - (cols - 1) * 0.233) / cols
    for i, (a, b) in enumerate(items):
        x = LM + (i % cols) * (w + 0.233)
        y = y0 + (i // cols) * (h + 0.15)
        box(s, x, y, w, h, fill=PANEL)
        numdot(s, x + 0.2, y + 0.18, i + 1, d=0.36, size=13)
        text(s, x + 0.72, y + 0.14, w - 0.9, 0.45, P(R(a, bold=True)), size=tsize)
        text(s, x + 0.72, y + 0.55, w - 0.9, h - 0.6, b, size=bsize, color=MUTED)

# ---- 1. the part in 3D -------------------------------------------------------------------------
s = new_slide("New geometry dataset: one part with internal voids", "Meet 10  ·  The data")
pic = picture(s, F / "fig3d_part.png", LM - 0.1, 1.2, w=8.4)
text(s, 8.75, 1.35, 4.0, 3.6, [
    P(R("Part: ", bold=True), R("20 \u00d7 20 mm (length \u00d7 breadth) block, 10 mm high: 10 layers of 1 mm on a 3-layer substrate."), space_after=8),
    P(R("Voids: ", bold=True), R("a hole through the part that widens near the top, and a slot from layer 6."), space_after=8),
    P(R("Built element by element: ", bold=True), R("3,844 elements of 1 mm (each one a “bead”), heated 1 s each, 5 s cooling per layer."))], size=13.5)
tiles = [("Steady", "one steady solve per element; 65 °C sink only outside a 3.4 mm hemisphere around it (local domain, Lc = √(αt))", TEAL),
         ("Transient", "6 time sub-steps per element; base held at 180 °C; convection and radiation", RED),
         ("Pairs", "matched by element number: 3,844 complete steady ↔ transient pairs", DARK)]
for i, (a, b, col) in enumerate(tiles):
    x = LM + i * 4.1
    box(s, x, 5.15, 3.93, 1.75, fill=PANEL)
    text(s, x + 0.25, 5.23, 3.5, 0.4, P(R(a, bold=True)), size=16, font=HEAD, color=col)
    text(s, x + 0.25, 5.65, 3.5, 1.2, b, size=11.5, color=MUTED)
pn(s, 1)

# ---- 2. how the data is read ----------------------------------------------------------------------
s = new_slide("How the data is read", "Meet 10  ·  From files to one number per block")
pic = picture(s, F / "fig_how_values_picked.png", LM - 0.1, 1.45, w=8.15)
box(s, 8.75, 1.25, 3.98, 5.6, fill=PANEL)
text(s, 8.97, 1.38, 3.6, 0.4, P(R("Why the highest value?", bold=True)), size=17, font=HEAD, color=RED)
text(s, 8.97, 1.9, 3.6, 4.9, [
    P(R("Sir's instruction: ", bold=True), R("map the max steady temperature to the max transient temperature for each bead.", color=MUTED), bullet=True, space_after=8),
    P(R("Defects come from overheating: ", bold=True), R("hot spots near walls and voids, as in the Ranjan paper; an average would hide them (most points stay at 65 °C).", color=MUTED), bullet=True, space_after=8),
    P(R("One fair number per block: ", bold=True), R("1 steady file and 6 transient files become two comparable numbers.", color=MUTED), bullet=True, space_after=8),
    P(R("Checked: ", bold=True), R("the highest point is on the new block in all 3,844 steady files and transient steps.", color=MUTED), bullet=True, space_after=8),
    P(R("For training ", bold=True), R("the whole field around each block can still be used; the peak is for checking the data.", color=MUTED), bullet=True)], size=12.5)
pn(s, 2)

# ---- 2. peaks in 3D ------------------------------------------------------------------------------
s = new_slide("Geometry shows up in both steady and transient peaks", "Meet 10  ·  Main finding")
IW = 9.6
pic = picture(s, F / "fig3d_peaks.png", LM + (CW - IW) / 2, 1.15, w=IW)
yb = 1.15 + pic.height / 914400 + 0.08
rows = [["Element next to…", "Steady peak", "Transient peak"], ["no empty side", "141,853 °C", "117,892 °C"],
        ["1 empty side", "165,602 °C", "120,189 °C"], ["2 empty sides", "202,351 °C", "129,543 °C"]]
table(s, LM, yb, [2.2, 1.7, 1.7], rows, row_h=(6.92 - yb) / 4, size=11.5, head_size=11)
text(s, LM + 5.9, yb, 6.23, 6.92 - yb, [
    P(R("Hotter next to walls and voids in both runs: ", bold=True), R("the hole and the outer edges stand out.", color=MUTED), space_after=4),
    P(R("Steady and transient peaks move together: ", bold=True), R("correlation 0.65, and 0.88 for blocks after 1,364.", color=MUTED), space_after=4),
    P(R("Dark band and the rise with height in the steady view: ", bold=True), R("a steady-script bug, not physics (problem 3, slide 5).", color=MUTED))], size=11.5, anchor="middle")
pn(s, 3)

# ---- 3. advantages -----------------------------------------------------------------------------------
s = new_slide("Advantages of this dataset", "Meet 10  ·  What is better than before")
cards(s, [
    ("Steady field depends on geometry", "3,033 different steady peaks over 3,844 elements (SS_new: all identical). Part of this variation comes from the script bug (problem 3)."),
    ("A physically closer steady model", "as designed, heat only has to reach a local domain (3.4 mm hemisphere, Lc = \u221a(\u03b1t)), not the whole part; the script must be fixed first (problem 3)."),
    ("Walls and voids heat up", "in both runs: steady 142k → 203k °C and transient 118k → 130k °C with 0 → 2 empty neighbours."),
    ("Steady and transient agree", "both are hotter next to walls and voids; peak correlation 0.88 for blocks after 1,364, even with the bug."),
    ("Much more data from one build", "3,844 element pairs, against 26 runs for the single bead (they come from one part, so they are not fully independent)."),
    ("Ready to use", "complete pairing by element number, same file format; loader and analysis scripts done.")], y0=1.35, h=1.62, cols=2, tsize=17, bsize=14)
pn(s, 4)

# ---- 4. problems ----------------------------------------------------------------------------------------
s = new_slide("Problems to fix before training", "Meet 10  ·  Questions for Rajit")
cards(s, [
    ("Temperatures not physical", "peaks of 95,000–318,000 °C; the heat input looks too large for 1 mm elements."),
    ("Different reference temperatures", "transient starts at and holds the base at 180 °C; the steady sink is 65 °C."),
    ("Local domain in the wrong place", "the hemisphere centre misses the running block count: after block 484 it sits 9\u201325 mm from the heated block (right for only 484 of 3,844). This causes the dark band."),
    ("Negative temperatures", "151 transient steps go below 0 °C (minimum −1,145 °C), mostly in layer 1."),
    ("Diffusivity mismatch", "the hemisphere uses 5.18 mm²/s; the material files give 39.3 mm²/s."),
    ("Only one part", "3,844 pairs to train on, but no second shape to test on.")], y0=1.3, h=1.3, cols=2, bsize=11.5)
box(s, LM, 5.75, CW, 1.15, fill=DARK)
text(s, LM + 0.35, 5.75, CW - 0.7, 1.15, [
    P(R("Questions: ", bold=True, color="E8344E"), R("Is the heat input intended? Can the steady script be fixed (hemisphere index, layer sizes, same base temperature) and rerun? Which diffusivity sets the hemisphere? How many parts will there be?", color=WHITE), space_after=6),
    P(R("Next: ", bold=True, color="E8344E"), R("element-wise pairs (local steady field → transient field) for the diffusion model.", color="C9CCD3"))], size=13, anchor="middle")
pn(s, 5)
# ---- 6. problem 3 in detail ----------------------------------------------------------------------
s = new_slide("Problem 3: the local domain is in the wrong place", "Meet 10  ·  Steady-script bug")
box(s, LM, 1.3, 6.55, 1.25, fill=DARK, radius=0.06)
text(s, LM + 0.25, 1.3, 6.2, 1.25, [
    P(R("heated block:  ", color="C9CCD3"), R("ele-(sub_ele_no + ele_number + ele + 1)", color=WHITE), R("   ✔", color="6FCF97")),
    P(R("hemisphere:    ", color="C9CCD3"), R("new_id = sub_ele_no + ele + 1", color=WHITE)),
    P(R("               \u2718 missing ele_number (the running block count)", color="E8344E"))],
    size=11, font="Courier New", anchor="middle")
text(s, LM, 2.75, 6.55, 4.15, [
    P(R("Should: ", bold=True), R("hold everything outside a 3.4 mm hemisphere around the new block at 65 °C.", color=MUTED), bullet=True, space_after=6),
    P(R("Does: ", bold=True), R("after block 484 the hemisphere is centred on the wrong block, 9–25 mm away. Correct for only 484 of 3,844 blocks (13%).", color=MUTED), bullet=True, space_after=6),
    P(R("Cold floor: ", bold=True), R("only the substrate, layer 1 and 88 blocks of layer 2 can be held at 65 °C.", color=MUTED), bullet=True, space_after=6),
    P(R("Layer 1 ", bold=True), R("is normal (heated before the bug starts). ", color=MUTED), R("Layer 2 ", bold=True),
      R("sits on the cold floor: lowest steady peak, the dark band. Higher layers are farther away, so the steady peak rises with height.", color=MUTED), bullet=True, space_after=6),
    P(R("The transient peak stays flat (~118k) in every layer: ", bold=True), R("the pattern comes from the steady script, not the geometry.", color=MUTED), bullet=True)], size=12.5)
picture(s, F / "fig_bug_layers.png", 7.3, 1.3, w=5.45)
box(s, 7.3, 4.95, 5.45, 1.9, fill="FBE9EC")
text(s, 7.5, 4.95, 5.05, 1.9, [
    P(R("Fix: ", bold=True, color=RED), R("add the running count ele_number to the hemisphere index (and use the real part layer sizes), then rerun the steady run.")),
    P(R("The transient data is fine to use.", color=MUTED), space_after=0)], size=12.5, anchor="middle")
pn(s, 6)
prs.save(OUT_FILE)
print("saved", OUT_FILE)
