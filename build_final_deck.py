"""Build the final BTP presentation (storyline from Ayush's handwritten notes, final_presentation_outline.md).

    python make_final_assets.py && python build_final_deck.py

The LPBF clip (final_deck_assets/lpbf_video.mp4) is the CC BY 4.0 Zenodo video 10.5281/zenodo.5864463,
re-encoded to 720p with ffmpeg; lpbf_video_poster.png is its frame at 9.5 s.

Motivation -> generative AI -> Case 1 (1D) -> Case 2 (single bead) -> physics of the local domain ->
Case 3 (geometry) -> key message. Numbers are read from the saved results where a chart needs them.
"""

import csv
import json
from pathlib import Path

import numpy as np
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path("/home/ayush/Desktop/Ayush_btp")
A = ROOT / "final_deck_assets"
OUT_FILE = ROOT / "BTP_final_presentation.pptx"
# PowerPoint plays the MP4 (H.264). LibreOffice on this laptop has no H.264 decoder, so a second copy embeds WebM.
VIDEO, VIDEO_MIME = "lpbf_video.mp4", "video/mp4"
if "--libreoffice" in __import__("sys").argv:
    VIDEO, VIDEO_MIME = "lpbf_video.webm", "video/webm"
    OUT_FILE = ROOT / "BTP_final_presentation_LibreOffice.pptx"

# ---- design system -------------------------------------------------------
DARK, INK, MUTED, FAINT = "15161B", "1F2128", "5E6573", "8A909C"
LINE, PANEL, WHITE = "DADDE3", "F2F3F5", "FFFFFF"
RED, RED_DK, TEAL, GRAY, AMBER = "C8102E", "8E0B21", "1F7A8C", "B9BEC7", "C27C0E"
HEAD, BODY = "Arial", "Arial"
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
prs.core_properties.title = "BTP final presentation: steady to transient temperature mapping in LPBF"
prs.core_properties.author = "Ayush Gupta"
SLIDES = []


def new_slide(title, section=None, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[5])  # Title Only
    SLIDES.append((s, dark))
    if dark:
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(DARK)
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(LM), Inches(0.6), Inches(CW), Inches(0.7)
    fill_text(t.text_frame, P(R(title, bold=True)), size=28, color=WHITE if dark else INK, font=HEAD, anchor="top")
    if section:
        label(s, LM, 0.32, 9, section, color="E8344E" if dark else RED, size=12)
    return s


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def pic(s, name, x, y, w=None, h=None, alt=""):
    p = picture(s, A / name, x, y, w=w, h=h)
    p._element.nvPicPr.cNvPr.set("descr", alt)
    return p


def band(s, y, h, paras, size=18, x=LM, w=CW):
    box(s, x, y, w, h, fill=DARK)
    text(s, x + 0.35, y, w - 0.7, h, paras, size=size, color=WHITE, anchor="middle")


def rows(s, x, y, w, items, step, hs=17, bs=14, fill=RED):
    for i, (a, b) in enumerate(items):
        yy = y + i * step
        numdot(s, x, yy, i + 1, d=0.42, size=14, fill=fill)
        text(s, x + 0.6, yy - 0.02, w - 0.6, 0.35, P(R(a, bold=True)), size=hs)
        text(s, x + 0.6, yy + 0.34, w - 0.6, step - 0.4, b, size=bs, color=MUTED)


def tile(s, x, y, w, h, head, body, color=RED, hs=17, bs=14):
    box(s, x, y, w, h, fill=PANEL)
    text(s, x + 0.22, y + 0.14, w - 0.44, 0.4, P(R(head, bold=True)), size=hs, color=color)
    text(s, x + 0.22, y + 0.58, w - 0.44, h - 0.66, body, size=bs, color=INK)


def xy_chart(s, x, y, w, h, series, styles, kind=XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, size=12, legend=True, **axis):
    cd = XyChartData()
    for name, xs, ys in series:
        ser = cd.add_series(name)
        for a, b in zip(xs, ys):
            ser.add_data_point(float(a), float(b))
    chart = s.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    style_chart(chart, size, legend)
    for ser, st in zip(chart.plots[0].series, styles):
        ser.smooth = False
        if st.get("line"):
            ser.format.line.color.rgb = rgb(st["line"])
            ser.format.line.width = Pt(st.get("width", 2.25))
            if st.get("dash"):
                ser.format.line.dash_style = MSO_LINE.DASH
        else:
            ser.format.line.fill.background()
        if st.get("marker"):
            ser.marker.style = XL_MARKER_STYLE.CIRCLE
            ser.marker.size = st.get("msize", 9)
            ser.marker.format.fill.solid()
            ser.marker.format.fill.fore_color.rgb = rgb(st["marker"])
            ser.marker.format.line.color.rgb = rgb(WHITE)
        else:
            ser.marker.style = XL_MARKER_STYLE.NONE
    chart.has_title = False
    style_axes(chart, size=size, **axis)
    return chart


_bar_chart = bar_chart


def bar_chart(*args, **kw):
    chart = _bar_chart(*args, **kw)
    chart.has_title = False
    return chart


def chart_title(s, x, y, w, t):
    text(s, x, y, w, 0.32, P(R(t, bold=True)), size=14, color=INK)


# ================================================================================================
# 1  Title
s = new_slide("Mapping Steady-State Temperature to Transient Temperature in LPBF", dark=True)
t = s.shapes.title
t.left, t.top, t.width, t.height = Inches(LM), Inches(1.75), Inches(7.1), Inches(2.5)
fill_text(t.text_frame, P(R("Mapping Steady-State Temperature to Transient Temperature in LPBF", bold=True)), size=36, color=WHITE, font=HEAD, anchor="top", line_spacing=1.05)
label(s, LM, 1.2, 7, "BTP final presentation", color="E8344E", size=13)
text(s, LM, 4.5, 7.0, 0.8, "Can a fast steady-state temperature field tell us the slow transient one?", size=20, color="D5D8DE")
text(s, LM, 5.75, 7, 0.8, [P(R("Ayush Gupta", bold=True), size=18, color=WHITE, space_after=4), P("Faculty coordinator: Rajit Ranjan", size=16, color="D5D8DE")])
pic(s, "fig_title.png", 8.25, 1.2, w=4.5, alt="Heat map of steady peak temperature on the top layer of the geometry part, with a hole and a slot")
text(s, 8.25, 5.8, 4.5, 0.3, "Steady peak temperature, top layer of the Case 3 part", size=12, color=FAINT)
notes(s, "Good morning. My project asks one question: laser powder bed fusion needs a very slow transient temperature simulation. A steady-state simulation is very fast. Can the fast one tell us the slow one?")

# 2  LPBF and why thermal
s = new_slide("LPBF: heat decides the quality of the part", "Motivation")
vid = s.shapes.add_movie(str(A / VIDEO), Inches(LM), Inches(1.5), Inches(7.3), Inches(7.3 * 9 / 16),
                         poster_frame_image=str(A / "lpbf_video_poster.png"), mime_type=VIDEO_MIME)
vid._element.nvPicPr.cNvPr.set("descr", "Real LPBF machine footage: a recoater spreads tool-steel powder, then the laser melts the pattern with sparks of spatter")
text(s, LM, 5.72, 7.3, 0.6, [P(R("0–3 s: ", bold=True), R("the recoater spreads a fresh powder layer.   "), R("From 5 s: ", bold=True), R("the laser melts the pattern; the sparks are spatter."))], size=14)
text(s, LM, 6.42, 7.3, 0.4, "Video: E. Iakovakis, Univ. of Manchester, doi:10.5281/zenodo.5864463, CC BY 4.0", size=11, color=MUTED)
rows(s, 8.3, 1.6, 4.43, [("Surface defects", "Local overheating causes melt balls and dross."),
                        ("Weaker microstructure", "Slow cooling in hot zones changes the material."),
                        ("Build failure", "Residual stress and distortion can jam the recoater.")], 1.32)
band(s, 5.65, 1.1, P(R("We need to find the hot zones before printing.", bold=True)), x=8.3, w=4.43, size=17)
notes(s, "Play the video. This is a real LPBF machine. First the recoater blade spreads a thin layer of metal powder. Then the laser melts the shape of this layer; the sparks are spatter, small drops thrown out of the melt pool. LPBF means laser powder bed fusion. A blade spreads a thin layer of powder, a laser melts it along lines, and this repeats. The settings are laser power P, scan speed v and hatch spacing h. All the heat has to travel through the part to the baseplate. Loose powder conducts only about 1 percent as well as solid metal, so above overhangs heat gets trapped. Overheating gives melt balls and dross, poor microstructure, and even build failure. So we want to find hot zones before printing.")

# 3  Data
s = new_slide("The data comes from a layer-by-layer heat model", "Motivation  ·  How the data is generated")
pic(s, "fig_bvp.png", LM, 1.55, w=6.5, alt="Sketch of the thermal model: heat flux on the new layer, insulated sides, convection and radiation on top, fixed temperature baseplate")
rows(s, 7.6, 1.7, 5.13, [("Add material", "Elements are switched on, layer by layer."),
                        ("Apply heat", "New material receives a heat flux Q."),
                        ("Solve", "CalculiX solves the heat equation with these boundaries."),
                        ("Record", "Temperature at every node at every step.")], 1.13)
text(s, LM, 6.3, 6.5, 0.5, "Ranjan et al., Materials 13 (2020) 4576", size=12, color=MUTED)
notes(s, "All the LPBF data comes from the finite element model of the Ranjan paper. Material is added layer by layer. The new layer gets a heat flux. Sides are insulated because powder barely conducts. The baseplate is a heat sink. The top loses heat by convection and radiation. The solver gives temperature at every node and every step.")

# 4  Time
s = new_slide("Main motivation: transient analysis takes too long", "Motivation  ·  Time")
cw = 5.95
for x, img, head, col, big, cap in ((LM, "fig_ss_curve.png", "Steady state", TEAL, "2 min 9 s", "Analytical in nature: put t\u00a0=\u00a0∞. One solve, no time steps. Fast."),
                                    (LM + cw + 0.233, "fig_tr_curves.png", "Transient", RED, "21 h 28 min", "Numerical experiment: every time step is solved. Slow.")):
    box(s, x, 1.5, cw, 3.45, fill=PANEL)
    pic(s, img, x + 0.15, 1.75, w=3.1, alt=f"{head} temperature curve")
    text(s, x + 3.4, 1.7, 2.4, 0.4, P(R(head, bold=True)), size=18, color=col)
    text(s, x + 3.4, 2.15, 2.45, 0.6, P(R(big, bold=True)), size=24, color=col)
    text(s, x + 3.4, 2.85, 2.4, 2.2, cap, size=15)
    text(s, x + 0.25, 4.25, 3.0, 0.4, "Ranjan et al. test part, 20 cores", size=12, color=MUTED)
band(s, 5.2, 1.6, [P(R("Question: can we go from steady state to transient?", bold=True), size=24, space_after=4),
                     P("Steady is 599× faster, but its peak temperature was off by 65%, so the paper calls it qualitative only.", size=16, color="D5D8DE")])
notes(s, "This is the main motivation: time. In the paper's test part the full transient run took 21 hours 28 minutes. The steady-state version took 2 minutes 9 seconds, about 600 times faster. Steady state is like setting time to infinity, so there is no time stepping. But the steady peak temperature was 65 percent wrong, so it is only qualitative. So our question: can a model learn to turn the fast answer into the slow answer?")

# 5  Diffusion and FNO
s = new_slide("GenAI: diffusion is our choice; it might not be the best", "Method")
def node(s, x, y, col, d=0.3):
    return box(s, x, y, d, d, fill=col, shape=MSO_SHAPE.OVAL)
def link(s, x1, y1, x2, y2, col):
    c = s.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(col); c.line.width = Pt(2)
pic(s, "../deck_assets/fig_forward_process.png", LM, 1.42, w=7.6, alt="A temperature field becoming pure noise over 400 steps")
text(s, 8.55, 1.5, 4.18, 1.45, [P(R("How diffusion works: ", bold=True), R("add noise to the true field, then train a network to remove it, guided by the steady field."))], size=16, anchor="middle")
for x, head, col, sub, pts in ((LM, "Diffusion", RED, "One input, many answers", [
        "Learns a probability distribution", "Our problem is deterministic: one input, one answer"]),
                               (LM + 5.95 + 0.233, "FNO (neural operator)", TEAL, "One input, one answer", [
        "Learns a direct map between fields", "Others use it; possibly the best pick for physics"])):
    cw = 5.95
    box(s, x, 3.15, cw, 2.5, fill=PANEL)
    text(s, x + 0.3, 3.27, 3.6, 0.45, P(R(head, bold=True)), size=20, color=col)
    text(s, x + 0.3, 3.72, 3.3, 0.35, sub, size=14, color=MUTED)
    cx, cy = x + 4.2, 3.55
    node(s, cx, cy, DARK)
    for e in ([-0.3, 0, 0.3] if head == "Diffusion" else [0]):
        link(s, cx + 0.3, cy + 0.15, cx + 1.2, cy + 0.15 + e, col)
        node(s, cx + 1.2, cy + e, col)
    text(s, x + 0.3, 4.3, cw - 0.6, 1.3, [P(p, bullet=True, bullet_color=col, space_after=6) for p in pts], size=16)
cw = 5.95
band(s, 5.85, 0.95, P(R("It might not be the best. We are trying it, and we say where it does not fit.", bold=True)), size=19)
notes(s, "Diffusion is a generative AI method. In training we destroy the true field with noise and the network learns to undo it, guided by the steady field. To predict, we start from noise and clean it step by step. To be honest about the method: diffusion learns a probability distribution, but our problem is deterministic. Others use Fourier neural operators, which learn a direct map between fields and may be the best choice for physics. Diffusion is the choice we made, and we tested how far it goes.")

# 6  Three cases
s = new_slide("Three cases, each closer to a real part", "The work")
cards3 = [("Case 1", "1D rod", "Does the idea work at all?", "1,600 training runs, 300 unseen"),
          ("Case 2", "Single-bead deposition", "Does it work on LPBF data?", "27 settings of P, v, h"),
          ("Case 3", "Geometry", "Does geometry carry the information?", "One part, 3,844 element pairs")]
for i, (a, b, c, d) in enumerate(cards3):
    x = LM + i * 4.1
    box(s, x, 1.55, 3.93, 5.1, fill=PANEL)
    text(s, x + 0.3, 1.72, 3.3, 0.4, P(R(a.upper(), bold=True, spc=100)), size=14, color=RED)
    text(s, x + 0.3, 2.1, 3.4, 0.5, P(R(b, bold=True)), size=22)
    py = 3.1
    if i == 0:
        box(s, x + 0.45, py + 0.5, 3.0, 0.3, fill=GRAY, shape=MSO_SHAPE.RECTANGLE)
        box(s, x + 1.85, py - 0.1, 0.2, 0.6, fill=RED, shape=MSO_SHAPE.ISOSCELES_TRIANGLE).rotation = 180
    elif i == 1:
        box(s, x + 0.45, py + 0.55, 3.0, 0.55, fill=GRAY, shape=MSO_SHAPE.RECTANGLE)
        box(s, x + 1.05, py + 0.3, 1.8, 0.25, fill=RED, shape=MSO_SHAPE.RECTANGLE)
    else:
        box(s, x + 0.95, py - 0.1, 2.0, 1.2, fill=GRAY, shape=MSO_SHAPE.RECTANGLE)
        box(s, x + 1.25, py + 0.15, 0.6, 0.6, fill=PANEL, shape=MSO_SHAPE.OVAL)
        box(s, x + 2.15, py + 0.55, 0.55, 0.22, fill=PANEL, shape=MSO_SHAPE.RECTANGLE)
    text(s, x + 0.3, 4.6, 3.35, 0.9, P(R(c, bold=True)), size=17)
    text(s, x + 0.3, 5.6, 3.35, 0.8, d, size=15, color=MUTED)
notes(s, "We worked through three cases. Case 1 is a 1D rod, simple enough to generate lots of data, to check the idea. Case 2 is the real single-bead LPBF simulation. Case 3 is a part with holes, where geometry changes from place to place.")

# 7  Case 1 setup and network
s = new_slide("Case 1: 1D rod and the network", "Case 1  ·  1D")
pic(s, "fig_rod.png", LM, 1.42, w=5.6, alt="Rod with both ends at 65 C heated by a moving laser")
stat(s, 6.5, 1.5, 1.95, 2.0, "1,600", "training runs", num_size=26, cap_size=14)
stat(s, 8.64, 1.5, 1.95, 2.0, "300", "unseen test runs", num_size=26, cap_size=14)
stat(s, 10.78, 1.5, 1.95, 2.0, "Hidden", "material: only steady shows it", num_color=TEAL, num_size=22, cap_size=14)
def blk(x, y, w_, h, lines, fill=PANEL, color=INK, size=14):
    c = box(s, x, y, w_, h, fill=fill)
    c.text_frame.margin_left = c.text_frame.margin_right = Inches(0.08)
    fill_text(c.text_frame, lines, size=size, color=color, align="center", anchor="middle")
blk(0.6, 3.72, 4.4, 0.52, [P(R("Noisy transient", bold=True), R(": 99 values"))], fill="FBE9EC")
blk(0.6, 4.32, 4.4, 0.52, [P(R("Steady + ln P, ln v", bold=True), R(" → encoder: 64 + 2"))], fill="DDEEF1")
blk(0.6, 4.92, 4.4, 0.52, [P(R("Noise level", bold=True), R(" → code: 64"))])
blk(5.45, 3.72, 1.5, 1.72, [P(R("Join", bold=True)), P("229 values")])
blk(7.4, 3.72, 2.6, 1.72, [P(R("Denoiser MLP", bold=True), color=WHITE), P("229 → 256 → 256 → 99", color=WHITE), P("SiLU", color=WHITE)], fill=DARK, color=WHITE)
blk(10.45, 3.72, 2.28, 1.72, [P(R("Clean transient", bold=True), color=WHITE), P("99 values", color=WHITE)], fill=RED, color=WHITE)
for x in (5.05, 7.0, 10.05):
    arrow(s, x, 4.47, 0.35, 0.22)
band(s, 5.7, 1.1, P(R("197 k parameters  ·  8,000 updates  ·  400 noise levels  ·  50 cleaning steps, 8 predictions averaged", bold=True)), size=17)
notes(s, "The rod is 10 millimetres long with both ends at 65 degrees and a laser moves along it. The model gets the maximum steady curve plus P and v, and predicts the maximum transient curve. The material is not given as a number, so only the steady curve shows it. The network is small and fully connected. Three things go in: the noisy curve, the compressed steady curve with log P and log v, and the noise level. They are joined into 229 numbers, pass through two hidden layers of 256, and the output is the clean curve. We model 99 of the 101 points because the two ends are fixed.")

# 8  Case 1 results and lessons
d = np.load(ROOT / "rod1d_outputs/rod1d_data.npz")
pr = np.load(ROOT / "rod1d_outputs/pred_test_N1600.npy")
pk = np.abs(pr.max(1) - d["tr_test"].max(1)) / (d["tr_test"].max(1) - 65) * 100
i = int(np.argsort(pk)[150])
res = list(csv.DictReader(open(ROOT / "rod1d_outputs/rod1d_results.csv")))
ws = [r for r in res if r["name"] == "with_ss"]
ap = {(r["name"], r["n_train"]): r for r in json.load(open(ROOT / "rod1d_outputs/rod1d_allparams.json"))}
no = [r for r in res if r["name"] == "no_ss" and r["n_train"] == "1600"][0]
s = new_slide("Case 1: 1.0% peak error on unseen runs", "Case 1  ·  Results and lessons")
chart_title(s, LM, 1.45, 6.3, f"A typical unseen run (P = {d['test_P'][i]:.0f} W, v = {d['test_v'][i]:.0f} mm/s)")
xy_chart(s, LM, 1.77, 6.3, 3.6, [("Steady field (input)", d["x"], d["ss_test"][i]), ("True transient", d["x"], d["tr_test"][i]), ("Diffusion prediction", d["x"], pr[i])],
         [dict(line=TEAL), dict(line=DARK, width=4), dict(line=RED, dash=True)], vtitle="Max temperature (°C)", htitle="Position along the rod (mm)", vmin=0, vfmt="#,##0").category_axis.maximum_scale = 10
chart_title(s, 7.13, 1.45, 5.6, "The steady field helps only if it adds information")
bar_chart(s, 7.13, 1.77, 5.6, 3.6, ["Material hidden", "All numbers given"],
          [("With steady field", [float(ws[-1]["mean_abs_peak_err_pct"]), ap[("all_params+ss", 1600)]["mean_abs_peak_err_pct"]]),
           ("Without steady field", [float(no["mean_abs_peak_err_pct"]), ap[("all_params_only", 1600)]["mean_abs_peak_err_pct"]])],
          [TEAL, GRAY], fmt="0.0", vtitle="Mean peak error (%)", vmin=0)
band(s, 5.55, 1.3, [P(R("Using steady as the answer overshoots the peak by 174% on average; the model is within 1.0%.", color="D5D8DE"), size=16, space_after=4),
                    P(R("Lesson: the steady field is valuable only when it carries something P and v do not.", bold=True), size=19)])
notes(s, "On 300 runs the model never saw, the average peak error is 1 percent. Using the steady field directly would overshoot by 174 percent on average. In this typical run the prediction lies on top of the truth. The important lesson is on the right. When the material is hidden, removing the steady field raises the error from 1 percent to 18 percent. When all numbers are given, the steady field adds almost nothing. If asked about data: error falls from 3 percent with 25 runs to 1 percent with 1,600. Keep this lesson in mind for Case 2.")

# 9  Case 2 setup and results
z = np.load(ROOT / "maxmap_outputs/cv_predictions.npz")
tp, pp = z["true"].max(1), z["pred"].max(1)
s = new_slide("Case 2: single bead, and it looks very accurate", "Case 2  ·  Transient vs steady")
pic(s, "fig_beads.png", LM, 1.45, w=7.4, alt="Single bead on a plate, transient with moving laser versus steady with laser held at each position")
for i, (a, b, c) in enumerate((("1.3%", "mean peak error", RED), ("5.1%", "worst peak error", INK), ("27", "settings: 3 P × 3 v × 3 h", TEAL))):
    stat(s, LM + i * 2.53, 4.1, 2.34, 2.65, a, b, num_color=c, cap_size=15)
chart_title(s, 8.3, 1.45, 4.4, "Peak temperature, held-out settings")
lim = [2000, 10000]
ch = xy_chart(s, 8.2, 1.77, 4.53, 5.0, [("Held-out run", tp, pp), ("Perfect", lim, lim)], [dict(marker=RED), dict(line=GRAY, width=1.5)],
              kind=XL_CHART_TYPE.XY_SCATTER, vtitle="Predicted (°C)", htitle="True (°C)", vmin=2000, vmax=10000, vfmt="#,##0")
ca = ch.category_axis
ca.minimum_scale, ca.maximum_scale, ca.major_unit = 2000, 10000, 2000
ca.tick_labels.number_format, ca.tick_labels.number_format_is_linked = "#,##0", False
notes(s, "Case 2 uses real simulations of one bead on a plate. In the transient run the laser moves and we solve in time. In the steady run the laser is held at each position, with no time. The task: max steady plus P, v, h gives max transient. We hold out one setting at a time. The peak error is 1.3 percent on average and 5.1 percent at worst. This looks great. But where did the accuracy come from?")

# 10  Problem
order = np.argsort(tp)
s = new_slide("Case 2 problem: the steady field never changes", "Case 2  ·  The failure")
pic(s, "fig_ss_same.png", LM, 1.5, w=6.45, alt="Steady fields look identical for two very different settings while transient fields differ strongly")
chart_title(s, 7.35, 1.5, 5.4, "Peak temperature in each run (sorted)")
ca = xy_chart(s, 7.25, 1.82, 5.5, 3.6, [("Steady", np.arange(26), z["ss"].max(1)[order]), ("Transient", np.arange(26), tp[order])],
         [dict(line=TEAL, width=3), dict(line=RED, width=3)], vtitle="Peak (°C)", htitle="Run (sorted)", vmin=0, vmax=10000, vfmt="#,##0").category_axis
ca.minimum_scale, ca.maximum_scale, ca.major_unit = 0, 25, 5
band(s, 5.55, 1.3, [P(R("Why? ", bold=True, color="F6C453"), R("All 27 runs use one standard geometry: the same bead on the same plate. The local steady model sees the same domain every time, so it gives the same field (within 0.12 °C).", color=WHITE), size=16, space_after=4),
                    P(R("So the accuracy came from P, v, h, not from the steady field.", bold=True), size=18)])
notes(s, "Here is the problem. Look at the top row: 60 watts and 120 watts, very different settings, but the steady fields are the same. Peak 6,667 degrees in every run. Across all 27 runs, the steady field differs by at most 0.12 degrees. The transient, at the bottom, changes a lot. So the steady field carries no information about the setting. All the accuracy came from the three numbers P, v and h. Remember the lesson from Case 1. To understand why this happens, we need some physics.")

# 11  Physics
s = new_slide("Steady state means waiting for infinite time", "Physics  ·  Why a local domain")
pic(s, "fig_weld.png", LM, 1.5, w=4.65, alt="Welding sketch with heat source, melt pool, isothermal lines and heat flowing from the new bead to older beads")
pic(s, "fig_ss_local.png", 5.4, 1.5, w=7.33, alt="Left: heat reaches the whole part after infinite time. Right: heat only reaches a local hemisphere")
for i, (a, b, c) in enumerate((("New bead heats old beads", "Heat is concentrated around the melt pool.", INK),
                               ("Not physical", "The next bead arrives long before heat fills the part.", RED),
                               ("Closer to reality", "Solve steady state only on the local domain.", TEAL))):
    tile(s, LM + i * 4.1, 4.3, 3.93, 2.45, a, b, color=c, hs=20, bs=18)
notes(s, "Think of welding. Under the heat source there is a melt pool, with isothermal lines around it, and the new bead heats the older beads. A steady-state temperature means: deposit the bead and wait for infinite time. Then heat would reach the whole part. That does not happen in a real build, because the next bead comes very soon. So we localise: solve the steady problem only near the new bead. The Ranjan paper makes the same point: only the domain close to the top layer should be considered.")

# 12  Lc
s = new_slide("How far does heat go? The characteristic length", "Physics  ·  Lc")
pic(s, "fig_lc.png", LM, 1.5, w=6.8, alt="Temperature in a bar with a 100 C end at three times, with the characteristic length marked")
box(s, 7.7, 1.55, 5.03, 1.9, fill=DARK)
text(s, 7.95, 1.68, 4.55, 0.95, P(R("Lc = √(α t)", bold=True)), size=38, color=WHITE, anchor="middle", align="center")
text(s, 7.95, 2.7, 4.55, 0.5, "α = k / (ρ c)   in m²/s", size=20, color="F6C453", align="center")
text(s, 7.7, 3.65, 5.03, 1.6, [P(R("Bar at 20 °C, one end at 100 °C. ", bold=True), R("Heat moves down like a wave, and Lc says how far it has gone."), space_after=8),
                               P(R("It slows down: ", bold=True), R("4× the time gives only 2× the distance."))], size=16)
tile(s, 7.7, 5.35, 5.03, 1.45, "In our Case 3 data", "α = 5.18 mm²/s, t = 1 s: Lc = 2.28 mm. Domain diameter = 3 Lc = 6.83 mm.", color=TEAL, hs=16, bs=14)
notes(s, "How big is the local domain? Take a bar at 20 degrees and hold one end at 100. After time t, how far has the heat gone? That distance is the characteristic length, Lc equals the square root of alpha t. Alpha is the thermal diffusivity, k over rho c, in metres squared per second. Heat moves like a wave, but one that slows down: four times the time, only twice the distance. In the Case 3 script, alpha is 5.18 and the heating time is 1 second, so Lc is 2.28 millimetres, and the hemisphere has diameter 3 Lc, 6.83 millimetres, radius 3.41.")

# 13  Same vs different geometry
s = new_slide("Why Case 2 failed: one standard geometry", "Physics  ·  Geometry decides the local domain")
text(s, LM, 1.42, 6.2, 0.35, P(R("Same geometry: same domain for every bead", bold=True)), size=17, color=RED)
pic(s, "fig_same_geom.png", LM, 1.8, w=6.2, alt="Three beads on a block, each with an identical hemisphere domain; the end wall lies outside")
box(s, LM, 4.45, 6.2, 1.05, fill=PANEL)
text(s, LM + 0.22, 4.45, 5.76, 1.05, "Case 2 had one standard geometry: every bead sees the same domain (the end wall lies outside it), so every run gives the same steady result.", size=15, anchor="middle")
text(s, 7.2, 1.42, 5.53, 0.35, P(R("Different geometry: different domains", bold=True)), size=17, color=TEAL)
pic(s, "fig_diff_geom.png", 7.45, 1.8, h=3.65, alt="Block with a hole: domains are cut near the hole and at the wall and full in the bulk")
band(s, 5.65, 1.15, P(R("Geometry comes in as the input: different shapes give different local domains, so different results.", bold=True)), size=19)
notes(s, "Now we can explain Case 2. On a plain block every bead sees the same hemisphere below it. Even a bead near the end wall does not feel the wall, because the wall is outside the domain. Same domain, same result. Along the middle of the bead the steady peak is about 3,206 degrees; only the two ends of the bead differ. If asked about power: the 27 steady runs also did not respond to P, v or h, so the heat input in those steady runs was effectively the same. Now take a part with a hole. A bead above the hole, or at a wall, has part of its domain missing. So domains differ, and the steady field picks up the geometry.")

# 14  Case 3
s = new_slide("Case 3: geometry appears in both fields", "Case 3  ·  Block with a hole and a slot")
pic(s, "fig_part3d.png", LM, 1.5, w=3.3, alt="3D view of the block with a hole and a slot on a substrate")
text(s, LM, 4.5, 3.3, 0.9, "20 × 20 mm, 10 layers; 3,844 matched element pairs", size=14, color=MUTED)
pic(s, "fig_geo_maps.png", 4.05, 1.5, h=3.85, alt="Maps of steady and transient peaks on layers 6 and 10, hotter near walls, the hole and the slot")
chart_title(s, 9.1, 1.45, 3.63, "Median peak (10³ °C)")
bar_chart(s, 9.0, 1.77, 3.73, 3.6, ["0", "1", "2"], [("Steady", [141.9, 165.6, 202.4]), ("Transient", [117.9, 120.2, 129.5])], [TEAL, RED], fmt="0", vmin=0, htitle="Empty neighbours")
band(s, 5.6, 1.2, P(R("Peaks rise next to walls and voids in both fields; they correlate at 0.88 after element 1,364.", bold=True)), size=19)
notes(s, "For Case 3 we have a block of 20 by 20 millimetres, 10 layers high, with a hole and a slot, built one element at a time. Each element is a bead, and the steady solve uses a local hemisphere of radius 3.41 millimetres. This gives 3,844 pairs. Now the steady field changes with position. Both steady and transient peaks are higher next to the walls, hole and slot. From element 1,365 on they correlate at 0.88; over all elements it is 0.65. Absolute temperatures are not physical, so read these as patterns. Part of the spread in the steady peaks also comes from the script error on the next slide.")

# 15  Case 3 result
r0 = json.load(open(ROOT / "tts_outputs/model_results.json"))
chk = json.load(open(ROOT / "tts_outputs/model_checks.json"))
st = [r0["diffusion"]["mae_C"]] + [chk[f"steady_seed{k}"]["mae_C"] for k in (1, 2)]
sh = [chk[f"shape_only_seed{k}"]["mae_C"] for k in (0, 1, 2)]
s = new_slide("Case 3 result: the model learns the geometry effect", "Case 3  ·  Unseen layers 9–10")
chart_title(s, LM, 1.45, 7.0, f"Mean error on unseen layers 9–10 ({r0['n_test']} elements; train on layers 1–8)")
cats = ["Guess the training average", "Straight line from steady peak", "Diffusion: steady patch", "Diffusion: shape only"]
vals = [r0["mean_of_training"]["mae_C"], r0["linear_centre_steady"]["mae_C"], float(np.mean(st)), float(np.mean(sh))]
ch = bar_chart(s, LM, 1.77, 7.0, 3.6, cats[::-1], [("Mean error (°C)", vals[::-1])], [GRAY], kind=XL_CHART_TYPE.BAR_CLUSTERED,
               legend=False, fmt="#,##0", vmin=0, gap=50, vtitle="Mean error (°C, dataset units)", vfmt="#,##0")
for k, c in enumerate([TEAL, RED, GRAY, GRAY]):
    pt = ch.plots[0].series[0].points[k]
    pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(c)
text(s, LM, 5.42, 7.0, 0.35, f"Diffusion: mean of 3 seeds (steady {min(st):,.0f}–{max(st):,.0f}; shape only {min(sh):,.0f}–{max(sh):,.0f}).", size=12, color=MUTED)
box(s, 7.95, 1.5, 4.78, 3.9, fill=PANEL)
text(s, 8.2, 1.62, 4.3, 0.4, P(R("Read with care", bold=True)), size=18, color=RED)
text(s, 8.2, 2.1, 4.35, 3.2, [P(x, bullet=True, space_after=7) for x in (
    "Temperatures are ~10⁵ °C (not physical); errors are in these units.",
    "Local domain misplaced for most elements (script error).",
    "Different references: 180 °C vs 65 °C.",
    "One part: the test is new layers, not a new part.")], size=15)
band(s, 5.85, 0.95, P(R("Error 2.5× below an average guess, and the shape alone is enough: Case 3 is a geometry mapping.", bold=True)), size=18)
notes(s, "Here is the Case 3 result. Same diffusion code. Each sample is one element with the 5 by 5 patch of steady peaks around it. We train on layers 1 to 8 and test on layers 9 and 10, where the hole widens into a round hole the model has never seen. The average error is about 1,075 degrees in the dataset's units, against 2,634 for guessing the average and 1,490 for a straight line from the steady peak. So about 2.5 times better than guessing. Interesting check: if we give only the shape, where the solid and the holes are, without steady values, it does as well, about 890. So in this dataset the information is the geometry itself; the steady values, with the misplaced local domain, add nothing extra yet. That supports the key message. Remember the units are not physical, and it is one part.")

# 16  Key message
s = new_slide("Key message", "Conclusion", dark=True)
text(s, LM, 1.45, CW, 1.75, P(R("Steady → transient makes sense not as a P, v, h mapping, but as a ", bold=True), R("geometry mapping.", bold=True, color="F6C453")), size=34, color=WHITE, anchor="middle")
concl = [("Case 1: the idea works", "1.0% peak error, when the steady field holds information the numbers do not."),
         ("Case 2: the expected failure", "One standard geometry, same local domain, same steady field: accuracy came from P, v, h."),
         ("Case 3: geometry is the signal", "On unseen layers, error 2.5× below an average guess; shape alone is enough.")]
for i, (a, b) in enumerate(concl):
    x = LM + i * 4.1
    box(s, x, 3.5, 3.93, 2.55, fill="262830")
    text(s, x + 0.25, 3.68, 3.45, 0.75, P(R(a, bold=True)), size=18, color=WHITE)
    text(s, x + 0.25, 4.5, 3.45, 1.45, b, size=16, color="D5D8DE")
text(s, LM, 6.3, CW, 0.5, P(R("Thank you. Questions are welcome.", bold=True)), size=20, color="F6C453")
notes(s, "To finish. The idea works when the steady field holds information the numbers do not, as in Case 1. In Case 2, the same local domain gives the same steady field, so it adds nothing. In Case 3, geometry changes the local domain, and both fields respond. So the key message: steady to transient makes sense as a geometry mapping, not as a P, v, h mapping. Thank you.")

# 17  References
s = new_slide("References", "References")
refs = ["R. Ranjan, C. Ayas, M. Langelaar, F. van Keulen. Fast detection of heat accumulation in powder bed fusion using computationally efficient thermal models. Materials 13 (2020) 4576.",
        "J. Ho, A. Jain, P. Abbeel. Denoising diffusion probabilistic models. NeurIPS 2020.",
        "J. Song, C. Meng, S. Ermon. Denoising diffusion implicit models. ICLR 2021.",
        "Z. Li et al. Fourier neural operator for parametric partial differential equations. ICLR 2021."]
for i, r in enumerate(refs):
    numdot(s, LM, 1.7 + i * 1.1, i + 1, d=0.42)
    text(s, LM + 0.65, 1.72 + i * 1.1, CW - 0.8, 0.9, r, size=17)
notes(s, "These are the main references.")

# ---- footers -----------------------------------------------------------------------------------
total = len(SLIDES)
for n, (sl, dark) in enumerate(SLIDES, 1):
    if n == 1:
        continue
    text(sl, LM, 7.05, 8, 0.25, "Ayush Gupta  ·  BTP final presentation", size=11, color=FAINT)
    text(sl, SW - LM - 1.2, 7.05, 1.2, 0.25, f"{n} / {total}", size=11, color=FAINT, align="right")

prs.save(OUT_FILE)
print("saved", OUT_FILE, total, "slides")
