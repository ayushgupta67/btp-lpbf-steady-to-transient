"""Build the BTP final report (Word) from the project's results.

    cd report
    ../.venv/bin/python make_report_figures.py          # figures -> report/figures/
    PYTHONPATH=<python-docx> ../.venv/bin/python build_report.py   # -> .build/BTP_report_draft.docx
    /usr/bin/python3 finalize.py .build/BTP_report_draft.docx BTP_report.docx BTP_report.pdf

The content is written once in `content(d)`; it runs twice: a dry pass numbers the figures, tables,
equations and references (references in order of first citation), the second pass writes the document.
In the text: **bold**, *italic*, _{sub}, ^{sup}, [@ref], @fig:key, @tab:key, @eq:key, {{placeholder}}.
"""
import json
import re
from pathlib import Path

from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX, WD_LINE_SPACING
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor

from docxkit import (Document, equation, field, mbar, mfrac, mhat, mparen, mr, msqrt, msub, msubsup, msum, msup,
                     mt, page_numbers)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / "figures"
FONT = "Times New Roman"

# ---- facts used in the text, read from the result files ------------------------------------------------
MAX = json.load(open(ROOT / "maxmap_outputs/summary.json"))
ROD = json.load(open(ROOT / "rod1d_outputs/rod1d_summary.json"))
C3 = json.load(open(ROOT / "tts_outputs/model_results.json"))
C3K = json.load(open(ROOT / "tts_outputs/model_checks.json"))
C3_ST = [C3["diffusion"]["mae_C"]] + [C3K[f"steady_seed{k}"]["mae_C"] for k in (1, 2)]
C3_SH = [C3K[f"shape_only_seed{k}"]["mae_C"] for k in (0, 1, 2)]
C3_ST_NV = [C3["diffusion"]["mae_next_to_void_or_wall_C"]] + [C3K[f"steady_seed{k}"]["mae_next_to_void_or_wall_C"] for k in (1, 2)]
C3_SH_NV = [C3K[f"shape_only_seed{k}"]["mae_next_to_void_or_wall_C"] for k in (0, 1, 2)]
C3_ST_B = [C3["diffusion"]["mae_in_bulk_C"]] + [C3K[f"steady_seed{k}"]["mae_in_bulk_C"] for k in (1, 2)]
C3_SH_B = [C3K[f"shape_only_seed{k}"]["mae_in_bulk_C"] for k in (0, 1, 2)]
mean = lambda v: sum(v) / len(v)

REFS = {
    "ranjan": "R. Ranjan, C. Ayas, M. Langelaar, F. van Keulen. Fast detection of heat accumulation in powder bed fusion using computationally efficient thermal models. Materials 13(20), 4576 (2020).",
    "king": "W. E. King, A. T. Anderson, R. M. Ferencz, N. E. Hodge, C. Kamath, S. A. Khairallah, A. M. Rubenchik. Laser powder bed fusion additive manufacturing of metals; physics, computational, and materials challenges. Applied Physics Reviews 2, 041304 (2015).",
    "debroy": "T. DebRoy, H. L. Wei, J. S. Zuback, T. Mukherjee, J. W. Elmer, J. O. Milewski, A. M. Beese, A. Wilson-Heid, A. De, W. Zhang. Additive manufacturing of metallic components – process, structure and properties. Progress in Materials Science 92, 112–224 (2018).",
    "hodge": "N. E. Hodge, R. M. Ferencz, J. M. Solberg. Implementation of a thermomechanical model for the simulation of selective laser melting. Computational Mechanics 54, 33–51 (2014).",
    "rosenthal": "D. Rosenthal. The theory of moving sources of heat and its application to metal treatments. Transactions of the ASME 68, 849–866 (1946).",
    "carslaw": "H. S. Carslaw, J. C. Jaeger. Conduction of Heat in Solids, 2nd ed. Oxford University Press (1959).",
    "dhondt": "G. Dhondt. The Finite Element Method for Three-Dimensional Thermomechanical Applications. Wiley (2004).",
    "sohl": "J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. Proceedings of ICML (2015).",
    "ho": "J. Ho, A. Jain, P. Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems 33 (NeurIPS 2020).",
    "ddim": "J. Song, C. Meng, S. Ermon. Denoising diffusion implicit models. International Conference on Learning Representations (ICLR 2021).",
    "fno": "Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar. Fourier neural operator for parametric partial differential equations. ICLR (2021).",
    "deeponet": "L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence 3, 218–229 (2021).",
    "pinn": "M. Raissi, P. Perdikaris, G. E. Karniadakis. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics 378, 686–707 (2019).",
    "pidm": "J.-H. Bastek, W. Sun, D. M. Kochmann. Physics-informed diffusion models. International Conference on Learning Representations (ICLR 2025).",
    "shu": "D. Shu, Z. Li, A. Barati Farimani. A physics-informed diffusion model for high-fidelity flow field reconstruction. Journal of Computational Physics 478, 111972 (2023).",
    "meltpool": "Inexpensive high-fidelity melt pool models in additive manufacturing using generative deep diffusion. arXiv:2311.16168 (2023).",
    "monitor": "Diffusion generative model-based learning for smart layer-wise monitoring of additive manufacturing. ASME Journal of Computing and Information Science in Engineering (2024).",
    "mozaffar": "M. Mozaffar, A. Paul, R. Al-Bahrani, S. Wolff, A. Choudhary, A. Agrawal, K. Ehmann, J. Cao. Data-driven prediction of the high-dimensional thermal history in directed energy deposition processes via recurrent neural networks. Manufacturing Letters 18, 35–39 (2018).",
    "adamw": "I. Loshchilov, F. Hutter. Decoupled weight decay regularization. ICLR (2019).",
    "silu": "S. Elfwing, E. Uchibe, K. Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks 107, 3–11 (2018).",
    "vaswani": "A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin. Attention is all you need. NeurIPS (2017).",
    "video": "E. Iakovakis. Selective laser melting process of tool steel powder [video]. University of Manchester, Zenodo, doi:10.5281/zenodo.5864463 (2022). CC BY 4.0.",
}


class Doc:
    def __init__(self, dry):
        self.dry = dry
        self.num = {}          # "fig:key" -> number
        self.refs = []         # citation keys in order of first use
        self.counters = {"fig": 0, "tab": 0, "eq": 0}
        self.d = None if dry else self._new()

    # ---- document setup ---------------------------------------------------------------------------
    def _new(self):
        d = Document()
        sec = d.sections[0]
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        sec.left_margin, sec.right_margin, sec.top_margin, sec.bottom_margin = Cm(3.0), Cm(2.5), Cm(2.5), Cm(2.5)
        st = d.styles
        n = st["Normal"]
        n.font.name, n.font.size = FONT, Pt(12)
        n.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        pf = n.paragraph_format
        pf.line_spacing, pf.space_after, pf.space_before = 1.2, Pt(5), Pt(0)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        bt = st["Body Text"]
        bt.base_style = n
        for name, size, before, after, italic in (("Heading 1", 18, 0, 18, False), ("Heading 2", 14, 14, 6, False), ("Heading 3", 12, 10, 4, True)):
            h = st[name]
            h.font.name, h.font.size, h.font.bold, h.font.italic = FONT, Pt(size), True, italic
            h.font.color.rgb = RGBColor(0x15, 0x16, 0x1B)
            h.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
            h.element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
            h.element.rPr.rFonts.set(qn("w:ascii"), FONT)
            for att in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                h.element.rPr.rFonts.attrib.pop(qn(att), None)
            hp = h.paragraph_format
            hp.space_before, hp.space_after, hp.keep_with_next = Pt(before), Pt(after), True
            hp.line_spacing = 1.15
            hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
        cap = st["Caption"]
        cap.font.name, cap.font.size, cap.font.italic, cap.font.bold = FONT, Pt(10.5), False, False
        cap.font.color.rgb = RGBColor(0x1F, 0x21, 0x28)
        cap.paragraph_format.line_spacing, cap.paragraph_format.space_after, cap.paragraph_format.space_before = 1.15, Pt(10), Pt(2)
        cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for name in ("List Bullet", "List Number"):
            lb = st[name]
            lb.font.name, lb.font.size = FONT, Pt(12)
            lb.paragraph_format.line_spacing, lb.paragraph_format.space_after = 1.2, Pt(2)
        for name in ("TOC 1", "TOC 2", "TOC 3"):
            try:
                t = st[name]
            except KeyError:
                t = st.add_style(name, 1)
            t.font.name, t.font.size = FONT, Pt(12)
            t.paragraph_format.line_spacing, t.paragraph_format.space_after = 1.0, Pt(1)
        settings = d.settings.element
        settings.append(parse_xml(f'<w:updateFields {nsdecls("w")} w:val="true"/>'))
        return d

    # ---- inline markup -----------------------------------------------------------------------------
    def _sub(self, text):
        def ref(m):
            keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
            out = []
            for k in keys:
                assert k in REFS, k
                if k not in self.refs:
                    self.refs.append(k)
                out.append(str(self.refs.index(k) + 1))
            return "[" + ", ".join(out) + "]"
        text = re.sub(r"\[(@[\w]+(?:;\s*@[\w]+)*)\]", ref, text)

        def cross(m):
            key = f"{m.group(1)}:{m.group(2)}"
            if self.dry:
                return "0"
            assert key in self.num, key
            n = self.num[key]
            return {"fig": f"Figure {n}", "tab": f"Table {n}", "eq": f"Eq. ({n})"}[m.group(1)]
        return re.sub(r"@(fig|tab|eq):([\w]+)", cross, text)

    def _runs(self, p, text, size=None, color=None):
        text = self._sub(text)
        if self.dry:
            return
        for tok in re.split(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\}|\{\{.+?\}\})", text):
            if not tok:
                continue
            r = None
            if tok.startswith("**"):
                r = p.add_run(tok[2:-2]); r.bold = True
            elif tok.startswith("{{"):
                r = p.add_run(tok[2:-2]); r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            elif tok.startswith("*"):
                r = p.add_run(tok[1:-1]); r.italic = True
            elif tok.startswith("_{"):
                r = p.add_run(tok[2:-1]); r.font.subscript = True
            elif tok.startswith("^{"):
                r = p.add_run(tok[2:-1]); r.font.superscript = True
            else:
                r = p.add_run(tok)
            if size:
                r.font.size = Pt(size)
            if color:
                r.font.color.rgb = RGBColor.from_string(color)

    # ---- blocks --------------------------------------------------------------------------------------
    def p(self, text, align=None, size=None, space_after=None, indent=None, keep=False, color=None, bold=False, style=None):
        if self.dry:
            self._sub(text); return
        para = self.d.add_paragraph(style=style)
        self._runs(para, f"**{text}**" if bold else text, size, color)
        if align:
            para.alignment = {"c": WD_ALIGN_PARAGRAPH.CENTER, "l": WD_ALIGN_PARAGRAPH.LEFT, "r": WD_ALIGN_PARAGRAPH.RIGHT}[align]
        if space_after is not None:
            para.paragraph_format.space_after = Pt(space_after)
        if indent is not None:
            para.paragraph_format.left_indent = Cm(indent)
        if keep:
            para.paragraph_format.keep_with_next = True
        return para

    def bullets(self, items, numbered=False):
        for it in items:
            if self.dry:
                self._sub(it); continue
            para = self.d.add_paragraph(style="List Number" if numbered else "List Bullet")
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            self._runs(para, it)

    def chapter(self, number, title):
        if self.dry:
            return
        h = self.d.add_heading(level=1)
        # front matter, chapter 1 and the back matter start a page; later chapters run on
        h.paragraph_format.page_break_before = not isinstance(number, int) or number == 1
        if not h.paragraph_format.page_break_before:
            h.paragraph_format.space_before = Pt(24)
        h.add_run(f"{number}  {title}" if number else title)

    def h2(self, text):
        if not self.dry:
            self.d.add_heading(text, level=2)

    def h3(self, text):
        if not self.dry:
            self.d.add_heading(text, level=3)

    def _caption(self, kind, key, text):
        self.counters[kind] += 1
        n = self.counters[kind]
        self.num[f"{kind}:{key}"] = n
        if self.dry:
            self._sub(text); return
        label = {"fig": "Figure", "tab": "Table"}[kind]
        c = self.d.add_paragraph(style="Caption")
        r = c.add_run(f"{label} "); r.bold = True
        field(c, f"SEQ {label} \\* ARABIC", str(n), run_props="<w:rPr><w:b/></w:rPr>")
        r = c.add_run(". "); r.bold = True
        self._runs(c, text)
        return c

    def fig(self, key, file, width_cm, caption):
        if not self.dry:
            para = self.d.add_paragraph()
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.keep_with_next = True
            para.paragraph_format.space_before = Pt(6)
            para.paragraph_format.line_spacing = 1.0
            para.add_run().add_picture(str(FIG / file), width=Cm(width_cm if width_cm <= 11 else 11 + (width_cm - 11) * 0.6))
        self._caption("fig", key, caption)

    def table(self, key, caption, rows, widths_cm, align=None, size=10.5, bold_last=False, highlight=()):
        c = self._caption("tab", key, caption)
        for row in rows:
            for cell in row:
                self._sub(str(cell))
        if self.dry:
            return
        c.paragraph_format.keep_with_next = True
        t = self.d.add_table(rows=len(rows), cols=len(rows[0]))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblPr = t._tbl.tblPr
        tblPr.append(parse_xml(f'<w:tblBorders {nsdecls("w")}><w:top w:val="single" w:sz="12" w:color="15161B"/>'
                               f'<w:bottom w:val="single" w:sz="12" w:color="15161B"/></w:tblBorders>'))
        for i, row in enumerate(rows):
            tr = t.rows[i]._tr
            trPr = tr.get_or_add_trPr()
            trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
            if i == 0:
                trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.width = Cm(widths_cm[j])
                tcPr = cell._tc.get_or_add_tcPr()
                if i == 0:
                    tcPr.append(parse_xml(f'<w:tcBorders {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:color="15161B"/></w:tcBorders>'))
                if i in highlight:
                    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="F6E3E6"/>'))
                para = cell.paragraphs[0]
                para.paragraph_format.line_spacing = 1.0
                para.paragraph_format.space_after = Pt(2)
                para.paragraph_format.space_before = Pt(2)
                para.paragraph_format.keep_with_next = i < len(rows) - 1
                a = (align[j] if align else ("l" if j == 0 else "c"))
                para.alignment = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT}[a]
                txt = str(val)
                if i == 0 or (bold_last and i == len(rows) - 1) or i in highlight:
                    txt = f"**{txt}**" if txt else txt
                self._runs(para, txt, size=size)
        sp = self.d.add_paragraph()
        sp.paragraph_format.space_after = Pt(4)
        sp.paragraph_format.line_spacing = 1.0

    def eq(self, key, omml):
        self.counters["eq"] += 1
        self.num[f"eq:{key}"] = self.counters["eq"]
        if not self.dry:
            equation(self.d, omml, str(self.counters["eq"]), style="Normal")

    def page_break(self):
        if not self.dry:
            self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def section(self, fmt, start=None, show=True):
        if self.dry:
            return
        s = self.d.add_section(WD_SECTION.NEW_PAGE)
        page_numbers(s, fmt, start, show)

    def toc(self, instr, placeholder):
        if not self.dry:
            para = self.d.add_paragraph()
            field(para, instr, placeholder)


# ======================================================================================================
def front(d):
    if d.dry:
        return
    doc = d.d
    page_numbers(doc.sections[0], "lowerRoman", 1, show=False)
    def c(text, size, bold=False, after=6, italic=False, color=None):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(after); p.paragraph_format.line_spacing = 1.15
        d._runs(p, text, size=size, color=color)
        for r in p.runs:
            r.bold = r.bold or bold
            r.italic = r.italic or italic
        return p
    c("", 12, after=30)
    c("Mapping Steady-State Temperature to Transient Temperature in Laser Powder Bed Fusion", 22, bold=True, after=10)
    c("Conditional diffusion models, from a 1D rod to part geometry", 14, italic=True, after=60)
    c("B.Tech Project Final Report", 13, after=70)
    c("Ayush Gupta", 16, bold=True, after=2)
    c("B23ME1014", 12, after=60)
    c("Supervisor", 12, after=4)
    c("Dr. Rajit Ranjan", 14, bold=True, after=110)
    c("Department of Mechanical Engineering", 12, after=2)
    c("Indian Institute of Technology Jodhpur", 13, bold=True, after=2)
    c("November 2026", 12, after=0)



def abstract(d):
    d.chapter("", "Abstract")
    d.p("Laser powder bed fusion (LPBF) builds metal parts layer by layer, and heat that cannot escape to the baseplate causes local overheating, a common source of defects. A transient thermal simulation predicts these hot zones accurately but takes about 21.5 hours on 20 cores for one part [@ranjan]; a steady-state analysis takes about 2 minutes but overestimates the peak temperature by 65%. This project asks whether a conditional denoising diffusion model can turn the fast steady-state field into the slow transient one.")
    d.p("The model is applied to three cases. In **Case 1**, a laser moving along a 1D rod, it predicts the peak temperature of unseen runs within 1.0% on average, where the steady field itself is off by 174%; the steady field helps only when it carries information that the process numbers do not. In **Case 2**, a single bead simulated for 27 combinations of power *P*, speed *v* and hatch *h*, the model reaches 1.3% peak error, but the localised steady fields of all runs are identical to within 0.12 °C: with one standard geometry the steady model sees the same local domain every time, and the accuracy comes only from *P*, *v* and *h*. Heat-conduction physics explains this: a steady state needs infinite time, whereas heat travels only a characteristic length *L*_{c} = √(*αt*) before the next bead arrives, so the steady analysis is local and can change only with the geometry. In **Case 3**, a block with a hole and a slot, steady and transient peaks both rise next to walls and voids, and a model trained on layers 1–8 predicts the unseen layers 9–10 with about 2.5 times less error than a constant guess.")
    d.p("The main conclusion is that the steady-to-transient mapping is meaningful as a **geometry mapping**, not as a mapping of process parameters. Limitations of the datasets, including non-physical temperature levels in Case 3, are stated.")
    d.p("**Keywords:** laser powder bed fusion; steady-state thermal model; diffusion model; characteristic length; part geometry.", space_after=0)


def nomenclature(d):
    d.chapter("", "Nomenclature")
    d.table("nom", "Symbols and abbreviations used in this report.", [
        ["Symbol", "Meaning", "Unit"],
        ["*T*, *T*_{0}", "temperature; reference (ambient or base) temperature", "°C"],
        ["*k*, *ρ*, *c*_{p}", "thermal conductivity, density, specific heat", "W/m·K, kg/m³, J/kg·K"],
        ["*α* = *k*/(*ρc*_{p})", "thermal diffusivity", "m²/s (mm²/s)"],
        ["*L*_{c} = √(*αt*)", "characteristic (diffusion) length after time *t*", "m (mm)"],
        ["*P*, *v*, *h*", "laser power, scan speed, hatch spacing", "W, mm/s, mm"],
        ["*q* = *P*/*h*", "power per hatch spacing (single-bead runs)", "W/mm"],
        ["*Q*", "volumetric heat source", "W/m³"],
        ["*x*_{0}, *x*_{τ}", "clean target field; noisy target at noise level *τ*", "–"],
        ["*ᾱ*_{τ}", "cumulative signal fraction of the noise schedule", "–"],
        ["*c*, *s*", "conditioning field (steady state) and conditioning scalars", "–"],
        ["*δ*", "peak-temperature error relative to the true peak rise", "%"],
        ["MAE, RMSE", "mean absolute error, root-mean-square error", "°C"],
        ["LPBF", "laser powder bed fusion", ""],
        ["SS", "steady state (steady-state thermal analysis)", ""],
        ["DDPM, DDIM", "denoising diffusion probabilistic / implicit model", ""],
        ["FNO", "Fourier neural operator", ""],
        ["MLP, SiLU", "multilayer perceptron; sigmoid-weighted linear unit", ""],
    ], [3.6, 9.0, 2.9], align="lcl")


# ======================================================================================================
def ch_intro(d):
    d.chapter(1, "Introduction")
    d.h2("1.1  Laser powder bed fusion and heat accumulation")
    d.p("Laser powder bed fusion (LPBF) is the most widely used additive manufacturing process for metal parts [@king; @debroy]. A recoater blade spreads a layer of metal powder a few tens of micrometres thick; a laser, steered by scanning mirrors, melts the cross-section of the part in that layer; the build plate is lowered and the cycle repeats until the part is complete. @fig:lpbf shows one layer being processed in a commercial machine. The process can make shapes that are impossible to machine or cast, but it is governed by heat: a small, intense heat source melts the powder, and the deposited energy must flow through the already-built part to the baseplate, which acts as the heat sink.")
    d.fig("lpbf", "fig_lpbf_frame.png", 11.5, "Inside an LPBF machine during processing: the laser melts the current layer of tool-steel powder; the sparks are spatter ejected from the melt pool. Frame from the video by Iakovakis [@video] (CC BY 4.0).")
    d.p("Where the heat cannot escape, the material locally overheats. Loose powder conducts heat about a hundred times worse than the solid metal, so overhangs, thin connections and regions above voids trap heat. Local overheating enlarges the melt pool and causes surface defects such as melt balls and dross, slow cooling changes the microstructure, and the associated residual stresses can distort the part enough to collide with the recoater and stop the build [@ranjan]. @fig:overheat illustrates the mechanism. Identifying such zones before printing allows the designer to change the geometry or the machine operator to adapt the process parameters.")
    d.fig("overheat", "fig_overheat.png", 9.5, "Heat accumulation in LPBF. In the solid column heat flows to the baseplate; in the overhang it is trapped above loose powder and the region overheats.")
    d.h2("1.2  The cost of thermal simulation")
    d.p("Part-scale thermal simulation of LPBF solves the transient heat equation while material is added layer by layer, so the computational domain grows throughout the analysis [@hodge; @ranjan]. Even with the usual simplifications (conduction only through the part, many real layers lumped into one simulated layer) a single part takes hours. Ranjan et al. [@ranjan] quantified a sequence of further physics-based simplifications for detecting overheating. Their reference transient analysis of a test part took 21 h 28 min on 20 cores. The most far-reaching simplification replaces the transient analysis by a steady-state analysis of a local domain near the newly added material and runs in 2 min 9 s, a speed-up of about 600 (Section 1.3).")
    d.p("This creates the question addressed in this project. The steady-state result is cheap but only qualitative; the transient result is accurate but expensive. If a model could learn the relationship between the two, a designer could run the cheap analysis and recover the accurate answer.")
    d.h2("1.3  Simplified thermal models")
    d.p("Ranjan et al. [@ranjan] start from a layer-by-layer reference model and introduce a ladder of simplifications, each faster and less accurate, and measure each against the reference by the error in maximum temperature (*δ*) and by the Jaccard index (*J*) of the hot-spot map. @tab:ranjan reproduces their results. Removing radiation, convection and temperature-dependent properties (S1–S3) changes little in cost. Decoupling the analysis in time (S4, each layer solved separately) and in space (S5, only a domain near the top layer) brings the cost down to minutes. Replacing the transient analysis of that local domain by a steady-state analysis (S6) gives a gain of 599 but with *δ* = 65% and *J* = 75%.")
    d.table("ranjan", "Simplified thermal models of Ranjan et al. [@ranjan] (their Table 2): wall-clock time on 20 cores, maximum-temperature error *δ*, Jaccard index *J* of the hot-spot map, and computational gain *η*.", [
        ["Model", "Description", "Wall-clock time", "δ (%)", "J (%)", "η"],
        ["R", "reference transient model", "21 h 28 min", "0", "100", "1"],
        ["S1", "R without radiation", "20 h 11 min", "22.4", "90.6", "1.06"],
        ["S2", "R without convection", "20 h 39 min", "4.2", "96.4", "1.03"],
        ["S3", "R without radiation, convection, T-dependence", "15 h 3 min", "18.8", "75.4", "1.7"],
        ["S4", "S3, temporally decoupled", "15 min 7 s", "−5.8", "89.9", "85.2"],
        ["S5", "S4, spatially decoupled", "8 min 6 s", "−7.2", "89.9", "144.2"],
        ["S6", "S5 with a steady-state analysis", "2 min 9 s", "65.3", "74.8", "599.3"],
    ], [1.3, 6.2, 2.8, 1.6, 1.6, 1.6], highlight=(7,))
    d.p("Two observations from [@ranjan] are central to this project. First, the steady-state response of a rod heated at one end is *T*_{ss} = *QL*/*k*: it depends on the conductivity of the whole path to the heat sink, including regions far from the heat source. In a real build, however, only regions close to the melt zone influence the temperature there within the short time available, so the steady analysis must be limited to the domain close to the top layer. Second, the characteristic time of the heat equation, *τ* = *L*²/*α*, sets how quickly a domain of size *L* reaches equilibrium. These two observations lead to the local-domain argument of Chapter 6.")
    d.h2("1.4  Objective")
    d.p("The objective is to learn the mapping from the steady-state temperature field of an LPBF simulation (with process parameters where they vary) to the corresponding transient temperature field, using a conditional denoising diffusion model, and to establish under which conditions this mapping is meaningful. Specifically, the work aims to:")
    d.bullets([
        "build a conditional diffusion model that takes a steady-state field and scalar conditions and generates a transient field;",
        "test the model honestly, on simulations it has never seen, against simple baselines;",
        "apply it to three cases of increasing realism: a 1D rod (Case 1), a single deposited bead with 27 process settings (Case 2) and a part with internal voids (Case 3);",
        "explain, using the physics of heat conduction, why the mapping works in some cases and fails in others."])
    d.h2("1.5  Organisation of the report")
    d.p("The report follows the order of the final presentation. Chapter 2 describes the thermal model that generates the data and the datasets used. Chapter 3 presents the generative-AI method: diffusion models, why they were chosen and where they may not be the best choice, and the model and evaluation protocol used throughout. Chapters 4 and 5 present Case 1 (a 1D rod) and Case 2 (a single bead). Chapter 6 develops the physics of the local domain and the characteristic length, which explains the Case 2 result, and Chapter 7 presents Case 3 (a part with voids). Chapter 8 states the key message, discusses the limitations and concludes. Earlier studies that shaped the method are summarised in the Appendix.")


def ch_data(d):
    d.chapter(2, "Thermal Model and Data Generation")
    d.h2("2.1  The thermal model")
    d.p("The temperature field in an LPBF part is governed by transient heat conduction with a moving heat source. Analytical solutions exist for idealised cases, the best known being Rosenthal's solution for a point source moving over a semi-infinite body [@rosenthal], and classical solutions of the heat equation such as those collected by Carslaw and Jaeger [@carslaw] give the temperature in simple geometries. Real parts require numerical solution, usually by the finite element method with element activation (\"birth and death\") to represent the growing part [@hodge; @dhondt]. Resolving the scan path at part scale is prohibitively expensive, so part-scale models apply heat to a whole (lumped) layer at a time; studies cited in [@ranjan] report speed-ups of 4–20 times from such lumping, but simulations still take hours.")
    d.p("All LPBF data in this project come from the thermal model of Ranjan et al. [@ranjan], solved with a multithreaded build of the CalculiX finite element code [@dhondt]. The temperature *T* in the part satisfies the heat equation")
    d.eq("heat", mr("ρ") + msub(mr("c"), mr("p")) + mfrac(mt("∂") + mr("T"), mt("∂") + mr("t")) + mt(" = ∇·") + mparen(mr("k") + mt("∇") + mr("T")) + mt(" + ") + mr("Q") + mt(",  "))
    d.p("with the boundary conditions sketched in @fig:bvp: the lateral faces in contact with powder are insulated, because the effective conductivity of the powder is only about 1% of the bulk value; the baseplate is held at the preheat temperature *T*_{0} and acts as the heat sink; and the top surface loses heat by convection, *q* = *h*_{conv}(*T* − *T*_{a}), and radiation, *q* = *σε*(*T*^{4} − *T*_{a}^{4}). Each new (lumped) layer starts at *T*_{0}, is heated for the time needed to scan it and then cools for the inter-layer time. For a layer of area *A* and thickness *l* the heating time and the volumetric source are")
    d.eq("th", msub(mr("t"), mr("h")) + mt(" = ") + mfrac(mr("A"), mr("h") + mr("v")) + mt(",      ") + mr("Q") + mt(" = ") + mfrac(mr("γ") + mr("P"), mr("A") + mr("l")) + mt(","))
    d.p("where *γ* is the absorptivity. Elements are activated as material is added, so the domain grows during the analysis. The steady-state variant drops the time derivative in @eq:heat and solves for the equilibrium temperature of each added increment; in its localised form (Chapter 6) the sink temperature is applied on the boundary of a local domain around the new material instead of only at the baseplate.")
    d.fig("bvp", "fig_bvp.png", 10.5, "Boundary-value problem of the layer-by-layer thermal model [@ranjan]: heat flux on the newly added layer, insulated sides, convection and radiation on the top surface, and the baseplate held at a fixed temperature.")
    d.p("Each simulation run is configured by plain-text parameter files (power, velocity, hatch, material, ambient temperature, number of layers, part height, inter-layer time and the mesh), and each stored time step is written as a text file holding the temperature of every node. The loaders written for this project read the parameters from these files, never from folder names; this choice found several labelling errors (Section 2.3).")
    d.h2("2.2  Datasets")
    d.p("Five datasets were used during the project. @tab:datasets summarises them; the 1D rod data are generated by a separate, much simpler solver described in Chapter 4.")
    d.table("datasets", "Datasets used in this project.", [
        ["Dataset", "Content", "Size", "Used in"],
        ["Wall", "one build of a thin wall on a plate; transient and steady fields", "480 frame pairs, 3,696 cells (1,094 material)", "Appendix"],
        ["1D rod", "laser moving along a rod; max steady and max transient fields", "1,600 + 300 runs, 101 nodes", "Case 1"],
        ["Single bead, snapshots", "27 settings (3 P × 3 v × 3 h); steady + transient at whole-number times", "26 usable runs, 5,166 nodes (1,094 active)", "Case 2"],
        ["Single bead, SS_new", "localised steady state, one solve per laser position", "27 runs × 320 frames", "Case 2"],
        ["Geometry (part_1)", "20 × 20 × 10 mm block with a hole and a slot, built element by element", "3,844 element steps; 7,406 nodes", "Case 3"],
    ], [3.0, 6.0, 4.4, 1.8], align="llll")
    d.h2("2.3  Data checks")
    d.p("Every dataset was analysed before any model was trained, and several findings changed the experiments:")
    d.bullets([
        "**Air nodes.** In the single-bead mesh 4,072 of the 5,166 nodes stay at exactly 65.0 °C in every frame of every run; they represent gas, not material. Errors are reported on all nodes and on the active nodes separately, because the constant nodes make any model look better than it is.",
        "**Duplicate runs.** Some runs are bit-for-bit copies of others: runs with the same ratio *q* = *P*/*h* and the same speed are physically identical in this model (for example *P* = 60 W, *h* = 0.10 mm and *P* = 90 W, *h* = 0.15 mm), and one run labelled *v* = 50 mm/s is a copy of the *v* = 10 mm/s run. Identical runs are always held out together, so the test never contains a copy of a training run; the mislabelled run is excluded.",
        "**Steady-state input.** The steady-state fields of the localised model (SS_new) are identical across all 27 single-bead settings to within 0.12 °C (Chapter 5).",
        "**Geometry dataset.** The temperature levels, reference temperatures and the placement of the local domain in the geometry dataset have problems that are documented in Chapter 7."])


def ch_method(d):
    d.chapter(3, "Generative-AI Method")
    d.h2("3.1  Machine-learning surrogates for thermal fields")
    d.p("Data-driven surrogates replace an expensive solver by a model trained on its outputs. Recurrent neural networks have been used to predict thermal histories in directed energy deposition [@mozaffar]. Physics-informed neural networks add the residual of the governing equation to the training loss [@pinn]. Neural operators learn a map between function spaces rather than between fixed vectors: DeepONet [@deeponet] and the Fourier neural operator (FNO) [@fno] are trained on many input–output pairs of a PDE and then evaluate new inputs in milliseconds. Neural operators are deterministic, which suits problems in which one input has exactly one correct output, as is the case for a thermal simulation.")
    d.h2("3.2  Denoising diffusion models")
    d.p("Diffusion models are generative models that learn to reverse a gradual noising process [@sohl; @ho]. In the forward process a clean sample is mixed with Gaussian noise over many steps until nothing of it remains; a neural network is trained to undo one step at a time, and new samples are generated by starting from pure noise and applying the network repeatedly. Denoising diffusion implicit models (DDIM) [@ddim] use the same trained network but take larger, deterministic steps, so that a sample needs tens rather than hundreds of network evaluations. Conditioning the network on additional inputs turns the generator into a conditional model: given a condition, it samples from the distribution of outputs consistent with it.")
    d.p("Diffusion models have recently been applied to engineering fields. Shu et al. [@shu] reconstruct high-fidelity flow fields from low-fidelity inputs with a physics-informed diffusion model. In additive manufacturing, diffusion models have been used to upscale a coarse melt-pool simulation to a fine one at a fraction of the cost [@meltpool] and to monitor printed layers by learning what a defect-free layer looks like [@monitor]. Bastek et al. [@pidm] add the residual of the governing equations to the diffusion training loss and report about a hundred-fold reduction in physics violation; they also note that the physics term is only meaningful when the network's estimate of the clean field is reliable, that is, at low noise levels.")
    d.h2("3.3  Diffusion is our choice; it might not be the best")
    d.p("The steady-to-transient problem is a fidelity-enhancement problem of the same kind as [@meltpool; @shu]: a cheap, low-fidelity field is the input and an expensive, high-fidelity field is the output. A diffusion model was chosen because it is a general conditional generator that can take a whole field as its condition. The choice has a known tension with the problem: a diffusion model learns a probability distribution, while a thermal simulation is deterministic. This project therefore uses the model in a mode close to regression (averaging several samples, Chapter 4) and states plainly where a deterministic operator such as the FNO [@fno] may be the better tool (Section 8.2). The contribution of the project lies less in the network than in establishing when the steady field carries the information needed for the mapping.")


    d.h2("3.4  Problem statement")
    d.p("Each training example consists of a condition and a target. The condition is a steady-state temperature field *c* (a vector of nodal temperatures) and, where they vary, scalar process parameters *s*; the target *x*_{0} is the transient field at the same nodes. In Cases 1 and 2 the fields are the **maximum over time** at each node, so that each run gives one pair; this choice, made with the supervisor, targets the quantity that matters for overheating (the hottest temperature each point reaches) and removes the need to align time instants between the steady and transient runs. In Case 3 the target is the peak transient temperature of each element step. The model must learn the conditional distribution *p*(*x*_{0} | *c*, *s*); because the problem is deterministic, this distribution should be narrow, and the prediction is taken as the mean of several samples.")
    d.h2("3.5  Forward and reverse processes")
    d.p("Following [@ho], a noisy version of the target at noise level *τ* ∈ {0, …, 399} is")
    d.eq("fwd", msub(mr("x"), mr("τ")) + mt(" = ") + msqrt(msub(mbar(mr("α")), mr("τ"))) + mt(" ") + msub(mr("x"), mt("0")) + mt(" + ") + msqrt(mt("1 − ") + msub(mbar(mr("α")), mr("τ"))) + mt(" ") + mr("ε") + mt(",    ") + mr("ε") + mt(" ~ 𝒩(0, I),    ") + msub(mbar(mr("α")), mr("τ")) + mt(" = ") + msubsup(mt("∏"), mr("i") + mt("=0"), mr("τ")) + mparen(mt("1 − ") + msub(mr("β"), mr("i"))) + mt(","))
    d.p("with a linear schedule *β*_{i} from 10^{−4} to 0.02 over 400 steps; @fig:forward shows the process on a field from the single-bead data. A network *f*_{θ} receives the noisy target, the noise level and the condition, and is trained to output the clean target directly (\"*x*_{0}-prediction\"):")
    d.eq("loss", mi_loss := (mr("ℒ") + mparen(mr("θ")) + mt(" = 𝔼") + msub(mt(""), mr("x") + mt("₀,") + mr("τ") + mt(",") + mr("ε")) + mparen(mt("‖ ") + msub(mr("f"), mr("θ")) + mparen(msub(mr("x"), mr("τ")) + mt(", ") + mr("τ") + mt(", ") + mr("c") + mt(", ") + mr("s")) + mt(" − ") + msub(mr("x"), mt("0")) + mt(" ‖²"), "[", "]") + mt(".")))
    d.fig("forward", "fig_forward.png", 15.5, "Forward diffusion process applied to a single-bead temperature field: the signal fraction falls from 100% to 13% over 400 noise levels while the noise grows to 99%. The network learns to reverse this process.")
    d.p("Predicting *x*_{0} rather than the added noise was essential in this project: most nodes of an LPBF field sit at a constant temperature, and the standard noise-prediction network did not train on the single-bead data (its loss stayed at 0.96, where 1.0 means no information). Samples are generated with DDIM [@ddim] in 50 deterministic steps. Starting from *x* ~ 𝒩(0, I) at *τ* = 399, each step computes the network's estimate *x̂*_{0}, the implied noise and the input of the next, lower noise level *τ*′:")
    d.eq("ddim", mhat(mr("ε")) + mt(" = ") + mfrac(msub(mr("x"), mr("τ")) + mt(" − ") + msqrt(msub(mbar(mr("α")), mr("τ"))) + msub(mhat(mr("x")), mt("0")), msqrt(mt("1 − ") + msub(mbar(mr("α")), mr("τ")))) + mt(",     ") + msub(mr("x"), mr("τ′")) + mt(" = ") + msqrt(msub(mbar(mr("α")), mr("τ′"))) + msub(mhat(mr("x")), mt("0")) + mt(" + ") + msqrt(mt("1 − ") + msub(mbar(mr("α")), mr("τ′"))) + mt(" ") + mhat(mr("ε")) + mt("."))
    d.p("The estimate *x̂*_{0} is clipped to the range seen in training at each step. Eight samples are drawn from different starting noise and averaged; their spread is recorded as a check that the model is not producing noise (in Case 2 the draws differ by at most 0.12 °C, confirming that the sampler behaves as a regressor).")
    d.h2("3.6  Temperature transform")
    d.p("Temperatures span three to four orders of magnitude between the cold parts of a field and the melt pool. They are mapped to a compact range by")
    d.eq("asinh", mr("z") + mt(" = ") + mfrac(mt("1"), mt("4")) + mt(" asinh") + mparen(mfrac(mr("T") + mt(" − ") + msub(mr("T"), mt("0")), mr("κ"))) + mt(",    ") + mr("κ") + mt(" = 10 °C,"))
    d.p("which is linear for small rises and logarithmic for large ones, and inverted after sampling. Scalars are converted to logarithms (for example ln *P*, ln *v*, ln *h* and ln *P*/*h*), because their effects on temperature are multiplicative, and standardised with statistics of the training set only.")
    d.h2("3.7  Network architecture")
    d.p("The network is a fully connected denoiser (@fig:arch). The condition field and the scalars are compressed by an encoder of two linear layers with a SiLU activation [@silu] to a code of *d*_{c} values. The noise level is encoded by 64 sine and cosine features [@vaswani] followed by a linear layer and SiLU. The noisy target, the condition code, the scalars and the noise code are concatenated and passed through a multilayer perceptron with two hidden layers of width *h* and SiLU activations, whose output is the clean target. The same code is used in every case; only the sizes change (@tab:configs).")
    d.fig("arch", "fig_architecture.png", 15.5, "Architecture of the conditional denoiser. *n*: number of modelled nodes; *m*: number of scalar conditions; *d*_{c}: size of the condition code; *h*: hidden width.")
    d.table("configs", "Model sizes and training settings in the three cases.", [
        ["", "Case 1 (rod)", "Case 2 (bead, max fields)", "Case 3 (geometry)"],
        ["modelled nodes *n*", "99", "1,252", "25 (5 × 5 patch)"],
        ["scalars *m*", "2 (ln *P*, ln *v*)", "4 (ln *P*, ln *v*, ln *h*, ln *q*)", "1 (layer)"],
        ["code *d*_{c} / hidden *h*", "64 / 256", "128 / 512", "64 / 512"],
        ["parameters", "196,835", "2,005,668", "382,361"],
        ["training pairs", "25 – 1,600", "25 per fold (26 runs)", "3,132 elements"],
        ["optimisation", "8,000 updates, batch 32", "1,500 epochs, batch 8", "82 epochs, batch 32"],
        ["training time (CPU)", "≈ 20–45 s", "41 s", "34 s"],
    ], [4.0, 3.4, 4.6, 3.6], align="lccc")
    d.p("Training uses AdamW [@adamw] with learning rate 10^{−3}, weight decay 10^{−4}, a cosine learning-rate decay and gradient clipping at norm 1.0. All training was done on the CPU of a laptop (12th-generation Intel Core i5-12500H); the code uses a GPU automatically when one is present.")
    d.h2("3.8  Evaluation protocol")
    d.p("A score on data that the model has seen measures memorisation, not prediction. Every result in this report is therefore measured on simulations excluded from training: a separate set of 300 runs in Case 1; leave-one-setting-out cross-validation in Case 2, where each of the 24 distinct settings is held out in turn together with any physically identical twin; and held-out layers in Case 3. The earlier studies (Appendix) showed how much this matters: with a random split the same linear model scored an error of 0.97, against 2.16 with whole time blocks held out.")
    d.p("The main metric is the peak-temperature error, the quantity that decides whether a region overheats:")
    d.eq("delta", mr("δ") + mt(" = ") + mfrac(mt("max ") + mhat(mr("T")) + mt(" − max ") + mr("T"), mt("max ") + mr("T") + mt(" − ") + msub(mr("T"), mt("0"))) + mt(" × 100%,"))
    d.p("that is, the error of the predicted peak relative to the true peak rise above the reference temperature. RMSE and MAE over the nodes are also reported, together with the overlap (Jaccard index) of the predicted and true hot-spot regions, defined as the nodes with at least half of the true peak rise. Every learned model is compared with simple baselines that use no deep learning: a constant (the training mean), the nearest training setting, linear regression, and the steady field itself used as the answer. A model is only useful if it beats these baselines.")


def ch_case1(d):
    with_ss = {r["n_train"]: r for r in ROD["runs"] if r["name"] == "with_ss"}
    no_ss = {r["n_train"]: r for r in ROD["runs"] if r["name"] == "no_ss"}
    ap = {(r["name"], r["n_train"]): r for r in json.load(open(ROOT / "rod1d_outputs/rod1d_allparams.json"))}
    g = ROD["steady_as_answer"]
    d.chapter(4, "Case 1: A Laser Moving Along a Rod")
    d.h2("4.1  Model problem")
    d.p("Case 1 reproduces the structure of the LPBF task in one dimension, where thousands of runs can be generated in seconds. A rod of length 10 mm, discretised by 101 nodes with both ends held at *T*_{0} = 65 °C, is heated by a Gaussian source of power *P* and radius *r* moving from *x* = 1 mm to *x* = 9 mm at speed *v* (@fig:rod). The rod also loses heat to its surroundings at a rate *β*:")
    d.eq("rod", mfrac(mt("∂") + mr("T"), mt("∂") + mr("t")) + mt(" = ") + mr("α") + mfrac(msup(mt("∂"), mt("2")) + mr("T"), mt("∂") + msup(mr("x"), mt("2"))) + mt(" − ") + mr("β") + mparen(mr("T") + mt(" − ") + msub(mr("T"), mt("0"))) + mt(" + ") + mr("S") + mparen(mr("x") + mt(" − ") + msub(mr("x"), mr("L")) + mparen(mr("t"))) + mt("."))
    d.fig("rod", "fig_rod.png", 11.0, "Case 1 set-up: a laser of power *P* moves at speed *v* along a 10 mm rod whose ends are held at 65 °C.")
    d.p("The **transient** run is solved in time with implicit Euler steps (the laser advances half a node per step, followed by 1 s of cooling), and the maximum temperature reached at each node is recorded. The **steady** run holds the laser still at each of 81 positions, solves @eq:rod with ∂*T*/∂*t* = 0, and records the maximum over the positions, mirroring the localised steady data of Case 2. Without time, the steady result cannot depend on *v*. @fig:rodmax shows both for one run: the steady maximum is a broad plateau far above the transient maximum, because a laser held still pumps heat into one place until equilibrium.")
    d.fig("rodmax", "fig_rod_max.png", 15.0, "Maximum-temperature fields of one rod run. Left: transient snapshots with the laser at three positions and their maximum over time (red). Right: steady solutions for the laser held at three positions and their maximum over 81 positions (dashed).")
    d.p(f"The parameters are sampled log-uniformly: *P* from 20 to 200 W, *v* from 2 to 50 mm/s, *r* from 0.15 to 0.40 mm, *α* from 3 to 12 mm²/s and *β* from 0.5 to 5 s^{{−1}}. Generating 1,600 training runs and 300 test runs took {ROD['generate_s']:.1f} s. The model receives the steady field (99 interior nodes) and ln *P*, ln *v*; the material properties *α*, *β* and *r* are **not** given as numbers, so they can be seen only through the steady field. This mimics the situation in a real part, where the steady field carries information (such as geometry) that no scalar describes.")
    d.h2("4.2  Results")
    d.p(f"On the 300 unseen runs the model trained on 1,600 runs predicts the peak of the maximum transient field with a mean error of {with_ss[1600]['mean_abs_peak_err_pct']:.1f}% (median {with_ss[1600]['median_abs_peak_err_pct']:.1f}%, 95% of runs within {with_ss[1600]['p95_abs_peak_err_pct']:.1f}%), and the whole field with an RMSE of {with_ss[1600]['rmse_C']:.0f} °C for peaks between about 140 and 5,100 °C. The steady field used directly as the answer overestimates the peak by {g['mean_peak_overestimate_pct']:.0f}% on average (median {g['median_abs_peak_err_pct']:.0f}%, worst {g['max_peak_overestimate_pct']:,.0f}%) with an RMSE of {g['rmse_C']:.0f} °C. @fig:case1 (a) shows a test run of median error: the prediction lies on the true curve.")
    d.fig("case1", "fig_case1_results.png", 15.8, "Case 1 results on 300 unseen runs. (a) A test run of median error: the steady maximum (input), the true transient maximum and the diffusion prediction. (b) Mean peak error with and without the steady field, when the material properties are hidden from the model and when every property is given as a number.")
    rows = [["Training runs", "Mean peak error (%)", "95th percentile (%)", "RMSE (°C)"]]
    for n in (25, 50, 100, 200, 400, 800, 1600):
        r = with_ss[n]
        rows.append([f"{n:,}", f"{r['mean_abs_peak_err_pct']:.2f}", f"{r['p95_abs_peak_err_pct']:.2f}", f"{r['rmse_C']:.1f}"])
    d.p("@tab:datasize shows how the error depends on the amount of training data, with the number of parameter updates held fixed at 8,000. The error falls quickly up to a few hundred runs and then levels off at about 1%.")
    d.table("datasize", "Case 1: error on the 300 unseen runs against the number of training runs.", rows, [3.2, 3.8, 3.8, 3.0], align="cccc")
    d.h2("4.3  When does the steady field help?")
    d.p(f"Two ablations isolate the role of the steady field (@fig:case1 (b)). When the material is hidden, removing the steady field from the input raises the mean peak error from {with_ss[1600]['mean_abs_peak_err_pct']:.1f}% to {no_ss[1600]['mean_abs_peak_err_pct']:.1f}% and the RMSE from {with_ss[1600]['rmse_C']:.0f} °C to {no_ss[1600]['rmse_C']:.0f} °C, and more data does not help ({no_ss[100]['mean_abs_peak_err_pct']:.1f}% with 100 runs, {no_ss[1600]['mean_abs_peak_err_pct']:.1f}% with 1,600): without the steady field the model cannot know the material. When all five physical inputs (*P*, *v*, *α*, *β*, *r*) are given as numbers, the model reaches {ap[('all_params+ss', 1600)]['mean_abs_peak_err_pct']:.2f}% with the steady field and {ap[('all_params_only', 1600)]['mean_abs_peak_err_pct']:.2f}% without it: the steady field then adds nothing.")
    d.p("**The steady field is valuable exactly when it carries information that the scalar inputs do not.** This lesson frames the rest of the report. In Case 2 everything that varies between runs is already in *P*, *v* and *h*; in a real part the information that no scalar captures is the geometry.")


def ch_case2(d):
    d.chapter(5, "Case 2: Single-Bead Deposition")
    d.h2("5.1  Data")
    d.p("Case 2 uses the thermal model of Chapter 2 for a single bead deposited on a plate, for 27 combinations of laser power *P* ∈ {60, 90, 120} W, scan speed *v* ∈ {10, 50, 100} mm/s and hatch spacing *h* ∈ {0.10, 0.15, 0.18} mm. The mesh has 41 × 42 × 3 = 5,166 nodes, of which 1,094 are material. Each transient run stores 326 frames; heating ends at *t* = 2 + 3.2/*v* s and the part is back at 65 °C one second later. Two kinds of steady-state data were provided (@fig:beads): an earlier set, solved once per run, and the localised steady-state data SS_new, which solve the steady problem once for each of 320 laser positions along the bead.")
    d.fig("beads", "fig_beads.png", 14.5, "Case 2: in the transient run the laser moves and the problem is solved in time, so history matters; in the steady run the laser is held at each position and solved without time.")
    d.h2("5.2  Maximum-temperature mapping")
    d.p(f"With the localised steady data SS_new the task is the same as in Case 1 (an earlier formulation that predicted transient snapshots is summarised in Appendix A.3): the per-node maximum of the steady fields over all 320 laser positions, plus ln *P*, ln *v*, ln *h* and ln(*P*/*h*), is mapped to the per-node maximum of the transient run. This gives one pair per run: 26 pairs (one copied run excluded) covering 24 distinct settings, with 1,252 modelled nodes. The model ({MAX['parameters']:,} parameters) is trained for {MAX['epochs']:,} epochs in {MAX['final_train_s']:.0f} s and tested by leave-one-setting-out cross-validation ({MAX['folds']} folds, {MAX['cv_wall_s'] / 60:.0f} min in total).")
    d.table("max", "Case 2, maximum-temperature mapping on unseen settings (24-fold leave-one-setting-out).", [
        ["Quantity", "Value"],
        ["mean / median / worst peak error", f"{MAX['mean_abs_peak_err_pct']:.1f}% / {MAX['median_abs_peak_err_pct']:.1f}% / {MAX['max_abs_peak_err_pct']:.1f}%"],
        ["RMSE / MAE over all 5,166 nodes", f"{MAX['rmse_all_C']:.1f} °C / {MAX['mae_all_C']:.1f} °C"],
        ["hot-spot overlap (Jaccard index)", f"{MAX['mean_hotspot_jaccard']:.2f}"],
        ["relative L2 error", f"{MAX['mean_rel_l2']:.2f}"],
        ["error on training data (not a test)", f"{MAX['in_sample_rmse_all_C']:.1f} °C RMSE"],
    ], [8.0, 6.0])
    d.p("The results (@tab:max, @fig:case2 (a)) look excellent: a mean peak error of 1.3% for peaks between 2,650 and 9,310 °C, and a perfect overlap of the hot-spot regions.")
    d.h2("5.3  The failure: the steady field never changes")
    d.p("The question is where the accuracy comes from. @fig:case2 (b) plots the peak of the steady input against the peak of the transient target for every run: the transient peak varies from 2,652 to 9,311 °C, but the steady peak is 6,666.6 °C in every run. Over every node and every one of the 320 frames, the 27 SS_new runs differ by at most 0.12 °C, even between *P* = 60 W, *h* = 0.18 mm and *P* = 120 W, *h* = 0.10 mm, which differ in heat input by a factor of 3.6 (@fig:sssame). The steady field therefore carries **no information about the setting**; the model's accuracy comes entirely from the three numbers *P*, *v* and *h*, and the steady input could be replaced by a constant.")
    d.fig("case2", "fig_case2_results.png", 15.8, "Case 2, maximum-temperature mapping. (a) Predicted against true peak for the 26 held-out runs. (b) Peak of the steady input and of the transient target in each run, sorted by the transient peak: the steady peak does not change.")
    d.fig("sssame", "fig_ss_same.png", 12.5, "Two very different settings (left: *P* = 60 W, *v* = 100 mm/s, *h* = 0.18 mm; right: *P* = 120 W, *v* = 10 mm/s, *h* = 0.10 mm). The steady fields (top) are identical; the transient fields (bottom) differ by a factor of 3.5 in peak.")
    d.p("Other properties of SS_new confirm that it is a different kind of field: the steady hot zone is about nine times wider than the transient one (280 against 32 nodes with a rise above 100 °C at the same frame), and the steady peak occurs at the start of the bead (6,666.6 °C) and its end (5,892 °C), with about 3,206 °C along the middle, edge effects of the bead ends. By the lesson of Case 1, a steady field that is the same for every run cannot help a model, however accurate the model appears.")
    d.p("The explanation is the geometry. **All 27 runs use one standard geometry: the same bead on the same plate.** The localised steady model solves for the equilibrium temperature in a local domain around the laser, and in this geometry that local domain is the same for every run and every position along the bead; the steady result does not respond to *P*, *v* or *h* either, so the only thing that could change it is the shape around the bead, and that never changes. Chapter 6 develops this argument from the physics of heat conduction, and Chapter 7 tests its converse.")


def ch_physics(d):
    d.chapter(6, "Physics of the Local Domain")
    d.h2("6.1  Heat around a new bead")
    d.p("@fig:weld shows the familiar picture from welding and LPBF. Under the heat source lies the melt pool; around it the temperature falls along isothermal lines; and the heat of the new bead flows into the older beads and the substrate. Over the short time between two beads, the temperature rise is concentrated near the source.")
    d.fig("weld", "fig_weld.png", 12.0, "Heat around a newly deposited bead: melt pool under the heat source, isothermal lines, and heat flowing from the new bead into older beads.")
    d.h2("6.2  Steady state means infinite time")
    d.p("A steady-state temperature is the temperature reached when the heat input continues and one waits for infinite time. Heat then reaches the whole part, including corners far from the bead and the baseplate (@fig:sslocal, left). This is not what happens in a build: the next bead arrives long before the heat fills the part. Using the steady response of the whole part therefore gives temperatures that are too high and that depend on features far from the heat source, which is exactly the limitation identified in [@ranjan]. The remedy, closer to reality, is to localise the analysis: solve the steady problem only on the domain that heat can reach in the time available (@fig:sslocal, right), and hold the boundary of that domain at the reference temperature.")
    d.fig("sslocal", "fig_ss_local.png", 15.0, "Left: in a steady state reached after infinite time, heat reaches the whole part. Right: in reality heat reaches only a local domain around the new bead before the next bead arrives.")
    d.h2("6.3  Characteristic length")
    d.p("How large is the local domain? Consider a semi-infinite bar initially at *T*_{i} whose end is raised to *T*_{s} at *t* = 0. The solution of the heat equation is [@carslaw]")
    d.eq("erfc", mr("T") + mparen(mr("x") + mt(", ") + mr("t")) + mt(" = ") + msub(mr("T"), mr("i")) + mt(" + ") + mparen(msub(mr("T"), mr("s")) + mt(" − ") + msub(mr("T"), mr("i"))) + mt(" erfc") + mparen(mfrac(mr("x"), mt("2") + msqrt(mr("α") + mr("t")))) + mt("."))
    d.p("The temperature profile depends on position only through *x*/√(*αt*), so the distance over which heat has penetrated after time *t* is set by the characteristic length")
    d.eq("lc", msub(mr("L"), mr("c")) + mt(" = ") + msqrt(mr("α") + mr("t")) + mt(",     ") + mr("α") + mt(" = ") + mfrac(mr("k"), mr("ρ") + msub(mr("c"), mr("p"))) + mt("."))
    d.p("At *x* = *L*_{c} the temperature rise is still erfc(0.5) ≈ 48% of the imposed rise. Heat travels like a wave, but one that slows down: four times the time gives only twice the distance. @fig:lc shows the profile in a bar with one end at 100 °C and the rest at 20 °C, for *α* = 5.18 mm²/s: after 0.1 s, 1 s and 4 s the characteristic length is 0.7 mm, 2.3 mm and 4.6 mm. This is the value of *α* used by the steady-state script of the geometry dataset, in which each element is heated for *t* = 1 s, so *L*_{c} = √(5.18 × 1) = 2.28 mm; the script takes the diameter of the local hemisphere as 3*L*_{c} = 6.83 mm, a radius of 3.41 mm, and holds all active nodes outside it at 65 °C. This is the same idea as the characteristic time *τ* = *L*²/*α* of [@ranjan], read the other way round.")
    d.fig("lc", "fig_lc.png", 10.0, "Temperature in a bar whose end is held at 100 °C, after 0.1 s, 1 s and 4 s (*α* = 5.18 mm²/s, @eq:erfc). The dashed line marks the characteristic length *L*_{c} = √(*αt*).")
    d.h2("6.4  Why geometry is what the steady field can carry")
    d.p("The local domain explains Case 2 and predicts Case 3 (@fig:geom).")
    d.bullets([
        "**Same geometry.** On a plain plate every bead sees the same hemisphere of material beneath it. Even a bead near the end wall does not feel the wall, because the wall lies outside the local domain. The same domain gives the same steady result for every bead and every run: the Case 2 failure.",
        "**Different geometry.** In a part with holes and walls, a bead above a hole or next to a wall has part of its local domain missing; a bead in the bulk has the full domain. Less material around the bead means less room for heat to spread and a hotter result. The local domains differ from bead to bead, and the steady field encodes the geometry around each bead."])
    d.fig("geom", "fig_same_geom.png", 13.0, "Same geometry: every bead on a plain block sees the same hemispherical local domain, and the end wall lies outside it.")
    d.fig("geom2", "fig_diff_geom.png", 9.5, "Different geometry: in a block with a hole the local domain is cut above the hole and at the wall, and complete in the bulk.")
    d.p("Hence the central argument of this report: **the steady-to-transient mapping makes sense as a geometry mapping, not as a mapping of P, v and h.** Process parameters are scalars that can be given to any model directly; geometry is what a scalar cannot describe and what the localised steady field can.")


def ch_case3(d):
    d.chapter(7, "Case 3: Geometry")
    d.h2("7.1  Data")
    d.p("The geometry dataset (part_1) is a block of 20 × 20 elements in plan and 10 layers of 1 mm on a three-layer substrate, made of Inconel 718 and processed with *P* = 260, *v* = 1 and *h* = 0.11 in the units of the parameter files. The block contains a 2 × 2 hole through layers 1–8 that widens to a round hole in layers 9–10, and a 6 × 2 slot from layer 6 upwards (@fig:part). The part is built element by element: 3,844 elements, each heated for 1 s, with 5 s of cooling after each layer. The steady-state run performs one steady solve per element with the 65 °C sink outside a 3.41 mm hemisphere around the new element (Section 6.3); the transient run solves six time sub-steps per element with the base held at 180 °C. Pairing the runs by element number gives 3,844 complete pairs; for each element the peak steady and peak transient temperatures are extracted.")
    d.fig("part", "fig_part3d.png", 15.0, "Geometry dataset: the part on its substrate (left) and a top view (right). The hole widens near the top and a slot opens from layer 6.")
    d.h2("7.2  Geometry appears in both fields")
    d.p("Unlike SS_new, the steady peak now changes from element to element: 3,033 distinct values over 3,844 elements, a 16% coefficient of variation. Both steady and transient peaks are higher next to walls and voids (@fig:geomaps, @fig:neigh): the median steady peak rises from 141,853 to 165,602 and 202,351 °C for elements with 0, 1 and 2 empty in-plane neighbours, and the median transient peak from 117,892 to 120,189 and 129,543 °C. The correlation between steady and transient peaks is 0.65 over all elements and 0.88 from element 1,365 onwards.")
    d.fig("geomaps", "fig_geo_maps.png", 10.5, "Peak temperature of each element on layers 6 and 10, steady (top) and transient (bottom); white cells are voids. Both are hotter next to the outer walls, the hole and the slot.")
    d.fig("neigh", "fig_case3_neighbours.png", 10.0, "Median steady and transient peak against the number of empty in-plane neighbours (void or outside the part).")
    d.h2("7.3  Problems in the dataset")
    d.p("The analysis also found problems that limit what can be claimed (@tab:issues). The most important is in the steady-state script: when the sink sets are built, the centre of the hemisphere is computed without the running element count, so after the first group of 484 elements the hemisphere is centred 9 to 25 mm away from the heated element. The local domain is in the correct place for only 484 of the 3,844 elements. Because the same wrong indices mark the nodes allowed to be held at 65 °C, only the substrate, layer 1 and part of layer 2 can act as a sink, which creates a cold floor under layer 2 and makes the steady peak rise with height.")
    d.table("issues", "Case 3: problems found in the geometry dataset.", [
        ["Problem", "Finding"],
        ["Temperature level", "steady peaks of 95,000–318,000 °C, transient peaks of about 118,000 °C: not physical, probably from the heat input of a 1 mm element; only relative patterns can be read"],
        ["Reference temperatures", "transient base held at 180 °C, steady sink at 65 °C; film and radiation in the transient run only"],
        ["Local domain misplaced", "hemisphere centre misses the running element count: correct for 484 of 3,844 elements; causes the dark band in layer 2 and the rise with height"],
        ["Negative temperatures", "333 transient element steps go below the 180 °C base, 151 below 0 °C (minimum −1,145 °C), mostly in layer 1"],
        ["Diffusivity", "the script uses 5.18 mm²/s; the material files give 39.3 mm²/s"],
        ["One part", "3,844 pairs from one geometry; no second part to test on"],
    ], [4.0, 11.6], align="ll")
    d.h2("7.4  Learning the transient peak from the steady field")
    d.p(f"To obtain a quantitative result despite these problems, the diffusion model of Chapter 3 was trained to map local patches. For each element the input is the 5 × 5 patch of steady peaks centred on it in its own layer (voids and the outside of the part marked as empty) and the layer number; the target is the 5 × 5 patch of transient peaks, and the score uses the centre value. Temperatures are shifted and scaled before the transform of @eq:asinh because of their 10^{{5}} °C level. The model ({C3['parameters']:,} parameters) is trained on layers 1–8 ({C3['n_train']:,} elements) and tested on layers 9–10 ({C3['n_test']} elements), where the hole widens into a round hole that does not occur in training. Two baselines use the same split: the mean transient peak of the training elements, and a straight line fitted from the centre steady peak and the layer. To check what the steady values contribute, the model was also trained on the **shape only**: the same patches with every solid cell set to one constant, so that only the position of walls and voids is visible. Each diffusion variant was trained with three random seeds.")
    d.table("c3", "Case 3: mean absolute error of the transient peak on the unseen layers 9–10 (°C, dataset units; diffusion: mean of three seeds, range in brackets).", [
        ["Method", "All elements", "Next to a void or wall", "In the bulk"],
        ["Guess the training mean", f"{C3['mean_of_training']['mae_C']:,.0f}", f"{C3['mean_of_training']['mae_next_to_void_or_wall_C']:,.0f}", f"{C3['mean_of_training']['mae_in_bulk_C']:,.0f}"],
        ["Straight line from the steady peak", f"{C3['linear_centre_steady']['mae_C']:,.0f}", f"{C3['linear_centre_steady']['mae_next_to_void_or_wall_C']:,.0f}", f"{C3['linear_centre_steady']['mae_in_bulk_C']:,.0f}"],
        ["Diffusion, steady patch", f"{mean(C3_ST):,.0f} ({min(C3_ST):,.0f}–{max(C3_ST):,.0f})", f"{mean(C3_ST_NV):,.0f}", f"{mean(C3_ST_B):,.0f}"],
        ["Diffusion, shape only", f"{mean(C3_SH):,.0f} ({min(C3_SH):,.0f}–{max(C3_SH):,.0f})", f"{mean(C3_SH_NV):,.0f}", f"{mean(C3_SH_B):,.0f}"],
    ], [5.4, 3.6, 3.6, 3.0], align="lccc")
    d.p(f"@tab:c3 and @fig:case3 give the results. The diffusion model with the steady patch reaches a mean error of {mean(C3_ST):,.0f} °C, about {C3['mean_of_training']['mae_C'] / mean(C3_ST):.1f} times less than the constant guess ({C3['mean_of_training']['mae_C']:,.0f} °C) and below the straight line ({C3['linear_centre_steady']['mae_C']:,.0f} °C); the gain is largest next to walls and voids, where the transient peak varies most ({mean(C3_ST_NV):,.0f} against {C3['linear_centre_steady']['mae_next_to_void_or_wall_C']:,.0f} °C for the straight line). The predicted map of layer 10 reproduces the hot rims around the round hole and the slot, which the model never saw in this form. The spread over seeds is large ({min(C3_ST):,.0f}–{max(C3_ST):,.0f} °C), so the ranking against the straight line is clear but the exact value is not.")
    d.fig("case3", "fig_case3_results.png", 15.8, "Case 3 on the unseen layers 9–10. (a) Mean absolute error of the transient peak (diffusion: mean of three seeds, bars show the range). (b–d) Layer 10: true transient peak, diffusion prediction from the steady patch, and the straight-line prediction from the steady peak.")
    d.p(f"The shape-only model is at least as accurate ({mean(C3_SH):,.0f} °C on average). In this dataset, therefore, **the information that predicts the transient peak is the geometry itself**, the position of walls and voids around each element; the steady values, computed with the misplaced local domain, add nothing beyond it. This result supports the central argument from a different direction: what makes the mapping work is the geometry, and the purpose of the localised steady model is to deliver that geometry information in a physically meaningful form. A correctly placed local domain would be needed to show whether the steady values then add information beyond the shape.")


def ch_discussion(d):
    d.chapter(8, "Key Message and Conclusions")
    d.p("**Key message: the mapping from steady-state to transient temperature makes sense not as a mapping of the process parameters P, v and h, but as a geometry mapping.** This chapter explains how the three cases support this message, reflects on the choice of method and its limitations, and concludes.")
    d.h2("8.1  What the three cases show together")
    d.p("The three cases form one argument. Case 1 shows that the method can learn the steady-to-transient mapping to about 1% in peak temperature when data are plentiful, and that the steady field is useful only when it carries information the scalar inputs do not. Case 2 shows a model that appears equally accurate but whose steady input is a constant, because a single standard geometry gives a single local domain. The physics of Chapter 6 explains why: a localised steady field responds to the geometry around the heat source, and in Case 2 that geometry never changes. Case 3 shows the converse: once the geometry varies, both fields respond to walls and voids, and a model trained on part of the build predicts the transient peaks of unseen layers. The conclusion that the mapping is a geometry mapping is thus supported by an experiment in which it should fail (Case 2) and one in which it should work (Case 3).")
    d.h2("8.2  Diffusion for a deterministic problem")
    d.p("Diffusion was the method chosen for this project, and it has been used honestly here: it learns a probability distribution, whereas a thermal simulation gives exactly one answer for one input. In every case with enough data the draws of the model collapsed to nearly the same field (spread below 0.12 °C in Case 2), so the model acted as an elaborate regressor, and 400 noise levels with 50 sampling steps cost more than a single evaluation of a deterministic network. Where data were scarce (the wall and the snapshot task) it lost to simple baselines. Neural operators such as the FNO [@fno] or DeepONet [@deeponet] are deterministic, are designed to map one field to another, and are being used by others for physics problems of this kind; they may well be the better tool for the steady-to-transient mapping. A conditional generative model would earn its cost if the mapping were genuinely uncertain, for example if the steady field were known only on part of the domain.")
    d.h2("8.3  Limitations")
    d.bullets([
        "**Data volume.** Case 2 has 26 runs and 24 distinct settings, Case 3 one part. Case 1 shows that errors fall substantially up to a few hundred runs.",
        "**Case 3 data quality.** The temperature levels are not physical, the two runs use different reference temperatures, and the local domain is misplaced for most elements (@tab:issues). The Case 3 results are therefore in the dataset's own units and show patterns, not physical temperatures.",
        "**Generalisation.** The Case 3 test uses new layers of the same part, not a new part, and neighbouring elements are correlated. A test on an unseen geometry is required to show that the mapping transfers between parts.",
        "**Steady response to process parameters.** The SS_new runs do not respond to *P*, *v* or *h*, which means that the heat input of the localised steady model was effectively the same in every run; the steady field cannot then carry the effect of the process parameters, which must enter the model as scalars.",
        "**Seeds and uncertainty.** Most results come from one training seed; in Case 3 the spread over three seeds is large compared with the differences between the diffusion variants."])


    d.h2("8.4  Conclusions")
    d.p("This project studied whether a machine-learning model can turn the fast steady-state temperature field of a laser powder bed fusion simulation into the slow transient field, using a conditional denoising diffusion model, on three cases of increasing realism. The main conclusions are:")
    d.bullets([
        "A conditional diffusion model can learn the mapping from the maximum steady field to the maximum transient field. On a 1D model problem with 1,600 training runs it predicts the peak temperature of unseen runs within 1.0% on average, where the steady field taken directly is off by 174%.",
        "The steady field is useful only when it carries information that the scalar inputs do not. Removing it raised the error from 1.0% to 18% when the material was hidden, but changed nothing when all properties were given as numbers.",
        "For a single bead with 27 process settings the model reached 1.3% mean peak error on unseen settings, but the localised steady fields of all runs were identical to within 0.12 °C. With one standard geometry the local domain is the same for every run, and all of the accuracy comes from *P*, *v* and *h*.",
        "A steady state corresponds to infinite time, while heat penetrates only a characteristic length *L*_{c} = √(*αt*) before the next bead arrives. The steady analysis must therefore be localised, and a localised steady field can vary only with the geometry around the heat source.",
        "In a block with a hole and a slot both steady and transient peaks rise next to walls and voids, and a model trained on layers 1–8 predicts the transient peaks of the unseen layers 9–10 with about 2.5 times less error than a constant guess; the shape of the part alone is as informative as the steady values in the present data.",
        "**The steady-to-transient mapping makes sense not as a mapping of P, v and h, but as a geometry mapping.**"])
    d.p("The datasets used have limitations, documented in Chapters 2 and 7, that bound these conclusions: the Case 3 temperature levels are not physical, its local domain is misplaced for most elements, and only one part was available. Diffusion was a workable but not a natural choice for a deterministic problem, and a deterministic neural operator may serve the same purpose at lower cost.")


def appendix_earlier(d):
    d.chapter("", "Appendix: Earlier Studies")
    d.p("Three studies before the three cases set up the method. They are not part of the presentation storyline, but their results determined the design described in Chapter 3 and are reported here for completeness.")
    d.h2("A.1  First 1D prototype")
    d.p("A first prototype was built on 1,000 synthetic 1D rods of 11 nodes with random piecewise conductivity, giving 8,000 pairs of transient snapshots and steady-state hints. A residual MLP of three layers of width 256 predicted the added noise, and a physics-informed variant added the residual (∂*T*/∂*t* − *α* ∂²*T*/∂*x*²)², following [@pidm]. The results were poor. The physics loss reached values in the millions at high noise levels, because reconstructing the clean field divides by √*ᾱ*_{τ}, which is about 0.003 at the noisiest step; restricting the penalty to noise levels *τ* < 200 with a weight of 0.01 reduced the error a hundred-fold, but predictions still oscillated between −400 and +300 °C for a rod near 180 °C. Small errors in the first of 1,000 reverse steps compounded through the chain. Finally, the result was plotted for a training sample, so there was no fair test.")
    d.p("Four changes followed: predict the clean field and sample with 50 DDIM steps instead of 1,000 noise-prediction steps; apply any physics term only at low noise; always test on held-out simulations; and move from synthetic toy data to the LPBF thermal model.")
    d.h2("A.2  Wall dataset")
    d.p("The first LPBF data were one simulated build of a thin wall: a grid of 8 × 22 × 21 = 3,696 cells, of which 2,602 are gas that never leaves 65.0 °C and 1,094 are material (a plate of 924 cells and a wall of 170 cells, 2 cells thick and 17 tall). The 480 stored frames form 80 blocks of six nearly identical sub-steps, so the effective number of independent samples is closer to 80 than to 480. @tab:wall compares methods on the last 20 blocks, with the transient field as input and the steady field as output (the direction first requested).")
    d.table("wall", "Wall dataset, transient → steady mapping: mean absolute error on the 1,094 material cells of the last 20 time blocks (temperature units of the dataset).", [
        ["Method", "MAE"],
        ["Ridge regression (linear)", "2.16"],
        ["Neural network + physics term", "2.36"],
        ["Neural network", "2.48"],
        ["Per-cell mean (ignores the input)", "2.63"],
        ["Copy the transient field", "4.38"],
        ["Diffusion + physics term", "5.31"],
        ["Diffusion", "5.42"],
    ], [9.0, 3.0], highlight=(6, 7))
    d.p("Diffusion was the worst method, worse even than ignoring the input; its repeated draws spread by about 12 units, nearly twice the spread of the field itself, so it generated noise rather than structure. With about 80 effective samples for a 1,094-value output, a generative model cannot learn the distribution; a linear model is the appropriate tool. Repeating the experiment in the useful direction, steady field in and transient field out, with the architecture of Chapter 3 (1.80 million parameters, 400 epochs, 71 s), gave an MAE of 1.48, level with ridge regression (1.51) and 9% below the no-skill per-cell mean (1.63). The model located the laser hot spot but one cell short of its true position, because the wall front at that position occurs only in the held-out part of the build. One build cannot show more than this.")
    d.p("Three lessons carried forward: honest splits are mandatory (a random split scored 0.97 where held-out blocks gave 2.16); a learned model must be compared with simple baselines; and the amount of independent data, not the network, limits what a diffusion model can learn. These lessons motivated Case 1, in which data are cheap enough to vary the amount.")
    d.h2("A.3  Single bead: transient snapshots")
    d.p("The first formulation, before SS_new existed, predicted the transient field at whole-number times *t* = 2, …, 7 s from the earlier steady field, *P*, *v*, *h* and *t*, giving 156 training rows from 26 usable runs. @tab:snap lists the cross-validated errors (24 folds, leave one setting out). Plain diffusion (6.85 °C RMSE over all nodes and times) was worse than copying the nearest training setting (4.46 °C). Analysis showed why: only *t* = 2 s carries a signal for every run (by *t* = 4 s every field is back at 65 °C), and the *t* = 2 s fields are one fixed spatial shape multiplied by an amplitude (the first singular component holds 99.7% of the variation). A model with an amplitude fitted in ln(*P*/*h*) and ln *v* times a regressed shape reached 0.22 °C, and a hybrid in which the diffusion model learns only the residual of this regression matched it. The lesson is the same as in the preliminary studies: a model must beat simple, physically structured baselines, and with 26 runs the structure has to be supplied rather than learned.")
    d.table("snap", "Case 2, transient snapshots: cross-validated RMSE (°C) over all 5,166 nodes, at *t* = 2 s and over all six times (24-fold leave-one-setting-out).", [
        ["Method", "t = 2 s", "All times"],
        ["Ambient 65 °C everywhere", "69.21", "28.26"],
        ["Mean of training runs", "43.23", "17.65"],
        ["Diffusion (plain)", "16.77", "6.85"],
        ["Ridge regression in ln q, ln v", "16.04", "6.55"],
        ["Nearest training setting", "10.93", "4.46"],
        ["Amplitude × shape (same v)", "1.73", "0.71"],
        ["Amplitude × shape (linear in ln q, same v)", "0.55", "0.22"],
        ["Hybrid: regression + diffusion residual", "0.55", "0.22"],
    ], [8.6, 2.6, 2.6], highlight=(3,))


def references(d):
    d.chapter("", "References")
    for i, k in enumerate(d.refs, 1):
        if d.dry:
            continue
        para = d.d.add_paragraph()
        para.paragraph_format.left_indent = Cm(1.0)
        para.paragraph_format.first_line_indent = Cm(-1.0)
        para.paragraph_format.line_spacing = 1.15
        para.paragraph_format.space_after = Pt(6)
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        para.add_run(f"[{i}]\t")
        d._runs(para, REFS[k])


def content(d):
    front(d)
    d.section("lowerRoman", None)
    abstract(d)
    if not d.dry:
        d.chapter("", "Contents")
        d.toc('TOC \\o "1-2" \\h \\z \\u', "Contents (updated when the document is finalised)")
    d.section("decimal", 1)
    for ch in (ch_intro, ch_data, ch_method, ch_case1, ch_case2, ch_physics, ch_case3, ch_discussion):
        ch(d)
    references(d)
    appendix_earlier(d)


if __name__ == "__main__":
    dry = Doc(dry=True)
    content(dry)
    real = Doc(dry=False)
    real.num, real.refs = dry.num, dry.refs
    real.counters = {"fig": 0, "tab": 0, "eq": 0}
    content(real)
    out = HERE / ".build" / "BTP_report_draft.docx"  # intermediate; finalize.py writes the real BTP_report.docx
    out.parent.mkdir(exist_ok=True)
    real.d.save(out)
    print("saved", out, "| figures", dry.counters["fig"], "tables", dry.counters["tab"], "equations", dry.counters["eq"], "references", len(dry.refs))
