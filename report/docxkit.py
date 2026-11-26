"""Small helpers on top of python-docx for the BTP report: fields (TOC, SEQ, PAGE), OMML equations,
page-number sections, captions. Kept separate so build_report.py reads as content."""
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor

M = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---- fields -------------------------------------------------------------------------------------
def field(paragraph, instr, result="", run_props=None):
    """Complex field: begin / instrText / separate / result / end."""
    rpr = run_props or ""
    xml = (f'<w:r {nsdecls("w")}>{rpr}<w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
           f'<w:r {nsdecls("w")}>{rpr}<w:instrText xml:space="preserve"> {_esc(instr)} </w:instrText></w:r>'
           f'<w:r {nsdecls("w")}>{rpr}<w:fldChar w:fldCharType="separate"/></w:r>'
           f'<w:r {nsdecls("w")}>{rpr}<w:t xml:space="preserve">{_esc(result)}</w:t></w:r>'
           f'<w:r {nsdecls("w")}>{rpr}<w:fldChar w:fldCharType="end"/></w:r>')
    for el in parse_xml(f'<w:p {nsdecls("w")}>{xml}</w:p>'):
        paragraph._p.append(el)


# ---- OMML ---------------------------------------------------------------------------------------
def mr(text, plain=False):
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ""
    return f"<m:r>{sty}<m:t xml:space=\"preserve\">{_esc(text)}</m:t></m:r>"


def mt(text):
    """Upright text (units, words, operators)."""
    return mr(text, plain=True)


def mfrac(num, den):
    return f"<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"


def msqrt(x):
    return f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{x}</m:e></m:rad>'


def msub(base, s):
    return f"<m:sSub><m:e>{base}</m:e><m:sub>{s}</m:sub></m:sSub>"


def msup(base, s):
    return f"<m:sSup><m:e>{base}</m:e><m:sup>{s}</m:sup></m:sSup>"


def msubsup(base, sb, sp):
    return f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sb}</m:sub><m:sup>{sp}</m:sup></m:sSubSup>"


def mparen(x, l="(", r=")"):
    return f'<m:d><m:dPr><m:begChr m:val="{_esc(l)}"/><m:endChr m:val="{_esc(r)}"/></m:dPr><m:e>{x}</m:e></m:d>'


def mbar(x):
    return f'<m:acc><m:accPr><m:chr m:val="̄"/></m:accPr><m:e>{x}</m:e></m:acc>'


def mhat(x):
    return f'<m:acc><m:accPr><m:chr m:val="̂"/></m:accPr><m:e>{x}</m:e></m:acc>'


def msum(sub, sup, body):
    return (f'<m:nary><m:naryPr><m:chr m:val="∑"/></m:naryPr><m:sub>{sub}</m:sub>'
            f'<m:sup>{sup}</m:sup><m:e>{body}</m:e></m:nary>')


def equation(doc, omml, number=None, style="Body Text"):
    """Centred equation with an optional right-aligned number, using tab stops."""
    p = doc.add_paragraph(style=style)
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    sec = doc.sections[-1]
    width = sec.page_width - sec.left_margin - sec.right_margin
    ts = p.paragraph_format.tab_stops
    ts.add_tab_stop(int(width / 2), WD_TAB_ALIGNMENT.CENTER)
    ts.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT)
    p.add_run("\t")
    p._p.append(parse_xml(f"<m:oMath {M}>{omml}</m:oMath>"))
    if number:
        p.add_run(f"\t({number})")
    return p


def inline_math(paragraph, omml):
    paragraph._p.append(parse_xml(f"<m:oMath {M}>{omml}</m:oMath>"))


# ---- sections -----------------------------------------------------------------------------------
def page_numbers(section, fmt="decimal", start=None, show=True, font="Times New Roman", size=11):
    sectPr = section._sectPr
    for old in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(old)
    pg = parse_xml(f'<w:pgNumType {nsdecls("w")} w:fmt="{fmt}"' + (f' w:start="{start}"' if start else "") + "/>")
    sectPr.append(pg)
    section.footer.is_linked_to_previous = False
    fp = section.footer.paragraphs[0]
    for r in list(fp.runs):
        r._r.getparent().remove(r._r)
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if show:
        field(fp, "PAGE", "1", run_props=f'<w:rPr><w:rFonts w:ascii="{font}" w:hAnsi="{font}"/><w:sz w:val="{size * 2}"/></w:rPr>')
