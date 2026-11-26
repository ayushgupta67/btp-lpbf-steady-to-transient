"""Open a .docx in LibreOffice (UNO), update every index and field (contents, figures, tables,
page numbers), then save the updated .docx and a PDF next to it.

    /usr/bin/python3 finalize.py BTP_report_draft.docx BTP_report.docx BTP_report.pdf
(needs the system Python, which has LibreOffice's uno module)."""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue


def prop(n, v):
    p = PropertyValue(); p.Name = n; p.Value = v
    return p


def main(src, out_docx, out_pdf):
    prof = tempfile.mkdtemp(prefix="lo_profile_")
    port = 2311
    proc = subprocess.Popen(["soffice", "--headless", "--invisible", "--norestore", "--nologo",
                             f"-env:UserInstallation=file://{prof}",
                             f"--accept=socket,host=127.0.0.1,port={port};urp;"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        local = uno.getComponentContext()
        resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
        for _ in range(60):
            try:
                ctx = resolver.resolve(f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext")
                break
            except Exception:
                time.sleep(0.5)
        desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(str(Path(src).resolve())), "_blank", 0, (prop("Hidden", True),))
        paras = doc.getStyleFamilies().getByName("ParagraphStyles")
        for name in ("Contents 1", "Contents 2"):  # compact table of contents (LibreOffice's own entry styles)
            if paras.hasByName(name):
                st = paras.getByName(name)
                ls = st.ParaLineSpacing; ls.Mode = 0; ls.Height = 100; st.ParaLineSpacing = ls
                st.ParaTopMargin, st.ParaBottomMargin, st.CharHeight = 0, 0, 11
        for _ in range(2):  # twice: page numbers settle after the indexes have their final length
            doc.getTextFields().refresh()
            idx = doc.getDocumentIndexes()
            for i in range(idx.getCount()):
                idx.getByIndex(i).update()
        doc.storeToURL(uno.systemPathToFileUrl(str(Path(out_docx).resolve())), (prop("FilterName", "MS Word 2007 XML"),))
        doc.storeToURL(uno.systemPathToFileUrl(str(Path(out_pdf).resolve())), (prop("FilterName", "writer_pdf_Export"),))
        n = idx.getCount()
        doc.close(True)
        print("indexes updated:", n)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main(*sys.argv[1:4])
