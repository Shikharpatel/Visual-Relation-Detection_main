"""
patch_research_paper_v2.py
==========================
Patches Research_Paper.docx:
  1. Fig. 1  — system architecture  (full PDF image, portrait)
  2. Fig. 3  — FusionModule         (Stage 4 crop from PDF)
  3. Results comparison — Exp A vs B side-by-side table (2.1" each, smaller)

Run:
    python scripts/patch_research_paper_v2.py
"""

import sys, io, copy, shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from docx import Document
from docx.shared import Inches, Pt, Emu, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from lxml import etree
import fitz
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent.parent
OUTPUTS  = BASE / "outputs"
DIAGRAMS = BASE / "outputs" / "diagrams"
PLOTS    = BASE / "outputs" / "plots"
DESKTOP  = Path("C:/Users/harsh/OneDrive/Desktop")
PDF_SRC   = Path("C:/Users/harsh/Downloads/fusion_module.pdf")
PNG_ARCH  = Path("C:/Users/harsh/Downloads/ib_vrd_ieee_preview.png")

SRC_DOCX = DESKTOP / "Research_Paper.docx"   # fresh from generate_paper.js

# ── Step 0: Render PDF images ─────────────────────────────────────────────────

def render_pdf_images():
    """Prepare figure images:
       - Fig. 1 → simplified architecture PNG (ib_vrd_ieee_preview.png)
       - Fig. 3 → full page render of fusion_module PDF (no crop)
    """
    # Fig. 1: use the simplified architecture PNG directly
    print("Preparing Fig. 1 from simplified architecture PNG...")
    arch_img = Image.open(str(PNG_ARCH)).convert("RGB")
    print(f"  PNG size: {arch_img.size}")
    fig1_path = DIAGRAMS / "fig1_sysarch_simplified.png"
    arch_img.save(str(fig1_path))
    print(f"  Saved Fig. 1: {fig1_path.name}")

    # Fig. 3: full-page render of the fusion module PDF (no cropping)
    print("Rendering fusion_module PDF at 10x zoom for Fig. 3 (full page)...")
    doc = fitz.open(str(PDF_SRC))
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(10, 10), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    # Add a small white border so the diagram doesn't touch the edges
    padded = Image.new("RGB", (img.width + 40, img.height + 40), (255, 255, 255))
    padded.paste(img, (20, 20))
    fig3_path = DIAGRAMS / "fig3_fusionmodule_full.png"
    padded.save(str(fig3_path))
    print(f"  Saved Fig. 3: {fig3_path.name}  ({padded.size})")

    return fig1_path, fig3_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clear_para(para):
    """Remove all w:r and w:hyperlink children, keep w:pPr."""
    p = para._element
    pPr = p.find(qn("w:pPr"))
    for child in list(p):
        if child is not pPr:
            p.remove(child)


def replace_placeholder_with_image(doc, placeholder, image_path, width_in):
    """Find exact-text placeholder paragraph and replace with centred image."""
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip() == placeholder.strip():
            print(f"  [Para {i}] Replacing: {repr(para.text[:55])}")
            _clear_para(para)
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = para.add_run()
            run.add_picture(str(image_path), width=Inches(width_in))
            return i
    print(f"  WARNING: not found — {repr(placeholder)}")
    return None


def _make_pPr_center():
    pPr = OxmlElement("w:pPr")
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center")
    pPr.append(jc)
    sp = OxmlElement("w:spacing")
    sp.set(qn("w:before"), "80")
    sp.set(qn("w:after"), "80")
    pPr.append(sp)
    return pPr


def _make_italic_run(text, size_half_pts=18):
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rPr.append(OxmlElement("w:i"))
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(size_half_pts))
    rPr.append(sz)
    r.append(rPr)
    t = OxmlElement("w:t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    r.append(t)
    return r


def _make_bold_run(text, size_half_pts=20):
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rPr.append(OxmlElement("w:b"))
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(size_half_pts))
    rPr.append(sz)
    r.append(rPr)
    t = OxmlElement("w:t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    r.append(t)
    return r


def _clone_image_drawing(doc, image_path, width_in):
    """Use a temporary paragraph to get the drawing XML for an image."""
    tmp = doc.add_paragraph()
    run = tmp.add_run()
    run.add_picture(str(image_path), width=Inches(width_in))
    # Collect all drawing elements
    drawings = []
    for child in list(tmp._element):
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "r":
            for sub in child:
                sub_tag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                if sub_tag == "drawing":
                    drawings.append(copy.deepcopy(child))
    # Remove temp paragraph
    doc.element.body.remove(tmp._element)
    return drawings


SECTION_E_TEXT = (
    "Fig.\u00a09 and Fig.\u00a010 show the top-5 predicted triplets on a held-out "
    "test image from Experiment\u00a0A and Experiment\u00a0B respectively. "
    "The visual-only model handles high-frequency predicates reliably \u2014 "
    "\u2018wearing\u2019 and \u2018has\u2019 are recovered correctly in both runs \u2014 "
    "but defaults to the generic predicate \u2018on\u2019 for several pairs where a "
    "more specific relation holds. "
    "Experiment\u00a0B\u2019s prior shifts probability mass toward \u2018carrying\u2019 "
    "for person\u2013bag pairs and toward \u2018holding\u2019 for person\u2013bottle "
    "pairs, matching ground-truth annotations that Experiment\u00a0A misses. "
    "The FusionModule does not blindly trust the prior: for object pairs with weak "
    "co-occurrence statistics, the learned \u03b1 down-weights the prior contribution "
    "and visual logits dominate. "
    "These side-by-side outputs show where the +4.02\u00a0pp mR@50 gain concentrates: "
    "mid-frequency predicates whose object-pair statistics are distinctive enough to "
    "resolve visual ambiguity."
)


def insert_side_by_side_after(doc, after_para_idx,
                              img_a_path, cap_a_text,
                              img_b_path, cap_b_text,
                              img_width_in=2.1,
                              heading_text="E.  Qualitative Comparison"):
    """
    Insert a sub-section heading + 2-column table (side-by-side images)
    immediately after the paragraph at after_para_idx.
    Inserts in reverse order (last element first) so final order is correct.
    """
    body = doc.element.body
    ref_elem = doc.paragraphs[after_para_idx]._element

    # ── Table: 2 columns, 1 row, no borders ──────────────────────────────────
    tbl = doc.add_table(rows=1, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    # No named style — borders removed manually below

    # Remove all borders from all cells
    nb_val = {"w:val": "none", "w:sz": "0", "w:space": "0", "w:color": "auto"}

    def _no_border(cell):
        tcPr = cell._element.find(qn("w:tcPr"))
        if tcPr is None:
            tcPr = OxmlElement("w:tcPr")
            cell._element.insert(0, tcPr)
        tcBorders = OxmlElement("w:tcBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            b = OxmlElement(f"w:{side}")
            b.set(qn("w:val"), "none")
            b.set(qn("w:sz"), "0")
            b.set(qn("w:space"), "0")
            b.set(qn("w:color"), "auto")
            tcBorders.append(b)
        tcPr.append(tcBorders)

    cells = tbl.rows[0].cells
    for cell in cells:
        _no_border(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP

    # Also strip table-level border
    tblPr = tbl._tbl.find(qn("w:tblPr"))
    if tblPr is not None:
        for old_borders in tblPr.findall(qn("w:tblBorders")):
            tblPr.remove(old_borders)
        tbl_borders = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            b = OxmlElement(f"w:{side}")
            b.set(qn("w:val"), "none")
            b.set(qn("w:sz"), "0")
            b.set(qn("w:space"), "0")
            b.set(qn("w:color"), "auto")
            tbl_borders.append(b)
        tblPr.append(tbl_borders)

    # Fill cells with image + caption
    for cell, img_path, cap in [
        (cells[0], img_a_path, cap_a_text),
        (cells[1], img_b_path, cap_b_text),
    ]:
        # Image paragraph
        img_para = cell.paragraphs[0]
        img_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        img_run = img_para.add_run()
        img_run.add_picture(str(img_path), width=Inches(img_width_in))
        # Caption paragraph
        cap_para = cell.add_paragraph()
        cap_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_run = cap_para.add_run(cap)
        cap_run.italic = True
        cap_run.font.size = Pt(8)

    # Move table XML just after ref_elem
    tbl_elem = tbl._tbl
    body.remove(tbl_elem)
    ref_elem.addnext(tbl_elem)

    # ── Body text paragraph (inserted between heading and table) ────────────
    txt_p = OxmlElement("w:p")
    pPr_t = OxmlElement("w:pPr")
    jc_t = OxmlElement("w:jc")
    jc_t.set(qn("w:val"), "both")   # justified
    pPr_t.append(jc_t)
    sp_t = OxmlElement("w:spacing")
    sp_t.set(qn("w:before"), "60")
    sp_t.set(qn("w:after"), "80")
    pPr_t.append(sp_t)
    txt_p.append(pPr_t)

    r_t = OxmlElement("w:r")
    rPr_t = OxmlElement("w:rPr")
    sz_t = OxmlElement("w:sz")
    sz_t.set(qn("w:val"), "20")   # 10pt
    rPr_t.append(sz_t)
    r_t.append(rPr_t)
    t_t = OxmlElement("w:t")
    t_t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t_t.text = SECTION_E_TEXT
    r_t.append(t_t)
    txt_p.append(r_t)

    # Insert: ref_elem → heading → text → table (insert in reverse order)
    tbl_elem.addprevious(txt_p)   # txt_p just before table

    # ── Subsection heading (inserted after ref_elem, before text+table) ─────
    h_p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    sp = OxmlElement("w:spacing")
    sp.set(qn("w:before"), "240")
    sp.set(qn("w:after"), "80")
    pPr.append(sp)
    h_p.append(pPr)
    h_p.append(_make_bold_run(heading_text, size_half_pts=20))
    ref_elem.addnext(h_p)   # heading goes immediately after ref_elem

    return tbl


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Render PDF images
    fig1_path, fig3_path = render_pdf_images()

    print(f"\nLoading: {SRC_DOCX}")
    doc = Document(str(SRC_DOCX))
    print(f"Paragraphs: {len(doc.paragraphs)}")

    # ── Fig. 1: Full system architecture from PDF ─────────────────────────────
    print("\n[Fig. 1] System architecture (full PDF image)")
    replace_placeholder_with_image(
        doc,
        placeholder="[ Fig. 1 \u2014 Architecture Diagram ]",
        image_path=fig1_path,
        width_in=3.0,    # simplified portrait: narrower for clean fit in column
    )

    # ── Fig. 3: FusionModule crop from PDF ────────────────────────────────────
    print("\n[Fig. 3] FusionModule Stage-4 crop from PDF")
    replace_placeholder_with_image(
        doc,
        placeholder="[ Fig. 3 \u2014 Architecture Diagram ]",
        image_path=fig3_path,
        width_in=3.2,    # full page, sized to stay within a single column
    )

    # ── Results comparison: side-by-side after Fig. 8 ─────────────────────────
    print("\n[Results] Exp A vs Exp B side-by-side comparison")
    fig8_idx = None
    for i, para in enumerate(doc.paragraphs):
        if para.text.startswith("Fig. 8.") and "Training loss" in para.text:
            fig8_idx = i
            break

    if fig8_idx is None:
        # Fallback: insert just before Discussion
        for i, para in enumerate(doc.paragraphs):
            if "DISCUSSION" in para.text.upper() and para.text.strip().startswith("VI"):
                fig8_idx = i - 1
                break

    if fig8_idx is not None:
        print(f"  Inserting after para {fig8_idx}: "
              f"{repr(doc.paragraphs[fig8_idx].text[:55])}")
        insert_side_by_side_after(
            doc,
            after_para_idx=fig8_idx,
            img_a_path=OUTPUTS / "sample_vis.png",
            cap_a_text="Fig. 9. Exp. A — visual-only predictions on a sample scene.",
            img_b_path=OUTPUTS / "exp_b_vis.png",
            cap_b_text="Fig. 10. Exp. B — same scene with prior-guided FusionModule.",
            img_width_in=2.1,
            heading_text="E.  Qualitative Comparison",
        )
        print("  Done.")
    else:
        print("  ERROR: could not find insertion point.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = SRC_DOCX  # overwrite in place
    doc.save(str(out))
    size_kb = out.stat().st_size // 1024
    print(f"\nSaved: {out}  ({size_kb} KB)")

    # ── PDF ───────────────────────────────────────────────────────────────────
    print("Converting to PDF...")
    try:
        from docx2pdf import convert
        pdf_out = DESKTOP / "Research_Paper.pdf"
        convert(str(out), str(pdf_out))
        print(f"PDF: {pdf_out}  ({pdf_out.stat().st_size // 1024} KB)")
    except Exception as e:
        print(f"PDF conversion skipped: {e}")


if __name__ == "__main__":
    main()
