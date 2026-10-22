"""
patch_research_paper.py
=======================
Inserts Fig. 1, Fig. 3, and a Qualitative Comparison section into
Research_Paper.docx without modifying any existing text.

Run:
    python scripts/patch_research_paper.py
"""

import sys, io, copy, shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from lxml import etree

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent.parent
OUTPUTS   = BASE / "outputs"
DIAGRAMS  = BASE / "outputs" / "diagrams"
PLOTS     = BASE / "outputs" / "plots"
DESKTOP   = Path("C:/Users/harsh/OneDrive/Desktop")
SRC_DOCX  = DESKTOP / "Research_Paper.docx"
OUT_DOCX  = DESKTOP / "Research_Paper.docx"   # overwrite in place

FIG1_IMG  = DIAGRAMS / "fig1_system_arch_hq.png"   # system architecture
FIG3_IMG  = DIAGRAMS / "fig_4_4_fusion_module.png"  # FusionModule diagram
EXPA_IMG  = OUTPUTS  / "sample_vis.png"             # Exp A qualitative output
EXPB_IMG  = OUTPUTS  / "exp_b_vis.png"              # Exp B qualitative output
OVERALL   = PLOTS    / "overall_comparison.png"      # quantitative bar chart
RK_COMP   = PLOTS    / "rk_comparison.png"           # R@K comparison

# ── Helpers ───────────────────────────────────────────────────────────────────

def _clear_and_center(para):
    """Remove all runs from a paragraph and set alignment to center."""
    p_elem = para._element
    pPr = p_elem.find(qn("w:pPr"))
    for child in list(p_elem):
        if child is not pPr:
            p_elem.remove(child)
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _set_para_center(para):
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER


def replace_placeholder_with_image(doc, placeholder_text, image_path,
                                   width_inches=5.0):
    """Find a paragraph exactly matching placeholder_text, clear it, embed image."""
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip() == placeholder_text.strip():
            print(f"  Replacing para {i}: {repr(para.text[:60])}")
            _clear_and_center(para)
            run = para.add_run()
            run.add_picture(str(image_path), width=Inches(width_inches))
            return True
    print(f"  WARNING: placeholder not found: {repr(placeholder_text)}")
    return False


def new_para_after(ref_elem, parent_body):
    """Create a new empty w:p element after ref_elem and return it."""
    new_p = OxmlElement("w:p")
    ref_elem.addnext(new_p)
    return new_p


def insert_centered_image_after(doc, after_para_idx, image_path,
                                 caption_text, fig_num, width_inches=5.5):
    """
    Insert (image paragraph + caption paragraph) immediately after the
    paragraph at after_para_idx.  Inserts in reverse order so final order
    is correct (image then caption).
    """
    body = doc.element.body
    ref_para = doc.paragraphs[after_para_idx]
    ref_elem = ref_para._element

    # -- Caption paragraph (inserted first, will be pushed down by image) --
    cap_p = OxmlElement("w:p")
    ref_elem.addnext(cap_p)

    # Build caption paragraph XML manually
    pPr = OxmlElement("w:pPr")
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center")
    pPr.append(jc)
    cap_p.append(pPr)

    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    # Italic caption style
    i_elem = OxmlElement("w:i")
    rPr.append(i_elem)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "18")   # 9pt × 2
    rPr.append(sz)
    r.append(rPr)

    t = OxmlElement("w:t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = caption_text
    r.append(t)
    cap_p.append(r)

    # -- Image paragraph (inserted after ref_elem, before caption) --
    img_p = OxmlElement("w:p")
    ref_elem.addnext(img_p)     # img before caption in DOM

    pPr2 = OxmlElement("w:pPr")
    jc2 = OxmlElement("w:jc")
    jc2.set(qn("w:val"), "center")
    pPr2.append(jc2)
    # Spacing before image
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), "120")
    spacing.set(qn("w:after"), "60")
    pPr2.append(spacing)
    img_p.append(pPr2)

    # Use a temporary paragraph to call add_picture via python-docx
    from docx.text.paragraph import Paragraph as DParagraph
    from docx.oxml import OxmlElement as OE
    tmp_p = doc.add_paragraph()
    tmp_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = tmp_p.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))

    # Move the picture run XML into img_p
    tmp_elem = tmp_p._element
    for child in list(tmp_elem):
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag in ("r", "hyperlink"):
            img_p.append(copy.deepcopy(child))

    # Remove the temp paragraph from body
    body.remove(tmp_elem)

    return img_p, cap_p


def insert_subsection_heading_after(ref_elem, heading_text, label="E."):
    """Insert a subsection heading paragraph after ref_elem."""
    h_p = OxmlElement("w:p")
    ref_elem.addnext(h_p)

    pPr = OxmlElement("w:pPr")
    sp = OxmlElement("w:spacing")
    sp.set(qn("w:before"), "240")
    sp.set(qn("w:after"), "80")
    pPr.append(sp)
    h_p.append(pPr)

    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    bold = OxmlElement("w:b")
    rPr.append(bold)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "20")  # 10pt
    rPr.append(sz)
    r.append(rPr)

    t = OxmlElement("w:t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = f"{label}  Qualitative Comparison"
    r.append(t)
    h_p.append(r)
    return h_p


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading: {SRC_DOCX}")
    doc = Document(str(SRC_DOCX))
    n_paras = len(doc.paragraphs)
    print(f"Paragraphs: {n_paras}")

    # ── Step 1: Replace Fig. 1 placeholder ────────────────────────────────────
    print("\n[Fig. 1] System architecture image")
    replace_placeholder_with_image(
        doc,
        placeholder_text="[ Fig. 1 \u2014 Architecture Diagram ]",
        image_path=FIG1_IMG,
        width_inches=4.2,    # portrait image, keep narrow
    )

    # ── Step 2: Replace Fig. 3 placeholder ────────────────────────────────────
    print("\n[Fig. 3] FusionModule diagram")
    replace_placeholder_with_image(
        doc,
        placeholder_text="[ Fig. 3 \u2014 Architecture Diagram ]",
        image_path=FIG3_IMG,
        width_inches=5.5,    # landscape diagram, wider
    )

    # ── Step 3: Add qualitative comparison after Fig. 8 caption ───────────────
    # Fig. 8 caption is the last figure in the Results section.
    # Find it by its known text prefix.
    print("\n[Results comparison] Inserting after Fig. 8 caption")

    fig8_idx = None
    for i, para in enumerate(doc.paragraphs):
        if para.text.startswith("Fig. 8.") and "Training loss" in para.text:
            fig8_idx = i
            break

    if fig8_idx is None:
        print("  WARNING: Fig. 8 caption not found, inserting before Discussion instead")
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip().startswith("VI.") and "DISCUSSION" in para.text.upper():
                fig8_idx = i - 1
                break

    if fig8_idx is not None:
        print(f"  Inserting after para {fig8_idx}: {repr(doc.paragraphs[fig8_idx].text[:60])}")

        body = doc.element.body
        ref_elem = doc.paragraphs[fig8_idx]._element

        # Insert in REVERSE order so the final reading order is correct:
        # heading → Exp A image → Exp A caption → bar chart → bar caption
        # → Exp B image → Exp B caption

        # -- Exp B qualitative output --
        _, cap_b = insert_centered_image_after(
            doc, fig8_idx, EXPB_IMG,
            "Fig. 10. Exp. B (prior-guided): same scene with FusionModule active. "
            "Mid-frequency predicates such as 'holding' and 'carrying' are now correctly ranked.",
            fig_num=10,
            width_inches=3.4,
        )

        # -- Exp A qualitative output --
        _, cap_a = insert_centered_image_after(
            doc, fig8_idx, EXPA_IMG,
            "Fig. 9. Exp. A (visual only): top-5 predicted triplets overlaid on a sample image. "
            "Detections from YOLO-World; embeddings from frozen CLIP ViT-B/32.",
            fig_num=9,
            width_inches=3.4,
        )

        # -- Subsection heading (inserted last so it appears first) --
        h_elem = insert_subsection_heading_after(
            ref_elem,
            heading_text="Qualitative Comparison",
            label="E.",
        )

        print("  Qualitative comparison section inserted.")
    else:
        print("  ERROR: could not find insertion point.")

    # ── Save ──────────────────────────────────────────────────────────────────
    doc.save(str(OUT_DOCX))
    size_kb = OUT_DOCX.stat().st_size // 1024
    print(f"\nSaved: {OUT_DOCX}  ({size_kb} KB)")

    # Convert to PDF
    print("Converting to PDF...")
    try:
        from docx2pdf import convert
        pdf_out = DESKTOP / "Research_Paper.pdf"
        convert(str(OUT_DOCX), str(pdf_out))
        pdf_kb = pdf_out.stat().st_size // 1024
        print(f"PDF: {pdf_out}  ({pdf_kb} KB)")
    except Exception as e:
        print(f"PDF conversion failed: {e}")


if __name__ == "__main__":
    main()
