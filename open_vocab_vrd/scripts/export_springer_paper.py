"""
export_springer_paper.py
========================
Produces two exports of the Springer LLNCS paper:
  1. Springer_Paper.txt  — clean plain-text extracted from the compiled PDF
  2. Springer_Paper.docx — formatted Word document matching the paper content

Run from open_vocab_vrd/:
    python scripts/export_springer_paper.py
"""

import sys, io
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import fitz                                     # PyMuPDF
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── Paths ─────────────────────────────────────────────────────────────────────
DESKTOP   = Path("C:/Users/harsh/OneDrive/Desktop")
PAPER_DIR = Path(__file__).resolve().parent.parent / "outputs/springer_paper"
PDF_SRC   = PAPER_DIR / "main.pdf"
FIGS      = PAPER_DIR / "figures"

OUT_TXT   = DESKTOP / "Springer_Paper.txt"
OUT_DOCX  = DESKTOP / "Springer_Paper.docx"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TXT export — extract text from compiled PDF
# ─────────────────────────────────────────────────────────────────────────────

def export_txt():
    print("Exporting TXT from PDF …")
    doc = fitz.open(str(PDF_SRC))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append(f"{'─'*72}\n  Page {i+1}\n{'─'*72}\n{text}")
    full = "\n".join(pages)
    OUT_TXT.write_text(full, encoding="utf-8")
    print(f"  Saved: {OUT_TXT}  ({OUT_TXT.stat().st_size // 1024} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DOCX export — recreate document with python-docx
# ─────────────────────────────────────────────────────────────────────────────

# ── Style helpers ─────────────────────────────────────────────────────────────

def _set_font(run, name="Times New Roman", size_pt=10, bold=False, italic=False,
              color=None):
    run.font.name        = name
    run.font.size        = Pt(size_pt)
    run.font.bold        = bold
    run.font.italic      = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_title(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(6)
    r = p.add_run(text)
    _set_font(r, size_pt=14, bold=True)
    return p


def add_authors(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    _set_font(r, size_pt=11)
    return p


def add_affiliation(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text)
    _set_font(r, size_pt=9, italic=True)
    return p


def add_section(doc, num, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after  = Pt(4)
    r = p.add_run(f"{num}  {title}")
    _set_font(r, size_pt=12, bold=True)
    return p


def add_subsection(doc, num, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after  = Pt(3)
    r = p.add_run(f"{num}  {title}")
    _set_font(r, size_pt=10, bold=True)
    return p


def add_subsubsection(doc, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(2)
    r = p.add_run(title)
    _set_font(r, size_pt=10, bold=True, italic=True)
    return p


def add_body(doc, text, first=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    if not first:
        p.paragraph_format.first_line_indent = Inches(0.2)
    r = p.add_run(text)
    _set_font(r, size_pt=10)
    return p


def add_abstract_block(doc, text, keywords):
    # "Abstract" label
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(2)
    r = p.add_run("Abstract. ")
    _set_font(r, size_pt=10, bold=True)
    r2 = p.add_run(text)
    _set_font(r2, size_pt=10)
    # Keywords
    pk = doc.add_paragraph()
    pk.paragraph_format.space_after = Pt(10)
    rk1 = pk.add_run("Keywords: ")
    _set_font(rk1, size_pt=10, bold=True)
    rk2 = pk.add_run(keywords)
    _set_font(rk2, size_pt=10)


def add_figure(doc, img_path, caption, label_num, width_in=4.5):
    if not Path(img_path).exists():
        add_body(doc, f"[Figure {label_num}: {caption}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    r = p.add_run()
    r.add_picture(str(img_path), width=Inches(width_in))
    # caption
    pc = doc.add_paragraph()
    pc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pc.paragraph_format.space_after = Pt(8)
    rc = pc.add_run(f"Fig. {label_num}. {caption}")
    _set_font(rc, size_pt=9, italic=True)


def add_table_caption(doc, num, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after  = Pt(2)
    r = p.add_run(f"Table {num}. {text}")
    _set_font(r, size_pt=9, bold=True)


def add_ruled_table(doc, headers, rows):
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.style = "Table Grid"
    # header row
    hrow = tbl.rows[0]
    for j, h in enumerate(headers):
        cell = hrow.cells[j]
        cell.text = h
        for run in cell.paragraphs[0].runs:
            _set_font(run, size_pt=9, bold=True)
    # data rows
    for i, row in enumerate(rows):
        drow = tbl.rows[i + 1]
        for j, val in enumerate(row):
            cell = drow.cells[j]
            cell.text = str(val)
            for run in cell.paragraphs[0].runs:
                _set_font(run, size_pt=9)
    doc.add_paragraph()   # spacer after table


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    _set_font(r, size_pt=10)


def add_ref(doc, num, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent    = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(f"[{num}] {text}")
    _set_font(r, size_pt=9)


def add_rule(doc):
    """Horizontal rule via a bottom border on an empty paragraph."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    pPr = p._element.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "888888")
    pBdr.append(bottom)
    pPr.append(pBdr)


# ── Build DOCX ────────────────────────────────────────────────────────────────

def export_docx():
    print("Building DOCX …")
    doc = Document()

    # Page margins (1 inch all sides)
    from docx.oxml import OxmlElement as OE
    section = doc.sections[0]
    section.top_margin    = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin   = Inches(1.25)
    section.right_margin  = Inches(1.25)

    # ── Title & Authors ───────────────────────────────────────────────────────
    add_title(doc,
        "Prior-Guided Visual Relationship Detection Using "
        "Open-Vocabulary Detectors and Frozen CLIP Encoders")

    add_authors(doc, "Harshul Bagri · Shikhar Patel · C. Muralidharan")
    add_affiliation(doc,
        "Department of Computing Technologies, SRM Institute of Science and Technology, "
        "Kattankulathur, Tamil Nadu 603 203, India")

    add_rule(doc)

    # ── Abstract ──────────────────────────────────────────────────────────────
    add_abstract_block(doc,
        "Visual relationship detection (VRD) is the task of identifying (subject, predicate, object) "
        "triplets in images, forming the backbone of scene understanding systems used in robotics, "
        "image retrieval, and visual question answering. Most existing VRD methods train end-to-end "
        "on closed-vocabulary detectors, making them brittle when object categories shift and prone "
        "to predicting only high-frequency predicates. We address both problems with a modular "
        "pipeline: YOLO-World provides open-vocabulary bounding-box proposals; a frozen CLIP "
        "ViT-B/32 encoder converts each crop into a 512-dimensional embedding; a 13-dimensional "
        "spatial feature vector captures relative geometry; and a lightweight three-layer MLP "
        "predicts predicate logits over 16 categories. An optional FusionModule blends visual logits "
        "with a Visual Genome co-occurrence prior through two learnable scalars, α and τ, adding "
        "only two trainable parameters. Trained on 18,244 filtered Visual Genome triplets, the "
        "visual-only model achieves R@50 of 73.01% and mR@50 of 30.28%. Enabling the prior raises "
        "these to 76.73% and 34.30% respectively — a +4.02 percentage-point gain in mean recall "
        "that concentrates on mid-frequency predicates such as carrying (+13.3 pp) and holding "
        "(+11.1 pp). The system generalises naturally to novel object categories because neither "
        "the detector nor the encoder is fine-tuned.",
        "Visual relationship detection · Scene graph generation · Open-vocabulary detection · "
        "CLIP embeddings · Co-occurrence prior")

    add_rule(doc)

    # ── 1. Introduction ───────────────────────────────────────────────────────
    add_section(doc, "1", "Introduction")

    add_body(doc,
        "Recognising objects in isolation is no longer the bottleneck in computer vision. "
        "The harder and more useful problem is understanding how objects relate to one another: "
        "a person riding a bicycle is semantically distant from a person standing near one, even "
        "though both scenes contain the same two object categories. Visual relationship detection "
        "(VRD) makes that distinction explicit by predicting triplets of the form "
        "(subject, predicate, object), and scene graphs built from these triplets have become a "
        "standard intermediate representation for tasks such as image captioning, visual question "
        "answering, and robotic manipulation planning.", first=True)

    add_body(doc,
        "Two persistent problems limit current VRD systems. First, predicate distributions in real "
        "datasets are severely long-tailed: in Visual Genome [1], the five most common predicates "
        "account for more than 60% of all annotated relationships, while rare predicates such as "
        "carrying and standing on appear in fewer than 1% of images. Models trained with standard "
        "cross-entropy collapse to the modal prediction — nearly always on or has — and achieve "
        "competitive Recall@K only because those metrics reward correct head-predicate predictions. "
        "Mean Recall@K (mR@K) exposes the failure more honestly; state-of-the-art scene-graph "
        "models often report mR@50 below 15% in the predicate-classification setting [2, 3].")

    add_body(doc,
        "Second, most published systems couple the object detector tightly to the relation model, "
        "training both end-to-end on fixed object-category vocabularies drawn from COCO or Visual "
        "Genome. Deploying such a system on a new domain requires retraining from scratch if the "
        "target categories differ from the training vocabulary.")

    add_body(doc,
        "We sidestep both issues. Our pipeline, IB-VRD, uses YOLO-World [9] as a frozen, "
        "open-vocabulary detector and a frozen CLIP [8] ViT-B/32 encoder for visual features. "
        "The only components trained from scratch are a small relation-head MLP and — optionally — "
        "the two scalar parameters of a FusionModule that blends visual logits with a Visual Genome "
        "co-occurrence prior. Because neither the detector nor the encoder is fine-tuned, IB-VRD "
        "accepts arbitrary object categories at inference time without any retraining.")

    add_body(doc, "The contributions of this work are:")
    add_bullet(doc,
        "A fully modular VRD pipeline that replaces the traditional Faster R-CNN backbone with "
        "open-vocabulary, frozen components, requiring no detector fine-tuning on the target dataset.")
    add_bullet(doc,
        "A lightweight FusionModule that injects a statistical co-occurrence prior through two "
        "learnable scalars, improving mR@50 by 4.02 pp over the visual-only baseline.")
    add_bullet(doc,
        "An analysis of the clean-embedding constraint — discarding training samples whose CLIP "
        "embeddings fall below a cosine-similarity threshold — and its effect on training stability "
        "and per-predicate recall.")

    # ── 2. Literature Review ──────────────────────────────────────────────────
    add_section(doc, "2", "Literature Review")

    add_subsection(doc, "2.1", "Scene Graph Generation")
    add_body(doc,
        "Early VRD methods treated relationship prediction as a simple classification problem "
        "applied to pairs of region proposals from a Faster R-CNN [6] detector. Xu et al. [11] "
        "introduced iterative message passing between a graph of object nodes and relationship "
        "edges. Zellers et al. [2] identified repeating structural patterns (motifs) in scene "
        "graphs and proposed a BiLSTM that processes objects in a fixed order determined by "
        "detection confidence. Tang et al. [3] replaced the linear sequence with a tree-structured "
        "context encoder, and Li et al. [4] introduced bipartite graph neural networks that "
        "alternate between object and relation representations. All of these methods share a common "
        "limitation: they are trained end-to-end with a fixed object vocabulary and require "
        "expensive detector retraining for new domains.", first=True)

    add_subsection(doc, "2.2", "Statistical Priors and Frequency Bias")
    add_body(doc,
        "The frequency-bias model, proposed implicitly in [2] and made explicit in subsequent "
        "work, simply adds the log-frequency of (subject-category, predicate, object-category) "
        "triplets from training data to the classifier output. This cheap addition improves R@K "
        "substantially but amplifies the head-predicate problem. Tang et al. [5] addressed this "
        "through causal inference, using counterfactual reasoning to subtract the direct effect "
        "of the frequency prior at inference time. Our FusionModule takes a softer approach: a "
        "learnable temperature τ controls how sharply the prior is peaked, and a learnable weight "
        "α controls how much it contributes, so the model can self-regulate without manual tuning.", first=True)

    add_subsection(doc, "2.3", "Open-Vocabulary Detection and Vision-Language Models")
    add_body(doc,
        "Closed-vocabulary detectors such as Faster R-CNN [6] and DETR [7] require a fixed "
        "category list at training time. Bansal et al. [13] proposed transferring region embeddings "
        "to unseen categories via semantic word vectors, and DETIC [14] scaled this with "
        "image-level supervision across 20,000 categories. YOLO-World [9] conditions the detection "
        "head on CLIP text embeddings, achieving open-vocabulary detection at real-time throughput. "
        "CLIP [8] learns aligned image-text representations from 400 million pairs; Yuan et al. [15] "
        "show that such pre-training transfers to VRD when relation descriptions are included in "
        "the text side. We use CLIP purely as a frozen crop encoder, keeping it general-purpose "
        "and free of task-specific overfitting.", first=True)

    # ── 3. Methodology ────────────────────────────────────────────────────────
    add_section(doc, "3", "Methodology")

    add_subsection(doc, "3.1", "Pipeline Overview")
    add_body(doc,
        "Figure 1 shows the five stages of IB-VRD. Given an input image, YOLO-World detects up "
        "to ten object bounding boxes with open-vocabulary class labels. CLIP encodes each box "
        "crop into a 512-dimensional L2-normalised embedding. A 13-dimensional spatial feature "
        "vector is computed for each ordered pair of detected objects. The relation-head MLP "
        "concatenates subject embedding, object embedding, and spatial features into a "
        "1037-dimensional vector and maps it to 16 predicate logits. When the FusionModule is "
        "active, visual logits are combined with a pre-computed co-occurrence prior before softmax.", first=True)

    add_figure(doc,
        FIGS / "fig1_simplified.png",
        "IB-VRD system architecture. Shaded modules (YOLO-World, CLIP) are frozen at training "
        "and inference time. Only the Relation Head MLP and, in Experiment B, the two "
        "FusionModule scalars are trained.",
        label_num=1, width_in=3.2)

    add_subsection(doc, "3.2", "Open-Vocabulary Object Detection")
    add_body(doc,
        "We use YOLOv8s-World with the standard MS-COCO object vocabulary extended with the "
        "80 most common Visual Genome object categories, giving 160 distinct classes. Detection "
        "confidence is thresholded at 0.30; at most ten boxes per image are retained after "
        "non-maximum suppression (IoU threshold 0.50). The detector weights are never updated "
        "during training.", first=True)

    add_subsection(doc, "3.3", "Frozen Visual Encoding")
    add_body(doc,
        "Each bounding box is cropped from the original image with a five-pixel padding on each "
        "side and resized to 224×224 pixels. The CLIP ViT-B/32 image encoder processes each crop "
        "independently, producing a 512-dimensional embedding that is L2-normalised before use. "
        "The encoder weights are frozen throughout. We found that 45.9% of training triplets "
        "produced degenerate embeddings — typically because the corresponding Visual Genome "
        "bounding box was outside the image boundary or reduced to a single-pixel crop after "
        "padding. These samples are discarded before training.", first=True)

    add_subsection(doc, "3.4", "Spatial Feature Vector")
    add_body(doc,
        "The geometric relationship between a subject box and an object box carries information "
        "the visual encoder cannot easily capture. We compute a 13-dimensional spatial feature "
        "vector: [b_s; b_o; Δc_x; Δc_y; log(w_s/w_o); log(h_s/h_o); IoU(b_s, b_o)], where "
        "b_s and b_o are normalised box coordinates, Δc_x and Δc_y are normalised centroid "
        "offsets, and IoU is the standard intersection-over-union. The log-ratio terms encode "
        "relative scale, which is predictive for predicates such as riding and wearing.", first=True)

    add_subsection(doc, "3.5", "Relation Head MLP")
    add_body(doc,
        "Subject embedding (512-D), object embedding (512-D), and spatial features (13-D) are "
        "concatenated into a single 1037-dimensional vector. The relation head is a three-layer "
        "MLP: LayerNorm(1037) → Linear(1037→512) → GELU → Dropout(0.3) → Linear(512→256) → "
        "GELU → Dropout(0.3) → Linear(256→16). The output is a vector of 16 visual predicate "
        "logits. Total trainable parameters in Experiment A (visual only) are 666,896.", first=True)

    add_subsection(doc, "3.6", "Co-occurrence Prior and FusionModule")
    add_body(doc,
        "For each ordered pair (subject category, object category) seen during training, we "
        "compute the empirical distribution over predicates from all matching triplets in the "
        "training split. For unseen pairs a uniform distribution is used. The FusionModule "
        "combines visual logits v with the prior vector p for the detected category pair:",
        first=True)
    add_body(doc, "    l̂ = v + α · (p / τ)")
    add_body(doc,
        "where α = exp(log α₀) and τ = softplus(log τ₀) + ε are scalar parameters initialised "
        "to 1.0. The FusionModule adds only two trainable parameters, bringing the Experiment B "
        "total to 666,898.")

    add_figure(doc,
        FIGS / "fusion_module.png",
        "FusionModule block diagram. Visual logits v and scaled prior logits p/τ are combined "
        "with learned weight α. At initialisation (α=1.0, τ=1.0) the output equals v + p. "
        "Only log α and log τ are trained.",
        label_num=2, width_in=3.0)

    add_subsection(doc, "3.7", "Training Objective")
    add_body(doc,
        "We use focal loss with γ = 2.0, which down-weights easy examples and concentrates "
        "gradient signal on samples where the model is uncertain or wrong — in practice, "
        "rare predicates and visually ambiguous configurations. Standard cross-entropy "
        "(γ = 0) converges faster in the first five epochs but plateaus at a lower mR@50.", first=True)

    # ── 4. Experimental Setup ─────────────────────────────────────────────────
    add_section(doc, "4", "Experimental Setup")

    add_subsection(doc, "4.1", "Dataset")
    add_body(doc,
        "We train and evaluate on a filtered subset of Visual Genome (VG) [1], restricting to "
        "the 16 most informative predicates. Table 1 summarises the split statistics. The top-3 "
        "predicates (on, has, wearing) together account for 61.2% of all triplets, while the "
        "bottom-5 predicates each appear in fewer than 1% of training examples.", first=True)

    add_table_caption(doc, 1,
        "Dataset split statistics for the 16-predicate Visual Genome subset.")
    add_ruled_table(doc,
        headers=["Split", "Images", "Raw triplets", "Filtered triplets"],
        rows=[
            ["Train",      "86,613", "33,741", "18,244 (54.1%)"],
            ["Validation",  "5,000",  "2,204",  "—"],
            ["Test",        "5,000",  "2,204",  "—"],
            ["Total",      "96,613", "38,149",  "—"],
        ])

    add_subsection(doc, "4.2", "Implementation Details")
    add_body(doc,
        "Table 2 lists the key hyperparameters. All experiments were run on a single NVIDIA GPU. "
        "The YOLO-World detector and CLIP encoder are used in evaluation mode with gradients "
        "disabled throughout. CLIP crop embeddings for the entire training set are pre-computed "
        "and stored in a sharded disk cache, reducing per-epoch I/O to simple array lookups.", first=True)

    add_table_caption(doc, 2, "Hyperparameter settings shared by both experiments.")
    add_ruled_table(doc,
        headers=["Hyperparameter", "Value"],
        rows=[
            ["Detector",       "YOLOv8s-World, conf ≥ 0.30, NMS IoU = 0.50"],
            ["CLIP encoder",   "ViT-B/32, frozen, L2-normalised"],
            ["MLP architecture","LN(1037)→512→256→16, GELU, Drop(0.3)"],
            ["Loss",           "Focal loss, γ = 2.0"],
            ["Optimiser",      "Adam, lr = 0.001, weight decay = 1e-4"],
            ["Scheduler",      "Cosine annealing"],
            ["Batch size",     "128"],
            ["Epochs",         "30"],
            ["Prior α init",   "1.0 (Exp. B only)"],
            ["Prior τ init",   "1.0 (Exp. B only)"],
        ])

    add_subsection(doc, "4.3", "Evaluation Protocol")
    add_body(doc,
        "We follow the standard predicate-classification (PredCls) protocol: ground-truth "
        "bounding boxes and object labels are provided to the model at test time, and only "
        "predicate prediction is evaluated. We report Recall@K (R@K) and mean Recall@K (mR@K) "
        "for K ∈ {20, 50, 100}. Experiment A is the visual-only baseline (MLP + focal loss, no "
        "prior). Experiment B adds the FusionModule with learnable α and τ.", first=True)

    # ── 5. Results and Analysis ───────────────────────────────────────────────
    add_section(doc, "5", "Results and Analysis")

    add_subsection(doc, "5.1", "Overall Performance")
    add_body(doc,
        "Table 3 compares Experiment A and Experiment B. Adding the FusionModule improves every "
        "metric. The gap is larger for mR@K than for R@K, which is expected: the prior provides "
        "the most information for predicates that appear infrequently and are therefore hard to "
        "recover from visual features alone.", first=True)

    add_table_caption(doc, 3,
        "Quantitative results on the Visual Genome 16-predicate test set. "
        "Δ: absolute gain in percentage points.")
    add_ruled_table(doc,
        headers=["Method", "R@20", "R@50", "R@100", "mR@20", "mR@50", "mR@100"],
        rows=[
            ["Exp. A (visual only)", "72.80", "73.01", "73.01", "30.28", "30.28", "30.28"],
            ["Exp. B (+ prior)",     "76.42", "76.73", "76.73", "34.23", "34.30", "34.30"],
            ["Δ (B − A)",            "+3.62", "+3.72", "+3.72", "+3.95", "+4.02", "+4.02"],
        ])

    add_figure(doc,
        FIGS / "overall_comparison.png",
        "R@K and mR@K comparison between Experiment A (visual only) and "
        "Experiment B (prior-guided).",
        label_num=3, width_in=4.2)

    add_subsection(doc, "5.2", "Per-Predicate Analysis")
    add_body(doc,
        "Figure 4 shows the per-predicate R@50 change from Experiment A to Experiment B. "
        "Three predicates gain the most: carrying (+13.3 pp), holding (+11.1 pp), and "
        "behind (+10.1 pp). These predicates have moderate frequency and distinctive "
        "co-occurrence patterns — carrying strongly co-occurs with person–bag pairs, and "
        "holding with person–bottle or person–phone pairs. Predicates that show no gain or "
        "slight regression include on (−0.3 pp) and wearing (−0.1 pp); the learned τ grows "
        "for these cases, effectively flattening the prior and reducing its influence.", first=True)

    add_figure(doc,
        FIGS / "predicate_delta.png",
        "Per-predicate R@50 gain from Experiment A to Experiment B. The three largest "
        "gains are for carrying, holding, and behind.",
        label_num=4, width_in=5.0)

    add_subsection(doc, "5.3", "Head vs. Tail Predicate Groups")
    add_body(doc,
        "Partitioning the 16 predicates into head (top-8 by training frequency) and tail "
        "(bottom-8) groups reveals an asymmetric improvement. Head-predicate mR@50 improves "
        "by a modest 7.9% relative (from ≈65.3% to ≈70.5%). Tail-predicate mR@50 improves "
        "by 26.1% relative (from ≈12.4% to ≈15.6%). This confirms the core motivation for "
        "the FusionModule: the prior is most informative precisely where the visual model "
        "is weakest. Four tail predicates — riding, hanging from, standing on, and walking on "
        "— reach R@50 of 0.0% in both experiments, likely due to low training frequency "
        "combined with high visual diversity.", first=True)

    # ── 6. Conclusion ─────────────────────────────────────────────────────────
    add_section(doc, "6", "Conclusion")

    add_body(doc,
        "We presented IB-VRD, a modular visual relationship detection pipeline that replaces "
        "the standard Faster R-CNN backbone with frozen, open-vocabulary components. Using "
        "YOLO-World for detection, CLIP for visual encoding, and a lightweight MLP for predicate "
        "classification, the visual-only baseline achieves 73.01% R@50 and 30.28% mR@50 on a "
        "16-predicate Visual Genome subset. Adding the FusionModule — two learnable scalars that "
        "blend visual logits with a co-occurrence prior — raises mR@50 to 34.30%, a gain of "
        "4.02 percentage points. The gain concentrates on mid-frequency tail predicates such as "
        "carrying and holding, precisely where statistical co-occurrence provides information "
        "that visual appearance cannot.", first=True)

    add_body(doc,
        "The system generalises to novel object categories out of the box because neither "
        "YOLO-World nor CLIP is fine-tuned, a practical advantage for deployment in new domains. "
        "Current limitations include the failure on very rare predicates (riding, walking on) "
        "and sensitivity to the quality of YOLO-World detections on unusual objects.")

    add_body(doc,
        "Future directions include using a language-model derived prior instead of the "
        "frequency-based co-occurrence table, exploring lightweight fine-tuning of the CLIP "
        "encoder with adapters, and extending the 16-predicate vocabulary to the full 50-predicate "
        "Visual Genome set.")

    add_subsubsection(doc, "Acknowledgements")
    add_body(doc,
        "The authors thank the Department of Computing Technologies at SRM Institute of Science "
        "and Technology for providing computational resources and laboratory access throughout "
        "this project.", first=True)

    # ── References ────────────────────────────────────────────────────────────
    add_section(doc, "", "References")

    refs = [
        ("1",  "Krishna, R. et al.: Visual Genome: connecting language and vision using crowdsourced "
               "dense image annotations. Int. J. Comput. Vis. 123(1), 32–73 (2017)."),
        ("2",  "Zellers, R., Yatskar, M., Thomson, S., Choi, Y.: Neural motifs: scene graph parsing "
               "with global context. In: CVPR, pp. 5831–5840. IEEE (2018)."),
        ("3",  "Tang, K., Zhang, H., Wu, B., Luo, W., Liu, W.: Learning to compose dynamic tree "
               "structures for visual contexts. In: CVPR, pp. 6619–6628. IEEE (2019)."),
        ("4",  "Li, R., Zhang, S., Wan, B., He, X.: Bipartite graph network with adaptive message "
               "passing for unbiased scene graph generation. In: CVPR, pp. 11109–11119. IEEE (2021)."),
        ("5",  "Tang, K., Niu, Y., Huang, J., Shi, J., Zhang, H.: Unbiased scene graph generation "
               "from biased training. In: CVPR, pp. 3716–3725. IEEE (2020)."),
        ("6",  "Ren, S., He, K., Girshick, R., Sun, J.: Faster R-CNN: towards real-time object "
               "detection with region proposal networks. IEEE TPAMI 39(6), 1137–1149 (2017)."),
        ("7",  "Carion, N. et al.: End-to-end object detection with transformers. In: ECCV, "
               "LNCS vol. 12346, pp. 213–229. Springer (2020)."),
        ("8",  "Radford, A. et al.: Learning transferable visual models from natural language "
               "supervision. In: ICML, pp. 8748–8763. PMLR (2021)."),
        ("9",  "Cheng, T. et al.: YOLO-World: real-time open-vocabulary object detection. "
               "In: CVPR, pp. 16901–16911. IEEE (2024)."),
        ("10", "Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollar, P.: Focal loss for dense "
               "object detection. IEEE TPAMI 42(2), 318–327 (2020)."),
        ("11", "Xu, D., Zhu, Y., Choy, C.B., Fei-Fei, L.: Scene graph generation by iterative "
               "message passing. In: CVPR, pp. 5410–5419. IEEE (2017)."),
        ("12", "Lin, X., Ding, C., Zeng, J., Tao, D.: GPS-Net: graph property sensing network "
               "for scene graph generation. In: CVPR, pp. 3746–3753. IEEE (2020)."),
        ("13", "Bansal, A. et al.: Zero-shot object detection. In: ECCV, LNCS vol. 11205, "
               "pp. 397–414. Springer (2018)."),
        ("14", "Zhou, X. et al.: Detecting twenty-thousand classes using image-level supervision. "
               "In: ECCV, LNCS vol. 13669, pp. 350–368. Springer (2022)."),
        ("15", "Yuan, Z. et al.: RLIP: relational language-image pre-training for human-object "
               "interaction detection. In: NeurIPS, vol. 35, pp. 37166–37179 (2022)."),
        ("16", "Yang, X., Tang, K., Zhang, H., Cai, J.: Auto-encoding scene graphs for image "
               "captioning. In: CVPR, pp. 10685–10694. IEEE (2019)."),
        ("17", "Teney, D., Liu, L., van den Hengel, A.: Graph-structured representations for "
               "visual question answering. In: CVPR, pp. 1–9. IEEE (2017)."),
        ("18", "Zhu, Y., Fathi, A., Fei-Fei, L.: Reasoning about object affordances in a "
               "knowledge base representation. In: ECCV, LNCS vol. 8690, pp. 408–424. "
               "Springer (2014)."),
    ]
    for num, text in refs:
        add_ref(doc, num, text)

    doc.save(str(OUT_DOCX))
    print(f"  Saved: {OUT_DOCX}  ({OUT_DOCX.stat().st_size // 1024} KB)")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    export_txt()
    export_docx()
    print("\nDone.")
