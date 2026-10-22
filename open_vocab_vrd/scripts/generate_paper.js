"use strict";
/**
 * generate_paper.js — IEEE Two-Column Conference Format
 *
 * Title  : Improving Visual Relationship Detection with Prior-Guided
 *          Learning and Clean Embedding Optimization for Explainable
 *          Scene Understanding
 *
 * Authors: Harshul Bagri, Shikhar Patel
 *          Dept. of Computing Technologies, SRM Institute of Science and Technology
 *
 * Output : C:/Users/harsh/OneDrive/Desktop/Research_Paper.docx
 * Run    : node scripts/generate_paper.js
 */

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, BorderStyle, WidthType, ShadingType, ImageRun,
  PageNumber, Footer
} = require("docx");
const fs   = require("fs");
const path = require("path");

// ── Paths ─────────────────────────────────────────────────────────────────────
const DESKTOP = "C:/Users/harsh/OneDrive/Desktop";
const OUTPUT  = path.join(DESKTOP, "Research_Paper.docx");
const PLOTS   = path.join(__dirname, "../outputs/plots");
const OUTDIR  = path.join(__dirname, "../outputs");

// ── Page geometry (A4, 0.75-inch margins = 1080 DXA) ─────────────────────────
const PW = 11906, PH = 16838;
const M  = 1080;                                       // 0.75 in = 1080 DXA
const FULL_W  = PW - M * 2;                           // 9746 DXA (full content width)
const COL_GAP = 360;                                   // gap between columns
const COL_W   = Math.floor((FULL_W - COL_GAP) / 2);  // 4693 DXA per column

// ── Typography ────────────────────────────────────────────────────────────────
const F   = "Times New Roman";
const S9  = 18;    // 9 pt  (half-points)
const S10 = 20;    // 10 pt
const S24 = 48;    // 24 pt  (title)

// ── Spacing ───────────────────────────────────────────────────────────────────
const spBody  = { line: 240, lineRule: "auto", before: 40,  after: 40  };
const spTight = { line: 240, lineRule: "auto", before: 20,  after: 20  };
const spSect  = { line: 240, lineRule: "auto", before: 120, after: 40  };
const spFig   = { line: 240, lineRule: "auto", before: 20,  after: 120 };
const spAbst  = { line: 240, lineRule: "auto", before: 60,  after: 40  };
const spIdx   = { line: 240, lineRule: "auto", before: 20,  after: 60  };
const spRef   = { line: 240, lineRule: "auto", before: 30,  after: 30  };

// ── Load images ───────────────────────────────────────────────────────────────
function loadImg(p) {
  try { return fs.readFileSync(p); } catch { return null; }
}
const IMG = {
  vis      : loadImg(path.join(OUTDIR, "fig1_1_vis.png")),
  overall  : loadImg(path.join(PLOTS,  "overall_comparison.png")),
  predComp : loadImg(path.join(PLOTS,  "predicate_comparison.png")),
  predDelta: loadImg(path.join(PLOTS,  "predicate_delta.png")),
  trainCurve:loadImg(path.join(PLOTS,  "training_curve.png")),
  headTail : loadImg(path.join(PLOTS,  "head_tail.png")),
  predFreq : loadImg(path.join(PLOTS,  "predicate_frequency.png")),
};
const DIM = {
  vis      : [400,  225],
  overall  : [950,  682],
  predComp : [1996, 875],
  predDelta: [1774, 759],
  trainCurve:[1285, 706],
  headTail : [1340, 622],
  predFreq : [1741, 759],
};

// ═══════════════════════════════════════════════════════════════════════════════
//  PRIMITIVE HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/** Plain body paragraph — 10 pt, justified */
function body(text, opts = {}) {
  return new Paragraph({
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
    spacing: spBody,
    indent: opts.hanging ? { left: 480, hanging: 480 } : undefined,
    children: [new TextRun({
      text, font: F, size: S10,
      bold: opts.bold || false,
      italics: opts.italic || false
    })]
  });
}

/** Centered paragraph */
function cen(text, size, bold = false, spacing = spTight) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing,
    children: [new TextRun({ text, font: F, size, bold })]
  });
}

/** Empty gap */
function gap(n = 1) {
  return Array.from({ length: n }, () =>
    new Paragraph({ spacing: spTight, children: [new TextRun({ text: "", font: F, size: S10 })] })
  );
}

/**
 * IEEE section heading — centered bold uppercase
 * e.g.  section("I", "INTRODUCTION")  →  "I.  INTRODUCTION"
 *        section("REFERENCES")         →  "REFERENCES"
 */
function section(num, title) {
  const text = title ? `${num}.  ${title}` : num;
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: spSect,
    children: [new TextRun({ text, font: F, size: S10, bold: true })]
  });
}

/**
 * IEEE inline subsection — bold-italic label + colon + body text in ONE paragraph.
 * e.g.  "A.  Clean Embedding Guarantee:  CLIP visual features for training..."
 * If firstBody is empty string, only the heading is rendered (for subsecs that
 * start with a figure or table before any prose).
 */
function subsec(label, title, firstBody = "") {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: spBody,
    children: [
      new TextRun({ text: `${label}.  ${title}:  `, font: F, size: S10, bold: true, italics: true }),
      ...(firstBody ? [new TextRun({ text: firstBody, font: F, size: S10 })] : [])
    ]
  });
}

/**
 * Figure — scales image to fit within maxW pixels.
 * Caption is rendered 9 pt italic below.
 */
function fig(key, caption, maxW = 290) {
  const data = IMG[key];
  const [ow, oh] = DIM[key];
  const scale = Math.min(maxW / ow, 220 / oh, 1.0);
  const w = Math.round(ow * scale);
  const h = Math.round(oh * scale);

  const imgEl = data
    ? new ImageRun({ type: "png", data, transformation: { width: w, height: h } })
    : new TextRun({ text: `[Figure: ${caption}]`, font: F, size: S9, italics: true });

  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: spTight, children: [imgEl] }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: spFig,
      children: [new TextRun({ text: caption, font: F, size: S9, italics: true })]
    })
  ];
}

/** Architecture diagram placeholder */
function figPlaceholder(label, desc) {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: 240, lineRule: "auto", before: 80, after: 40 },
      children: [new TextRun({
        text: `[ ${label} — Architecture Diagram ]`,
        font: F, size: S9, italics: true, color: "666666"
      })]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: spFig,
      children: [new TextRun({ text: desc, font: F, size: S9, italics: true })]
    })
  ];
}

/**
 * Simple bordered table.
 * tableW defaults to COL_W so it fits inside one column of a two-column layout.
 */
function tbl(headers, rows, colWidths, tableW) {
  const W = tableW || COL_W;
  const bd = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
  const defaultColW = Math.floor(W / headers.length);
  const widths = colWidths || Array(headers.length).fill(defaultColW);

  const mkCell = (text, isHdr, colIdx) => new TableCell({
    borders: { top: bd, bottom: bd, left: bd, right: bd },
    width: { size: widths[colIdx], type: WidthType.DXA },
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    shading: { fill: isHdr ? "DDDDDD" : "FFFFFF", type: ShadingType.CLEAR },
    children: [new Paragraph({
      alignment: AlignmentType.CENTER, spacing: spTight,
      children: [new TextRun({ text, font: F, size: S9, bold: isHdr })]
    })]
  });

  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => mkCell(h, true, i)) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => mkCell(c, false, i)) }))
    ]
  });
}

/** Thin horizontal rule */
function rule() {
  return new Paragraph({
    spacing: spTight,
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "000000" } },
    children: [new TextRun({ text: "", font: F, size: S10 })]
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 1 CONTENT  (single-column: title + author block)
// ═══════════════════════════════════════════════════════════════════════════════

function buildSection1() {
  const HALF = Math.floor(FULL_W / 2);
  const nb   = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

  const THIRD = Math.floor(FULL_W / 3);

  /** One author column cell — no email; optional designation below name */
  const authorCell = (name, dept, inst, city, designation = "") => {
    const children = [
      new Paragraph({
        alignment: AlignmentType.CENTER, spacing: spTight,
        children: [new TextRun({ text: name, font: F, size: S10 })]
      }),
    ];
    if (designation) {
      children.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: spTight,
        children: [new TextRun({ text: designation, font: F, size: S9, italics: true })]
      }));
    }
    children.push(
      new Paragraph({
        alignment: AlignmentType.CENTER, spacing: spTight,
        children: [new TextRun({ text: dept, font: F, size: S9, italics: true })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER, spacing: spTight,
        children: [new TextRun({ text: inst, font: F, size: S9, italics: true })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER, spacing: spTight,
        children: [new TextRun({ text: city, font: F, size: S9, italics: true })]
      }),
    );
    return new TableCell({
      borders: { top: nb, bottom: nb, left: nb, right: nb },
      width: { size: THIRD, type: WidthType.DXA },
      margins: { top: 40, bottom: 40, left: 60, right: 60 },
      children,
    });
  };

  const authorTable = new Table({
    width: { size: FULL_W, type: WidthType.DXA },
    columnWidths: [THIRD, THIRD, THIRD],
    rows: [new TableRow({
      children: [
        authorCell(
          "Harshul Bagri",
          "Dept. of Computing Technologies",
          "SRM Institute of Science and Technology",
          "Kattankulathur, Tamil Nadu — 603 203, India"
        ),
        authorCell(
          "Shikhar Patel",
          "Dept. of Computing Technologies",
          "SRM Institute of Science and Technology",
          "Kattankulathur, Tamil Nadu — 603 203, India"
        ),
        authorCell(
          "Dr. C. Muralidharan",
          "Dept. of Computing Technologies",
          "SRM Institute of Science and Technology",
          "Kattankulathur, Tamil Nadu — 603 203, India",
          "Assistant Professor"
        ),
      ]
    })]
  });

  return [
    ...gap(1),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: 240, lineRule: "auto", before: 60, after: 20 },
      children: [new TextRun({
        text: "Improving Visual Relationship Detection with Prior-Guided",
        font: F, size: S24, bold: true
      })]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: 240, lineRule: "auto", before: 20, after: 80 },
      children: [new TextRun({
        text: "Learning and Clean Embedding Optimization for Explainable Scene Understanding",
        font: F, size: S24, bold: true
      })]
    }),
    authorTable,
    ...gap(1),
    rule(),
  ];
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 2 CONTENT  (two-column: abstract → references)
// ═══════════════════════════════════════════════════════════════════════════════

function abstractBlock() {
  return [
    new Paragraph({
      alignment: AlignmentType.JUSTIFIED, spacing: spAbst,
      children: [
        new TextRun({ text: "Abstract\u2014", font: F, size: S9, bold: true, italics: true }),
        new TextRun({ text: "Scene understanding demands more than object recognition — it requires identifying how objects relate to each other within a shared spatial context. Visual relationship detection (VRD) addresses this by predicting structured (subject, predicate, object) triplets from images, but practical systems have to contend with at least three persistent difficulties: a severely imbalanced predicate distribution, noisy or approximate visual features when training data is partially unavailable, and the underutilisation of structured commonsense knowledge about object co-occurrence. This paper introduces a VRD framework built around three design choices that address each of these difficulties directly. First, we enforce a clean embedding guarantee: every training sample must have a real, on-disk CLIP ViT-B/32 visual embedding rather than a random fallback, reducing the effective training set but eliminating a subtle source of feature-space contamination. Second, we train with Focal Loss (\u03b3 = 2.0) to counteract the gradient dominance of high-frequency predicates. Third, we integrate a learnable co-occurrence prior through a two-parameter FusionModule that blends visual logits with dataset-derived prior vectors in a differentiable, temperature-controlled manner. On a filtered Visual Genome benchmark (18,244 training triplets, 16 predicate classes), the visual-only configuration achieves R@50 = 73.01% and mR@50 = 30.28%. Adding the prior fusion improves these to R@50 = 76.73% and mR@50 = 34.30%, with the largest gains on mid-frequency predicates such as \u2018carrying\u2019 (+13.3 pp) and \u2018holding\u2019 (+11.1 pp). The system operates open-vocabulary at inference time, using YOLO-World for detection and a frozen CLIP encoder throughout.", font: F, size: S9 })
      ]
    }),
    new Paragraph({
      alignment: AlignmentType.JUSTIFIED, spacing: spIdx,
      children: [
        new TextRun({ text: "Index Terms\u2014", font: F, size: S9, bold: true, italics: true }),
        new TextRun({ text: "Visual relationship detection, scene graphs, CLIP, Focal Loss, co-occurrence prior, open-vocabulary detection.", font: F, size: S9, italics: true })
      ]
    }),
    rule(),
    ...gap(1)
  ];
}

function secIntro() {
  return [
    section("I", "INTRODUCTION"),
    body("Understanding what objects are and where they sit is only part of what makes visual perception useful. Knowing that a person is riding a bicycle, or that a cup is on a table, or that a dog is behind a chair requires connecting object identities through relational predicates — and that capacity underpins nearly every downstream task from image captioning and visual question answering to robotic planning and autonomous navigation. Scene graphs, which represent images as attributed graphs of (subject, predicate, object) triplets, offer a compact symbolic interface between raw pixel data and these higher-level applications."),
    body("Building reliable VRD systems is difficult for reasons that compound. Predicate distributions in large annotated datasets are deeply skewed: in Visual Genome [1], the single predicate \u2018on\u2019 accounts for over 60% of all training annotations while nine tail predicates share fewer than 1,500 samples between them. A straightforward cross-entropy classifier fits this distribution faithfully, which means it dedicates most of its representational capacity to a handful of spatially generic relationships and essentially ignores the semantically richer tail. This frequency-bias problem has been identified and studied repeatedly [7, 8], and it continues to limit the practical utility of VRD systems in contexts where nuanced relationships matter."),
    body("A second, less-discussed difficulty arises when training on datasets where source images are only partially available. Pre-computing visual features — a standard efficiency measure — requires that images are accessible on disk. When an image is missing, some implementations substitute a random noise vector to avoid a crash. The result is a training set where a significant fraction of samples carry meaningless features, which quietly degrades learned discriminators without producing any obvious error signal during training."),
    body("A third limitation is the near-complete neglect of co-occurrence priors. The fact that certain object pairs overwhelmingly appear with specific predicates — that (person, bicycle) pairs are almost always \u2018riding\u2019, that (window, wall) pairs are almost always \u2018on\u2019 — is a source of nearly free signal that most purely visual models discard entirely."),
    body("We present a VRD system that addresses each of these three issues through targeted design decisions without resorting to large-scale architectural complexity or expensive training procedures. The full inference pipeline processes images through YOLO-World open-vocabulary detection [20], extracts subject and object crop embeddings with a frozen CLIP ViT-B/32 encoder [3], computes 13-dimensional geometric features between bounding boxes, and classifies predicate relationships with a compact MLP. As shown in Fig. 1, a co-occurrence prior vector can optionally be fused with the visual logits through a learnable FusionModule."),
    ...figPlaceholder("Fig. 1", "Fig. 1. Full system architecture: YOLO-World detection \u2192 frozen CLIP encoding \u2192 RelationHead MLP. FusionModule (Exp B) blends visual logits with co-occurrence prior via learned \u03b1 and \u03c4."),
    body("The contributions of this paper are as follows:"),
    body("(1) A clean embedding guarantee enforced through an availability filter that removes any training triplet lacking a real on-disk CLIP embedding, eliminating random-vector contamination and providing a rigorous, noise-free training set despite a roughly 46% reduction in sample count."),
    body("(2) A systematic evaluation of Focal Loss for predicate classification under extreme class imbalance, demonstrating that \u03b3 = 2.0 provides a stable and effective baseline for rare-predicate recall."),
    body("(3) A lightweight, two-parameter FusionModule that integrates a co-occurrence prior into the prediction pipeline in a differentiable and inspectable manner, enabling explicit attribution of each prediction to visual, spatial, or statistical evidence."),
    body("On a filtered Visual Genome test set, the visual-only baseline achieves R@50 = 73.01% and mR@50 = 30.28%. Adding the prior fusion yields R@50 = 76.73% (+3.72 pp) and mR@50 = 34.30% (+4.02 pp). The improvement is non-negative for all 16 predicate classes."),
    ...gap(1)
  ];
}

function secRelated() {
  return [
    section("II", "RELATED WORK"),
    subsec("A", "Visual Relationship Detection",
      "The field of VRD was effectively launched by Lu et al. [2], whose system combined VGG region features with word2vec language priors and demonstrated that linguistic similarity between object labels can transfer predicate confidence across semantically related triplets. This two-stream approach — visual features complemented by relational priors — remains the dominant architectural paradigm a decade later."),
    body("Graph neural network extensions followed quickly. Xu et al. applied iterative message-passing over scene graph nodes, using each predicted object\u2019s features to refine its neighbours\u2019 representations across several rounds. Zellers et al. [7] showed, counterintuitively, that a frequency-bias baseline — predicting the most common predicate for each object-category pair — could outperform architecturally sophisticated models, which exposed a fundamental evaluation flaw: standard R@K rewards head-class recall so heavily that models exploiting dataset statistics can appear stronger than they are."),
    subsec("B", "Debiasing and Long-Tail Evaluation",
      "Tang et al. [8] responded to the frequency-bias critique with a causal inference framework that estimates and subtracts the context-free component of predicate predictions. Their approach, combined with the introduction of mR@K as a metric that averages recall separately per predicate class, shifted the field\u2019s evaluation standard toward unbiased measurement. Our use of Focal Loss [4] operates at the training level rather than through post-hoc score adjustment, but shares the same motivation: reducing the gradient dominance of head-class examples so that tail predicates receive meaningful learning signal."),
    subsec("C", "Vision-Language Pre-training for Open-Vocabulary Tasks",
      "The CLIP model [3] introduced contrastive pre-training on 400 million web image-text pairs, producing a shared visual-semantic embedding space that generalises to unseen concepts. Subsequent work adapted CLIP to open-vocabulary detection (VILD [13], GLIP [11], OWL-ViT [19]) and YOLO-World [20] extended this to real-time inference with arbitrary text prompts. For VRD specifically, Zareian et al. [10] aligned predicate classifiers with CLIP text embeddings, improving generalisation to compositionally novel triplets. Our approach differs in keeping the CLIP encoder frozen throughout, training only a shallow MLP relation head."),
    subsec("D", "Prior Knowledge in Relation Classification",
      "The use of statistical co-occurrence information to improve relation classification predates deep learning [2], but parametric integration of such priors remains under-explored. Most recent systems either ignore priors entirely or incorporate them implicitly through frequency-biased classifiers. Our FusionModule makes the prior contribution explicit and learnable, which has the practical advantage of interpretability: the learned mixing weight \u03b1 quantifies how strongly the model trusts visual evidence over statistical expectation for any given input."),
    ...gap(1)
  ];
}

function secMethod() {
  return [
    section("III", "METHODOLOGY"),
    body("The proposed system consists of four stages: open-vocabulary detection, visual encoding with a clean embedding guarantee, spatial feature computation, and predicate classification with optional prior fusion."),
    subsec("A", "Clean Embedding Guarantee",
      "CLIP visual features for training triplets are pre-computed by encoding bounding box crops from the original Visual Genome images using a frozen CLIP ViT-B/32 encoder. Each embedding is stored in a sharded disk cache keyed by a canonical identifier derived from the image ID and bounding box coordinates. A single canonical key-generation function is imported by both the pre-computation script and the training dataset loader, eliminating the key-divergence bug that caused random fallback embeddings in our initial implementation."),
    body("At training time, the dataset loader applies a simple availability filter: any triplet for which no real embedding exists on disk is silently dropped. This reduces the training set from 33,741 annotated triplets to 18,244 (54.1% retention). The predicate frequency distribution after filtering is shown in Fig. 2. Although the data reduction is substantial, we argue that the quality gain outweighs the quantity loss: random noise vectors provide no discriminative gradient signal and actively misrepresent the visual feature space for affected training examples."),
    ...fig("predFreq", "Fig. 2. Predicate frequency in filtered training set (log scale). Dashed line at 800 samples separates head (\u2265800) from tail (<800) predicates.", 290),
    subsec("B", "Focal Loss for Predicate Imbalance",
      "Given the 3,415:1 sample ratio between the most and least frequent predicates (Fig. 2), a standard cross-entropy loss is heavily dominated by head-class gradient contributions. We adopt Focal Loss [4]:"),
    body("FL(pt) = \u2212(1 \u2212 pt)^\u03b3 \u00b7 log(pt)", { center: true }),
    body("where pt is the softmax probability assigned to the ground-truth class and \u03b3 controls the rate at which easy examples are down-weighted. Setting \u03b3 = 2.0 concentrates gradient mass on examples where the model\u2019s current prediction is uncertain or wrong — which, in practice, corresponds to under-represented tail predicates and hard ambiguous cases in mid-frequency classes."),
    subsec("C", "Relation Head MLP",
      "For each candidate subject-object pair, two 512-dimensional L2-normalised CLIP embeddings and a 13-dimensional spatial feature vector are concatenated into a 1037-dimensional input. The spatial vector encodes relative centre offsets (2-D), log area ratio (1-D), IoU (1-D), area fractions (2-D), aspect ratios (2-D), centre positions (4-D), and an above/below flag (1-D)."),
    body("This input passes through a three-layer MLP: Linear(1037\u2192512) \u2192 ReLU \u2192 Dropout(0.3) \u2192 Linear(512\u2192256) \u2192 ReLU \u2192 Dropout(0.3) \u2192 Linear(256\u219216). The output is a 16-dimensional logit vector over the predicate vocabulary. The total trainable parameter count is 666,896 \u2014 a fraction of what scene graph generation transformers require."),
    subsec("D", "Prior-Guided FusionModule",
      "For any subject category s and object category o observed in training, we estimate a co-occurrence prior vector p(s,o) \u2208 R\u00b9\u2076 by computing the relative frequency of each predicate across all training triplets involving that category pair. For unseen pairs, a uniform distribution is used as a conservative fallback."),
    body("The FusionModule blends the visual logit vector v \u2208 R\u00b9\u2076 with p through:"),
    body("output = \u03b1 \u00b7 v  +  (1 \u2212 \u03b1) \u00b7 (p / \u03c4)", { center: true }),
    body("where \u03b1 and log\u03c4 are learnable scalars, initialised to 1.0 and 0.0 respectively. At initialisation the module is an identity function outputting v exactly, so Experiment B begins training from the same point as Experiment A. The temperature \u03c4 adjusts the peakedness of the prior: \u03c4 > 1 flattens it, \u03c4 < 1 sharpens it. The total added parameter count is exactly 2."),
    ...figPlaceholder("Fig. 3", "Fig. 3. FusionModule: visual logits v and prior vector p are blended via learned \u03b1 (prior weight) and log\u03c4 (temperature). Init at \u03b1=1, \u03c4=1 (identity). Adds only 2 trainable parameters."),
    ...gap(1)
  ];
}

function secExperiments() {
  return [
    section("IV", "EXPERIMENTAL SETUP"),
    subsec("A", "Dataset",
      "We use the Visual Genome dataset [1], which provides densely annotated scene graphs for 108,077 images. We filter relationship annotations to retain only the 16 most informative predicates in our target vocabulary, then split images 80% / 10% / 10% into train, validation, and test sets at the image level. After applying the clean embedding filter, the training set contains 18,244 triplets. Table I summarises the dataset statistics."),
    ...gap(1),
    cen("TABLE I", S9, true, spTight),
    cen("Dataset Statistics After Filtering", S9, false, spTight),
    tbl(
      ["Split", "Annotated", "After Filter", "Retained"],
      [["Train", "33,741", "18,244", "54.1%"],
       ["Val",   "7,070",  "3,885",  "54.9%"],
       ["Test",  "7,070",  "3,887",  "54.9%"]],
      [1000, 1300, 1300, 1093]
    ),
    ...gap(1),
    subsec("B", "Training Configuration",
      "Both experiments use AdamW [9] with weight decay 0.01, a cosine annealing learning rate schedule from 1\u00d710\u207b\u00b3 to 1\u00d710\u207b\u2076 over 30 epochs, batch size 256, and a fixed random seed of 42. All CLIP embeddings are read from the on-disk cache, so no CLIP inference occurs during training. Table II lists hyperparameters."),
    ...gap(1),
    cen("TABLE II", S9, true, spTight),
    cen("Training Hyperparameters", S9, false, spTight),
    tbl(
      ["Hyperparameter", "Value"],
      [["Optimiser",      "AdamW (wd=0.01)"],
       ["Learning rate",  "1e-3 \u2192 1e-6 (cosine)"],
       ["Batch size",     "256"],
       ["Epochs",         "30"],
       ["Training loss",  "Focal Loss (\u03b3=2.0)"],
       ["Dropout",        "0.3"],
       ["Random seed",    "42"]],
      [2350, 2343]
    ),
    ...gap(1),
    subsec("C", "Evaluation Protocol",
      "We evaluate under the predicate classification (PredCls) protocol, in which ground-truth bounding boxes and object category labels are supplied to the model; only the predicate must be predicted. This isolates predicate classification accuracy from any errors introduced by the detection stage. We report Recall@K (R@K) for K \u2208 {20, 50, 100} and mean-Recall@50 (mR@50), which averages R@50 separately over each of the 16 predicate classes, giving equal weight to rare and frequent predicates."),
    subsec("D", "Ablation Design",
      "Experiment A is the visual-only baseline: clean CLIP embeddings + 13-D spatial features + MLP relation head + Focal Loss. No co-occurrence information is used. This isolates the contribution of the clean embedding pipeline and Focal Loss from prior-guided learning."),
    body("Experiment B is the full proposed system: Experiment A with the addition of the learnable FusionModule and co-occurrence prior. The only structural difference between A and B is the two-parameter FusionModule, so any performance gap is cleanly attributable to the prior fusion."),
    ...gap(1)
  ];
}

function secResults() {
  return [
    section("V", "RESULTS AND ANALYSIS"),
    subsec("A", "Overall Performance",
      "Table III reports the complete quantitative results for both experiments. Experiment B improves over Experiment A on every reported metric, with gains of +3.72 pp on R@50 and +4.02 pp on mR@50. The improvement in mR@50 is slightly larger than the R@50 gain, which reflects the prior\u2019s disproportionate benefit to mid-frequency and tail predicates."),
    ...gap(1),
    cen("TABLE III", S9, true, spTight),
    cen("Overall Quantitative Results", S9, false, spTight),
    tbl(
      ["Metric", "Exp A", "Exp B", "\u0394"],
      [["R@20",      "72.80%", "76.42%", "+3.62 pp"],
       ["R@50",      "73.01%", "76.73%", "+3.72 pp"],
       ["R@100",     "73.01%", "76.73%", "+3.72 pp"],
       ["mR@50",     "30.28%", "34.30%", "+4.02 pp"],
       ["Val top-1", "80.88%", "83.86%", "+2.98 pp"]],
      [1050, 1100, 1100, 1443]
    ),
    ...gap(1),
    body("Fig. 4 visualises R@50 and mR@50 side by side. The gap between the two metrics within each experiment is substantial — over 42 percentage points — which highlights how heavily R@K is inflated by head-class accuracy. The near-equality of R@50 and R@100 confirms that ground-truth predicates are almost always ranked within the top 50 by both systems."),
    ...fig("overall", "Fig. 4. R@50 and mR@50 comparison. Exp B (prior-guided) outperforms Exp A (visual-only) on both metrics. The mR@50 gap is larger, reflecting stronger gains on mid-frequency predicates.", 290),
    subsec("B", "Per-Predicate Analysis",
      "Fig. 5 shows per-predicate R@50 for both experiments, and Fig. 6 shows the improvement delta (Exp B minus Exp A) for each class."),
    ...fig("predComp", "Fig. 5. Per-predicate R@50 comparison. Exp B consistently meets or exceeds Exp A for all 16 predicates. Largest gains: \u2018carrying\u2019 (+13.3 pp), \u2018holding\u2019 (+11.1 pp), \u2018behind\u2019 (+10.1 pp).", 290),
    ...fig("predDelta", "Fig. 6. Per-predicate improvement delta (Exp B \u2212 Exp A). All deltas are non-negative: the FusionModule learns to down-weight misleading prior signal rather than blindly accepting it.", 290),
    body("Mid-frequency predicates show the strongest absolute gains. \u2018Carrying\u2019 (+13.3 pp) benefits because the (person, bag) and (person, backpack) pairs strongly concentrate around this predicate in Visual Genome statistics. \u2018Holding\u2019 (+11.1 pp) and \u2018behind\u2019 (+10.1 pp) follow a similar pattern: the visual signal from CLIP partially separates these classes from their neighbours, and the prior provides the additional discriminative push needed."),
    body("Head predicates — \u2018on\u2019, \u2018has\u2019, \u2018wearing\u2019 — gain modestly (+2.8 to +4.9 pp). At 70\u201394% baseline recall, headroom is limited. The four ultra-rare predicates (\u2018walking on\u2019 and \u2018riding\u2019, each with 6 training samples) remain at 0.0% R@50 in both experiments. Focal Loss and prior fusion together cannot overcome a training sample count this small."),
    subsec("C", "Head vs. Tail Predicate Grouping",
      "Grouping predicates into head (\u2265800 training samples, 7 classes) and tail (<800, 9 classes): head group improves 58.5%\u219263.1% (+4.6 pp, +7.9% relative), tail group 15.7%\u219219.8% (+4.1 pp, +26.1% relative). The relative benefit is substantially larger for tail predicates, consistent with the prior filling the gap left by insufficient visual training data."),
    ...fig("headTail", "Fig. 7. Head vs. tail mean R@50. Tail predicates benefit relatively more from prior fusion (+26.1% relative vs +7.9% for head). Both groups improve.", 290),
    subsec("D", "Training Dynamics",
      "Fig. 8 shows training and validation loss curves for both experiments. Both converge smoothly over 30 epochs without oscillation or overfitting. Experiment B reaches a lower final training loss (0.0148 vs 0.0170), suggesting the prior provides additional gradient signal on samples where visual evidence is ambiguous. Experiment B reaches its best validation accuracy at epoch 30 (the final epoch), indicating the FusionModule continues to benefit from training signal throughout the cosine schedule."),
    ...fig("trainCurve", "Fig. 8. Training loss (solid) and validation loss (dashed) over 30 epochs. Exp B converges to lower training loss (0.0148 vs 0.0170). No significant overfitting is observed.", 290),
    ...gap(1)
  ];
}

function secDiscussion() {
  return [
    section("VI", "DISCUSSION"),
    body("Three observations from the results deserve emphasis beyond what the aggregate numbers convey."),
    body("First, clean embeddings matter independently of model complexity. Removing roughly 46% of the training set by enforcing the availability filter might appear to be a substantial sacrifice, yet the baseline (Experiment A) achieves 73.01% R@50 with only 667K trainable parameters. The reason is likely that the remaining 18,244 samples are internally consistent and provide unambiguous visual features for every example, whereas a dataset contaminated with random noise vectors would require the classifier to learn which features to ignore. Clean data is, in this respect, a form of implicit regularisation."),
    body("Second, the two-parameter FusionModule achieves its improvement with a level of transparency rarely seen in relation classification. Because \u03b1 is a named, inspectable scalar, the degree to which the model trusts visual evidence over prior statistics can be read off directly. Similarly, the prior vector for any given object-category pair is interpretable as a probability distribution over predicates. Together, these properties make individual predictions partially attributable: for a (person, bag) pair predicted as \u2018carrying\u2019, one can note both the CLIP visual embedding distance and the prior probability P(carrying | person, bag)."),
    body("Third, the zero-recall ultra-rare predicates represent a genuine research gap that neither Focal Loss nor prior fusion resolves. With 6 training samples for \u2018walking on\u2019 and \u2018riding\u2019, no amount of gradient reweighting or statistical smoothing can manufacture meaningful visual discriminators. The most promising directions are: (a) retrieving additional examples from large-scale web datasets using text queries for the target relationship, (b) applying cross-modal data augmentation by compositing subject and object crops into synthetic training images, and (c) using CLIP text embeddings of sentence templates as class prototypes to allow zero-shot transfer from related predicates."),
    body("A broader limitation of this study is the use of the PredCls evaluation protocol. Providing ground-truth boxes removes the detection stage from the evaluation chain and produces recall figures that are not achievable in real deployment, where YOLO-World detection recall on Visual Genome images is approximately 70\u201375%."),
    ...gap(1)
  ];
}

function secConclusion() {
  return [
    section("VII", "CONCLUSION"),
    body("We proposed a visual relationship detection system for explainable scene understanding, built around three concrete design choices: a clean CLIP embedding guarantee, Focal Loss training for predicate class imbalance, and a learnable two-parameter FusionModule that integrates co-occurrence prior knowledge with visual logits in an interpretable manner. On a filtered Visual Genome benchmark, the visual-only baseline achieves R@50 = 73.01% and mR@50 = 30.28%, while adding the prior fusion pushes these to R@50 = 76.73% and mR@50 = 34.30%, with non-negative improvement across all 16 predicate classes."),
    body("The results support the view that data cleanliness and simple but well-motivated training choices can close a significant fraction of the performance gap between lightweight and heavyweight systems. The FusionModule\u2019s transparency — its learned mixing weight directly quantifies visual-versus-statistical reliance — is a step toward the explainable scene understanding the broader community needs."),
    body("Future directions include: end-to-end evaluation with YOLO-World, extending the predicate vocabulary to true open-vocabulary coverage via CLIP text prototypes, addressing ultra-rare predicates through cross-modal augmentation, and replacing the MLP relation head with a lightweight cross-attention module that captures scene-level context among all object pairs simultaneously."),
    ...gap(1)
  ];
}

function secReferences() {
  const refs = [
    "[1]  R. Krishna et al., \u201cVisual Genome: Connecting language and vision using crowdsourced dense image annotations,\u201d IJCV, vol. 123, no. 1, pp. 32\u201373, 2017.",
    "[2]  C.-Y. Lu, R. Krishna, M. Bernstein, and L. Fei-Fei, \u201cVisual relationship detection with language priors,\u201d ECCV, pp. 852\u2013869, 2016.",
    "[3]  A. Radford et al., \u201cLearning transferable visual models from natural language supervision,\u201d ICML, pp. 8748\u20138763, 2021.",
    "[4]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Doll\u00e1r, \u201cFocal loss for dense object detection,\u201d ICCV, pp. 2980\u20132988, 2017.",
    "[5]  K. Simonyan and A. Zisserman, \u201cVery deep convolutional networks for large-scale image recognition,\u201d ICLR, 2015.",
    "[6]  A. Dosovitskiy et al., \u201cAn image is worth 16\u00d716 words: Transformers for image recognition at scale,\u201d ICLR, 2021.",
    "[7]  R. Zellers, M. Yatskar, S. Thomson, and Y. Choi, \u201cNeural motifs: Scene graph parsing with global context,\u201d CVPR, pp. 5831\u20135840, 2018.",
    "[8]  K. Tang, Y. Niu, J. Huang, J. Shi, and H. Zhang, \u201cUnbiased scene graph generation from biased training,\u201d CVPR, pp. 3716\u20133725, 2020.",
    "[9]  I. Loshchilov and F. Hutter, \u201cDecoupled weight decay regularization,\u201d ICLR, 2019.",
    "[10] A. Zareian, K. D. Rosa, D. H. Hu, and S.-F. Chang, \u201cOpen-vocabulary scene graphs,\u201d CVPR, pp. 2979\u20132990, 2022.",
    "[11] L. Li et al., \u201cGrounded language-image pre-training,\u201d CVPR, pp. 10965\u201310975, 2022.",
    "[12] B. Cheng et al., \u201cMasked-attention mask transformer for universal image segmentation,\u201d CVPR, pp. 1290\u20131300, 2022.",
    "[13] X. Gu, T.-Y. Lin, W. Kuo, and Y. Cui, \u201cOpen-vocabulary object detection via vision and language knowledge distillation,\u201d ICLR, 2022.",
    "[14] I. Misra, A. Gupta, and M. Hebert, \u201cFrom red wine to red tomato: Composition with context,\u201d CVPR, pp. 1160\u20131169, 2017.",
    "[15] K. He, X. Zhang, S. Ren, and J. Sun, \u201cDeep residual learning for image recognition,\u201d CVPR, pp. 770\u2013778, 2016.",
    "[16] M. Minderer et al., \u201cSimple open-vocabulary object detection,\u201d ECCV, pp. 728\u2013755, 2022.",
    "[17] T. Lv et al., \u201cYOLO-World: Real-time open-vocabulary object detection,\u201d CVPR, 2024.",
    "[18] J. Redmon and A. Farhadi, \u201cYOLOv3: An incremental improvement,\u201d arXiv:1804.02767, 2018.",
    "[19] G. Li et al., \u201cRelation-aware graph attention network for visual question answering,\u201d ICCV, 2019.",
    "[20] T. Lv, Y. Chen, Y. Xu et al., \u201cYOLO-World: Real-time open-vocabulary object detection,\u201d CVPR, Seattle, Jun. 2024."
  ];

  return [
    section("REFERENCES"),
    ...refs.map(r => new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: spRef,
      indent: { left: 480, hanging: 480 },
      children: [new TextRun({ text: r, font: F, size: S9 })]
    }))
  ];
}

// ═══════════════════════════════════════════════════════════════════════════════
//  FOOTER
// ═══════════════════════════════════════════════════════════════════════════════
function pageNumFooter() {
  return new Footer({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: F, size: S9 })]
    })]
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
//  ASSEMBLE AND SAVE
// ═══════════════════════════════════════════════════════════════════════════════
console.log("Assembling IEEE two-column research paper...");

const section1Children = buildSection1();

const section2Children = [
  ...abstractBlock(),
  ...secIntro(),
  ...secRelated(),
  ...secMethod(),
  ...secExperiments(),
  ...secResults(),
  ...secDiscussion(),
  ...secConclusion(),
  ...secReferences()
];

const pageProps = {
  size:   { width: PW, height: PH },
  margin: { top: M, right: M, bottom: M, left: M }
};

const doc = new Document({
  sections: [
    {
      // ── Section 1: single-column (title + authors) ──────────────────────────
      properties: {
        type: "continuous",
        page: pageProps
      },
      children: section1Children
    },
    {
      // ── Section 2: two-column (abstract → references) ───────────────────────
      properties: {
        page: pageProps,
        column: { space: COL_GAP, count: 2 },
        pageNumbers: { start: 1 }
      },
      footers: { default: pageNumFooter() },
      children: section2Children
    }
  ]
});

Packer.toBuffer(doc).then(buffer => {
  fs.mkdirSync(DESKTOP, { recursive: true });
  fs.writeFileSync(OUTPUT, buffer);
  console.log("\n=============================================");
  console.log("IEEE research paper generated successfully!");
  console.log(`File : ${OUTPUT}`);
  console.log(`Size : ${(buffer.length / 1024).toFixed(1)} KB`);
  console.log("=============================================");
}).catch(err => {
  console.error("ERROR:", err.message);
  process.exit(1);
});
