"""
generate_diagrams.py
====================
Generates all 11 publication-quality architecture diagrams for the IB-VRD
B.Tech report using matplotlib (no external native library required).

Run:
    cd open_vocab_vrd/
    python scripts/generate_diagrams.py

Output: outputs/diagrams/*.png  (300 DPI)
"""

import sys
import io
# Force UTF-8 output so Unicode diagram labels don't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE    = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "outputs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (B&W-safe)
# ─────────────────────────────────────────────────────────────────────────────
C_DARK   = "#1a1a1a"
C_MED    = "#333333"
C_HEAD   = "#444444"
C_BOX    = "#f0f0f0"
C_BOX2   = "#e4e4e4"
C_BOX3   = "#d4d4d4"
C_WHITE  = "#ffffff"
C_ACCENT = "#888888"
C_OUT    = "#2a2a2a"

# ─────────────────────────────────────────────────────────────────────────────
#  DRAWING PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def _fig(w, h, dpi=150):
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.axis("off"); ax.set_aspect("equal")
    fig.patch.set_facecolor(C_WHITE)
    return fig, ax

def box(ax, x, y, w, h, fill=C_BOX, ec=C_MED, lw=1.2, alpha=1.0, zorder=2, radius=0.12):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=fill, edgecolor=ec, linewidth=lw,
                       alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p

def txt(ax, x, y, s, size=8, color=C_DARK, bold=False, italic=False,
        ha="center", va="center", zorder=5, wrap=False):
    weight = "bold" if bold else "normal"
    style  = "italic" if italic else "normal"
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            fontstyle=style, ha=ha, va=va, zorder=zorder,
            fontfamily="DejaVu Sans")

def arr(ax, x1, y1, x2, y2, color=C_DARK, lw=1.2, style="-", hw=0.12, hl=0.18,
        label="", label_off=(0, 0.1)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=f"-|>,head_width={hw},head_length={hl}",
                                color=color, lw=lw, linestyle=style))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        txt(ax, mx+label_off[0], my+label_off[1], label, size=6.5, italic=True, color=C_ACCENT)

def darr(ax, x1, y1, x2, y2, label=""):
    """Dashed arrow (optional path)."""
    arr(ax, x1, y1, x2, y2, color=C_ACCENT, lw=0.8, style="--", label=label)

def hdr(ax, x, y, w, h, title, size=8.5, fill=C_DARK):
    box(ax, x, y, w, h, fill=fill, ec=C_DARK, lw=0, radius=0.1)
    txt(ax, x+w/2, y+h/2, title, size=size, color=C_WHITE, bold=True)

def save_fig(fig, name, dpi=150):
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight",
                facecolor=C_WHITE, edgecolor="none")
    plt.close(fig)
    kb = p.stat().st_size // 1024
    print(f"  OK  {name}.png  ({kb} KB)")

# ─────────────────────────────────────────────────────────────────────────────
#  1.  FIG 1.2  —  SYSTEM OVERVIEW  (16 × 5.5 inches)
# ─────────────────────────────────────────────────────────────────────────────
def fig_1_2():
    W, H = 16, 5.5
    fig, ax = _fig(W, H)

    # Outer border + title bar
    box(ax, 0.15, 0.15, W-0.3, H-0.3, fill=C_WHITE, ec=C_DARK, lw=2, radius=0.2)
    hdr(ax, 0.15, H-0.85, W-0.3, 0.7,
        "IB-VRD  SYSTEM OVERVIEW — IMAGE BASED VISUAL RELATIONSHIP DETECTION",
        size=9.5, fill=C_DARK)

    # Box 1: Input
    bx, by, bw, bh = 0.4, 1.0, 2.2, 3.4
    box(ax, bx, by, bw, bh, fill=C_BOX, ec=C_MED)
    hdr(ax, bx, by+bh-0.52, bw, 0.52, "INPUT", size=8, fill=C_MED)
    txt(ax, bx+bw/2, by+bh/2+0.1, "Raw Image", bold=True, size=9)
    txt(ax, bx+bw/2, by+bh/2-0.35, "H × W × 3  (RGB)", size=8, italic=True)
    txt(ax, bx+bw/2, by+bh/2-0.7, "JPEG / PNG", size=7.5, italic=True, color=C_ACCENT)

    # Arrow 1→2
    arr(ax, bx+bw, by+bh/2+0.3, 3.1, by+bh/2+0.3, label="full image")

    # Box 2: Feature Extraction (split into 2 inner boxes)
    m2x, m2y, m2w, m2h = 3.1, 0.45, 5.5, 4.5
    box(ax, m2x, m2y, m2w, m2h, fill=C_BOX2, ec=C_MED, lw=1.5)
    hdr(ax, m2x, m2y+m2h-0.52, m2w, 0.52, "FEATURE EXTRACTION MODULE", size=8.5, fill=C_HEAD)

    # Sub-box: YOLO-World
    box(ax, m2x+0.2, m2y+2.45, 2.4, 1.7, fill=C_BOX, ec=C_MED, lw=1)
    hdr(ax, m2x+0.2, m2y+2.45+1.7-0.42, 2.4, 0.42, "YOLO-World", size=7.5, fill=C_MED)
    txt(ax, m2x+0.2+1.2, m2y+2.45+1.08, "yolov8s-world.pt", bold=True, size=7.5)
    txt(ax, m2x+0.2+1.2, m2y+2.45+0.7, "conf ≥ 0.3  |  top-k = 10", size=7, italic=True)
    txt(ax, m2x+0.2+1.2, m2y+2.45+0.35, "Open-vocabulary detection", size=7, italic=True)

    # Sub-box: CLIP
    box(ax, m2x+0.2, m2y+0.6, 2.4, 1.7, fill=C_BOX, ec=C_MED, lw=1)
    hdr(ax, m2x+0.2, m2y+0.6+1.7-0.42, 2.4, 0.42, "CLIP ViT-B/32", size=7.5, fill=C_MED)
    txt(ax, m2x+0.2+1.2, m2y+0.6+1.08, "FROZEN encoder", bold=True, size=7.5)
    txt(ax, m2x+0.2+1.2, m2y+0.6+0.7, "512-D, L2-normalised", size=7, italic=True)
    txt(ax, m2x+0.2+1.2, m2y+0.6+0.35, "per detected crop", size=7, italic=True)

    # Sub-box: Spatial
    box(ax, m2x+2.9, m2y+0.6, 2.4, 3.55, fill=C_BOX, ec=C_MED, lw=1)
    hdr(ax, m2x+2.9, m2y+0.6+3.55-0.42, 2.4, 0.42, "Spatial Features", size=7.5, fill=C_MED)
    txt(ax, m2x+2.9+1.2, m2y+0.6+2.7, "13-D per (subj, obj) pair", bold=True, size=7.5)
    txt(ax, m2x+2.9+1.2, m2y+0.6+2.25, "Δcx, Δcy", size=7, italic=True)
    txt(ax, m2x+2.9+1.2, m2y+0.6+1.85, "log(w-ratio), log(h-ratio)", size=7, italic=True)
    txt(ax, m2x+2.9+1.2, m2y+0.6+1.45, "IoU,  area fractions", size=7, italic=True)
    txt(ax, m2x+2.9+1.2, m2y+0.6+1.05, "aspect ratios", size=7, italic=True)
    txt(ax, m2x+2.9+1.2, m2y+0.6+0.65, "→ 512+512+13 = 1037-D", size=7.5, bold=True)
    txt(ax, m2x+2.9+1.2, m2y+0.6+0.30, "concatenated feature vector", size=7, italic=True)

    # Arrow 2→3
    arr(ax, m2x+m2w, m2y+m2h/2, 9.15, m2y+m2h/2, label="1037-D feature vector")

    # Box 3: Relation Reasoning
    m3x, m3y, m3w, m3h = 9.15, 0.45, 5.5, 4.5
    box(ax, m3x, m3y, m3w, m3h, fill=C_BOX2, ec=C_MED, lw=1.5)
    hdr(ax, m3x, m3y+m3h-0.52, m3w, 0.52, "RELATION REASONING MODULE", size=8.5, fill=C_HEAD)

    # Sub-box MLP
    box(ax, m3x+0.2, m3y+2.45, m3w-0.4, 1.7, fill=C_BOX, ec=C_MED, lw=1)
    hdr(ax, m3x+0.2, m3y+2.45+1.7-0.42, m3w-0.4, 0.42, "RelationHead MLP", size=7.5, fill=C_MED)
    txt(ax, m3x+0.2+(m3w-0.4)/2, m3y+2.45+1.08, "1037 → 512 → 256 → 16", bold=True, size=8)
    txt(ax, m3x+0.2+(m3w-0.4)/2, m3y+2.45+0.7, "GELU activation  |  Dropout(0.3)", size=7, italic=True)
    txt(ax, m3x+0.2+(m3w-0.4)/2, m3y+2.45+0.35, "666,896 trainable parameters (Exp A)", size=7, italic=True)

    # Sub-box FusionModule
    box(ax, m3x+0.2, m3y+0.6, m3w-0.4, 1.7, fill=C_BOX, ec=C_MED, lw=1)
    hdr(ax, m3x+0.2, m3y+0.6+1.7-0.42, m3w-0.4, 0.42,
        "FusionModule  (Experiment B only)", size=7.5, fill=C_MED)
    txt(ax, m3x+0.2+(m3w-0.4)/2, m3y+0.6+1.08, "output = v + α · (p / τ)", bold=True, size=8)
    txt(ax, m3x+0.2+(m3w-0.4)/2, m3y+0.6+0.7, "prior_vec p from llm_prior.json  (16-D)", size=7, italic=True)
    txt(ax, m3x+0.2+(m3w-0.4)/2, m3y+0.6+0.35, "α, log_τ learnable  (+2 params in Exp B)", size=7, italic=True)

    # Arrow 3→Output
    arr(ax, m3x+m3w/2, m3y, m3x+m3w/2, 0.38)
    box(ax, m3x+0.2, 0.15, m3w-0.4, 0.6, fill=C_OUT, ec=C_DARK, lw=1.5)
    txt(ax, m3x+0.2+(m3w-0.4)/2, 0.46, "TOP-K RANKED TRIPLETS", bold=True, color=C_WHITE, size=8)
    txt(ax, m3x+0.2+(m3w-0.4)/2, 0.24, "{subject,  predicate,  object,  confidence}", italic=True, color="#cccccc", size=7.5)

    return fig, "fig_1_2_system_overview"


# ─────────────────────────────────────────────────────────────────────────────
#  2.  FIG 2.1  —  TAXONOMY  (15 × 7 inches)
# ─────────────────────────────────────────────────────────────────────────────
def fig_2_1():
    W, H = 15, 7.0
    fig, ax = _fig(W, H)
    box(ax, 0.15, 0.15, W-0.3, H-0.3, fill=C_WHITE, ec=C_DARK, lw=2, radius=0.2)
    hdr(ax, 0.15, H-0.85, W-0.3, 0.7,
        "TAXONOMY OF VRD APPROACHES BY BACKBONE AND PRIOR KNOWLEDGE TYPE",
        size=9.5, fill=C_DARK)

    # Column x positions (backbone label + 4 prior types)
    cols = [0.2, 2.2, 5.2, 8.2, 11.2, 14.8]
    heads = ["Backbone \\ Prior", "None / Frequency", "Language Prior",
             "Causal Inference", "Learnable Fusion"]
    rows_y = [H-1.0, H-2.3, H-3.65, H-5.1, H-6.55]   # top y of each row
    row_h = 1.2

    # Header row
    for ci, (hd, cx, nx) in enumerate(zip(heads, cols, cols[1:])):
        hdr(ax, cx, rows_y[0]-row_h, nx-cx-0.05, row_h, hd, size=8.5, fill=C_HEAD)

    # Row labels
    row_labels = ["CNN\n(VGG / ResNet)", "GNN\n(Graph-RCNN)", "CLIP\nTransformer"]
    for ri, (rl, ry) in enumerate(zip(row_labels, rows_y[1:])):
        hdr(ax, cols[0], ry-row_h, cols[1]-cols[0]-0.05, row_h, rl, size=8, fill=C_MED)

    # Cell content: (row, col, lines, highlight)
    cells = [
        # CNN row
        (1,1, ["Neural Motifs", "(Zellers 2018)", "R@50=65.2%", "mR@50=7.9%"], False),
        (1,2, ["VRD — Lu et al.", "(2016)", "word2vec prior", "R@50≈47.9%"], False),
        (1,3, ["Unbiased SGG", "(Tang 2020)", "Causal TDE", "mR@50=14.9%"], False),
        (1,4, ["—"], False),
        # GNN row
        (2,1, ["Graph-RCNN", "(Xu et al.)", "R@50=65.2%"], False),
        (2,2, ["—"], False),
        (2,3, ["—"], False),
        (2,4, ["—"], False),
        # CLIP row
        (3,1, ["—"], False),
        (3,2, ["OvSGTR", "(Zareian 2022)", "R@50≈52%", "mR@50≈18%"], False),
        (3,3, ["—"], False),
        (3,4, ["★  IB-VRD  [This Work]",
               "Exp A: R@50=73.01%","mR@50=30.28%",
               "Exp B: R@50=76.73%","mR@50=34.30%"], True),
    ]
    for (ri, ci, lines, highlight) in cells:
        cx, nx = cols[ci], cols[ci+1]
        ry = rows_y[ri]-row_h
        f = C_BOX if not highlight else "#e0e0e0"
        ec = C_DARK if highlight else "#aaaaaa"
        lw = 2.0 if highlight else 0.8
        box(ax, cx, ry, nx-cx-0.05, row_h, fill=f, ec=ec, lw=lw)
        cy = ry + row_h/2 + (len(lines)-1)*0.08
        for ln in lines:
            bold = highlight and ln.startswith("★")
            txt(ax, (cx+nx-0.05)/2, cy, ln, size=7.5, bold=bold)
            cy -= 0.22

    # Legend
    txt(ax, 0.5, 0.22, "★ = This Work", size=8, bold=True, ha="left")
    txt(ax, 0.5, 0.08, "Note: R@50 and mR@50 under PredCls protocol. "
        "IB-VRD outperforms all listed methods on both metrics.", size=7.5, italic=True, ha="left")

    return fig, "fig_2_1_taxonomy"


# ─────────────────────────────────────────────────────────────────────────────
#  3.  FIG 4.1  —  HIGH-LEVEL SYSTEM ARCHITECTURE  (18 × 10 inches)
# ─────────────────────────────────────────────────────────────────────────────
def fig_4_1():
    W, H = 18, 10.5
    fig, ax = _fig(W, H)
    box(ax, 0.15, 0.15, W-0.3, H-0.3, fill=C_WHITE, ec=C_DARK, lw=2, radius=0.2)
    hdr(ax, 0.15, H-0.85, W-0.3, 0.7,
        "HIGH-LEVEL SYSTEM ARCHITECTURE — IB-VRD (IMAGE BASED VISUAL RELATIONSHIP DETECTION)",
        size=10, fill=C_DARK)

    # ── Three module columns ───────────────────────────────────────────────────
    MH = 8.5; MY = 0.6
    m1x, m1w = 0.3,  5.0
    m2x, m2w = 5.8,  5.0
    m3x, m3w = 11.2, 6.5

    box(ax, m1x, MY, m1w, MH, fill=C_BOX, ec=C_MED, lw=1.8)
    hdr(ax, m1x, MY+MH-0.55, m1w, 0.55, "DETECTION MODULE", size=9, fill=C_HEAD)

    box(ax, m2x, MY, m2w, MH, fill=C_BOX, ec=C_MED, lw=1.8)
    hdr(ax, m2x, MY+MH-0.55, m2w, 0.55, "ENCODING MODULE", size=9, fill=C_HEAD)

    box(ax, m3x, MY, m3w, MH, fill=C_BOX, ec=C_MED, lw=1.8)
    hdr(ax, m3x, MY+MH-0.55, m3w, 0.55, "RELATION REASONING MODULE", size=9, fill=C_HEAD)

    # ── Detection Module ───────────────────────────────────────────────────────
    # Input
    box(ax, 0.5, 9.52, 4.6, 0.58, fill=C_BOX2, ec=C_MED, lw=1)
    txt(ax, 2.8, 9.81, "Input Image  (H × W × 3)", bold=True, size=8.5)
    arr(ax, 2.8, 9.52, 2.8, 9.12)

    box(ax, 0.5, 7.8, 4.6, 1.25, fill=C_BOX2, ec=C_MED, lw=1)
    hdr(ax, 0.5, 7.8+1.25-0.38, 4.6, 0.38, "YOLO-World  (yolov8s-world.pt)", size=7.5, fill=C_MED)
    txt(ax, 2.8, 8.38, "Open-vocabulary object detector", bold=True, size=8)
    txt(ax, 2.8, 8.07, "conf_thresh = 0.3  |  top_k = 10  |  NMS IoU = 0.5", size=7.5, italic=True)
    arr(ax, 2.8, 7.8, 2.8, 7.42, label="")
    txt(ax, 3.6, 7.65, "N detections", size=7, italic=True, color=C_ACCENT, ha="left")

    box(ax, 0.5, 6.3, 4.6, 1.05, fill=C_BOX3, ec=C_MED, lw=1)
    hdr(ax, 0.5, 6.3+1.05-0.38, 4.6, 0.38, "Detection Output", size=7.5, fill=C_MED)
    txt(ax, 2.8, 6.7, "N × {label,  [x1, y1, x2, y2],  score}", size=7.5)
    txt(ax, 2.8, 6.44, "sorted by confidence (descending)", size=7, italic=True)
    arr(ax, 2.8, 6.3, 2.8, 5.85, label="")

    box(ax, 0.5, 4.8, 4.6, 1.0, fill=C_BOX2, ec=C_MED, lw=1)
    hdr(ax, 0.5, 4.8+1.0-0.38, 4.6, 0.38, "Bounding Box Crops", size=7.5, fill=C_MED)
    txt(ax, 2.8, 5.2, "Crop each detected region from image", size=7.5)
    txt(ax, 2.8, 4.95, "Resize to 224×224 (CLIP standard)", size=7, italic=True)

    # ── Encoding Module ────────────────────────────────────────────────────────
    box(ax, 6.0, 7.8, 4.6, 1.25, fill=C_BOX2, ec=C_MED, lw=1)
    hdr(ax, 6.0, 7.8+1.25-0.38, 4.6, 0.38, "CLIP ViT-B/32  (FROZEN)", size=7.5, fill=C_MED)
    txt(ax, 8.3, 8.38, "Visual Transformer encoder", bold=True, size=8)
    txt(ax, 8.3, 8.07, "L2-normalised 512-D per crop", size=7.5, italic=True)
    txt(ax, 8.3, 7.82, "requires_grad = False  throughout", size=7, italic=True)

    box(ax, 6.0, 6.3, 4.6, 1.0, fill=C_BOX2, ec=C_MED, lw=1)
    hdr(ax, 6.0, 6.3+1.0-0.38, 4.6, 0.38, "DiskCache", size=7.5, fill=C_MED)
    txt(ax, 8.3, 6.68, "SHA-256 sharded on-disk embeddings", size=7.5)
    txt(ax, 8.3, 6.44, "Pre-computed at warmup  |  ~100% hit rate", size=7, italic=True)
    arr(ax, 8.3, 7.8, 8.3, 7.32)
    arr(ax, 8.3, 6.3, 8.3, 5.85)

    box(ax, 6.0, 4.8, 4.6, 1.0, fill=C_BOX2, ec=C_MED, lw=1)
    hdr(ax, 6.0, 4.8+1.0-0.38, 4.6, 0.38, "Object Embeddings", size=7.5, fill=C_MED)
    txt(ax, 8.3, 5.18, "N × 512-D  (subject + object crops)", size=7.5, bold=True)
    txt(ax, 8.3, 4.94, "L2-normalised unit vectors", size=7, italic=True)
    arr(ax, 8.3, 4.8, 8.3, 4.22)

    # Spatial features
    box(ax, 6.0, 3.15, 4.6, 1.0, fill=C_BOX3, ec=C_MED, lw=1)
    hdr(ax, 6.0, 3.15+1.0-0.38, 4.6, 0.38, "Spatial Feature Encoder", size=7.5, fill=C_MED)
    txt(ax, 8.3, 3.53, "compute_spatial_features(subj_box, obj_box, W, H)", size=7.5)
    txt(ax, 8.3, 3.28, "13-D vector: Δcx, Δcy, log(w/h), IoU, area fracs…", size=7, italic=True)
    arr(ax, 8.3, 3.15, 8.3, 2.6)

    box(ax, 6.0, 1.55, 4.6, 1.0, fill=C_OUT, ec=C_DARK, lw=1.5)
    txt(ax, 8.3, 2.07, "Feature Concatenation", bold=True, color=C_WHITE, size=8)
    txt(ax, 8.3, 1.78, "[subj_emb ‖ obj_emb ‖ spatial]  =  1037-D", color="#dddddd", size=7.5)

    # ── Reasoning Module ───────────────────────────────────────────────────────
    box(ax, 11.4, 7.2, 6.1, 1.7, fill=C_BOX2, ec=C_MED, lw=1)
    hdr(ax, 11.4, 7.2+1.7-0.38, 6.1, 0.38, "RelationHead MLP", size=7.5, fill=C_MED)
    txt(ax, 14.45, 8.22, "LayerNorm(1037) → Linear(1037→512) → GELU → Dropout(0.3)", size=7.5, bold=True)
    txt(ax, 14.45, 7.94, "→ Linear(512→256) → GELU → Dropout(0.3) → Linear(256→16)", size=7.5, bold=True)
    txt(ax, 14.45, 7.66, "Exp A:  666,896 parameters         Exp B:  666,898 parameters", size=7, italic=True)
    txt(ax, 14.45, 7.38, "Output: 16-D visual logits  (pre-softmax)", size=7, italic=True)
    arr(ax, 14.45, 7.2, 14.45, 6.78)

    box(ax, 11.4, 5.1, 6.1, 1.62, fill=C_BOX2, ec=C_MED, lw=1)
    hdr(ax, 11.4, 5.1+1.62-0.38, 6.1, 0.38, "FusionModule  [Experiment B only]", size=7.5, fill=C_MED)
    txt(ax, 14.45, 6.06, "prior_vec p  (16-D) from llm_prior.json", size=7.5)
    txt(ax, 14.45, 5.78, "output = visual_logits + α · (p / τ)", bold=True, size=8)
    txt(ax, 14.45, 5.5, "α = exp(log_α),  τ = softplus(log_τ) + ε  →  both learnable", size=7.5, italic=True)
    txt(ax, 14.45, 5.26, "Init: α=1, τ=1  →  output = visual_logits  (identity at start)", size=7, italic=True)
    arr(ax, 14.45, 5.1, 14.45, 4.64)

    box(ax, 11.4, 3.55, 6.1, 1.02, fill=C_BOX2, ec=C_MED, lw=1)
    hdr(ax, 11.4, 3.55+1.02-0.38, 6.1, 0.38, "Softmax  (inference only)", size=7.5, fill=C_MED)
    txt(ax, 14.45, 3.9, "Converts logits to probability distribution  (16 classes)", size=7.5)
    txt(ax, 14.45, 3.66, "Cross-entropy applied during validation; FocalLoss during training", size=7, italic=True)
    arr(ax, 14.45, 3.55, 14.45, 3.08)

    box(ax, 11.4, 2.0, 6.1, 1.0, fill=C_BOX3, ec=C_MED, lw=1)
    hdr(ax, 11.4, 2.0+1.0-0.38, 6.1, 0.38, "Rank & Select Top-K", size=7.5, fill=C_MED)
    txt(ax, 14.45, 2.32, "Sort N(N-1) pairs by max predicted probability", size=7.5)
    txt(ax, 14.45, 2.1, "Return top-K {subject, predicate, object, confidence}", size=7, italic=True)
    arr(ax, 14.45, 2.0, 14.45, 1.5)

    box(ax, 11.4, 0.85, 6.1, 0.6, fill=C_OUT, ec=C_DARK, lw=1.5)
    txt(ax, 14.45, 1.18, "OUTPUT TRIPLETS  —  Top-K (subject, predicate, object, confidence)", bold=True, color=C_WHITE, size=8)

    # ── Inter-module arrows ────────────────────────────────────────────────────
    # Detection → Encoding (crops)
    arr(ax, 5.12, 5.3, 5.98, 5.3, label="crop images")
    # Encoding → Reasoning
    arr(ax, 10.62, 2.05, 11.38, 4.5)
    txt(ax, 11.0, 3.2, "1037-D", size=7.5, italic=True, color=C_ACCENT)
    # Prior input (from above into Reasoning)
    box(ax, 13.3, 9.52, 3.3, 0.58, fill=C_BOX2, ec=C_MED, lw=1)
    txt(ax, 14.95, 9.81, "Co-occurrence Prior  (llm_prior.json,  16-D)", bold=True, size=7.5)
    darr(ax, 14.95, 9.52, 14.95, 8.92)

    return fig, "fig_4_1_system_arch"


# ─────────────────────────────────────────────────────────────────────────────
#  4.  FIG 4.2  —  DATA FLOW DIAGRAM  (10 × 16 inches, tall)
# ─────────────────────────────────────────────────────────────────────────────
def fig_4_2():
    W, H = 11, 18
    fig, ax = _fig(W, H)
    box(ax, 0.15, 0.15, W-0.3, H-0.3, fill=C_WHITE, ec=C_DARK, lw=2, radius=0.2)
    hdr(ax, 0.15, H-0.85, W-0.3, 0.7,
        "DATA FLOW DIAGRAM — RAW IMAGE TO PREDICTED TRIPLET",
        size=9.5, fill=C_DARK)

    BX, BW, BH = 1.0, 7.5, 0.82
    CX = BX + BW/2    # center x
    AX = BX + BW + 0.1  # annotation column x
    AW = W - AX - 0.2

    steps = [
        # (y-top, title, subtext, fill, ann_lines)
        (H-1.8, "Input Image", ["Raw RGB image — any resolution (H × W × 3)"],
         C_BOX, ["H × W × 3"]),
        (H-3.2, "YOLO-World Detection",
         ["yolov8s-world.pt  |  conf_thresh=0.3  |  top_k=10  |  NMS IoU=0.5"],
         C_BOX2, ["Open-vocab", "detector"]),
        (H-4.6, "N Bounding Boxes",
         ["N × {label,  [x1, y1, x2, y2],  score}   sorted by confidence"],
         C_BOX, ["N × 4", "(xyxy)"]),
        (H-6.0, "CLIP ViT-B/32 Encoding  (frozen)",
         ["Crop each box → resize 224×224 → CLIP forward  |  check DiskCache first"],
         C_BOX2, ["Frozen", "ViT-B/32"]),
        (H-7.4, "Object Embeddings",
         ["N × 512-D  (one embedding per crop)  —  L2-normalised to unit length"],
         C_BOX, ["N × 512-D"]),
        (H-8.8, "Pair Construction",
         ["Form N(N-1) ordered (subject, object) pairs  —  all i≠j combinations"],
         C_BOX2, ["N(N-1)", "pairs"]),
        (H-10.2, "Feature Concatenation",
         ["[subj_emb ‖ obj_emb ‖ spatial_feat]   per pair  =  1037-D"],
         C_BOX3, ["512+512+13", "= 1037-D"]),
        (H-11.6, "RelationHead MLP",
         ["LayerNorm → Linear(1037→512) → GELU → Dropout(0.3) → Linear(512→256) → GELU → Dropout(0.3) → Linear(256→16)"],
         C_BOX2, ["16-D", "logits"]),
        (H-13.0, "Visual Logits",
         ["16-D pre-softmax scores per (subject, object) pair"],
         C_BOX, ["16-D"]),
        (H-14.4, "Softmax  →  Probability Distribution",
         ["Convert logits to probabilities over 16 predicate classes"],
         C_BOX2, ["16-D", "probs"]),
        (H-15.8, "Rank & Select Top-K",
         ["Sort all N(N-1) pairs by max predicted probability  |  return top-K"],
         C_BOX3, ["sort &", "top-K"]),
        (H-17.2, "OUTPUT TRIPLETS",
         ["{subject,  predicate,  object,  confidence score}  — Top-K results"],
         C_OUT, ["Final", "output"]),
    ]

    prev_cy = None
    for i, (yt, title, subs, fill, ann) in enumerate(steps):
        dark = (fill == C_OUT)
        b = box(ax, BX, yt, BW, BH, fill=fill, ec=C_MED, lw=1.3)
        cy = yt + BH/2
        lbl_y = yt + BH - 0.16
        if dark:
            txt(ax, CX, lbl_y, title, bold=True, color=C_WHITE, size=8.5)
            if subs:
                txt(ax, CX, yt+0.22, subs[0], color="#cccccc", size=7.5, italic=True)
        else:
            txt(ax, CX, lbl_y, title, bold=True, size=8.5)
            if subs:
                txt(ax, CX, yt+0.22, subs[0], size=7.5, italic=True)

        # Annotation box on the right
        ann_h = len(ann)*0.28 + 0.12
        ann_y = yt + (BH-ann_h)/2
        box(ax, AX, ann_y, AW, ann_h, fill=C_BOX2, ec="#aaaaaa", lw=0.7, radius=0.05)
        for j, a in enumerate(ann):
            txt(ax, AX+AW/2, ann_y+ann_h-(j+0.6)*(ann_h/len(ann)), a, size=7.5, italic=True)

        # Arrow from previous step
        if prev_cy is not None:
            arr(ax, CX, prev_cy, CX, yt+BH, hw=0.1, hl=0.15)
        prev_cy = yt

    # FusionModule side branch (dashed, from Visual Logits step)
    fm_y = H-13.0 + BH/2
    box(ax, 0.2, fm_y-0.75, 0.72, 1.5, fill=C_BOX, ec="#888888", lw=0.8)
    txt(ax, 0.56, fm_y+0.42, "FusionModule", size=6.5, bold=True)
    txt(ax, 0.56, fm_y+0.18, "(Exp B only)", size=6, italic=True)
    txt(ax, 0.56, fm_y-0.08, "v + α·(p/τ)", size=6.5)
    txt(ax, 0.56, fm_y-0.35, "prior 16-D", size=6, italic=True)
    txt(ax, 0.56, fm_y-0.58, "from json", size=6, italic=True)
    darr(ax, BX, fm_y, 0.92, fm_y)
    darr(ax, 0.92, fm_y-0.2, BX, fm_y-0.2)

    # Spatial features side branch
    sp_y = H-10.2 + BH/2
    box(ax, 0.2, sp_y-0.8, 0.72, 1.6, fill=C_BOX, ec="#888888", lw=0.8)
    txt(ax, 0.56, sp_y+0.55, "Spatial", size=6.5, bold=True)
    txt(ax, 0.56, sp_y+0.30, "Features", size=6.5, bold=True)
    txt(ax, 0.56, sp_y+0.05, "13-D per pair", size=6, italic=True)
    txt(ax, 0.56, sp_y-0.22, "Δcx, Δcy,", size=6, italic=True)
    txt(ax, 0.56, sp_y-0.46, "IoU, ratio…", size=6, italic=True)
    txt(ax, 0.56, sp_y-0.68, "CPU only", size=6, italic=True)
    darr(ax, BX, sp_y, 0.92, sp_y)

    return fig, "fig_4_2_data_flow"


# ─────────────────────────────────────────────────────────────────────────────
#  5.  FIG 4.3  —  MLP ARCHITECTURE  (11 × 13 inches)
# ─────────────────────────────────────────────────────────────────────────────
def fig_4_3():
    W, H = 11, 13.5
    fig, ax = _fig(W, H)
    box(ax, 0.15, 0.15, W-0.3, H-0.3, fill=C_WHITE, ec=C_DARK, lw=2, radius=0.2)
    hdr(ax, 0.15, H-0.85, W-0.3, 0.7,
        "RELATIONHEAD MLP ARCHITECTURE  (1037 → 512 → 256 → 16)",
        size=9.5, fill=C_DARK)

    BX, BW = 0.4, 8.0
    CX = BX + BW/2

    # Param annotation column
    PX = BX + BW + 0.2

    def layer_block(ax, y, h, header, lines, fill=C_BOX2, params=""):
        box(ax, BX, y, BW, h, fill=fill, ec=C_MED, lw=1.3)
        hdr(ax, BX, y+h-0.42, BW, 0.42, header, size=8, fill=C_MED if fill != C_OUT else C_DARK)
        for i, ln in enumerate(lines):
            bold = (i == 0)
            size = 8.5 if bold else 7.5
            col = C_WHITE if fill == C_OUT else (C_DARK if bold else C_ACCENT)
            txt(ax, CX, y+h-0.55-i*0.35, ln, bold=bold, size=size, color=col)
        if params:
            box(ax, PX, y, W-PX-0.22, h, fill="#f0f0f0", ec="#aaaaaa", lw=0.7)
            txt(ax, PX+(W-PX-0.22)/2, y+h/2+0.12, params, size=8.5, bold=True)
            txt(ax, PX+(W-PX-0.22)/2, y+h/2-0.18, "params", size=7.5, italic=True)

    # Input
    layer_block(ax, H-2.0, 0.85,
                "INPUT FEATURE VECTOR",
                ["CLIP_subj (512-D)  ⊕  CLIP_obj (512-D)  ⊕  Spatial (13-D)  =  1037-D",
                 "One vector per candidate (subject, object) pair in the image"],
                fill=C_BOX)
    arr(ax, CX, H-2.0, CX, H-2.62, label="1037-D", label_off=(0.2, 0))

    # LayerNorm
    layer_block(ax, H-3.65, 0.75, "LayerNorm (1037)",
                ["Normalise across feature dimension; reduces covariate shift",
                 "No learnable weight / bias parameters counted in total"],
                fill=C_BOX2)
    arr(ax, CX, H-3.65, CX, H-4.3)

    # Hidden 1
    HL1_Y = H-6.4; HL1_H = 1.9
    box(ax, BX, HL1_Y, BW, HL1_H, fill=C_BOX3, ec=C_MED, lw=1.5)
    hdr(ax, BX, HL1_Y+HL1_H-0.42, BW, 0.42, "HIDDEN LAYER 1", size=8.5, fill=C_HEAD)
    # Sub-sections
    for xi, (sw, lbl) in enumerate([(BW/2-0.05, "Linear  (1037 → 512)\n531,456 parameters"),
                                     (BW/2-0.05, "")]):
        box(ax, BX+xi*(BW/2+0.05), HL1_Y+0.9, BW/2-0.1, 0.6, fill=C_BOX2, ec="#999", lw=0.8)
    box(ax, BX+0.05, HL1_Y+0.05, BW-0.1, 0.82, fill=C_BOX2, ec="#999", lw=0.8)
    txt(ax, CX, HL1_Y+0.48, "Linear  (1037 → 512)   —   531,456 parameters", bold=True, size=8.5)
    box(ax, BX+0.05, HL1_Y+0.9, (BW-0.2)/2, 0.6, fill=C_BOX, ec="#999", lw=0.8)
    txt(ax, BX+0.05+(BW-0.2)/4, HL1_Y+1.22, "GELU activation", size=8)
    box(ax, BX+0.15+(BW-0.2)/2, HL1_Y+0.9, (BW-0.2)/2, 0.6, fill=C_BOX, ec="#999", lw=0.8)
    txt(ax, BX+0.15+(BW-0.2)/2+(BW-0.2)/4, HL1_Y+1.22, "Dropout (p = 0.3)", size=8)
    box(ax, PX, HL1_Y, W-PX-0.22, HL1_H, fill="#f0f0f0", ec="#aaaaaa", lw=0.7)
    txt(ax, PX+(W-PX-0.22)/2, HL1_Y+HL1_H/2+0.1, "531K", size=9, bold=True)
    txt(ax, PX+(W-PX-0.22)/2, HL1_Y+HL1_H/2-0.2, "params", size=7.5, italic=True)

    arr(ax, CX, HL1_Y, CX, H-4.3, label="512-D", label_off=(0.2, 0))
    arr(ax, CX, HL1_Y-0.02, CX, H-7.18)

    # Hidden 2
    HL2_Y = H-9.3; HL2_H = 1.9
    box(ax, BX, HL2_Y, BW, HL2_H, fill=C_BOX3, ec=C_MED, lw=1.5)
    hdr(ax, BX, HL2_Y+HL2_H-0.42, BW, 0.42, "HIDDEN LAYER 2", size=8.5, fill=C_HEAD)
    box(ax, BX+0.05, HL2_Y+0.05, BW-0.1, 0.82, fill=C_BOX2, ec="#999", lw=0.8)
    txt(ax, CX, HL2_Y+0.48, "Linear  (512 → 256)   —   131,328 parameters", bold=True, size=8.5)
    box(ax, BX+0.05, HL2_Y+0.9, (BW-0.2)/2, 0.6, fill=C_BOX, ec="#999", lw=0.8)
    txt(ax, BX+0.05+(BW-0.2)/4, HL2_Y+1.22, "GELU activation", size=8)
    box(ax, BX+0.15+(BW-0.2)/2, HL2_Y+0.9, (BW-0.2)/2, 0.6, fill=C_BOX, ec="#999", lw=0.8)
    txt(ax, BX+0.15+(BW-0.2)/2+(BW-0.2)/4, HL2_Y+1.22, "Dropout (p = 0.3)", size=8)
    box(ax, PX, HL2_Y, W-PX-0.22, HL2_H, fill="#f0f0f0", ec="#aaaaaa", lw=0.7)
    txt(ax, PX+(W-PX-0.22)/2, HL2_Y+HL2_H/2+0.1, "131K", size=9, bold=True)
    txt(ax, PX+(W-PX-0.22)/2, HL2_Y+HL2_H/2-0.2, "params", size=7.5, italic=True)

    arr(ax, CX, HL2_Y, CX, H-7.18, label="256-D", label_off=(0.2, 0))
    arr(ax, CX, HL2_Y-0.02, CX, H-10.18)

    # Output layer
    layer_block(ax, H-11.4, 0.75, "OUTPUT LAYER  —  Linear (256 → 16)",
                ["4,112 parameters   |   16 = number of predicate classes in vocabulary"],
                fill=C_BOX2, params="4K")
    arr(ax, CX, H-11.4, CX, H-10.18, label="16-D logits", label_off=(0.2, 0))
    arr(ax, CX, H-11.4-0.02, CX, H-12.15)

    # Softmax
    layer_block(ax, H-13.0, 0.7, "Softmax  (inference only — NOT applied during training)",
                ["Converts 16-D logits to probability distribution over predicate classes"],
                fill=C_BOX)
    arr(ax, CX, H-13.0, CX, H-12.15)

    # Total params
    box(ax, PX, H-13.0, W-PX-0.22, 1.7, fill="#e8e8e8", ec=C_MED, lw=1)
    txt(ax, PX+(W-PX-0.22)/2, H-12.0, "TOTAL", size=8, bold=True)
    txt(ax, PX+(W-PX-0.22)/2, H-12.4, "Exp A:", size=8, bold=True)
    txt(ax, PX+(W-PX-0.22)/2, H-12.7, "666,896", size=9, bold=True)
    txt(ax, PX+(W-PX-0.22)/2, H-13.0+0.22, "Exp B: +2", size=7.5, italic=True)

    return fig, "fig_4_3_mlp_arch"


# ─────────────────────────────────────────────────────────────────────────────
#  6.  FIG 4.4  —  FUSIONMODULE  (14 × 10 inches)
# ─────────────────────────────────────────────────────────────────────────────
def fig_4_4():
    W, H = 14, 11
    fig, ax = _fig(W, H)
    box(ax, 0.15, 0.15, W-0.3, H-0.3, fill=C_WHITE, ec=C_DARK, lw=2, radius=0.2)
    hdr(ax, 0.15, H-0.85, W-0.3, 0.7,
        "FUSIONMODULE DESIGN — PRIOR-GUIDED LOGIT BLENDING",
        size=9.5, fill=C_DARK)

    # ── Left path: Visual ─────────────────────────────────────────────────────
    # Input header
    box(ax, 0.4, 8.95, 5.5, 0.72, fill=C_BOX, ec=C_MED, lw=1)
    txt(ax, 3.15, 9.33, "Input Features  (1037-D per pair)", bold=True, size=8.5)
    txt(ax, 3.15, 9.06, "[CLIP_subj ‖ CLIP_obj ‖ Spatial]", size=7.5, italic=True)
    arr(ax, 3.15, 8.95, 3.15, 8.35)

    box(ax, 0.4, 7.25, 5.5, 1.05, fill=C_BOX2, ec=C_MED, lw=1.3)
    hdr(ax, 0.4, 7.25+1.05-0.4, 5.5, 0.4, "RelationHead MLP", size=8, fill=C_HEAD)
    txt(ax, 3.15, 7.68, "1037-D → LayerNorm → 512 → 256 → 16", bold=True, size=8.5)
    txt(ax, 3.15, 7.4, "GELU activations,  Dropout(0.3)", size=7.5, italic=True)
    arr(ax, 3.15, 7.25, 3.15, 6.68, label="visual_logits v  (16-D)")

    box(ax, 0.4, 5.85, 5.5, 0.8, fill=C_BOX3, ec=C_MED, lw=1.2)
    txt(ax, 3.15, 6.27, "Visual Logits   v   (16-D)", bold=True, size=9)
    txt(ax, 3.15, 6.0, "Pre-softmax scores over 16 predicate classes", size=7.5, italic=True)

    # ── Right path: Prior ─────────────────────────────────────────────────────
    box(ax, 8.1, 8.95, 5.5, 0.72, fill=C_BOX, ec=C_MED, lw=1)
    txt(ax, 10.85, 9.33, "LLM Co-occurrence Prior", bold=True, size=8.5)
    txt(ax, 10.85, 9.06, "data/llm_prior.json  |  308 (subj, obj) entries", size=7.5, italic=True)
    arr(ax, 10.85, 8.95, 10.85, 8.35)

    box(ax, 8.1, 7.25, 5.5, 1.05, fill=C_BOX2, ec=C_MED, lw=1.3)
    hdr(ax, 8.1, 7.25+1.05-0.4, 5.5, 0.4, "PriorLookup", size=8, fill=C_HEAD)
    txt(ax, 10.85, 7.68, "Key: '{subj}__{obj}'  →  16-D probability vector", bold=True, size=8.5)
    txt(ax, 10.85, 7.4, "Uniform distribution returned for unseen pairs", size=7.5, italic=True)
    arr(ax, 10.85, 7.25, 10.85, 6.68, label="prior_logits p  (16-D)")

    box(ax, 8.1, 5.85, 5.5, 0.8, fill=C_BOX3, ec=C_MED, lw=1.2)
    txt(ax, 10.85, 6.27, "Prior Vector   p   (16-D)", bold=True, size=9)
    txt(ax, 10.85, 6.0, "Co-occurrence probabilities for this (subj, obj) pair", size=7.5, italic=True)
    arr(ax, 10.85, 5.85, 10.85, 5.25)

    # Temperature scaling
    box(ax, 8.1, 4.18, 5.5, 1.02, fill=C_BOX2, ec=C_MED, lw=1.3)
    hdr(ax, 8.1, 4.18+1.02-0.4, 5.5, 0.4, "Temperature Scaling", size=8, fill=C_MED)
    txt(ax, 10.85, 4.6, "scaled = p / τ", bold=True, size=9)
    txt(ax, 10.85, 4.34, "τ = softplus(log_τ) + ε  [τ > 0 guaranteed]", size=7.5, italic=True)
    txt(ax, 10.85, 4.1+0.15, "log_τ learnable  (init = 0  →  τ = 1 at start)", size=7, italic=True)
    arr(ax, 10.85, 4.18, 10.85, 3.58, label="scaled prior p/τ  (16-D)")

    # Alpha weighting
    box(ax, 8.1, 2.48, 5.5, 1.05, fill=C_BOX2, ec=C_MED, lw=1.3)
    hdr(ax, 8.1, 2.48+1.05-0.4, 5.5, 0.4, "Alpha Weighting", size=8, fill=C_MED)
    txt(ax, 10.85, 2.9, "weighted = α · (p / τ)", bold=True, size=9)
    txt(ax, 10.85, 2.63, "α = exp(log_α)  [α > 0 guaranteed]", size=7.5, italic=True)
    txt(ax, 10.85, 2.38, "log_α learnable  (init = 0  →  α = 1 at start)", size=7, italic=True)

    # ── Addition operation ─────────────────────────────────────────────────────
    # Draw + circle
    circ = plt.Circle((7.0, 1.68), 0.38, color=C_MED, fill=False, linewidth=2, zorder=5)
    ax.add_patch(circ)
    txt(ax, 7.0, 1.72, "+", size=18, bold=True)

    # Arrows into + circle
    arr(ax, 3.15, 5.85, 3.15, 2.0)
    arr(ax, 3.15, 2.0, 6.6, 1.68, label="v  (16-D)")
    arr(ax, 10.85, 2.48, 10.85, 1.68)
    arr(ax, 10.85, 1.68, 7.4, 1.68, label="α·(p/τ)  (16-D)")

    # Output
    arr(ax, 7.0, 1.3, 7.0, 0.98)

    # Formula highlight box
    box(ax, 2.5, 0.22, 9.0, 0.72, fill="#e8e8e8", ec=C_MED, lw=1.5)
    txt(ax, 7.0, 0.78, "FUSED LOGITS  (16-D)", bold=True, size=9.5)
    txt(ax, 7.0, 0.53, "output  =  v  +  α · (p / τ)    "
        "←   Init: α=1, τ=1  →  output = v  (identity at t=0)", size=8, italic=True)
    txt(ax, 7.0, 0.3, "Only 2 extra trainable parameters:  log_α  and  log_τ", size=7.5, italic=True)

    # Note box on left
    box(ax, 0.4, 2.48, 2.55, 3.8, fill="#f5f5f5", ec="#aaaaaa", lw=0.8, radius=0.1)
    txt(ax, 1.68, 5.9, "Key Properties", bold=True, size=8)
    notes = [
        "• Identity init: output = v",
        "  when α=1, τ=1",
        "• τ > 1: flattens prior",
        "• τ < 1: sharpens prior",
        "• Both α, τ optimised",
        "  by AdamW with MLP",
        "• Explainable: α reveals",
        "  visual vs. prior trust",
        "• Differentiable end-to-end",
    ]
    for i, n in enumerate(notes):
        txt(ax, 1.68, 5.6-i*0.37, n, size=7.5, ha="left" if n.startswith("  ") else "center",
            va="center",
            italic=n.startswith("•"))

    return fig, "fig_4_4_fusion_module"


# ─────────────────────────────────────────────────────────────────────────────
#  7.  FIG 5.2  —  SPATIAL FEATURES  (15 × 9 inches)
# ─────────────────────────────────────────────────────────────────────────────
def fig_5_2():
    W, H = 15, 9.5
    fig, ax = _fig(W, H)
    box(ax, 0.15, 0.15, W-0.3, H-0.3, fill=C_WHITE, ec=C_DARK, lw=2, radius=0.2)
    hdr(ax, 0.15, H-0.85, W-0.3, 0.7,
        "SPATIAL FEATURE COMPUTATION FROM SUBJECT AND OBJECT BOUNDING BOXES",
        size=9.5, fill=C_DARK)

    # ── Image frame ────────────────────────────────────────────────────────────
    IX, IY, IW, IH = 0.4, 0.3, 6.0, 7.9
    box(ax, IX, IY, IW, IH, fill="#f8f8f8", ec=C_MED, lw=2, radius=0.1)
    txt(ax, IX+IW/2, IY+IH-0.22, "Image  (W × H)", bold=True, size=9)

    # Subject box (solid, darker fill)
    SX, SY, SW, SH2 = IX+0.5, IY+2.5, 2.8, 3.5
    box(ax, SX, SY, SW, SH2, fill="#d8d8d8", ec=C_DARK, lw=2.5, radius=0.05)
    txt(ax, SX+SW/2, SY+SH2-0.22, "SUBJECT BOX", bold=True, size=8.5)

    # Object box (dashed)
    OX, OY, OW, OH2 = IX+2.3, IY+1.0, 3.2, 4.2
    from matplotlib.patches import Rectangle
    r = Rectangle((OX, OY), OW, OH2, fill=False, edgecolor=C_MED,
                  linewidth=2, linestyle="--", zorder=3)
    ax.add_patch(r)
    txt(ax, OX+OW/2, OY+0.22, "OBJECT BOX  (dashed)", bold=True, size=8.5, color=C_MED)

    # Centers
    SCX, SCY = SX+SW/2, SY+SH2/2
    OCX, OCY = OX+OW/2, OY+OH2/2
    ax.plot(SCX, SCY, 'o', color=C_DARK, ms=7, zorder=6)
    ax.plot(OCX, OCY, 'o', color=C_DARK, ms=7, zorder=6)
    txt(ax, SCX-0.2, SCY+0.18, "cs", size=8, ha="right", bold=True)
    txt(ax, OCX+0.2, OCY+0.18, "co", size=8, ha="left", bold=True)

    # Δcx arrow (horizontal, between centers)
    ax.annotate("", xy=(OCX, SCY), xytext=(SCX, SCY),
                arrowprops=dict(arrowstyle="<->", color=C_ACCENT, lw=1.5))
    txt(ax, (SCX+OCX)/2, SCY-0.22, "Δcx = (cxo−cxs)/W", size=7.5, italic=True)

    # Δcy arrow (vertical)
    ax.annotate("", xy=(OCX+0.15, OCY), xytext=(OCX+0.15, SCY),
                arrowprops=dict(arrowstyle="<->", color=C_ACCENT, lw=1.5))
    txt(ax, OCX+0.55, (SCY+OCY)/2, "Δcy =\n(cyo−cys)/H", size=7.5, italic=True, ha="left")

    # IoU overlap shading
    ovx = max(SX, OX); ovy = max(SY, OY)
    ovw = min(SX+SW, OX+OW)-ovx; ovh = min(SY+SH2, OY+OH2)-ovy
    if ovw > 0 and ovh > 0:
        from matplotlib.patches import Rectangle as R2
        ov = R2((ovx, ovy), ovw, ovh, fill=True, facecolor="#999999",
                alpha=0.35, edgecolor="#666666", linewidth=1, zorder=4)
        ax.add_patch(ov)
        txt(ax, ovx+ovw/2, ovy+ovh/2, "IoU\noverlap", size=7.5, bold=True, color="#444")

    # ── Feature table ────────────────────────────────────────────────────────
    FX = IX + IW + 0.35
    FW = W - FX - 0.25
    box(ax, FX, IY, FW, IH, fill="#f7f7f7", ec=C_MED, lw=1.5, radius=0.1)
    hdr(ax, FX, IY+IH-0.52, FW, 0.52, "13-D SPATIAL FEATURE VECTOR", size=9, fill=C_HEAD)

    feats = [
        ("Dims 0–3", "Subject box (normalised)",  "x1s/W,  y1s/H,  x2s/W,  y2s/H"),
        ("Dims 4–7", "Object box (normalised)",   "x1o/W,  y1o/H,  x2o/W,  y2o/H"),
        ("Dim  8",   "Centre offset  Δcx",        "(cxo − cxs) / W"),
        ("Dim  9",   "Centre offset  Δcy",        "(cyo − cys) / H"),
        ("Dim 10",   "Log width ratio",           "log(wo / ws)  clipped to [−5, +5]"),
        ("Dim 11",   "Log height ratio",          "log(ho / hs)  clipped to [−5, +5]"),
        ("Dim 12",   "Intersection over Union",   "IoU(subject_box,  object_box)"),
    ]
    RH = 0.97
    for i, (dim, name, formula) in enumerate(feats):
        fy = IY + IH - 0.55 - (i+1)*RH
        fill = "#f0f0f0" if i % 2 == 0 else "#e6e6e6"
        box(ax, FX+0.08, fy, FW-0.16, RH-0.05, fill=fill, ec="#cccccc", lw=0.8, radius=0.06)
        txt(ax, FX+1.1, fy+RH*0.72, dim, bold=True, size=8.5)
        txt(ax, FX+1.1, fy+RH*0.44, name, size=8)
        txt(ax, FX+1.1, fy+RH*0.18, formula, size=7.5, italic=True, color=C_ACCENT)

    # Arrows from diagram to table rows
    for i, (fy_offset, from_y) in enumerate([
        (0, IY+IH-0.55-0.5*RH),
        (1, IY+IH-0.55-1.5*RH),
        (2, SCY),
        (3, (SCY+OCY)/2),
        (4, SCX+SW+0.1),
        (5, SCX+SW+0.1),
        (6, ovy+ovh/2),
    ]):
        target_y = IY+IH-0.55-(i+0.5)*RH
        darr(ax, IX+IW, min(max(from_y, IY+0.3), IY+IH-0.4), FX+0.08, target_y)

    return fig, "fig_5_2_spatial_features"


# ─────────────────────────────────────────────────────────────────────────────
#  8–11.  SPRINT DIAGRAMS
# ─────────────────────────────────────────────────────────────────────────────
def fig_2_2():
    W, H = 15, 6.5
    fig, ax = _fig(W, H)
    box(ax, 0.15, 0.15, W-0.3, H-0.3, fill=C_WHITE, ec=C_DARK, lw=2, radius=0.2)
    hdr(ax, 0.15, H-0.85, W-0.3, 0.7,
        "SPRINT ASSIGNMENT MATRIX — USER STORIES US-01 to US-11",
        size=9.5, fill=C_DARK)

    COL = [0.2, 1.4, 8.0, 10.5, 12.2, 13.7, 14.8]
    HEADS = ["ID", "User Story (Summary)", "Priority", "Sprint", "Effort", "Status"]
    RH = 0.44; BASE_Y = H-1.65

    for ci, (h, cx, nx) in enumerate(zip(HEADS, COL, COL[1:])):
        hdr(ax, cx, BASE_Y, nx-cx-0.04, RH, h, size=8, fill=C_HEAD)

    rows = [
        ("US-01","Download and preprocess VG triplets to JSON","High","Sprint 1","3 days","Done"),
        ("US-02","Pre-compute all CLIP embeddings via warmup_cache.py","High","Sprint 1","2 days","Done"),
        ("US-03","Implement DiskCache with SHA-256 sharded directories","High","Sprint 1","2 days","Done"),
        ("US-04","Implement 13-D spatial feature computation","High","Sprint 1","1 day","Done"),
        ("US-05","Build PriorLookup from llm_prior.json","Medium","Sprint 1","1 day","Done"),
        ("US-06","VGRelationDataset: zero random-embedding guarantee","High","Sprint 1","2 days","Done"),
        ("US-07","RelationHead MLP (1037→512→256→16, GELU)","High","Sprint 2","2 days","Done"),
        ("US-08","FusionModule: learnable alpha and log_tau","Medium","Sprint 2","1 day","Done"),
        ("US-09","Training loop: FocalLoss, AdamW, cosine LR","High","Sprint 2","2 days","Done"),
        ("US-10","Evaluation: R@K, mR@50, per-predicate R@50","High","Sprint 2","2 days","Done"),
        ("US-11","Plot generation from JSON/CSV files","Medium","Sprint 2","1 day","Done"),
    ]
    for ri, row in enumerate(rows):
        ry = BASE_Y - (ri+1)*(RH+0.02)
        fills = ["#f5f5f5", "#ebebeb"]
        sp_fills = ["#d0dfd0", "#ffe8cc"]  # Sprint 2 green, Sprint 1 tan
        for ci, (val, cx, nx) in enumerate(zip(row, COL, COL[1:])):
            f = sp_fills[0 if "Sprint 2" in row[3] else 1] if ci == 3 else fills[ri % 2]
            box(ax, cx, ry, nx-cx-0.04, RH, fill=f, ec="#bbbbbb", lw=0.7)
            anchor = "left" if ci == 1 else "center"
            tx = cx+0.06 if ci == 1 else (cx+nx-0.04)/2
            txt(ax, tx, ry+RH/2, val, size=7.5, ha=anchor)

    # Legend
    box(ax, 0.3, 0.2, 0.35, 0.3, fill="#d0dfd0", ec="#888", lw=0.8)
    txt(ax, 0.82, 0.36, "Sprint 2", size=8, ha="left")
    box(ax, 1.6, 0.2, 0.35, 0.3, fill="#ffe8cc", ec="#888", lw=0.8)
    txt(ax, 2.12, 0.36, "Sprint 1", size=8, ha="left")

    return fig, "fig_2_2_sprint_matrix"


def fig_3_1():
    W, H = 15, 6.0
    fig, ax = _fig(W, H)
    box(ax, 0.15, 0.15, W-0.3, H-0.3, fill=C_WHITE, ec=C_DARK, lw=2, radius=0.2)
    hdr(ax, 0.15, H-0.85, W-0.3, 0.7,
        "SPRINT I EXECUTION TIMELINE — DATA PIPELINE AND CLIP EMBEDDING CACHE  (Weeks 3–6)",
        size=9.5, fill=C_DARK)

    tasks = [
        ("Week 3", "VG Download & Triplet Extraction",
         "Download VG v1.4 relationship annotations and image metadata (≈6 GB). "
         "Parse annotations, extract (subject, predicate, object) triplets with bounding boxes. "
         "Filter to 16-class vocab; split train/val/test at image level (80/10/10).",
         "33,741 train / 7,070 val / 7,070 test triplets extracted"),
        ("Week 4", "DiskCache & Spatial Feature Implementation",
         "Implement DiskCache with SHA-256 two-level sharded directories and atomic writes. "
         "Implement compute_spatial_features() — 13-D vector per (subject, object) box pair. "
         "Create canonical make_clip_cache_key() to prevent key divergence.",
         "DiskCache and Spatial encoder unit-tested and integrated"),
        ("Week 5", "PriorLookup & CLIP Cache Warmup",
         "Construct data/llm_prior.json — 308 (subject, object) entries with 16-D prior vectors. "
         "Implement PriorLookup class; run warmup_cache.py for all three splits. "
         "Verify 100% cache hit rate during first training epoch.",
         "CLIP embeddings cached for 18,244 train + 3,885 val + 3,887 test triplets"),
        ("Week 6", "VGRelationDataset & Integration Test",
         "Implement VGRelationDataset with _filter_available() — drops triplets with missing embeddings. "
         "Verify zero random-embedding guarantee across all DataLoader workers. "
         "End-to-end pipeline test: image → detector → encoder → dataset → DataLoader.",
         "Sprint I COMPLETE — data pipeline production-ready"),
    ]
    TH = 1.05
    for i, (week, title, desc, result) in enumerate(tasks):
        ty = H-1.0-(i+1)*(TH+0.12)
        box(ax, 0.3, ty, 2.0, TH, fill=C_HEAD, ec=C_DARK, lw=1.2, radius=0.1)
        txt(ax, 1.3, ty+TH*0.68, week, bold=True, color=C_WHITE, size=9)
        txt(ax, 1.3, ty+TH*0.32, "Sprint I", italic=True, color="#cccccc", size=8)
        arr(ax, 2.32, ty+TH/2, 2.52, ty+TH/2)
        box(ax, 2.55, ty, W-2.8, TH, fill=C_BOX if i%2==0 else C_BOX2, ec=C_MED, lw=1)
        txt(ax, 2.75, ty+TH*0.82, title, bold=True, size=9, ha="left")
        # wrap desc in two parts
        half = len(desc)//2; split = desc.rfind(" ", 0, half+40)
        txt(ax, 2.75, ty+TH*0.57, desc[:split], size=7.5, italic=True, ha="left")
        txt(ax, 2.75, ty+TH*0.36, desc[split+1:], size=7.5, italic=True, ha="left")
        txt(ax, 2.75, ty+TH*0.14, f"✓  {result}", size=7.5, bold=True, color="#3a3a3a", ha="left")

    return fig, "fig_3_1_sprint1_timeline"


def fig_3_2():
    W, H = 15, 7.5
    fig, ax = _fig(W, H)
    box(ax, 0.15, 0.15, W-0.3, H-0.3, fill=C_WHITE, ec=C_DARK, lw=2, radius=0.2)
    hdr(ax, 0.15, H-0.85, W-0.3, 0.7,
        "CLIP CACHE WARMUP WORKFLOW  —  warmup_cache.py",
        size=9.5, fill=C_DARK)

    steps = [
        ("Load\nTriplets", "Read train/val/test\nJSON annotations"),
        ("Group by\nImage ID", "Minimise\nimage re-reads"),
        ("Open Image\n(once per img)", "PIL.Image.open()\n+ CLIP preprocess"),
        ("Batch CLIP\nEncoding", "Encode all boxes\nin single forward"),
        ("Compute\nCache Key", "make_clip_cache_key()\n.1f format for ints"),
        ("Write to\nDiskCache", "SHA-256 sharded\nAtomic rename write"),
    ]
    n = len(steps)
    SW2, SH = 2.1, 2.6
    gap = 0.25
    total = n*SW2 + (n-1)*gap
    sx0 = (W-total)/2
    sy = 2.2

    for i, (title, sub) in enumerate(steps):
        sx = sx0 + i*(SW2+gap)
        fill = C_OUT if i == n-1 else (C_BOX3 if i == 0 else C_BOX2)
        box(ax, sx, sy, SW2, SH, fill=fill, ec=C_MED if i<n-1 else C_DARK, lw=1.5, radius=0.12)
        txt(ax, sx+SW2/2, sy+SH*0.78, title, bold=True, size=8.5,
            color=C_WHITE if i==n-1 else C_DARK)
        for j, ln in enumerate(sub.split("\n")):
            txt(ax, sx+SW2/2, sy+SH*0.5-j*0.38, ln, size=8,
                italic=True, color="#bbbbbb" if i==n-1 else C_ACCENT)
        if i < n-1:
            arr(ax, sx+SW2, sy+SH/2, sx+SW2+gap, sy+SH/2)

    # Cache-hit feedback arc
    ax.annotate("", xy=(sx0+3*(SW2+gap)+SW2/2, sy),
                xytext=(sx0+5*(SW2+gap)+SW2/2, sy),
                arrowprops=dict(arrowstyle="-|>,head_width=0.15,head_length=0.2",
                                color="#888888", lw=1.2,
                                connectionstyle="arc3,rad=-0.5"))
    txt(ax, sx0+4.5*(SW2+gap)+SW2/2, sy-0.7,
        "Cache Hit: skip encode if key exists  (idempotent)", size=8, italic=True, color=C_ACCENT)

    # Stats box
    box(ax, 0.4, 0.22, W-0.8, 1.7, fill=C_BOX, ec=C_MED, lw=1, radius=0.1)
    txt(ax, W/2, 1.68, "Warmup Results", bold=True, size=9)
    stats = [
        "Train: 18,244 triplets cached  |  Val: 3,885  |  Test: 3,887  |  100% hit rate after warmup",
        "Speedup: 12 min/epoch (on-the-fly CLIP)  →  3 min/epoch (cache-only)  —  4× faster training",
        "Atomic writes (write-then-rename) prevent corruption from interrupted runs",
    ]
    for i, s in enumerate(stats):
        txt(ax, W/2, 1.32-i*0.36, s, size=8, italic=True)

    return fig, "fig_3_2_clip_warmup"


def fig_3_3():
    W, H = 15, 6.0
    fig, ax = _fig(W, H)
    box(ax, 0.15, 0.15, W-0.3, H-0.3, fill=C_WHITE, ec=C_DARK, lw=2, radius=0.2)
    hdr(ax, 0.15, H-0.85, W-0.3, 0.7,
        "SPRINT II EXECUTION TIMELINE — MODEL TRAINING AND EVALUATION  (Weeks 7–8)",
        size=9.5, fill=C_DARK)

    tasks = [
        ("Week 7\nDays 1–2", "Model Implementation",
         "Implement RelationHead MLP: 1037→LayerNorm→512→GELU→Dropout(0.3)→256→GELU→Dropout(0.3)→16. "
         "Implement FusionModule: learnable log_α, log_τ; output = v + α·(p/τ); identity at init. "
         "Unit tests: verify MLP output shape; FusionModule identity at init; FocalLoss equals CE at γ=0.",
         "All model components unit-tested and integrated"),
        ("Week 7\nDays 3–5", "Training Loop Implementation",
         "FocalLoss(γ=2.0) for training; cross-entropy for validation. "
         "AdamW optimiser (wd=0.01); cosine annealing LR schedule (1e-3 → 1e-6 over 30 epochs). "
         "Checkpoint saved by best validation top-1 accuracy; log.csv records all epoch metrics.",
         "Training infrastructure tested on 5-epoch dry run"),
        ("Week 8\nDays 1–2", "Experiment A Training & Evaluation",
         "Train visual-only configuration (CLIP + spatial, no prior) for 30 epochs, batch=256. "
         "Best checkpoint at epoch 30: val top-1 = 80.88%. "
         "eval.py: R@20=72.80%, R@50=73.01%, R@100=73.01%, mR@50=30.28%. Saved to eval_results.json.",
         "Exp A: R@50=73.01%  |  mR@50=30.28%  |  Checkpoint: run_20260422_213310/best.pt"),
        ("Week 8\nDays 3–5", "Experiment B Training + All Evaluation + Figures",
         "Train prior-guided config (CLIP + spatial + FusionModule) for 30 epochs. "
         "Best checkpoint at epoch 30: val top-1 = 83.86%. "
         "eval.py: R@50=76.73%, mR@50=34.30%. plot_results.py generates 8 publication figures.",
         "Exp B: R@50=76.73%  |  mR@50=34.30%  |  Sprint II COMPLETE"),
    ]
    TH = 1.05
    for i, (week, title, desc, result) in enumerate(tasks):
        ty = H-1.0-(i+1)*(TH+0.12)
        fill = C_HEAD if i < 2 else C_MED
        box(ax, 0.3, ty, 2.0, TH, fill=fill, ec=C_DARK, lw=1.2, radius=0.1)
        txt(ax, 1.3, ty+TH*0.68, week, bold=True, color=C_WHITE, size=8.5)
        txt(ax, 1.3, ty+TH*0.32, "Sprint II", italic=True, color="#cccccc", size=8)
        arr(ax, 2.32, ty+TH/2, 2.52, ty+TH/2)
        box(ax, 2.55, ty, W-2.8, TH, fill=C_BOX if i%2==0 else C_BOX2, ec=C_MED, lw=1)
        txt(ax, 2.75, ty+TH*0.82, title, bold=True, size=9, ha="left")
        half = len(desc)//2; split = desc.rfind(" ", 0, half+40)
        txt(ax, 2.75, ty+TH*0.57, desc[:split], size=7.5, italic=True, ha="left")
        txt(ax, 2.75, ty+TH*0.36, desc[split+1:], size=7.5, italic=True, ha="left")
        txt(ax, 2.75, ty+TH*0.14, f"✓  {result}", size=7.5, bold=True, color="#2a2a2a", ha="left")

    return fig, "fig_3_3_sprint2_timeline"


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
ALL = [fig_1_2, fig_2_1, fig_2_2, fig_3_1, fig_3_2, fig_3_3,
       fig_4_1, fig_4_2, fig_4_3, fig_4_4, fig_5_2]

if __name__ == "__main__":
    print(f"\nGenerating {len(ALL)} diagrams -> {OUT_DIR}\n")
    errors = []
    for fn in ALL:
        try:
            fig, name = fn()
            save_fig(fig, name, dpi=150)
        except Exception as e:
            import traceback
            print(f"  FAIL  {fn.__name__}  ERROR: {e}")
            traceback.print_exc()
            errors.append(fn.__name__)
    print(f"\n{'All done!' if not errors else f'{len(errors)} errors: {errors}'}")
