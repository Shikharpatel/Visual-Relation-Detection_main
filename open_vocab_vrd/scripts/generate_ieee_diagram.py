"""
generate_ieee_diagram.py
========================
Generates a single full-width IEEE-paper system architecture diagram for the
IB-VRD paper. Designed to span both columns (7.16 inches wide) at 300 DPI.

Run:
    cd open_vocab_vrd/
    python scripts/generate_ieee_diagram.py

Output: outputs/diagrams/ieee_system_arch.png
"""

import sys, io
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
import matplotlib.patheffects as pe

BASE    = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "outputs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Color palette (B&W print-safe) ───────────────────────────────────────────
C_DARK   = "#222222"
C_MED    = "#555555"
C_LIGHT  = "#999999"
C_BOX    = "#F4F4F4"
C_HDR    = "#2B2B2B"
C_WHITE  = "#FFFFFF"
C_DASH   = "#666666"
C_SHADE  = "#E0E0E0"
C_FUSE   = "#EBEBEB"   # slightly different shade for FusionModule


def _round_box(ax, x, y, w, h, fill=C_BOX, ec=C_DARK, lw=1.0, radius=0.06, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=fill, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(p)
    return p


def _hdr(ax, x, y, w, hh, text, fontsize=7.5, zorder=4):
    """Dark header bar with white bold label."""
    p = FancyBboxPatch((x, y), w, hh,
                       boxstyle="round,pad=0,rounding_size=0.04",
                       facecolor=C_HDR, edgecolor=C_HDR, linewidth=0, zorder=zorder)
    ax.add_patch(p)
    ax.text(x + w / 2, y + hh / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=C_WHITE, zorder=zorder + 1)


def _inner(ax, x, y, w, h, lines, fontsize=6.5, zorder=3):
    """Light inner box with text lines centered."""
    _round_box(ax, x, y, w, h, fill=C_WHITE, ec=C_LIGHT, lw=0.6, radius=0.04, zorder=zorder)
    cy = y + h
    for i, line in enumerate(lines):
        cy -= h / (len(lines) + 0.5)
        ax.text(x + w / 2, cy, line,
                ha="center", va="center", fontsize=fontsize,
                color=C_DARK, zorder=zorder + 1)


def _arrow(ax, x1, y1, x2, y2, label="", lw=1.0, style="-", color=C_DARK,
           hw=0.04, hl=0.06, lblside="top", zorder=5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=f"-|>,head_width={hw},head_length={hl}",
                                color=color, lw=lw, linestyle=style,
                                connectionstyle="arc3,rad=0"),
                zorder=zorder)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dy = 0.055 if lblside == "top" else -0.055
        ax.text(mx, my + dy, label, ha="center", va="center",
                fontsize=5.8, color=C_MED, style="italic", zorder=zorder + 1)


def _dashed_arrow(ax, x1, y1, x2, y2, label="", color=C_DASH, lw=0.8):
    _arrow(ax, x1, y1, x2, y2, label=label, lw=lw, style="--", color=color,
           hw=0.035, hl=0.05, lblside="top")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────

def draw():
    # Canvas: 7.16 w x 3.0 h inches at 300 DPI
    W, H = 7.16, 3.0
    DPI  = 300
    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_aspect("equal")
    fig.patch.set_facecolor(C_WHITE)

    # ── Layout constants ──────────────────────────────────────────────────────
    # Five stage columns:  Input | Detection | Encoding+Spatial | RelHead | Output
    COL_X   = [0.08, 0.94, 2.05, 3.70, 5.85, 6.82]  # left edges + right edge
    STAGE_W = [COL_X[i+1] - COL_X[i] - 0.06 for i in range(5)]

    HDR_H   = 0.26
    BOX_Y   = 0.30          # bottom of main stage boxes
    BOX_H   = 2.30          # height of stage boxes
    HDR_Y   = BOX_Y + BOX_H - HDR_H   # top of each stage box

    # ── Stage 1: Input ────────────────────────────────────────────────────────
    s1x = COL_X[0]
    _round_box(ax, s1x, BOX_Y, STAGE_W[0], BOX_H, fill=C_BOX, ec=C_DARK, lw=1.0)
    _hdr(ax, s1x, HDR_Y, STAGE_W[0], HDR_H, "INPUT")
    # Image icon (simple grid)
    ix, iy, iw, ih = s1x + 0.05, BOX_Y + 1.05, STAGE_W[0] - 0.10, 0.72
    _round_box(ax, ix, iy, iw, ih, fill=C_SHADE, ec=C_MED, lw=0.8, radius=0.03, zorder=3)
    # mini grid lines
    for gx in [ix + iw*0.33, ix + iw*0.67]:
        ax.plot([gx, gx], [iy, iy + ih], color=C_LIGHT, lw=0.5, zorder=4)
    for gy in [iy + ih*0.33, iy + ih*0.67]:
        ax.plot([ix, ix + iw], [gy, gy], color=C_LIGHT, lw=0.5, zorder=4)
    ax.text(ix + iw/2, iy + ih/2, "RGB", ha="center", va="center",
            fontsize=7, fontweight="bold", color=C_MED, zorder=5)
    ax.text(s1x + STAGE_W[0]/2, BOX_Y + 0.82, "H \u00d7 W \u00d7 3",
            ha="center", va="center", fontsize=6.2, color=C_MED, style="italic", zorder=4)
    ax.text(s1x + STAGE_W[0]/2, BOX_Y + 0.52, "scene image",
            ha="center", va="center", fontsize=6.0, color=C_MED, zorder=4)

    # ── Stage 2: YOLO-World Detection ────────────────────────────────────────
    s2x = COL_X[1]
    _round_box(ax, s2x, BOX_Y, STAGE_W[1], BOX_H, fill=C_BOX, ec=C_DARK, lw=1.0)
    _hdr(ax, s2x, HDR_Y, STAGE_W[1], HDR_H, "DETECTION")
    _inner(ax, s2x+0.06, BOX_Y+1.35, STAGE_W[1]-0.12, 0.70,
           ["YOLO-World", "yolov8s-world"], fontsize=6.5, zorder=3)
    ax.text(s2x + STAGE_W[1]/2, BOX_Y + 1.23, "open-vocabulary",
            ha="center", va="center", fontsize=5.8, color=C_MED, style="italic")
    # Config labels
    for i, txt in enumerate(["conf \u2265 0.3", "top-k = 10", "NMS IoU = 0.5"]):
        ax.text(s2x + STAGE_W[1]/2, BOX_Y + 0.92 - i*0.24, txt,
                ha="center", va="center", fontsize=6.0, color=C_MED)
    # Output label
    ax.text(s2x + STAGE_W[1]/2, BOX_Y + 0.38,
            "N \u00d7 {label, box, score}",
            ha="center", va="center", fontsize=5.8, color=C_MED, style="italic")

    # ── Stage 3: Encoding + Spatial ───────────────────────────────────────────
    s3x = COL_X[2]
    _round_box(ax, s3x, BOX_Y, STAGE_W[2], BOX_H, fill=C_BOX, ec=C_DARK, lw=1.0)
    _hdr(ax, s3x, HDR_Y, STAGE_W[2], HDR_H, "ENCODING")

    # CLIP sub-box (top half)
    cx, cy2 = s3x + 0.07, BOX_Y + 1.32
    cw, ch  = STAGE_W[2] - 0.14, 0.78
    _inner(ax, cx, cy2, cw, ch,
           ["CLIP ViT-B/32", "(frozen)"], fontsize=6.5)
    ax.text(s3x + STAGE_W[2]/2, cy2 - 0.12,
            "512-D, L2-norm", ha="center", va="center",
            fontsize=5.8, color=C_MED, style="italic")

    # Spatial sub-box (bottom half)
    sx2, sy2 = s3x + 0.07, BOX_Y + 0.44
    sw, sh   = STAGE_W[2] - 0.14, 0.72
    _round_box(ax, sx2, sy2, sw, sh, fill=C_SHADE, ec=C_MED, lw=0.7, radius=0.04, zorder=3)
    ax.text(sx2 + sw/2, sy2 + sh/2 + 0.12, "Spatial Features",
            ha="center", va="center", fontsize=6.5, fontweight="bold", color=C_DARK, zorder=4)
    ax.text(sx2 + sw/2, sy2 + sh/2 - 0.10, "13-D geometric",
            ha="center", va="center", fontsize=5.8, color=C_MED, style="italic", zorder=4)
    ax.text(sx2 + sw/2, sy2 + sh/2 - 0.27,
            "\u0394cx, \u0394cy, log(w/h ratios), IoU",
            ha="center", va="center", fontsize=5.5, color=C_MED, zorder=4)

    # Separator line
    ax.plot([s3x + 0.07, s3x + STAGE_W[2] - 0.07],
            [BOX_Y + 1.24, BOX_Y + 1.24], color=C_LIGHT, lw=0.5, ls="--", zorder=3)

    # ── Stage 4: RelationHead + FusionModule ──────────────────────────────────
    s4x = COL_X[3]
    _round_box(ax, s4x, BOX_Y, STAGE_W[3], BOX_H, fill=C_BOX, ec=C_DARK, lw=1.0)
    _hdr(ax, s4x, HDR_Y, STAGE_W[3], HDR_H, "RELATION REASONING")

    # MLP sub-box
    mx, my = s4x + 0.07, BOX_Y + 1.20
    mw, mh = STAGE_W[3] - 0.14, 0.90
    _inner(ax, mx, my, mw, mh,
           ["RelationHead MLP", "LayerNorm(1037)", "Linear \u2192 512 \u2192 256 \u2192 16",
            "GELU + Dropout(0.3)"], fontsize=6.2)

    # Concat label
    ax.text(s4x + STAGE_W[3]/2, BOX_Y + 1.12,
            "concat: 512 + 512 + 13 = 1037-D",
            ha="center", va="center", fontsize=5.8, color=C_MED, style="italic")

    # FusionModule dashed sub-box
    fx, fy = s4x + 0.07, BOX_Y + 0.36
    fw, fh = STAGE_W[3] - 0.14, 0.68
    p = FancyBboxPatch((fx, fy), fw, fh,
                       boxstyle="round,pad=0,rounding_size=0.04",
                       facecolor=C_FUSE, edgecolor=C_MED, linewidth=0.8,
                       linestyle="--", zorder=3)
    ax.add_patch(p)
    ax.text(fx + fw/2, fy + fh - 0.14, "FusionModule",
            ha="center", va="center", fontsize=6.5, fontweight="bold", color=C_DARK, zorder=4)
    ax.text(fx + fw/2, fy + fh/2 - 0.04,
            r"$\mathbf{v} + \alpha \cdot (\mathbf{p}/\tau)$",
            ha="center", va="center", fontsize=7.0, color=C_DARK, zorder=4,
            math_fontfamily="dejavusans")
    ax.text(fx + fw/2, fy + 0.12,
            "\u03b1, \u03c4 learnable (+2 params)",
            ha="center", va="center", fontsize=5.6, color=C_MED, style="italic", zorder=4)

    # LLM Prior label (external, with dashed arrow coming in)
    px_lbl = s4x + STAGE_W[3] * 0.5
    py_lbl = BOX_Y + 0.08
    ax.text(px_lbl, py_lbl, "LLM prior vector p (16-D)",
            ha="center", va="center", fontsize=5.8, color=C_MED, style="italic")

    # ── Stage 5: Output ───────────────────────────────────────────────────────
    s5x = COL_X[4]
    _round_box(ax, s5x, BOX_Y, STAGE_W[4], BOX_H, fill=C_BOX, ec=C_DARK, lw=1.0)
    _hdr(ax, s5x, HDR_Y, STAGE_W[4], HDR_H, "OUTPUT")

    # Softmax box
    _inner(ax, s5x+0.07, BOX_Y+1.68, STAGE_W[4]-0.14, 0.42,
           ["Softmax"], fontsize=6.5)
    ax.text(s5x + STAGE_W[4]/2, BOX_Y + 1.58,
            "N(N-1) \u00d7 16",
            ha="center", va="center", fontsize=5.8, color=C_MED, style="italic")

    # Rank & select box
    _inner(ax, s5x+0.07, BOX_Y+1.00, STAGE_W[4]-0.14, 0.42,
           ["Rank & Top-K"], fontsize=6.5)

    # Triplet examples
    ax.text(s5x + STAGE_W[4]/2, BOX_Y + 0.82,
            "ranked triplets:", ha="center", va="center",
            fontsize=6.0, fontweight="bold", color=C_DARK)
    triplets = [
        "(person, wearing, hat)",
        "(dog, on, floor)",
        "(cup, near, bottle)",
    ]
    for i, t in enumerate(triplets):
        ax.text(s5x + STAGE_W[4]/2, BOX_Y + 0.65 - i * 0.18,
                t, ha="center", va="center",
                fontsize=5.6, color=C_MED, style="italic")

    # ── Inter-stage arrows ────────────────────────────────────────────────────
    MID_Y = BOX_Y + BOX_H * 0.62   # mid-height for horizontal arrows

    # Input -> Detection
    _arrow(ax, COL_X[0] + STAGE_W[0], MID_Y + 0.18,
           COL_X[1], MID_Y + 0.18, label="image", lw=1.1)

    # Detection -> Encoding (CLIP, upper arrow)
    _arrow(ax, COL_X[1] + STAGE_W[1], MID_Y + 0.50,
           COL_X[2], MID_Y + 0.50, label="N boxes", lw=1.1)

    # Detection -> Encoding (Spatial, lower arrow)
    _arrow(ax, COL_X[1] + STAGE_W[1], MID_Y - 0.26,
           COL_X[2], MID_Y - 0.26, label="N boxes", lw=0.9, color=C_MED)

    # Encoding (CLIP) -> RelHead
    _arrow(ax, COL_X[2] + STAGE_W[2], MID_Y + 0.50,
           COL_X[3], MID_Y + 0.50, label="N\u00d7512-D", lw=1.1)

    # Encoding (Spatial) -> RelHead
    _arrow(ax, COL_X[2] + STAGE_W[2], MID_Y - 0.26,
           COL_X[3], MID_Y - 0.26, label="N(N-1)\u00d713-D", lw=0.9, color=C_MED)

    # RelHead -> Output (visual logits -> softmax)
    _arrow(ax, COL_X[3] + STAGE_W[3], MID_Y + 0.50,
           COL_X[4], MID_Y + 0.50, label="16-D logits", lw=1.1)

    # Softmax -> Rank (within output stage)
    _arrow(ax, s5x + STAGE_W[4]/2, BOX_Y + 1.68,
           s5x + STAGE_W[4]/2, BOX_Y + 1.44,
           label="", lw=0.8)

    # Rank -> triplets (within output stage)
    _arrow(ax, s5x + STAGE_W[4]/2, BOX_Y + 1.00,
           s5x + STAGE_W[4]/2, BOX_Y + 0.88,
           label="", lw=0.8)

    # FusionModule dashed arrow from prior label (bottom)
    _dashed_arrow(ax, px_lbl, BOX_Y + 0.16, fx + fw/2, fy, color=C_MED)

    # Vertical internal arrow in RelHead: MLP -> FusionModule
    _arrow(ax, s4x + STAGE_W[3]/2, BOX_Y + 1.20,
           s4x + STAGE_W[3]/2, BOX_Y + 1.04 + fh,
           label="16-D", lw=0.8, hw=0.03, hl=0.05)

    # FusionModule -> out arrow to stage 5 (lower)
    _arrow(ax, COL_X[3] + STAGE_W[3], BOX_Y + 0.70,
           COL_X[4], BOX_Y + 0.70,
           label="fused 16-D", lw=0.9, color=C_MED)

    # Dashed "optional" label on FusionModule path
    ax.text((COL_X[3] + STAGE_W[3] + COL_X[4]) / 2,
            BOX_Y + 0.55,
            "(Exp. B only)", ha="center", va="center",
            fontsize=5.4, color=C_MED, style="italic")

    # ── Legend ────────────────────────────────────────────────────────────────
    lx, ly = 0.08, 0.10
    ax.plot([lx, lx + 0.20], [ly, ly], color=C_DARK, lw=1.1, zorder=5)
    ax.annotate("", xy=(lx + 0.20, ly), xytext=(lx + 0.17, ly),
                arrowprops=dict(arrowstyle="-|>,head_width=0.03,head_length=0.04",
                                color=C_DARK, lw=1.1), zorder=5)
    ax.text(lx + 0.22, ly, "data flow", va="center", fontsize=5.5, color=C_MED)

    lx2 = lx + 1.0
    ax.plot([lx2, lx2 + 0.20], [ly, ly], color=C_MED, lw=0.8, linestyle="--", zorder=5)
    ax.annotate("", xy=(lx2 + 0.20, ly), xytext=(lx2 + 0.17, ly),
                arrowprops=dict(arrowstyle="-|>,head_width=0.025,head_length=0.04",
                                color=C_MED, lw=0.8), zorder=5)
    ax.text(lx2 + 0.22, ly, "optional (prior, Exp. B)", va="center",
            fontsize=5.5, color=C_MED)

    # Figure title (small, below main boxes)
    ax.text(W / 2, 0.10, "", ha="center", va="center", fontsize=6, color=C_MED)

    plt.tight_layout(pad=0.1)
    out = OUT_DIR / "ieee_system_arch.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    kb = out.stat().st_size // 1024
    print(f"Saved: {out.name}  ({kb} KB)")
    return out


if __name__ == "__main__":
    path = draw()
    print(f"Done -> {path}")
