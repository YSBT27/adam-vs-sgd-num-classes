"""Report figure S2: block diagram of the two networks (models.SmallCNN and models.FCN).

The diagram is drawn by hand, not generated from the model objects: layer types, widths and output sizes are
written out below and must be kept in sync with models.py (SmallCNN widths 32/64/128, FCN hidden size 32).

Usage:  .venv/bin/python plot_architecture.py      ->  figures/figS2_architecture.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.family"] = "DejaVu Sans"
COL = {"io": "#F2F2F2", "conv": "#CFE0F3", "pool": "#D9EAD3", "lin": "#FBE5C8", "act": "#EADCF0"}
EDGE = "#444444"
fig, ax = plt.subplots(figsize=(12.3, 5.9)); ax.set_xlim(0, 12.3); ax.set_ylim(0, 5.9); ax.axis("off")
def box(x, y, w, h, text, kind, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=COL[kind], ec=EDGE, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")
def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12, lw=1.1, color=EDGE))
def shape(x, y, text): ax.text(x, y, text, ha="center", va="top", fontsize=9, color="#333333")
def chain(items, y, h, gap=0.3, x0=0.25):
    x = x0; cents = []
    for i, (w, text, kind) in enumerate(items):
        box(x, y, w, h, text, kind); cents.append(x + w / 2)
        if i < len(items) - 1: arrow(x + w + 0.03, x + w + gap - 0.03, y + h / 2)
        x += w + gap
    return cents

# ---- SmallCNN
ax.text(0.25, 5.62, "SmallCNN  (main experiment, CIFAR-100; about 0.3M parameters)", fontsize=11.5, fontweight="bold", va="center")
y, h = 4.25, 0.95
c = chain([(1.2, "Input image", "io"), (1.6, "Conv block 1\nwidth 32", "conv"), (1.6, "Conv block 2\nwidth 64", "conv"),
           (1.6, "Conv block 3\nwidth 128", "conv"), (1.6, "Global Average\nPooling", "pool"), (1.6, "Linear\n128 → C", "lin"), (0.9, "logits", "io")], y, h, gap=0.28)
for x, s in zip(c, ["3×32×32", "32×16×16", "64×8×8", "128×4×4", "128", "C", ""]): shape(x, y - 0.1, s)
ax.text(0.25, y - 0.48, "Below each box: output size (channels × height × width).  C = number of classes.", fontsize=9, color="#333333", va="center")
# conv block detail
ax.add_patch(FancyBboxPatch((0.25, 2.42), 11.78, 1.05, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec="#999999", lw=1.0, ls="--"))
ax.text(0.45, 2.945, "Conv block\nof width w:", fontsize=10, fontweight="bold", va="center")
chain([(1.9, "Conv 3×3, w\n(padding 1)", "conv"), (0.9, "ReLU", "act"), (1.9, "Conv 3×3, w\n(padding 1)", "conv"), (0.9, "ReLU", "act"), (1.6, "MaxPool 2×2", "pool")], 2.6, 0.69, gap=0.3, x0=2.1)
ax.text(10.75, 2.945, "no Batch\nNormalization", fontsize=9.5, style="italic", va="center", color="#333333")
# ---- FCN
ax.text(0.25, 1.92, "FCN  (auxiliary experiment, Fashion-MNIST; 6304 + 33C parameters)", fontsize=11.5, fontweight="bold", va="center")
y2, h2 = 0.7, 0.85
c = chain([(1.2, "Input image", "io"), (1.6, "AvgPool 2×2", "pool"), (1.3, "Flatten", "io"), (1.75, "Linear\n196 → 32", "lin"), (1.1, "ReLU", "act"),
           (1.75, "Linear\n32 → C", "lin"), (1.0, "logits", "io")], y2, h2, gap=0.38)
for x, s in zip(c, ["1×28×28", "1×14×14", "196", "32", "32", "C", ""]): shape(x, y2 - 0.1, s)
OUT = Path(__file__).parent / "figures" / "figS2_architecture.png"
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.12, facecolor="white")
print(OUT)
