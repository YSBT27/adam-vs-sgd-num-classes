"""Report figure S1: example of the label merge at C = 20 (random merge vs. the CIFAR-100 superclass merge).

Reads the raw CIFAR-100 training set (probes/raw/cifar-100-python, downloaded by data.load_base) and the label maps
of seed 0 written by data.get_label_map (probes/labelmap_cifar_merge_C20_s0.npy and ..._C20_sc_s0.npy).
Each row shows one merged class with one training image from each of the five original classes merged into it.

Usage:  .venv/bin/python plot_merge_example.py      ->  figures/figS1_merge_example.png
"""
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
PROBES, OUT = ROOT / "probes", ROOT / "figures" / "figS1_merge_example.png"
plt.rcParams["font.family"] = "DejaVu Sans"
raw = PROBES / "raw" / "cifar-100-python"
tr = pickle.load(open(raw / "train", "rb"), encoding="bytes")
meta = pickle.load(open(raw / "meta", "rb"), encoding="bytes")
X = tr[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1); fine = np.array(tr[b"fine_labels"]); coarse = np.array(tr[b"coarse_labels"])
fn = [n.decode() for n in meta[b"fine_label_names"]]; cn = [n.decode() for n in meta[b"coarse_label_names"]]
mr = np.load(PROBES / "labelmap_cifar_merge_C20_s0.npy"); ms = np.load(PROBES / "labelmap_cifar_merge_C20_sc_s0.npy")
c_of_f = np.array([coarse[fine == k][0] for k in range(100)])
assert (ms == c_of_f).all(), "superclass label map must equal the dataset coarse labels"
def pick_img(k): return X[np.flatnonzero(fine == k)[0]]
rand_groups = [0, 1, 2]
sc_groups = [cn.index("flowers"), cn.index("aquatic_mammals"), cn.index("vehicles_1")]
for g in rand_groups: print("random merged class", g, [fn[k] for k in np.flatnonzero(mr == g)])
for g in sc_groups: print("superclass", cn[g], [fn[k] for k in np.flatnonzero(ms == g)])
fig = plt.figure(figsize=(12.6, 5.2))
def panel(x0, title, groups, mp, names):
    fig.text(x0 + 0.235, 0.955, title, ha="center", va="center", fontsize=12.5, fontweight="bold")
    for r, g in enumerate(groups):
        members = np.flatnonzero(mp == g)
        y = 0.64 - r * 0.305
        fig.text(x0 + 0.002, y + 0.115, names(g), ha="left", va="center", fontsize=9.5, rotation=90, fontweight="bold", color="#333333")
        for c, k in enumerate(members):
            ax = fig.add_axes([x0 + 0.03 + c * 0.089, y, 0.082, 0.23])
            ax.imshow(pick_img(k), interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_color("#888888"); s.set_linewidth(0.8)
            ax.set_xlabel(fn[k].replace("_", " "), fontsize=8.5, labelpad=2)
panel(0.01, "Random merge (C = 20, seed 0)", rand_groups, mr, lambda g: f"class {g}")
panel(0.515, "Semantic merge: CIFAR-100 superclasses (C = 20)", sc_groups, ms, lambda g: cn[g].replace("_", " "))
fig.add_artist(plt.Line2D([0.503, 0.503], [0.03, 0.93], color="#bbbbbb", lw=1))
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor="white")
print(OUT)
