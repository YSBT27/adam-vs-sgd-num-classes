"""Training curves at the best LR: normalised train loss vs step, SGD vs Adam, per C (cifar_merge).
Standalone figure script; reads results/best_lr.csv, results/runs.csv and runs/{id}/metrics.csv."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"

ROOT = Path(__file__).parent
SMOOTH_WINDOW, ALPHA, PROTOCOL = 50, 0.7, "cifar_merge"


def main(root: Path = ROOT, out: Path | None = None) -> Path:
    runs = pd.read_csv(root / "results" / "runs.csv").drop_duplicates("run_id", keep="last")
    runs = runs[(runs.protocol == PROTOCOL) & (runs.steps == 8000) & (~runs.superclass.astype(bool))]
    best = pd.read_csv(root / "results" / "best_lr.csv")
    best = best[(best.protocol == PROTOCOL) & (~best.superclass.astype(bool))]
    Cs = sorted(best.C.unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True, sharey=True)
    for ax, C in zip(axes.flat, Cs):
        for opt, color in (("sgd", "C0"), ("adam", "C1")):
            lr = float(best[(best.C == C) & (best.optimizer == opt)].lr.iloc[0])
            sel = runs[(runs.C == C) & (runs.optimizer == opt) & np.isclose(runs.lr, lr) & (~runs.diverged.astype(bool))]
            curves = []
            for rid in sel.run_id:
                m = pd.read_csv(root / "runs" / rid / "metrics.csv")
                curves.append(m.train_loss.rolling(SMOOTH_WINDOW, min_periods=SMOOTH_WINDOW).mean().to_numpy() / math.log(C))
            Y = np.vstack(curves)
            x = np.arange(1, Y.shape[1] + 1)
            ax.fill_between(x, Y.min(0), Y.max(0), color=color, alpha=0.2, lw=0)
            ax.plot(x, Y.mean(0), color=color, lw=1.6, label=f"{opt} (lr={lr:g})")
        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax.axhline(ALPHA, color="k", ls="--", lw=0.8)
        ax.set_title(f"C={C}", fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
        ax.set_ylim(0, 1.05)
    for ax in axes[-1]:
        ax.set_xlabel("Training step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Train loss / log C")
    fig.suptitle(f"Training curves at best LR ({PROTOCOL}): 50-step running mean, mean over seeds, band = min-max; "
                 f"dashed = target {ALPHA} log C; dotted = chance loss")
    fig.tight_layout()
    out = out or (root / "figures" / "fig6_training_curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
