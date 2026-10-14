"""Control experiment (post-hoc; results in the report's results appendix, H2 section): SGD+momentum with a separately tuned last-layer learning rate.

Question: is Adam's speed advantage explained by the ~1/C shrinkage of last-layer gradients (a scale effect that a
larger head LR would fix), rather than by curvature structure?

Design (same data, model, seeds, batch order, warmup and step budget as the main sweep; nothing in runs/ or
results/runs.csv is touched):
  phase 1: seed 0, body LR = the tuned SGD LR of the cell (results/best_lr.csv), head LR = body LR * m, m in HEAD_MULTS
  phase 2: seeds 1, 2 at the best m per C (lowest final train loss among non-diverged runs; m = 1 is the existing
           baseline run and is read from runs/, not re-run)
Outputs: runs_lastlayer/{run_id}/metrics.csv and one row per run in results/lastlayer_runs.csv. Resumable.

Usage:  .venv/bin/python lastlayer_control.py            (about 30 runs, ~2 min each on MPS)
        .venv/bin/python lastlayer_control.py --dry-run  (50-step smoke test of one run, writes nothing permanent)
        .venv/bin/python lastlayer_control.py --mults 0.1 0.3 --Cs 20 50 100   (extend the grid below m = 1)
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from config import RunConfig, default_device
from constants import (BATCH_SIZE, C_VALUES, DIVERGE_FACTOR, DIVERGE_GRACE, EVAL_EVERY, FINAL_WINDOW_FRAC, RESULTS_DIR,
                       ROOT, SGD_MOMENTUM, STEPS, WARMUP_STEPS, WEIGHT_DECAY)
from data import batches, build
from models import build_model
from train import evaluate

PROTOCOL = "cifar_merge"
HEAD_MULTS = (3.0, 10.0, 30.0)  # m = 1 is the baseline SGD run that already exists in runs/
SEEDS_PHASE2 = (1, 2)
OUT_DIR = ROOT / "runs_lastlayer"
OUT_CSV = RESULTS_DIR / "lastlayer_runs.csv"


def run_id(C: int, lr: float, mult: float, seed: int) -> str:
    return f"{PROTOCOL}_C{C}_sgd_lr{lr:.1e}_head{mult:g}x_s{seed}"


def body_lr(C: int) -> float:
    best = pd.read_csv(RESULTS_DIR / "best_lr.csv")
    hit = best[(best.protocol == PROTOCOL) & (best.C == C) & (~best.superclass.astype(bool)) & (best.optimizer == "sgd")]
    return float(hit.lr.iloc[0])


def baseline_final_loss(C: int, lr: float, seed: int = 0) -> float:
    """final_train_loss of the existing m = 1 run (plain SGD at the tuned LR)."""
    runs = pd.read_csv(RESULTS_DIR / "runs.csv").drop_duplicates("run_id", keep="last")
    rid = f"{PROTOCOL}_C{C}_sgd_lr{lr:.1e}_s{seed}"
    return float(runs[runs.run_id == rid].final_train_loss.iloc[0])


def train_one(C: int, lr: float, mult: float, seed: int, device: str, steps: int, save: bool = True) -> dict:
    rid = run_id(C, lr, mult, seed)
    out_dir = OUT_DIR / rid
    if save and (out_dir / "metrics.csv").exists() and OUT_CSV.exists():
        done = pd.read_csv(OUT_CSV)
        hit = done[done.run_id == rid]
        if not hit.empty:
            return hit.iloc[-1].to_dict()
    cfg = RunConfig(protocol=PROTOCOL, C=C, superclass=False, seed=seed, optimizer="sgd", lr=lr, steps=steps,
                    batch_size=BATCH_SIZE)
    data = build(cfg, device)
    model = build_model(cfg).to(device)  # same init as the main runs at this seed
    head_ids = {id(p) for p in model.head.parameters()}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    groups = [{"params": body, "lr": lr, "base_lr": lr},
              {"params": list(model.head.parameters()), "lr": lr * mult, "base_lr": lr * mult}]
    opt = torch.optim.SGD(groups, lr=lr, momentum=SGD_MOMENTUM, weight_decay=WEIGHT_DECAY)
    model.train()
    rows, t0, diverged, step = [], time.time(), False, 0
    for step, (x, y) in enumerate(batches(data, cfg.batch_size, steps, cfg.seed), start=1):
        warm = min(1.0, step / WARMUP_STEPS)  # same linear warmup as train.lr_at, applied to both groups
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * warm
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        row = {"step": step, "train_loss": loss.item()}
        if step % EVAL_EVERY == 0 or step == steps:
            row["test_acc"], row["test_loss"] = evaluate(model, data.x_test, data.y_test, data.mean, data.std)
        rows.append(row)
        if not math.isfinite(row["train_loss"]) or (
            step > WARMUP_STEPS + DIVERGE_GRACE and row["train_loss"] > DIVERGE_FACTOR * cfg.chance_loss
        ):
            diverged = True
            break
    df = pd.DataFrame(rows)
    if "test_acc" not in df:
        df["test_acc"] = float("nan")
    tail = df.train_loss.tail(max(1, int(FINAL_WINDOW_FRAC * steps)))
    rec = {"run_id": rid, "C": C, "seed": seed, "lr_body": lr, "head_mult": mult, "steps": steps,
           "steps_run": int(step), "final_train_loss": float(tail.mean()),
           "best_test_acc": float(df.test_acc.max()) if df.test_acc.notna().any() else float("nan"),
           "diverged": diverged, "wall_time_s": time.time() - t0}
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "metrics.csv", index=False)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([rec]).to_csv(OUT_CSV, mode="a", index=False, header=not OUT_CSV.exists())
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--mults", type=float, nargs="+", default=list(HEAD_MULTS), help="head-LR multipliers for phase 1")
    ap.add_argument("--Cs", type=int, nargs="+", default=None, help="restrict to these C values")
    ap.add_argument("--dry-run", action="store_true", help="50 steps of one run; nothing is saved")
    a = ap.parse_args()
    device = a.device or default_device(PROTOCOL)
    if a.dry_run:
        rec = train_one(10, body_lr(10), 3.0, 0, device, steps=50, save=False)
        print("dry run ok:", rec)
        return
    steps = STEPS[PROTOCOL]
    Cs = tuple(a.Cs) if a.Cs else C_VALUES[PROTOCOL]
    todo1 = [(C, m) for C in Cs for m in a.mults]
    print(f"phase 1: {len(todo1)} runs (seed 0) on {device}", flush=True)
    for k, (C, m) in enumerate(todo1, 1):
        rec = train_one(C, body_lr(C), m, 0, device, steps)
        print(f"[{k}/{len(todo1)}] {rec['run_id']}: loss={rec['final_train_loss']:.4f} acc={rec['best_test_acc']:.4f} "
              f"diverged={rec['diverged']} {rec['wall_time_s']:.0f}s", flush=True)
    done = pd.read_csv(OUT_CSV)
    done = done[(done.seed == 0) & (done.steps == steps)].drop_duplicates("run_id", keep="last")
    best = {}
    for C in Cs:
        lr = body_lr(C)
        cand = {1.0: baseline_final_loss(C, lr)}
        ok = done[(done.C == C) & (~done.diverged.astype(bool))]
        cand.update({float(m): float(l) for m, l in zip(ok.head_mult, ok.final_train_loss)})
        best[C] = min(cand, key=cand.get)
        print(f"C={C}: final loss by head multiplier {({k: round(v, 4) for k, v in sorted(cand.items())})} -> best m={best[C]:g}",
              flush=True)
    todo2 = [(C, best[C], s) for C in Cs for s in SEEDS_PHASE2 if best[C] != 1.0]
    print(f"phase 2: {len(todo2)} runs (seeds {SEEDS_PHASE2}) at the best multiplier", flush=True)
    for k, (C, m, s) in enumerate(todo2, 1):
        rec = train_one(C, body_lr(C), m, s, device, steps)
        print(f"[{k}/{len(todo2)}] {rec['run_id']}: loss={rec['final_train_loss']:.4f} acc={rec['best_test_acc']:.4f} "
              f"diverged={rec['diverged']} {rec['wall_time_s']:.0f}s", flush=True)
    print(f"done -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
