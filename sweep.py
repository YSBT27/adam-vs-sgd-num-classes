"""Sweep runner: LR grid at seed 0, best-LR selection, edge extension, seeds, escalation. Imports analysis, never the reverse."""
from __future__ import annotations

import argparse
import math
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

from config import RunConfig, default_device, is_done
from constants import (BATCH_SIZE, C_VALUES, FALLBACK_SEEDS, GRID_FACTOR, LR_GRID, MAX_GRID_EXTENSIONS, RESULTS_DIR,
                       SEEDS, STEPS, TARGET_ALPHAS)

RUNS_CSV = RESULTS_DIR / "runs.csv"
NOTES_CSV = RESULTS_DIR / "sweep_notes.csv"
CELL = ["protocol", "C", "superclass", "optimizer"]


def conditions(protocol: str) -> list[tuple[int, bool]]:
    conds = [(C, False) for C in C_VALUES[protocol]]
    if protocol == "cifar_merge":
        conds.append((20, True))
    return conds


def make_cfg(protocol: str, C: int, superclass: bool, optimizer: str, lr: float, seed: int) -> RunConfig:
    return RunConfig(protocol=protocol, C=C, superclass=superclass, seed=seed, optimizer=optimizer,
                     lr=float(lr), steps=STEPS[protocol], batch_size=BATCH_SIZE)


def sweep_configs(protocol: str, seed: int = 0) -> list[RunConfig]:
    """All (C, superclass) conditions x optimizer x LR_GRID at one seed."""
    return [make_cfg(protocol, C, sc, opt, lr, seed)
            for (C, sc), opt in product(conditions(protocol), ("sgd", "adam")) for lr in LR_GRID[opt]]


def load_runs() -> pd.DataFrame:
    if not RUNS_CSV.exists():
        return pd.DataFrame(columns=["run_id"] + CELL + ["seed", "lr", "final_train_loss", "diverged"])
    df = pd.read_csv(RUNS_CSV)
    df["superclass"] = df["superclass"].astype(bool)
    df["diverged"] = df["diverged"].astype(bool)
    df = df[df.steps == df.protocol.map(STEPS)]  # drop dry runs / non-default lengths
    return df.drop_duplicates("run_id", keep="last").reset_index(drop=True)


def cell_rows(runs: pd.DataFrame, protocol: str, C: int, superclass: bool, optimizer: str, seed: int = 0) -> pd.DataFrame:
    m = (runs.protocol == protocol) & (runs.C == C) & (runs.superclass == bool(superclass)) & \
        (runs.optimizer == optimizer) & (runs.seed == seed)
    return runs[m].sort_values("lr")


def best_lr(runs: pd.DataFrame, protocol: str, C: int, superclass: bool, optimizer: str) -> tuple[float, str]:
    """(lr, status) with status in {ok, edge_low, edge_high, none}: argmin final_train_loss over non-diverged seed-0 rows.
    Edge = best is the smallest/largest LR *tried* in this cell (diverged runs count as tried)."""
    rows = cell_rows(runs, protocol, C, superclass, optimizer)
    ok = rows[~rows.diverged & np.isfinite(rows.final_train_loss)]
    if ok.empty:
        return float("nan"), "none"
    lr = float(ok.loc[ok.final_train_loss.idxmin(), "lr"])
    tried = np.sort(rows.lr.unique())
    if np.isclose(lr, tried.min()):
        return lr, "edge_low"
    if np.isclose(lr, tried.max()):
        return lr, "edge_high"
    return lr, "ok"


def extend_grid(cell_runs: pd.DataFrame, status: str) -> float:
    """Next LR one GRID_FACTOR beyond the current edge in the edge direction."""
    tried = cell_runs.lr
    if status == "edge_low":
        return float(tried.min() / GRID_FACTOR)
    if status == "edge_high":
        return float(tried.max() * GRID_FACTOR)
    raise ValueError(status)


def seed_configs(runs: pd.DataFrame, protocol: str, seeds) -> list[RunConfig]:
    """Best LR x seeds (skipping seed 0, which already exists from the sweep)."""
    cfgs = []
    for (C, sc), opt in product(conditions(protocol), ("sgd", "adam")):
        lr, status = best_lr(runs, protocol, C, sc, opt)
        if status == "none":
            print(f"WARNING: no non-diverged run for {protocol} C={C} sc={sc} {opt}; skipping seeds")
            continue
        cfgs += [make_cfg(protocol, C, sc, opt, lr, s) for s in seeds if s != 0]
    return cfgs


def best_lr_runs(protocol: str | None = None) -> list[str]:
    """run_ids (all seeds) at each cell's best LR."""
    runs = load_runs()
    ids = []
    for key, grp in runs.groupby(CELL):
        if protocol and key[0] != protocol:
            continue
        lr, status = best_lr(runs, *key)
        if status != "none":
            ids += grp[np.isclose(grp.lr, lr) & ~grp.diverged].run_id.tolist()
    return sorted(ids)


def escalation_check(gaps: pd.DataFrame, alpha: float | None = None, protocol: str = "cifar_merge",
                     metric: str = "delta_speed") -> tuple[bool, dict]:
    """Plan Section 8.1 step 4. Escalate to FALLBACK_SEEDS if either
    (a) se of the mean gap at any C exceeds half the spread (max - min) of the mean gap across C, or
    (b) the sign of Spearman(gap, C) flips under leave-one-seed-out."""
    g = gaps[(gaps.protocol == protocol) & (~gaps.superclass)]
    if alpha is not None:
        g = g[np.isclose(g.alpha, alpha)]
    g = g.dropna(subset=[metric])
    info = {"metric": metric, "n_C": int(g.C.nunique()), "n_seeds": int(g.seed.nunique())}
    if g.C.nunique() < 3 or g.seed.nunique() < 2:
        info["reason"] = "not enough conditions/seeds to evaluate the rule"
        return False, info
    means = g.groupby("C")[metric].mean()
    ses = g.groupby("C")[metric].sem()
    spread = float(means.max() - means.min())
    info.update({"max_se": float(ses.max()), "spread": spread, "rule_a": bool(ses.max() > 0.5 * spread)})
    full = stats.spearmanr(means.index, means.values).statistic
    signs = []
    for s in sorted(g.seed.unique()):
        m = g[g.seed != s].groupby("C")[metric].mean()
        signs.append(float(np.sign(stats.spearmanr(m.index, m.values).statistic)))
    info.update({"spearman_full": float(full), "loo_signs": signs,
                 "rule_b": bool(len(set(signs + [float(np.sign(full))])) > 1)})
    return bool(info["rule_a"] or info["rule_b"]), info


def execute(cfgs: list[RunConfig], device: str | None = None, dry_run: bool = False) -> None:
    from train import run  # only sweep.py drives train.py

    todo = [c for c in cfgs if not is_done(c)]
    print(f"{len(todo)}/{len(cfgs)} runs to do on {device or 'per-protocol default device'}", flush=True)
    for i, cfg in enumerate(todo, 1):
        rec = run(cfg, device=device, dry_run=dry_run)
        print(f"[{i}/{len(todo)}] {rec.run_id}: loss={rec.final_train_loss:.4f} acc={rec.best_test_acc:.4f} "
              f"diverged={rec.diverged} {rec.wall_time_s:.0f}s", flush=True)


def phase_sweep(protocol: str, device, dry_run=False) -> None:
    execute(sweep_configs(protocol), device, dry_run)


def phase_extend(protocol: str, device, dry_run=False) -> pd.DataFrame:
    """Extend the grid by GRID_FACTOR in the edge direction, at most MAX_GRID_EXTENSIONS times per cell."""
    notes = []
    for (C, sc), opt in product(conditions(protocol), ("sgd", "adam")):
        n_ext = 0
        while True:
            runs = load_runs()
            lr, status = best_lr(runs, protocol, C, sc, opt)
            if status in ("ok", "none") or n_ext >= MAX_GRID_EXTENSIONS:
                break
            new_lr = extend_grid(cell_rows(runs, protocol, C, sc, opt), status)
            print(f"extending {protocol} C={C} sc={sc} {opt}: best {lr:.2e} is {status} -> try {new_lr:.2e}")
            execute([make_cfg(protocol, C, sc, opt, new_lr, 0)], device, dry_run)
            n_ext += 1
        runs = load_runs()
        lr, status = best_lr(runs, protocol, C, sc, opt)
        notes.append({"protocol": protocol, "C": C, "superclass": sc, "optimizer": opt, "best_lr": lr,
                      "status": status, "extensions": n_ext, "under_tuned": status not in ("ok",)})
    df = pd.DataFrame(notes)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prev = pd.read_csv(NOTES_CSV) if NOTES_CSV.exists() else pd.DataFrame()
    if not prev.empty:
        prev = prev[prev.protocol != protocol]
    pd.concat([prev, df]).to_csv(NOTES_CSV, index=False)
    bad = df[df.under_tuned]
    if not bad.empty:
        print(f"WARNING: {len(bad)} cells still at a grid edge after {MAX_GRID_EXTENSIONS} extensions (see {NOTES_CSV}):")
        print(bad.to_string(index=False))
    return df


def phase_seeds(protocol: str, n_seeds: int, device, dry_run=False) -> None:
    seeds = (SEEDS + FALLBACK_SEEDS)[:n_seeds]
    execute(seed_configs(load_runs(), protocol, seeds), device, dry_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["cifar_merge", "fmnist_merge"])
    ap.add_argument("--phase", default="auto", choices=["sweep", "extend", "seeds", "auto"])
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true", help="50-step runs (smoke test of the whole pipeline)")
    a = ap.parse_args()
    if a.phase in ("sweep", "auto"):
        phase_sweep(a.protocol, a.device, a.dry_run)
    if a.phase in ("extend", "auto"):
        phase_extend(a.protocol, a.device, a.dry_run)
    if a.phase in ("seeds", "auto"):
        phase_seeds(a.protocol, a.seeds, a.device, a.dry_run)


if __name__ == "__main__":
    main()
