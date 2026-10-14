"""run(cfg) -> RunRecord. Logs per-step loss, checkpoints, one row in results/runs.csv."""
from __future__ import annotations

import argparse
import dataclasses
import math
import time
from dataclasses import asdict, dataclass, fields

import pandas as pd
import torch
import torch.nn.functional as F

from config import RunConfig, ckpt_path, default_device, is_done
from constants import (ADAM_BETAS, ADAM_EPS, DIVERGE_FACTOR, DIVERGE_GRACE, DRY_RUN_STEPS, EVAL_EVERY,
                       FINAL_WINDOW_FRAC, RESULTS_DIR, SGD_MOMENTUM, STEPS, WARMUP_STEPS, WEIGHT_DECAY)
from data import batches, build, normalize
from models import build_model

RUNS_CSV = RESULTS_DIR / "runs.csv"


@dataclass
class RunRecord:
    run_id: str
    protocol: str
    C: int
    superclass: bool
    seed: int
    optimizer: str
    lr: float
    steps: int
    final_train_loss: float
    best_test_acc: float
    final_test_acc: float
    diverged: bool
    wall_time_s: float


def make_optimizer(model, cfg: RunConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=SGD_MOMENTUM, weight_decay=WEIGHT_DECAY, foreach=True)
    return torch.optim.Adam(params, lr=cfg.lr, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY, foreach=True)


def lr_at(step: int, base_lr: float) -> float:
    """Linear warmup to WARMUP_STEPS, then constant."""
    return base_lr * min(1.0, step / WARMUP_STEPS)


@torch.no_grad()
def evaluate(model, x_uint8, y, mean, std, bs: int = 1000) -> tuple[float, float]:
    """(acc, loss) over the full set; eval mode; no grad."""
    was_training = model.training
    model.eval()
    correct, loss_sum = 0, 0.0
    for i in range(0, len(y), bs):
        logits = model(normalize(x_uint8[i : i + bs], mean, std))
        loss_sum += F.cross_entropy(logits, y[i : i + bs], reduction="sum").item()
        correct += (logits.argmax(1) == y[i : i + bs]).sum().item()
    model.train(was_training)
    return correct / len(y), loss_sum / len(y)


def save_checkpoint(model, cfg: RunConfig, step: int, frac: float) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "config_json": cfg.to_json(), "step": step}, ckpt_path(cfg, frac))


def append_record(rec: RunRecord) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([asdict(rec)])
    row.to_csv(RUNS_CSV, mode="a", index=False, header=not RUNS_CSV.exists())


def load_record(cfg: RunConfig) -> RunRecord | None:
    if not RUNS_CSV.exists():
        return None
    df = pd.read_csv(RUNS_CSV)
    hit = df[df.run_id == cfg.run_id]
    if hit.empty:
        return None
    row = hit.iloc[-1].to_dict()
    casts = {int: int, float: float, bool: bool, str: str}
    return RunRecord(**{f.name: casts[f.type if isinstance(f.type, type) else eval(f.type)](row[f.name])
                        for f in fields(RunRecord)})


def run(cfg: RunConfig, device: str | None = None, dry_run: bool = False) -> RunRecord:
    """Train one config. dry_run shortens to DRY_RUN_STEPS; the run then gets its own `_T50` run_id/directory."""
    device = device or default_device(cfg.protocol)
    if dry_run:
        cfg = dataclasses.replace(cfg, steps=min(DRY_RUN_STEPS, cfg.steps))
    if is_done(cfg):
        rec = load_record(cfg)
        if rec is not None:
            return rec
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "config.json").write_text(cfg.to_json())
    data = build(cfg, device)
    model = build_model(cfg).to(device)
    opt = make_optimizer(model, cfg)
    steps = cfg.steps
    ckpt_at = {s: f for f, s in cfg.checkpoint_steps.items()}  # step -> frac
    save_checkpoint(model, cfg, 0, 0.0)
    model.train()

    rows, t0, diverged, step = [], time.time(), False, 0
    for step, (x, y) in enumerate(batches(data, cfg.batch_size, steps, cfg.seed), start=1):
        lr = lr_at(step, cfg.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        row = {"step": step, "train_loss": loss.item(), "lr": lr}
        if step % EVAL_EVERY == 0 or step == steps:
            row["test_acc"], row["test_loss"] = evaluate(model, data.x_test, data.y_test, data.mean, data.std)
        row["wall_time"] = time.time() - t0
        rows.append(row)
        if step in ckpt_at and ckpt_at[step] != 1.0:
            save_checkpoint(model, cfg, step, ckpt_at[step])
        if not math.isfinite(row["train_loss"]) or (
            step > WARMUP_STEPS + DIVERGE_GRACE and row["train_loss"] > DIVERGE_FACTOR * cfg.chance_loss
        ):
            diverged = True
            break

    df = pd.DataFrame(rows)
    for col in ("test_acc", "test_loss"):
        if col not in df:
            df[col] = float("nan")
    if "test_acc" in df and df.test_acc.isna().all():  # diverged before first eval
        acc, tl = evaluate(model, data.x_test, data.y_test, data.mean, data.std)
        df.loc[df.index[-1], ["test_acc", "test_loss"]] = [acc, tl]
    df.to_csv(cfg.run_dir / "metrics.csv", index=False)
    save_checkpoint(model, cfg, step, 1.0)  # "done" marker even for diverged runs
    tail = df.train_loss.tail(max(1, int(FINAL_WINDOW_FRAC * steps)))
    rec = RunRecord(
        run_id=cfg.run_id, protocol=cfg.protocol, C=cfg.C, superclass=cfg.superclass, seed=cfg.seed,
        optimizer=cfg.optimizer, lr=cfg.lr, steps=steps,
        final_train_loss=float(tail.mean()), best_test_acc=float(df.test_acc.max()),
        final_test_acc=float(df.test_acc.dropna().iloc[-1]),
        diverged=diverged, wall_time_s=time.time() - t0,
    )
    append_record(rec)
    return rec


def main():
    ap = argparse.ArgumentParser(description="Train a single run.")
    ap.add_argument("--protocol", default="fmnist_merge", choices=["cifar_merge", "fmnist_merge"])
    ap.add_argument("--C", type=int, default=10)
    ap.add_argument("--superclass", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimizer", default="adam", choices=["sgd", "adam"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true", help=f"{DRY_RUN_STEPS} steps, writes a real run directory")
    a = ap.parse_args()
    cfg = RunConfig(protocol=a.protocol, C=a.C, superclass=a.superclass, seed=a.seed,
                    optimizer=a.optimizer, lr=a.lr, steps=a.steps or STEPS[a.protocol])
    rec = run(cfg, device=a.device, dry_run=a.dry_run)
    print(rec)


if __name__ == "__main__":
    main()
