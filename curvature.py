"""Curvature estimators on the fixed probe batch: last-layer closed form + full-network Hutchinson.
Never imports train.py; rebuilds models from runs/{id}/ckpt_{frac}.pt."""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.func import functional_call

from config import RunConfig, ckpt_path, default_device, sub_seed
from constants import CHECKPOINT_FRACS, EXACT_TOL, HUTCH_PROBES, RESULTS_DIR, RUNS_DIR, STEPS
from data import get_label_map, get_probe, load_base
from models import build_model

CURV_CSV = RESULTS_DIR / "curvature.csv"
METRICS = ("rho", "r_block", "block_norm_cv", "fro_sq", "e_diag", "e_diag_P", "e_diag_raw", "e_diag_se", "e_diag_P_se")


@dataclass
class CurvatureRow:
    run_id: str
    frac: float
    metric: str
    value: float


def _acc_device(device) -> str:
    """Device for float64 accumulation (MPS has no float64)."""
    return "cpu" if str(device).startswith("mps") else str(device)


# --------------------------------------------------------------------------- loading


def load_model_at(run_id: str, frac: float, device: str) -> tuple[nn.Module, RunConfig]:
    cfg = RunConfig.load(run_id)
    ck = torch.load(ckpt_path(cfg, frac), map_location="cpu", weights_only=True)
    assert RunConfig.from_json(ck["config_json"]) == cfg
    model = build_model(cfg)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), cfg


def load_probe(cfg: RunConfig, device: str) -> tuple[Tensor, Tensor]:
    """Normalized probe_x and merged probe_y for this config (same images for every C)."""
    base = load_base(cfg.protocol)
    probe = get_probe(cfg.protocol, cfg.seed, base)
    lm = torch.from_numpy(get_label_map(cfg.protocol, cfg.C, cfg.superclass, cfg.seed))
    y = lm[base["y_train_fine"][probe["indices"]]]
    return probe["x"].to(device), y.to(device)


# --------------------------------------------------------------------------- flat loss


def flat_loss_fn(model: nn.Module, probe_x: Tensor, probe_y: Tensor) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """(f, theta0): f(theta_flat) = mean CE of `model` with parameters theta_flat on the probe.
    Flattening order = named_parameters() order (matches models.class_block_indices)."""
    names, shapes, numels = [], [], []
    for n, p in model.named_parameters():
        names.append(n)
        shapes.append(p.shape)
        numels.append(p.numel())
    theta0 = torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def f(theta: Tensor) -> Tensor:
        params = {n: t.view(s) for n, t, s in zip(names, torch.split(theta, numels), shapes)}
        logits = functional_call(model, params, (probe_x,))
        return torch.nn.functional.cross_entropy(logits, probe_y)

    return f, theta0


# --------------------------------------------------------------------------- last layer


def block_norm_metrics(norms: Tensor) -> dict:
    """rho, r_block, block_norm_cv from the [C, C] matrix of block Frobenius norms."""
    C = norms.shape[0]
    diag = norms.diagonal()
    off = norms - torch.diag(diag)
    rho = off.norm() / diag.norm()
    mask = ~torch.eye(C, dtype=torch.bool, device=norms.device)
    r_block = (norms / torch.sqrt(diag[:, None] * diag[None, :]))[mask].median()
    cv = diag.std() / diag.mean() if C > 1 else torch.zeros((), dtype=norms.dtype)
    return {"rho": rho.item(), "r_block": r_block.item(), "block_norm_cv": cv.item(),
            "block_norms": diag.cpu().numpy()}


@torch.no_grad()
def last_layer_blocks(model: nn.Module, probe_x: Tensor, probe_y: Tensor | None = None) -> dict:
    """Closed-form last-layer Hessian blocks H_{cc'} = (1/n) sum_i A[i,c,c'] x_i x_i^T (x with bias 1).
    Uses ||H_{cc'}||_F^2 = (1/n^2) a_{cc'}^T G a_{cc'} with G = (X X^T)^{o2}, so no (d+1)^2 blocks are formed.
    Accumulates in float64 (on CPU when the model is on MPS)."""
    model.eval()
    acc = _acc_device(probe_x.device)
    feats = model.features(probe_x)
    logits = model.head(feats)
    x = feats.detach().to(acc).double()
    x = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype, device=acc)], 1)  # [n, d+1]
    p = torch.softmax(logits.detach().to(acc).double(), dim=1)  # [n, C]
    n, C = p.shape
    A = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]  # [n, C, C]
    G = (x @ x.T) ** 2  # [n, n]
    a = A.reshape(n, C * C)
    norms_sq = (a * (G @ a)).sum(0) / (n * n)
    norms = norms_sq.clamp_min(0).sqrt().view(C, C)
    return block_norm_metrics(norms)


# --------------------------------------------------------------------------- Hutchinson


def hvp(f: Callable[[Tensor], Tensor], theta: Tensor, v: Tensor) -> Tensor:
    """Hessian-vector product: forward-over-reverse (jvp of grad) with an autograd double-backward fallback."""
    try:
        _, Hv = torch.func.jvp(torch.func.grad(f), (theta,), (v,))
        return Hv
    except Exception:  # e.g. an op without forward-mode AD support
        th = theta.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(f(th), th, create_graph=True)
        (Hv,) = torch.autograd.grad(g, th, grad_outputs=v)
        return Hv.detach()


def hutchinson(f: Callable[[Tensor], Tensor], theta: Tensor, K: int, seed: int) -> dict:
    """Rademacher probes v_k (CPU generator -> device-independent), d_k = v_k * (H v_k).
    With the K x K Gram matrix G_kl = d_k . d_l:  G_kk = ||H v_k||^2, so fro_sq = mean_k G_kk, and
    E[G_kl] = ||diag H||^2 for k != l, so ||diag H||^2 = mean_{k != l} G_kl (unbiased U-statistic).
    e_diag_raw = ||mean_k d_k||^2 / fro_sq is the naive estimate, biased upward by ~(1 - e_diag)/K.
    e_diag_se is a leave-one-probe-out jackknife standard error of e_diag."""
    g = torch.Generator().manual_seed(seed)
    P = theta.numel()
    acc = _acc_device(theta.device)
    D = torch.empty(K, P, dtype=torch.float64, device=acc)
    for k in range(K):
        v = (torch.randint(0, 2, (P,), generator=g, dtype=theta.dtype) * 2 - 1).to(theta.device)
        Hv = hvp(f, theta, v)
        D[k] = (v.to(acc).double() * Hv.to(acc).double())
    G = D @ D.T  # [K, K]
    off = ~torch.eye(K, dtype=torch.bool, device=acc)
    fro_sq = G.diagonal().mean()
    diag_sq = G[off].mean() if K > 1 else torch.zeros((), dtype=torch.float64, device=acc)
    e_corr = diag_sq / fro_sq
    e_raw = (D.mean(0) ** 2).sum() / fro_sq
    se = float("nan")
    if K > 2:  # leave-one-probe-out jackknife of the ratio, in closed form
        tr, diag_G = G.diagonal().sum(), G.diagonal()
        row_off = G.sum(1) - diag_G  # sum of off-diagonal entries in row k
        sum_off = row_off.sum()
        loo_diag_sq = (sum_off - 2 * row_off) / ((K - 1) * (K - 2))
        loo_fro = (tr - diag_G) / (K - 1)
        loo = loo_diag_sq / loo_fro
        se = math.sqrt((K - 1) / K * ((loo - loo.mean()) ** 2).sum().item())
    return {"fro_sq": fro_sq.item(), "e_diag": e_corr.item(), "e_diag_P": (e_corr * P).item(),
            "e_diag_raw": e_raw.item(), "e_diag_se": se, "e_diag_P_se": se * P}


# --------------------------------------------------------------------------- measure


def measure(run_id: str, frac: float, device: str | None = None, K: int = HUTCH_PROBES) -> list[CurvatureRow]:
    cfg0 = RunConfig.load(run_id)
    device = device or default_device(cfg0.protocol)
    model, cfg = load_model_at(run_id, frac, device)
    px, py = load_probe(cfg, device)
    out = last_layer_blocks(model, px, py)
    np.save(RUNS_DIR / run_id / f"blocks_{frac:.1f}.npy", out.pop("block_norms"))
    f, theta0 = flat_loss_fn(model, px, py)
    out.update(hutchinson(f, theta0, K, sub_seed(cfg.seed, "hutchinson")))
    return [CurvatureRow(run_id, frac, m, float(out[m])) for m in METRICS]


def load_curvature_long() -> pd.DataFrame:
    if not CURV_CSV.exists():
        return pd.DataFrame(columns=["run_id", "frac", "metric", "value"])
    return pd.read_csv(CURV_CSV)


def is_measured(run_id: str, frac: float, curv: pd.DataFrame | None = None) -> bool:
    curv = load_curvature_long() if curv is None else curv
    hit = curv[(curv.run_id == run_id) & (np.isclose(curv.frac, frac))]
    return set(METRICS) <= set(hit.metric)


def append_rows(rows: list[CurvatureRow]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(r) for r in rows]).to_csv(CURV_CSV, mode="a", index=False, header=not CURV_CSV.exists())


def _assert_validated_for_cifar(run_ids: list[str]) -> None:
    """Stage 7 gate (plan Section 10). rho / r_block rows must all pass. e_diag is reported: at K = HUTCH_PROBES its
    Hutchinson estimate is unbiased but noisy, so its pass rate is printed as a warning rather than blocking."""
    if not any(r.startswith("cifar_merge") for r in run_ids):
        return
    p = RESULTS_DIR / "exact_validation.csv"
    assert p.exists(), "Stage 7 first: run `python exact.py --validate-all` before measuring CIFAR checkpoints."
    ev = pd.read_csv(p)
    gated = ev[ev.metric.isin(["rho", "r_block"])]
    bad = gated[~gated["pass"].astype(bool)]
    assert bad.empty, f"exact validation has failing rho/r_block rows:\n{bad.to_string()}"
    ed = ev[ev.metric == "e_diag"]
    if not ed.empty and not ed["pass"].astype(bool).all():
        print(f"WARNING: e_diag passed the {EXACT_TOL:.0%} exact check on only {ed['pass'].astype(bool).mean():.0%} of FCN "
              f"checkpoints (median rel err {ed.rel_err.median():.2f}). Treat e_diag as noisy at HUTCH_PROBES={HUTCH_PROBES}; "
              f"raise HUTCH_PROBES or read e_diag with its e_diag_se.")


def all_run_ids(include_dry: bool = False) -> list[str]:
    """Finished runs (ckpt_1.0.pt present). Dry / non-default-length runs (run_id ..._T{steps}) are excluded by default."""
    ids = []
    for p in sorted(RUNS_DIR.iterdir()):
        if not ((p / "config.json").exists() and (p / "ckpt_1.0.pt").exists()):
            continue
        cfg = RunConfig.load(p.name)
        if include_dry or cfg.steps == STEPS[cfg.protocol]:
            ids.append(p.name)
    return ids


def best_lr_run_ids() -> list[str]:
    from sweep import best_lr_runs  # sweep imports analysis; neither imports curvature
    return best_lr_runs()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-checkpoints", action="store_true")
    ap.add_argument("--best-lr-only", action="store_true")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--frac", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--K", type=int, default=HUTCH_PROBES)
    ap.add_argument("--skip-validation-check", action="store_true")
    a = ap.parse_args()
    device = a.device  # None -> per-protocol default inside measure()

    if a.run_id:
        ids = [a.run_id]
        fracs = [a.frac] if a.frac is not None else list(CHECKPOINT_FRACS)
    elif a.all_checkpoints:
        ids = best_lr_run_ids() if a.best_lr_only else all_run_ids()
        if a.protocol:
            ids = [i for i in ids if i.startswith(a.protocol)]
        fracs = list(CHECKPOINT_FRACS)
    else:
        ap.error("give --run-id ID [--frac F] or --all-checkpoints [--best-lr-only]")
    if not a.skip_validation_check:
        _assert_validated_for_cifar(ids)

    curv = load_curvature_long()
    todo = [(i, f) for i in ids for f in fracs if not is_measured(i, f, curv) and ckpt_path(RunConfig.load(i), f).exists()]
    print(f"{len(todo)} (run, frac) pairs to measure on {device or 'default device'}", flush=True)
    for k, (i, f) in enumerate(todo, 1):
        rows = measure(i, f, device, a.K)
        append_rows(rows)
        vals = {r.metric: r.value for r in rows}
        print(f"[{k}/{len(todo)}] {i} frac={f:.1f} rho={vals['rho']:.4f} r_block={vals['r_block']:.4f} "
              f"e_diag_P={vals['e_diag_P']:.3f}", flush=True)


if __name__ == "__main__":
    main()
