"""Stage 7: dense Hessian on the FCN; validates curvature.py's estimators. Runs before any CIFAR curvature."""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from config import RunConfig, default_device, sub_seed
from constants import CHECKPOINT_FRACS, C_VALUES, EXACT_TOL, HUTCH_PROBES, RESULTS_DIR, STEPS
from curvature import all_run_ids, block_norm_metrics, flat_loss_fn, hutchinson, last_layer_blocks, load_model_at, load_probe
from models import build_model, class_block_indices, param_count

VALIDATION_CSV = RESULTS_DIR / "exact_validation.csv"
THEORY_CSV = RESULTS_DIR / "theory_check.csv"
MAX_DENSE_PARAMS = 20_000


def dense_hessian(f, theta: Tensor) -> Tensor:
    """torch.func.hessian(f)(theta) as float64 [P, P]."""
    assert theta.numel() < MAX_DENSE_PARAMS, f"P={theta.numel()} too large for a dense Hessian"
    H = torch.func.hessian(f)(theta).double()
    return 0.5 * (H + H.T)  # symmetrize away float round-off


def exact_metrics(H: Tensor, blocks: list[Tensor]) -> dict:
    """rho / r_block / block_norm_cv on the last-layer sub-matrix sliced by `blocks`; e_diag on full H."""
    C = len(blocks)
    idx = torch.cat(blocks).to(H.device)
    Hll = H[idx][:, idx]
    sizes = [len(b) for b in blocks]
    assert len(set(sizes)) == 1
    d1 = sizes[0]
    norms = Hll.view(C, d1, C, d1).permute(0, 2, 1, 3).flatten(2).norm(dim=2)  # [C, C]
    out = block_norm_metrics(norms)
    fro_sq = (H**2).sum()
    e_diag = (H.diagonal() ** 2).sum() / fro_sq
    out.update({"fro_sq": fro_sq.item(), "e_diag": e_diag.item(), "e_diag_P": (e_diag * H.shape[0]).item(),
                "block_norms_matrix": norms.cpu().numpy()})
    return out


def _exact_on(model, px, py) -> dict:
    """Exact metrics for a (CPU, float64) copy of the model on probe (px, py)."""
    m64 = model.double().cpu()
    f, theta = flat_loss_fn(m64, px.cpu().double(), py.cpu())
    H = dense_hessian(f, theta)
    return exact_metrics(H, class_block_indices(m64))


def validate(run_id: str, frac: float, tol: float = EXACT_TOL, K: int = HUTCH_PROBES, device: str | None = None) -> dict:
    """Exact vs estimated (last_layer_blocks + hutchinson) on the same checkpoint.
    Returns {"pass": bool, "rows": [per-metric dicts with exact, estimated, rel_err, pass]}."""
    device = device or "cpu"
    model, cfg = load_model_at(run_id, frac, device)
    assert param_count(model) < MAX_DENSE_PARAMS
    px, py = load_probe(cfg, device)
    est = last_layer_blocks(model, px, py)
    f, theta = flat_loss_fn(model, px, py)
    est.update(hutchinson(f, theta, K, sub_seed(cfg.seed, "hutchinson")))
    ex = _exact_on(model, px, py)
    rows = []
    for m in ("rho", "r_block", "block_norm_cv", "fro_sq", "e_diag", "e_diag_P", "e_diag_raw", "e_diag_se"):
        if m == "e_diag_se":  # calibration: is the exact value within 2 jackknife se of the estimate?
            z = abs(est["e_diag"] - ex["e_diag"]) / est["e_diag_se"] if est["e_diag_se"] > 0 else float("nan")
            rows.append({"run_id": run_id, "frac": frac, "metric": "e_diag_z", "exact": ex["e_diag"],
                         "estimated": est["e_diag_se"], "rel_err": z, "pass": True})
            continue
        e = ex["e_diag"] if m == "e_diag_raw" else ex[m]
        rel = abs(est[m] - e) / abs(e) if e != 0 else float("nan")
        # acceptance (plan Section 10): rho and e_diag within tol; r_block within one grid point (factor sqrt(10)).
        if m in ("rho", "e_diag"):
            ok = rel < tol
        elif m == "r_block":
            ok = abs(math.log10(est[m] / e)) < 0.5 if e > 0 and est[m] > 0 else False
        else:
            ok = True  # reported, not gated
        rows.append({"run_id": run_id, "frac": frac, "metric": m, "exact": e, "estimated": est[m], "rel_err": rel, "pass": ok})
    return {"pass": all(r["pass"] for r in rows), "rows": rows}


def theory_check(protocol: str = "fmnist_merge", seed: int = 0) -> pd.DataFrame:
    """At init for C in C_VALUES[protocol]: exact rho and r_block vs theory.
    Large-C forms: 1/sqrt(C), 1/C. Exact uniform-softmax forms: 1/sqrt(C-1), 1/(C-1) (rho = 1 at C = 2)."""
    recs = []
    for C in C_VALUES[protocol]:
        cfg = RunConfig(protocol=protocol, C=C, superclass=False, seed=seed, optimizer="sgd", lr=0.1, steps=STEPS[protocol])
        model = build_model(cfg).eval()
        assert param_count(model) < MAX_DENSE_PARAMS
        px, py = load_probe(cfg, "cpu")
        ex = _exact_on(model, px, py)
        recs.append({"protocol": protocol, "C": C, "seed": seed, "rho": ex["rho"], "rho_theory": C**-0.5,
                     "rho_theory_exact": (C - 1) ** -0.5, "r_block": ex["r_block"], "r_block_theory": 1.0 / C,
                     "r_block_theory_exact": 1.0 / (C - 1), "e_diag_P": ex["e_diag_P"]})
    df = pd.DataFrame(recs)
    if len(df) > 1:
        lx = np.log(df.C.values)
        df.attrs["rho_slope"] = float(np.polyfit(lx, np.log(df.rho.values), 1)[0])
        df.attrs["r_block_slope"] = float(np.polyfit(lx, np.log(df.r_block.values), 1)[0])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-all", action="store_true", help="all FCN checkpoints -> results/exact_validation.csv")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--frac", type=float, default=None)
    ap.add_argument("--theory", action="store_true")
    ap.add_argument("--K", type=int, default=HUTCH_PROBES)
    ap.add_argument("--tol", type=float, default=EXACT_TOL)
    a = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if a.theory:
        df = theory_check()
        df.to_csv(THEORY_CSV, index=False)
        print(df.to_string(index=False))
        print(f"log-log slopes at init: rho {df.attrs.get('rho_slope', float('nan')):+.3f} (theory -0.5), "
              f"r_block {df.attrs.get('r_block_slope', float('nan')):+.3f} (theory -1)")

    if a.validate_all or a.run_id:
        ids = [a.run_id] if a.run_id else [r for r in all_run_ids() if r.startswith("fmnist_merge")]
        fracs = [a.frac] if a.frac is not None else list(CHECKPOINT_FRACS)
        rows, n_pass = [], 0
        for i in ids:
            for f in fracs:
                res = validate(i, f, a.tol, a.K)
                rows += res["rows"]
                n_pass += res["pass"]
                worst = max(r["rel_err"] for r in res["rows"] if r["metric"] in ("rho", "e_diag"))
                print(f"{'PASS' if res['pass'] else 'FAIL'} {i} frac={f:.1f} worst rel_err(rho,e_diag)={worst:.4f}")
        df = pd.DataFrame(rows)
        df.to_csv(VALIDATION_CSV, index=False)
        print(f"{n_pass}/{len(ids) * len(fracs)} checkpoints pass -> {VALIDATION_CSV}")


if __name__ == "__main__":
    main()
