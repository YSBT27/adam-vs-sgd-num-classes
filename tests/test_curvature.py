import math

import pytest
import torch

from curvature import flat_loss_fn, hutchinson, hvp, last_layer_blocks
from exact import dense_hessian, exact_metrics
from models import FCN, class_block_indices

torch.manual_seed(0)


@pytest.fixture(scope="module")
def setup():
    C, n = 5, 60
    model = FCN(C).double().eval()
    x = torch.randn(n, 1, 28, 28, dtype=torch.float64)
    y = torch.arange(n) % C
    f, theta = flat_loss_fn(model, x, y)
    H = dense_hessian(f, theta)
    return model, x, y, f, theta, H


def test_closed_form_blocks_match_dense_hessian(setup):
    model, x, y, f, theta, H = setup
    est = last_layer_blocks(model, x, y)
    ex = exact_metrics(H, class_block_indices(model))
    for k in ("rho", "r_block", "block_norm_cv"):
        assert abs(est[k] - ex[k]) < 1e-6 * max(1.0, abs(ex[k])), k
    assert torch.allclose(torch.as_tensor(est["block_norms"]), torch.as_tensor(ex["block_norms"]), rtol=1e-6, atol=1e-9)


def test_dense_hessian_symmetric_and_last_layer_is_gauss_newton(setup):
    model, x, y, f, theta, H = setup
    assert torch.allclose(H, H.T)
    idx = torch.cat(class_block_indices(model))
    Hll = H[idx][:, idx]
    evals = torch.linalg.eigvalsh(Hll)
    assert evals.min() > -1e-8  # last-layer Hessian is PSD (CE convex in logits, logits linear in W)


def test_hvp_matches_finite_differences(setup):
    model, x, y, f, theta, H = setup
    v = torch.randn_like(theta)
    Hv = hvp(f, theta, v)
    assert torch.allclose(Hv, H @ v, rtol=1e-6, atol=1e-8)
    eps = 1e-5
    th = theta.detach().requires_grad_(True)
    g = lambda t: torch.autograd.grad(f(t), t)[0]
    fd = (g((theta + eps * v).requires_grad_(True)) - g((theta - eps * v).requires_grad_(True))) / (2 * eps)
    assert torch.allclose(Hv, fd, rtol=1e-4, atol=1e-6)


def test_hvp_autograd_fallback_agrees(setup):
    model, x, y, f, theta, H = setup
    v = torch.randn_like(theta)
    th = theta.detach().requires_grad_(True)
    (g,) = torch.autograd.grad(f(th), th, create_graph=True)
    (Hv2,) = torch.autograd.grad(g, th, grad_outputs=v)
    assert torch.allclose(Hv2, H @ v, rtol=1e-6, atol=1e-8)


def test_hutchinson_matches_exact(setup):
    model, x, y, f, theta, H = setup
    ex = exact_metrics(H, class_block_indices(model))
    est = hutchinson(f, theta, K=4000, seed=123)
    assert abs(est["fro_sq"] - ex["fro_sq"]) / ex["fro_sq"] < 0.05
    # e_diag is ~0.006 here, so ||diag||^2 sits far below the Hutchinson noise floor; the bias-corrected
    # estimator is unbiased but has ~3% relative std at K=4000 (the raw one is >2x off at K=50).
    assert abs(est["e_diag"] - ex["e_diag"]) / ex["e_diag"] < 0.10
    # at K=50 the raw estimate is biased upward by ~(1-e)/K ~ 0.02 >> e; the corrected one must beat it
    small = hutchinson(f, theta, K=50, seed=5)
    assert abs(small["e_diag"] - ex["e_diag"]) < abs(small["e_diag_raw"] - ex["e_diag"])
    assert small["e_diag_raw"] > 2 * ex["e_diag"]


def test_hutchinson_deterministic_in_seed(setup):
    model, x, y, f, theta, H = setup
    a = hutchinson(f, theta, K=3, seed=7)
    b = hutchinson(f, theta, K=3, seed=7)
    c = hutchinson(f, theta, K=3, seed=8)
    assert a == b and a != c


def test_rho_at_init_scales_like_inv_sqrt_C():
    """Sanity: at uniform softmax A_cc=(C-1)/C^2, A_cc'=-1/C^2, so rho = 1/sqrt(C-1), r_block = 1/(C-1)
    exactly (the plan's 1/sqrt(C), 1/C are the large-C forms; at C=2 rho is exactly 1)."""
    rhos, rbs = [], []
    for C in (2, 5, 10):
        torch.manual_seed(0)
        m = FCN(C).double().eval()
        with torch.no_grad():
            m.head.weight.mul_(0.01); m.head.bias.zero_()  # near-uniform softmax
        x = torch.randn(200, 1, 28, 28, dtype=torch.float64)
        out = last_layer_blocks(m, x)
        rhos.append(out["rho"]); rbs.append(out["r_block"])
    for C, r, rb in zip((2, 5, 10), rhos, rbs):
        assert abs(math.log(r) - math.log((C - 1) ** -0.5)) < 0.15
        assert abs(math.log(rb) - math.log(1 / (C - 1))) < 0.15
