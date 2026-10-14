import math

import numpy as np
import pandas as pd
import pytest

import analysis as A
from sweep import best_lr, escalation_check, extend_grid


def test_steps_to_target_synthetic():
    C, steps = 10, 1000
    loss = math.log(C) * np.exp(-np.arange(1, steps + 1) / 300.0)  # smooth decay
    m = pd.DataFrame({"step": np.arange(1, steps + 1), "train_loss": loss})
    t = A.steps_to_target(m, C, alpha=0.5, window=1)
    assert t == 1 + int(np.ceil(300 * math.log(2))) or abs(t - 300 * math.log(2)) < 2
    t50 = A.steps_to_target(m, C, alpha=0.5, window=50)
    assert t50 > t  # smoothing over a decaying curve delays the crossing
    assert math.isnan(A.steps_to_target(m, C, alpha=0.001))  # never reached -> censored
    # noisy curve: window prevents a single dip from counting
    noisy = m.copy(); noisy.loc[10, "train_loss"] = 0.0
    assert A.steps_to_target(noisy, C, 0.5, window=1) == 11
    assert A.steps_to_target(noisy, C, 0.5, window=50) == t50


def test_loglog_slope_recovers_known_slope():
    x = np.array([2, 5, 10, 20, 50, 100], float)
    s, se = A.loglog_slope(x, 3.0 * x**-0.5)
    assert abs(s + 0.5) < 1e-9 and se < 1e-6
    s, _ = A.loglog_slope(x, 7.0 / x)
    assert abs(s + 1.0) < 1e-9


def _runs():
    rows = []
    for C in (2, 10):
        for opt, lrs in (("sgd", (0.3, 0.1, 0.03)), ("adam", (3e-3, 1e-3, 3e-4))):
            for i, lr in enumerate(lrs):
                rows.append(dict(run_id=f"p_C{C}_{opt}_lr{lr:.1e}_s0", protocol="p", C=C, superclass=False, seed=0,
                                 optimizer=opt, lr=lr, steps=10, final_train_loss=[0.5, 0.2, 0.4][i] * math.log(C),
                                 best_test_acc=0.5, final_test_acc=0.5, diverged=(opt == "sgd" and C == 2 and i == 0),
                                 wall_time_s=1.0))
    return pd.DataFrame(rows)


def test_best_lr_and_edges():
    runs = _runs()
    lr, status = best_lr(runs, "p", 10, False, "sgd")
    assert lr == 0.1 and status == "ok"
    # C=2 sgd: lr 0.3 diverged, best among rest is 0.1 which is not an edge (0.3 was tried)
    lr, status = best_lr(runs, "p", 2, False, "sgd")
    assert lr == 0.1 and status == "ok"
    r2 = runs.copy(); r2.loc[(r2.optimizer == "adam") & (r2.C == 10) & (r2.lr == 3e-3), "final_train_loss"] = 0.01
    lr, status = best_lr(r2, "p", 10, False, "adam")
    assert status == "edge_high" and abs(extend_grid(r2[(r2.optimizer == "adam") & (r2.C == 10)], status) - 3e-3 * 10**0.5) < 1e-9
    r3 = runs.copy(); r3.loc[(r3.optimizer == "adam") & (r3.C == 10) & (r3.lr == 3e-4), "final_train_loss"] = 0.01
    lr, status = best_lr(r3, "p", 10, False, "adam")
    assert status == "edge_low" and abs(extend_grid(r3[(r3.optimizer == "adam") & (r3.C == 10)], status) - 3e-4 / 10**0.5) < 1e-12
    lr, status = best_lr(runs, "p", 5, False, "adam")
    assert math.isnan(lr) and status == "none"


def test_paired_gaps_raises_on_missing_pair(monkeypatch):
    runs = _runs()
    runs = runs[~((runs.optimizer == "adam") & (runs.C == 2))]  # drop Adam at C=2 entirely
    monkeypatch.setattr(A, "load_metrics", lambda rid: pd.DataFrame({"step": [1, 2], "train_loss": [0.1, 0.1]}))
    with pytest.raises(ValueError, match="unpaired"):
        A.paired_gaps(runs)


def test_paired_gaps_and_choose_alpha(monkeypatch):
    runs = _runs()
    def fake_metrics(rid):
        C = int(rid.split("_C")[1].split("_")[0]); fast = "adam" in rid
        steps = 400
        tau = 60 if fast else 120
        loss = math.log(C) * np.exp(-np.arange(1, steps + 1) / tau)
        return pd.DataFrame({"step": np.arange(1, steps + 1), "train_loss": loss})
    monkeypatch.setattr(A, "load_metrics", fake_metrics)
    g = A.paired_gaps(runs)
    assert set(g.alpha.unique()) == set(A.TARGET_ALPHAS) and len(g) == 2 * len(A.TARGET_ALPHAS)
    assert (g.delta_speed > 0).all()  # Adam is faster in the synthetic curves
    assert not g.censored.any()
    assert A.choose_alpha(g) == 0.5  # primary alpha when uncensored
    g2 = g.copy(); g2.loc[np.isclose(g2.alpha, 0.5) & (g2.C == 2), "censored"] = True
    assert A.choose_alpha(g2) == 0.7  # fall back upward
    g3 = g.copy(); g3.loc[np.isclose(g3.alpha, 0.5) | np.isclose(g3.alpha, 0.7), "censored"] = True
    assert A.choose_alpha(g3) == 0.3


def test_escalation_check_rules():
    # tight, monotone gaps -> no escalation
    rows = [dict(protocol="cifar_merge", C=C, superclass=False, seed=s, alpha=0.5, delta_speed=math.log(C) + 0.01 * s)
            for C in (2, 5, 10, 20) for s in (0, 1, 2)]
    esc, info = escalation_check(pd.DataFrame(rows), alpha=0.5)
    assert not esc and not info["rule_a"] and not info["rule_b"]
    # huge seed noise -> rule (a)
    rows = [dict(protocol="cifar_merge", C=C, superclass=False, seed=s, alpha=0.5, delta_speed=0.01 * math.log(C) + 5.0 * (s - 1))
            for C in (2, 5, 10, 20) for s in (0, 1, 2)]
    esc, info = escalation_check(pd.DataFrame(rows), alpha=0.5)
    assert esc and info["rule_a"]
