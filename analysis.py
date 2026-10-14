"""Join tables, gap metrics, slopes, hypothesis tests, figures. Pure dataframe functions; make_figures has side effects."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from constants import (C_VALUES, CHECKPOINT_FRACS, FIGURES_DIR, RESULTS_DIR, RUNS_DIR, SMOOTH_WINDOW,
                       TARGET_ALPHAS)

RUNS_CSV = RESULTS_DIR / "runs.csv"
CURV_CSV = RESULTS_DIR / "curvature.csv"
NOTES_JSON = RESULTS_DIR / "analysis_notes.json"
CELL = ["protocol", "C", "superclass", "optimizer"]
COND = ["protocol", "C", "superclass"]


# --------------------------------------------------------------------------- loading


def load_runs() -> pd.DataFrame:
    """results/runs.csv restricted to full-length runs (dry runs carry a _T suffix and non-default steps)."""
    from sweep import load_runs as _load

    return _load()


def load_curvature() -> pd.DataFrame:
    """results/curvature.csv pivoted to one row per (run_id, frac)."""
    long = pd.read_csv(CURV_CSV)
    long = long.drop_duplicates(["run_id", "frac", "metric"], keep="last")
    wide = long.pivot(index=["run_id", "frac"], columns="metric", values="value").reset_index()
    wide.columns.name = None
    return wide


def load_metrics(run_id: str) -> pd.DataFrame:
    return pd.read_csv(RUNS_DIR / run_id / "metrics.csv")


# --------------------------------------------------------------------------- per-run quantities


def steps_to_target(metrics: pd.DataFrame, C: int, alpha: float, window: int = SMOOTH_WINDOW) -> float:
    """First step where rolling mean(train_loss, window) <= alpha*log C; NaN if never (censored)."""
    sm = metrics.train_loss.rolling(window, min_periods=window).mean()
    hit = metrics.step[sm <= alpha * math.log(C)]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def best_lr_table(runs: pd.DataFrame) -> pd.DataFrame:
    """(protocol, C, superclass, optimizer) -> lr, status in {ok, edge_low, edge_high, none}. Seed-0, non-diverged."""
    from sweep import best_lr  # sweep.py owns the selection rule; analysis only reports it

    rows = []
    for key, _ in runs.groupby(CELL):
        lr, status = best_lr(runs, *key)
        rows.append(dict(zip(CELL, key)) | {"lr": lr, "status": status})
    return pd.DataFrame(rows)


def best_lr_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Rows of `runs` at each cell's best LR (all seeds), non-diverged."""
    table = best_lr_table(runs).dropna(subset=["lr"])
    m = runs.merge(table[CELL + ["lr"]], on=CELL, suffixes=("", "_best"))
    m = m[np.isclose(m.lr, m.lr_best, rtol=1e-6)].drop(columns="lr_best")
    return m[~m.diverged].reset_index(drop=True)


def paired_gaps(runs: pd.DataFrame, alphas=TARGET_ALPHAS) -> pd.DataFrame:
    """One row per (protocol, C, superclass, seed, alpha): delta_speed, delta_loss, delta_acc from best-LR runs.
    SGD and Adam are joined on the condition + seed; a missing partner raises."""
    best = best_lr_runs(runs)
    recs = []
    for _, r in best.iterrows():
        m = load_metrics(r.run_id)
        for a in alphas:
            recs.append({**{k: r[k] for k in COND + ["seed", "optimizer", "lr", "final_train_loss", "best_test_acc"]},
                         "alpha": a, "steps_to_target": steps_to_target(m, int(r.C), a)})
    per = pd.DataFrame(recs)
    sgd = per[per.optimizer == "sgd"].drop(columns="optimizer")
    adam = per[per.optimizer == "adam"].drop(columns="optimizer")
    keys = COND + ["seed", "alpha"]
    g = sgd.merge(adam, on=keys, how="outer", suffixes=("_sgd", "_adam"), indicator=True)
    missing = g[g._merge != "both"]
    if not missing.empty:
        raise ValueError(f"unpaired best-LR runs:\n{missing[keys + ['_merge']].drop_duplicates().to_string()}")
    g = g.drop(columns="_merge")
    g["log_C"] = np.log(g.C)
    g["delta_speed"] = np.log(g.steps_to_target_sgd / g.steps_to_target_adam)
    g["delta_loss"] = (g.final_train_loss_sgd - g.final_train_loss_adam) / g.log_C
    g["delta_acc"] = g.best_test_acc_adam - g.best_test_acc_sgd
    g["censored"] = g.steps_to_target_sgd.isna() | g.steps_to_target_adam.isna()
    return g.sort_values(keys).reset_index(drop=True)


PRIMARY_ALPHA = 0.5


def choose_alpha(gaps: pd.DataFrame, protocol: str | None = None) -> float:
    """Plan Section 4.2 / 11.1: alpha = 0.5 is primary; if any best-LR run is censored at 0.5, take the smallest
    uncensored alpha above it. (The code plan's 'largest uncensored alpha' would always pick 0.7.)"""
    g = gaps if protocol is None else gaps[gaps.protocol == protocol]
    ok = [a for a in sorted(TARGET_ALPHAS) if (np.isclose(g.alpha, a)).any() and not g[np.isclose(g.alpha, a)].censored.any()]
    if not ok:
        raise ValueError("no alpha in TARGET_ALPHAS is reached by every best-LR run; see plan Section 13 fallback")
    above = [a for a in ok if a >= PRIMARY_ALPHA - 1e-9]
    return float(above[0]) if above else float(ok[-1])


def gap_summary(gaps: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """mean, se, n over seeds per condition at the chosen alpha."""
    g = gaps[np.isclose(gaps.alpha, alpha)]
    agg = g.groupby(COND).agg(
        delta_speed_mean=("delta_speed", "mean"), delta_speed_se=("delta_speed", "sem"),
        delta_loss_mean=("delta_loss", "mean"), delta_loss_se=("delta_loss", "sem"),
        delta_acc_mean=("delta_acc", "mean"), delta_acc_se=("delta_acc", "sem"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return agg


# --------------------------------------------------------------------------- curvature


def curvature_summary(curv: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """mean +- se of rho, r_block, e_diag_P (and e_diag, block_norm_cv) per (protocol, C, superclass, optimizer, frac).
    Curvature at frac 0 is optimizer-independent (same init), so a per-condition summary can be read off optimizer='sgd'."""
    j = curv.merge(runs[["run_id"] + CELL + ["seed"]], on="run_id")
    cols = [c for c in ("rho", "r_block", "e_diag_P", "e_diag", "block_norm_cv", "fro_sq") if c in j]
    agg = j.groupby(CELL + ["frac"])[cols].agg(["mean", "sem", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def loglog_slope(x, y) -> tuple[float, float]:
    """OLS slope and standard error of log y on log x."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    res = stats.linregress(lx, ly)
    return float(res.slope), float(res.stderr)


def hypothesis_tests(gaps: pd.DataFrame, curv: pd.DataFrame, runs: pd.DataFrame, alpha: float,
                     protocol: str = "cifar_merge") -> dict:
    """H1: log-log slopes at init. H2: Spearman(delta_speed_mean, -rho) over C. H3: Spearman(rho@0 per (C,seed), delta_speed)."""
    out = {"alpha": alpha, "protocol": protocol}
    j = curv.merge(runs[["run_id"] + CELL + ["seed"]], on="run_id")
    j = j[(j.protocol == protocol) & (~j.superclass)]
    init = j[np.isclose(j.frac, 0.0)].groupby(["C", "seed"], as_index=False)[["rho", "r_block", "e_diag_P"]].mean()
    if init.C.nunique() >= 3:
        for m, th in (("rho", -0.5), ("r_block", -1.0)):
            s, se = loglog_slope(init.C, init[m])
            out[f"H1_{m}_slope"], out[f"H1_{m}_se"], out[f"H1_{m}_theory"] = s, se, th
        s, se = loglog_slope(init.C, init.e_diag_P)
        out["secondary_e_diag_P_slope"], out["secondary_e_diag_P_se"] = s, se
    gs = gap_summary(gaps, alpha)
    gs = gs[(gs.protocol == protocol) & (~gs.superclass)]
    rho_C = init.groupby("C").rho.mean()
    both = gs.merge(rho_C.rename("rho0"), left_on="C", right_index=True).dropna(subset=["delta_speed_mean"])
    out["H2_n_C"] = int(len(both))  # censored C values (no delta_speed) are dropped
    if len(both) >= 3:
        r = stats.spearmanr(both.delta_speed_mean, -both.rho0)
        out["H2_spearman_speed_vs_neg_rho"], out["H2_p"] = float(r.statistic), float(r.pvalue)
        r2 = stats.spearmanr(both.delta_speed_mean, both.C)
        out["H2_spearman_speed_vs_C"], out["H2_C_p"] = float(r2.statistic), float(r2.pvalue)
    per = gaps[(gaps.protocol == protocol) & (~gaps.superclass) & np.isclose(gaps.alpha, alpha)]
    per = per.merge(init[["C", "seed", "rho"]], on=["C", "seed"]).dropna(subset=["delta_speed"])
    if len(per) >= 3:
        r = stats.spearmanr(per.rho, per.delta_speed)
        out["H3_spearman_rho0_vs_speed"], out["H3_p"], out["H3_n"] = float(r.statistic), float(r.pvalue), int(len(per))
    return out


# --------------------------------------------------------------------------- figures


def make_figures(gaps: pd.DataFrame, curv_summary: pd.DataFrame, runs: pd.DataFrame, alpha: float,
                 curv: pd.DataFrame | None = None, protocol: str = "cifar_merge") -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    gs = gap_summary(gaps, alpha)
    protos = [p for p in ("cifar_merge", "fmnist_merge") if (gs.protocol == p).any()]

    # ---- Fig 1: gaps vs C
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ylabels = {"delta_speed": r"$\Delta_{speed}$  (log of SGD/Adam steps to target)",
               "delta_loss": r"$\Delta_{loss}$  (final train-loss gap / log C)",
               "delta_acc": r"$\Delta_{acc}$  (best test-accuracy gap)"}
    for m, ax, lab in zip(("delta_speed", "delta_loss", "delta_acc"), axes,
                          (r"$\Delta_{speed}=\log(steps_{SGD}/steps_{Adam})$", r"$\Delta_{loss}=(L_{SGD}-L_{Adam})/\log C$",
                           r"$\Delta_{acc}=acc_{Adam}-acc_{SGD}$")):
        for p in protos:
            d = gs[(gs.protocol == p) & (~gs.superclass)].sort_values("C")
            ax.errorbar(d.C, d[f"{m}_mean"], yerr=d[f"{m}_se"], marker="o", capsize=3, label=p)
            sc = gs[(gs.protocol == p) & (gs.superclass)]
            if not sc.empty:
                ax.errorbar(sc.C, sc[f"{m}_mean"], yerr=sc[f"{m}_se"], marker="s", ls="none", capsize=3,
                            label=f"{p} superclass")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("Number of classes C"); ax.set_ylabel(ylabels[m])
        ax.set_title(lab, fontsize=10); ax.legend(fontsize=8)
    fig.suptitle(f"Adam-vs-SGD gaps at best LR (alpha={alpha}); mean +- se over seeds")
    fig.tight_layout(); paths.append(FIGURES_DIR / "fig1_gaps_vs_C.png"); fig.savefig(paths[-1], dpi=150); plt.close(fig)

    # ---- Fig 2: log rho, log r_block vs log C per checkpoint (optimizer-averaged)
    cs = curv_summary[(curv_summary.protocol == protocol) & (~curv_summary.superclass)]
    if not cs.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for m, ax, th, lab in zip(("rho", "r_block"), axes, (-0.5, -1.0),
                                  (r"$\rho$  (off-block / on-block Frobenius norm)",
                                   r"$r_{block}$  (median normalised off-block norm)")):
            for frac in CHECKPOINT_FRACS:
                d = cs[np.isclose(cs.frac, frac)].groupby("C", as_index=False)[[f"{m}_mean", f"{m}_sem"]].mean()
                if d.empty or d.C.nunique() < 2:
                    continue
                s, se = loglog_slope(d.C, d[f"{m}_mean"])
                ax.errorbar(d.C, d[f"{m}_mean"], yerr=d[f"{m}_sem"].fillna(0), marker="o", capsize=3,
                            label=f"frac {frac:.1f}: slope {s:+.2f}+-{se:.2f}")
            Cs = np.array(sorted(cs.C.unique()), float)
            ref = cs[np.isclose(cs.frac, 0.0)].groupby("C")[f"{m}_mean"].mean()
            if len(ref):
                scale = ref.iloc[0] / (Cs[0] ** th)
                ax.plot(Cs, scale * Cs**th, "k--", lw=1, label=f"theory slope {th:+.1f}")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Number of classes C"); ax.set_ylabel(lab)
            ax.legend(fontsize=8)
        fig.suptitle(f"Last-layer Hessian block structure vs C ({protocol}), per checkpoint fraction")
        fig.tight_layout(); paths.append(FIGURES_DIR / "fig2_rho_rblock_vs_C.png"); fig.savefig(paths[-1], dpi=150); plt.close(fig)

        # ---- Fig 3: e_diag*P vs C per checkpoint
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for frac in CHECKPOINT_FRACS:
            d = cs[np.isclose(cs.frac, frac)].groupby("C", as_index=False)[["e_diag_P_mean", "e_diag_P_sem"]].mean()
            if not d.empty:
                ax.errorbar(d.C, d.e_diag_P_mean, yerr=d.e_diag_P_sem.fillna(0), marker="o", capsize=3, label=f"frac {frac:.1f}")
        ax.axhline(1, color="gray", lw=0.8, ls=":", label="random-matrix null (1/P)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Number of classes C")
        ax.set_ylabel(r"$e_{diag}\cdot P$ = diagonal share of $\|H\|_F^2$, relative to 1/P"); ax.legend(fontsize=8)
        ax.set_title(f"Full-network diagonal dominance ({protocol})")
        fig.tight_layout(); paths.append(FIGURES_DIR / "fig3_ediag_vs_C.png"); fig.savefig(paths[-1], dpi=150); plt.close(fig)

    # ---- Fig 4: scatter init rho vs final delta_speed, one point per (C, seed)
    if curv is not None:
        j = curv.merge(runs[["run_id"] + CELL + ["seed"]], on="run_id")
        init = j[(j.protocol == protocol) & (~j.superclass) & np.isclose(j.frac, 0.0)]
        init = init.groupby(["C", "seed"], as_index=False).rho.mean()
        per = gaps[(gaps.protocol == protocol) & (~gaps.superclass) & np.isclose(gaps.alpha, alpha)]
        per = per.merge(init, on=["C", "seed"]).dropna(subset=["delta_speed"])  # censored (C, seed) pairs excluded
        if not per.empty:
            fig, ax = plt.subplots(figsize=(6, 4.2))
            scat = ax.scatter(per.rho, per.delta_speed, c=np.log10(per.C), cmap="viridis")
            for _, r in per.iterrows():
                ax.annotate(f"C{int(r.C)}s{int(r.seed)}", (r.rho, r.delta_speed), fontsize=7, xytext=(3, 3), textcoords="offset points")
            cb = fig.colorbar(scat); cb.set_label("log10 C")
            if len(per) >= 3:
                rs = stats.spearmanr(per.rho, per.delta_speed)
                ax.set_title(f"Init-time block ratio vs final speed gap, one point per (C, seed)\n"
                             f"Spearman {rs.statistic:+.2f}, p={rs.pvalue:.2g}, n={len(per)}", fontsize=10)
            ax.set_xscale("log"); ax.set_xlabel(r"$\rho$ at initialisation (last-layer off/on-block ratio)")
            ax.set_ylabel(r"$\Delta_{speed}$ = log(SGD steps / Adam steps to target)")
            fig.tight_layout(); paths.append(FIGURES_DIR / "fig4_init_rho_vs_speed.png"); fig.savefig(paths[-1], dpi=150); plt.close(fig)

    # ---- Fig 5: LR sweep curves per C (seed 0)
    sw = runs[(runs.protocol == protocol) & (runs.seed == 0)]
    conds = sorted(sw[COND].drop_duplicates().itertuples(index=False), key=lambda t: (t.C, t.superclass))
    if conds:
        ncol = min(4, len(conds)); nrow = math.ceil(len(conds) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow), squeeze=False)
        for ax, cond in zip(axes.flat, conds):
            d = sw[(sw.C == cond.C) & (sw.superclass == cond.superclass)]
            for opt, mk in (("sgd", "o"), ("adam", "s")):
                dd = d[d.optimizer == opt].sort_values("lr")
                ok = dd[~dd.diverged]; bad = dd[dd.diverged]
                ax.plot(ok.lr, ok.final_train_loss, marker=mk, label=opt)
                if not bad.empty:
                    ax.plot(bad.lr, np.full(len(bad), ax.get_ylim()[1] if ok.empty else ok.final_train_loss.max()),
                            marker="x", ls="none", color="red")
            ax.axhline(math.log(cond.C), color="gray", ls=":", lw=0.8)
            ax.set_xscale("log"); ax.set_title(f"C={cond.C}{' superclass' if cond.superclass else ''}", fontsize=9)
            ax.set_xlabel("Learning rate"); ax.set_ylabel("Final train loss"); ax.legend(fontsize=7)
        for ax in axes.flat[len(conds):]:
            ax.axis("off")
        fig.suptitle(f"LR sweep, seed 0 ({protocol}): final train loss = mean of last 5% of steps; "
                     f"red x = diverged; dotted = chance loss log C")
        fig.tight_layout(); paths.append(FIGURES_DIR / "fig5_lr_sweep.png"); fig.savefig(paths[-1], dpi=150); plt.close(fig)
    return paths


# --------------------------------------------------------------------------- CLI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=None, help="override target fraction (default: choose_alpha)")
    ap.add_argument("--protocol", default=None, help="protocol for curvature figures / tests (default: cifar if present)")
    a = ap.parse_args()
    runs = load_runs()
    gaps = paired_gaps(runs)
    protocol = a.protocol or ("cifar_merge" if (runs.protocol == "cifar_merge").any() else "fmnist_merge")
    alpha = a.alpha if a.alpha is not None else choose_alpha(gaps, protocol)
    notes = {"alpha": alpha, "alpha_source": "cli" if a.alpha is not None else "choose_alpha", "protocol": protocol}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_lr_table(runs).to_csv(RESULTS_DIR / "best_lr.csv", index=False)
    gaps.to_csv(RESULTS_DIR / "gaps.csv", index=False)
    gs = gap_summary(gaps, alpha)
    gs.to_csv(RESULTS_DIR / "gap_summary.csv", index=False)
    print(gs.to_string(index=False))
    curv = load_curvature() if CURV_CSV.exists() else None
    cs = curvature_summary(curv, runs) if curv is not None else pd.DataFrame(columns=CELL + ["frac"])
    if curv is not None:
        cs.to_csv(RESULTS_DIR / "curvature_summary.csv", index=False)
        notes["hypothesis_tests"] = hypothesis_tests(gaps, curv, runs, alpha, protocol)
        print(json.dumps(notes["hypothesis_tests"], indent=2))
    from sweep import escalation_check
    esc, esc_info = escalation_check(gaps, alpha=alpha, protocol=protocol)
    notes["escalation"] = {"escalate": esc, **esc_info}
    paths = make_figures(gaps, cs, runs, alpha, curv, protocol)
    notes["figures"] = [str(p) for p in paths]
    prev = json.loads(NOTES_JSON.read_text()) if NOTES_JSON.exists() else {}
    prev.update(notes)
    NOTES_JSON.write_text(json.dumps(prev, indent=2, default=float))
    print(f"alpha={alpha}; escalate={esc}; wrote {len(paths)} figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
