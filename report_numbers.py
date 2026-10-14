"""Regenerate every table and in-text number of the report (v10) from the saved results, and check each one
against the value printed in the report.

Reads only results/*.csv, results/analysis_notes.json and runs_lastlayer/*/metrics.csv, so it needs no training,
no GPU and no checkpoints (torch is used only for an optional parameter-count check). Run it after the pipeline in
README.md / REPRODUCE.md:

    .venv/bin/python report_numbers.py

Outputs
    results/report_tables/*.csv     one CSV per report table (S1-S11, Table 1) plus the in-text numbers
    results/report_numbers.md       every value next to the report's value, with OK / MISMATCH
The exit code is 1 if any value differs from the report by more than its printed precision.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import constants as K
from analysis import loglog_slope, steps_to_target

RES = K.RESULTS_DIR
OUT_DIR = RES / "report_tables"
MD = RES / "report_numbers.md"
ALPHA = 0.7  # target fraction used in the report (see REPRODUCE.md, "Choice of alpha")
CIFAR, FMNIST = "cifar_merge", "fmnist_merge"
CS = (2, 5, 10, 20, 50, 100)

CHECKS: list[dict] = []
SECTIONS: list[tuple[str, str]] = []  # (title, markdown body)


# --------------------------------------------------------------------------- helpers


def _tol(expected: str) -> float:
    """Half a unit in the last printed digit of `expected` (plus float slack), e.g. '0.576' -> 0.0005."""
    s = expected.replace(",", "").lstrip("+-")
    dec = len(s.split(".")[1]) if "." in s else 0
    return 0.5 * 10 ** (-dec) * 1.02 + 1e-12


def check(section: str, label: str, computed, expected: str, tol: float | None = None, rel: float | None = None,
          known: str | None = None):
    """Record one comparison. `expected` is the value as printed in the report. `known` documents a difference
    that is already understood (a rounding slip in the report); it is listed but does not fail the run."""
    exp = float(expected.replace(",", "").replace("−", "-"))
    if computed is None or (isinstance(computed, float) and math.isnan(computed)):
        ok = math.isnan(exp)
    elif rel is not None:
        ok = abs(computed - exp) <= rel * abs(exp)
    else:
        ok = abs(computed - exp) <= (tol if tol is not None else _tol(expected))
    CHECKS.append({"section": section, "quantity": label, "computed": computed, "report": expected, "ok": bool(ok),
                   "note": "" if ok or known is None else known, "known": (not ok) and known is not None})


def check_true(section: str, label: str, cond: bool, detail: str):
    CHECKS.append({"section": section, "quantity": label, "computed": detail, "report": "holds", "ok": bool(cond),
                   "note": "", "known": False})


def save(df: pd.DataFrame, name: str, title: str, floatfmt: str = "{:.4g}") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{name}.csv", index=False)
    show = df.copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    SECTIONS.append((title, f"`results/report_tables/{name}.csv`\n\n" + _md_table(show)))


def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def is_dry(run_id: pd.Series) -> pd.Series:
    return run_id.str.contains(r"_T\d+$")


# --------------------------------------------------------------------------- loading


def load():
    runs = pd.read_csv(RES / "runs.csv").drop_duplicates("run_id", keep="last")
    runs = runs[~is_dry(runs.run_id)]
    runs["superclass"] = runs.superclass.astype(bool)
    best = pd.read_csv(RES / "best_lr.csv")
    best["superclass"] = best.superclass.astype(bool)
    cell = ["protocol", "C", "superclass", "optimizer"]
    br = runs.merge(best[cell + ["lr"]], on=cell + ["lr"])  # best-LR runs, all seeds

    cur = pd.read_csv(RES / "curvature.csv")
    cur = cur[~is_dry(cur.run_id)]
    wide = cur.pivot_table(index=["run_id", "frac"], columns="metric", values="value").reset_index()
    curv = wide.merge(br[["run_id", "protocol", "C", "superclass", "optimizer", "seed"]], on="run_id")

    gaps = pd.read_csv(RES / "gaps.csv")
    gaps["superclass"] = gaps.superclass.astype(bool)
    notes = json.loads((RES / "analysis_notes.json").read_text())
    return runs, best, br, curv, gaps, notes


# --------------------------------------------------------------------------- tables


def table_S1():
    rows = [
        ("mu (SGD momentum)", K.SGD_MOMENTUM, "0.9"),
        ("beta1", K.ADAM_BETAS[0], "0.9"), ("beta2", K.ADAM_BETAS[1], "0.999"), ("eps", K.ADAM_EPS, "1e-8"),
        ("weight decay", K.WEIGHT_DECAY, "0"), ("warmup steps", K.WARMUP_STEPS, "200"),
        ("batch size", K.BATCH_SIZE, "128"), ("steps CIFAR", K.STEPS[CIFAR], "8000"),
        ("steps FMNIST", K.STEPS[FMNIST], "2000"), ("probe images per original class", K.PROBE_PER_CLASS, "10"),
        ("Hutchinson vectors q", K.HUTCH_PROBES, "50"), ("smoothing window", K.SMOOTH_WINDOW, "50"),
    ]
    for name, v, e in rows:
        check("Table S1", name, float(v), e, tol=1e-12)
    check_true("Table S1", "checkpoints", tuple(K.CHECKPOINT_FRACS) == (0.0, 0.1, 0.5, 1.0), str(K.CHECKPOINT_FRACS))
    check_true("Table S1", "SGD LR grid", tuple(K.LR_GRID["sgd"]) == (0.3, 0.1, 0.03, 0.01, 0.003), str(K.LR_GRID["sgd"]))
    check_true("Table S1", "Adam LR grid", tuple(K.LR_GRID["adam"]) == (3e-3, 1e-3, 3e-4, 1e-4, 3e-5), str(K.LR_GRID["adam"]))
    check_true("Table S1", "seeds", tuple(K.SEEDS) + tuple(K.FALLBACK_SEEDS) == (0, 1, 2, 3, 4), "0,1,2 (+3,4)")
    save(pd.DataFrame([(n, v) for n, v, _ in rows], columns=["parameter", "value"]), "tableS1_hyperparameters",
         "Table S1 - fixed hyperparameters (constants.py)")


def model_sizes():
    """P for SmallCNN and FCN, from the layer shapes in models.py (checked against torch if available)."""
    body = 0
    ch = 3
    for w in (32, 64, 128):
        body += (ch * w * 9 + w) + (w * w * 9 + w)
        ch = w
    P_cnn = {C: body + 129 * C for C in CS}
    P_fcn = {C: 196 * 32 + 32 + 33 * C for C in (2, 5, 10)}
    check("Methods", "SmallCNN parameters at C=100 (millions)", P_cnn[100] / 1e6, "0.3", tol=0.05)
    check("Methods", "FCN parameters = 6304 + 33C: constant part", 196 * 32 + 32, "6304", tol=0)
    check("Methods", "last layer share of parameters, C=2 (%)", 100 * 129 * 2 / P_cnn[2], "0.1")
    check("Methods", "last layer share of parameters, C=100 (%)", 100 * 129 * 100 / P_cnn[100], "4.3")
    try:
        from config import RunConfig
        from models import build_model, param_count
        for C in (2, 100):
            cfg = RunConfig(protocol=CIFAR, C=C, superclass=False, seed=0, optimizer="sgd", lr=0.1, steps=K.STEPS[CIFAR])
            check("Methods", f"SmallCNN param_count C={C} (torch)", float(param_count(build_model(cfg))), str(P_cnn[C]), tol=0)
    except ImportError:
        pass
    save(pd.DataFrame({"C": list(CS), "P_SmallCNN": [P_cnn[C] for C in CS],
                       "last_layer_pct": [100 * 129 * C / P_cnn[C] for C in CS]}), "model_sizes",
         "Model sizes (report section 3 / appendix ג.2)")


def table_S2(runs: pd.DataFrame):
    ll = pd.read_csv(RES / "lastlayer_runs.csv").drop_duplicates("run_id", keep="last")
    rows = []
    for p, d in runs.groupby("protocol"):
        rows.append({"protocol": p, "sweep_seed0": int((d.seed == 0).sum()), "extra_seeds": int((d.seed > 0).sum()),
                     "total": len(d), "diverged": int(d.diverged.astype(bool).sum()),
                     "median_s": d.wall_time_s.median(), "total_h": d.wall_time_s.sum() / 3600})
    rows.append({"protocol": "lastlayer_control", "sweep_seed0": int((ll.seed == 0).sum()),
                 "extra_seeds": int((ll.seed > 0).sum()), "total": len(ll), "diverged": int(ll.diverged.astype(bool).sum()),
                 "median_s": ll.wall_time_s.median(), "total_h": ll.wall_time_s.sum() / 3600})
    t = pd.DataFrame(rows)
    exp = {CIFAR: ("70", "56", "126", "1", "112", "4.4"), FMNIST: ("35", "12", "47", "0", None, None),
           "lastlayer_control": ("24", "10", "34", "0", "137", "1.7")}
    for r in t.itertuples():
        e = exp[r.protocol]
        for k, v in zip(("sweep_seed0", "extra_seeds", "total", "diverged"), e[:4]):
            check("Table S2", f"{r.protocol} {k}", float(getattr(r, k)), v, tol=0)
        if e[4]:
            check("Table S2", f"{r.protocol} median seconds/run", r.median_s, e[4])
            check("Table S2", f"{r.protocol} total hours", r.total_h, e[5])
    check_true("Table S2", "FMNIST median < 1 s/run", t.loc[t.protocol == FMNIST, "median_s"].iloc[0] < 1, "")
    check("Table S2", "total runs", float(t.total.sum()), "207", tol=0)
    check("Table S2", "total hours", float(t.total_h.sum()), "6.1",
          known="the report adds the rounded rows (4.4 + 1.7); the unrounded total is 6.05 h, i.e. about 6.0 h")
    div = runs[runs.diverged.astype(bool)].run_id.tolist()
    check_true("Table S2", "only diverged run is SGD lr 0.3 at C=20", div == ["cifar_merge_C20_sgd_lr3.0e-01_s0"], str(div))
    save(t, "tableS2_run_budget", "Table S2 - run budget (results/runs.csv, results/lastlayer_runs.csv)")


def table_S3(best: pd.DataFrame):
    notes = pd.read_csv(RES / "sweep_notes.csv")
    notes["superclass"] = notes.superclass.astype(bool)
    t = best.merge(notes[["protocol", "C", "superclass", "optimizer", "extensions"]],
                   on=["protocol", "C", "superclass", "optimizer"], how="left")
    exp = {(CIFAR, 2, False): ("0.1", "1e-3"), (FMNIST, 2, False): ("0.1", "3e-3"),
           (FMNIST, 5, False): ("0.03", "9.5e-3"), (FMNIST, 10, False): ("0.03", "9.5e-3")}
    for r in t.itertuples():
        e = exp.get((r.protocol, r.C, r.superclass), ("0.03", "1e-3"))[0 if r.optimizer == "sgd" else 1]
        check("Table S3", f"{r.protocol} C={r.C}{' sc' if r.superclass else ''} {r.optimizer} LR", r.lr, e, rel=0.01)
        check_true("Table S3", f"{r.protocol} C={r.C}{' sc' if r.superclass else ''} {r.optimizer} interior minimum",
                   r.status == "ok", r.status)
    ext = t[t.protocol == FMNIST].set_index(["C", "optimizer"]).extensions
    for C, e in ((2, "1"), (5, "2"), (10, "2")):
        check("Table S3", f"FMNIST C={C} Adam grid extensions", float(ext[(C, "adam")]), e, tol=0)
    save(t.sort_values(["protocol", "superclass", "C", "optimizer"]), "tableS3_best_lr",
         "Table S3 - chosen learning rate per cell (results/best_lr.csv, results/sweep_notes.csv)")


def curvature_tables(curv: pd.DataFrame, notes: dict):
    c = curv[(curv.protocol == CIFAR) & (~curv.superclass)]
    m = c.groupby(["C", "frac"])[["rho", "r_block", "e_diag_P", "e_diag", "block_norm_cv"]].mean().reset_index()
    n = c.groupby(["C", "frac"]).size()
    check_true("Table S4", "10 best-LR runs per C at init", (n.xs(0.0, level="frac") == 10).all(), str(n.xs(0.0, level="frac").to_dict()))
    piv = lambda col: m.pivot(index="C", columns="frac", values=col)
    rho, rb, ed, cv = piv("rho"), piv("r_block"), piv("e_diag_P"), piv("block_norm_cv")

    # Table S4 / Table 1 (curvature columns)
    t4 = pd.DataFrame({"C": CS, "rho_init": rho[0.0].values, "rho_theory": [1 / math.sqrt(C - 1) for C in CS],
                       "r_block_init": rb[0.0].values, "r_block_theory": [1 / (C - 1) for C in CS],
                       "rho_end": rho[1.0].values, "r_block_end": rb[1.0].values})
    exp4 = {2: ("1.000", "1.000", "1.000", "1.000"), 5: ("0.501", "0.247", "0.523", "0.247"),
            10: ("0.334", "0.111", "0.377", "0.112"), 20: ("0.230", "0.0525", "0.308", "0.0526"),
            50: ("0.143", "0.0204", "0.317", "0.0131"), 100: ("0.101", "0.0101", "0.355", "0.0019")}
    for r in t4.itertuples():
        for col, e in zip(("rho_init", "r_block_init", "rho_end", "r_block_end"), exp4[r.C]):
            check("Table S4 / Table 1", f"C={r.C} {col}", getattr(r, col), e)
    dev = max((t4.rho_init / t4.rho_theory - 1).abs().max(), (t4.r_block_init / t4.r_block_theory - 1).abs().max())
    check("Table S4", "largest deviation from exact theory at init (%)", 100 * dev, "1.2")
    end10 = t4[t4.C >= 10].rho_end
    check("Results H1", "rho at end, C>=10: min", end10.min(), "0.31")
    check("Results H1", "rho at end, C>=10: max", end10.max(), "0.38")
    save(t4, "tableS4_blocks_init_end", "Table S4 / Table 1 - block metrics at init vs theory and at the end (CIFAR)")

    # Table S5: log-log slopes on the per-C means, per checkpoint
    rows = []
    for f in K.CHECKPOINT_FRACS:
        s1, e1 = loglog_slope(rho.index, rho[f])
        s2, e2 = loglog_slope(rb.index, rb[f])
        rows.append({"frac": f, "rho_slope": s1, "rho_se": e1, "r_block_slope": s2, "r_block_se": e2})
    t5 = pd.DataFrame(rows)
    exp5 = {0.0: ("-0.58", "0.02", "-1.15", "0.05"), 0.1: ("-0.35", "0.06", "-1.23", "0.04"),
            0.5: ("-0.30", "0.07", "-1.37", "0.06"), 1.0: ("-0.25", "0.09", "-1.50", "0.12")}
    for r in t5.itertuples():
        for col, e in zip(("rho_slope", "rho_se", "r_block_slope", "r_block_se"), exp5[r.frac]):
            check("Table S5", f"frac={r.frac} {col}", getattr(r, col), e)
    exact_slopes = (loglog_slope(np.array(CS), [1 / math.sqrt(C - 1) for C in CS])[0],
                    loglog_slope(np.array(CS), [1 / (C - 1) for C in CS])[0])
    check("Appendix א.3", "exact-form log-log slope of rho over C=2..100", exact_slopes[0], "-0.576")
    check("Appendix א.3", "exact-form log-log slope of r_block over C=2..100", exact_slopes[1], "-1.152")
    save(t5, "tableS5_slopes_by_checkpoint", "Table S5 - log-log slopes of the block metrics per checkpoint (CIFAR)")

    # H1 slopes over all (C, seed) points at init (analysis.hypothesis_tests)
    h = notes["hypothesis_tests"]
    for k, e in (("H1_rho_slope", "-0.576"), ("H1_rho_se", "0.009"), ("H1_r_block_slope", "-1.151"),
                 ("H1_r_block_se", "0.019"), ("secondary_e_diag_P_slope", "-0.31"), ("secondary_e_diag_P_se", "0.06")):
        check("Results H1 / analysis_notes.json", k, h[k], e)
    check("Results H1", "rho at C=100, init", rho.loc[100, 0.0], "0.101")
    check("Results H1", "rho at C=100, end", rho.loc[100, 1.0], "0.355")

    # Table S7: e_diag * P
    t7 = ed[[0.0, 0.1, 1.0]].T.reset_index().rename(columns={"frac": "checkpoint"})
    exp7 = {0.0: (52700, 53800, 60600, 41500, 28000, 16200), 0.1: (10900, 1910, 1620, 1640, 1930, 1740),
            1.0: (2990, 2180, 2660, 3200, 3760, 3240)}
    for f, vals in exp7.items():
        for C, e in zip(CS, vals):
            check("Table S7", f"e_diag*P C={C} frac={f}", ed.loc[C, f], str(e), rel=0.006)
    drop = (ed[0.0] / ed[0.1])
    check("Appendix ה.3", "drop factor init -> 10%: min", drop.min(), "5", tol=0.5)
    check("Appendix ה.3", "drop factor init -> 10%: max", drop.max(), "37", tol=0.5)
    edt = m[m.frac >= 0.5].e_diag
    check_true("Appendix ג.5", "e_diag in the second half of training ~ 0.01", 0.005 < edt.min() and edt.max() < 0.02,
               f"{edt.min():.4f}-{edt.max():.4f}")
    save(t7, "tableS7_ediagP", "Table S7 - e_diag * P per C (CIFAR, mean over 10 runs)", "{:.4g}")

    # Table S10: block_norm_cv
    t10 = cv.loc[[5, 10, 20, 50, 100]].reset_index()
    exp10 = {5: (0.044, 0.160, 0.130, 0.143), 10: (0.045, 0.277, 0.201, 0.178), 20: (0.047, 0.362, 0.256, 0.233),
             50: (0.050, 0.498, 0.338, 0.351), 100: (0.052, 0.600, 0.431, 0.490)}
    for C, vals in exp10.items():
        for f, e in zip(K.CHECKPOINT_FRACS, vals):
            check("Table S10", f"cv_block C={C} frac={f}", cv.loc[C, f], f"{e:.3f}",
                  known="0.20048 rounds to 0.200, not 0.201" if (C, f) == (10, 0.5) else None)
    save(t10, "tableS10_block_cv", "Table S10 - cv_block per C and checkpoint (CIFAR)", "{:.3f}")

    # FCN erosion (appendix ה.2): exact last-layer blocks of the FCN
    fc = curv[(curv.protocol == FMNIST) & (curv.C == 10)].groupby("frac").rho.mean()
    check("Appendix ה.2", "FCN rho at C=10, init", fc[0.0], "0.34")
    check("Appendix ה.2", "FCN rho at C=10, half of training", fc[0.5], "0.69")


def table_S6():
    t = pd.read_csv(RES / "theory_check.csv")
    exp = {2: ("1.000", "1.000"), 5: ("0.511", "0.239"), 10: ("0.343", "0.108")}
    for r in t.itertuples():
        check("Table S6", f"FCN exact rho C={r.C}", r.rho, exp[r.C][0])
        check("Table S6", f"FCN exact r_block C={r.C}", r.r_block, exp[r.C][1])
    save(t, "tableS6_theory_check", "Table S6 - exact FCN Hessian at init vs theory (exact.py --theory)", "{:.3f}")


def gap_tables(gaps: pd.DataFrame, notes: dict):
    g = gaps[np.isclose(gaps.alpha, ALPHA)]
    cond = ["protocol", "C", "superclass"]
    s = g.groupby(cond).agg(
        delta_speed=("delta_speed", "mean"), delta_speed_se=("delta_speed", "sem"),
        n_speed=("delta_speed", "count"), delta_loss=("delta_loss", "mean"), delta_loss_se=("delta_loss", "sem"),
        delta_acc=("delta_acc", "mean"), delta_acc_se=("delta_acc", "sem"), seeds=("seed", "nunique")).reset_index()
    exp = {  # (protocol, C, sc): (speed, se, loss, se, acc, se, seeds)
        (CIFAR, 2, False): ("nan", None, "0.503", "0.023", "0.0097", "0.0016", "5"),
        (CIFAR, 5, False): ("0.625", "0.018", "0.442", "0.019", "0.0099", "0.0039", "5"),
        (CIFAR, 10, False): ("0.608", "0.022", "0.308", "0.017", "0.0147", "0.0037", "5"),
        (CIFAR, 20, False): ("0.561", "0.038", "0.218", "0.009", "0.0152", "0.0045", "5"),
        (CIFAR, 20, True): ("0.631", "0.021", "0.068", "0.009", "0.0167", "0.0042", "5"),
        (CIFAR, 50, False): ("0.601", "0.028", "0.112", "0.007", "0.0103", "0.0031", "5"),
        (CIFAR, 100, False): ("0.625", "0.030", "0.045", "0.005", "0.0020", "0.0042", "5"),
        (FMNIST, 2, False): ("-0.293", "0.009", "0.006", "0.003", "0.0017", "0.0012", "3"),
        (FMNIST, 5, False): ("0.202", "0.022", "0.000", "0.001", "0.0018", "0.0019", "3"),
        (FMNIST, 10, False): ("0.217", "0.019", "0.000", "0.001", "-0.0021", "0.0022", "3"),
    }
    cols = ("delta_speed", "delta_speed_se", "delta_loss", "delta_loss_se", "delta_acc", "delta_acc_se", "seeds")
    for r in s.itertuples():
        e = exp[(r.protocol, r.C, r.superclass)]
        name = f"{r.protocol} C={r.C}{' superclass' if r.superclass else ''}"
        for col, ev in zip(cols, e):
            if ev is not None:
                check("Table S8 / Table 1", f"{name} {col}", float(getattr(r, col)), ev)
    c5 = s[(s.protocol == CIFAR) & (s.C == 5) & (~s.superclass)].n_speed.iloc[0]
    check("Table S8", "CIFAR C=5: seeds where SGD reaches the target", float(c5), "3", tol=0)
    save(s, "tableS8_table1_gaps", f"Table S8 / Table 1 - optimizer gaps at alpha = {ALPHA} (mean, s.e. over seeds)")

    # Table S9: steps to target, final loss, best test accuracy (CIFAR merged)
    c = g[(g.protocol == CIFAR) & (~g.superclass)]
    rows = []
    for C, d in c.groupby("C"):
        ok = d.dropna(subset=["steps_to_target_sgd"])  # C = 5: only the seeds where SGD reaches the target
        rows.append({"C": C, "steps_sgd": ok.steps_to_target_sgd.mean() if len(ok) else np.nan,
                     "steps_adam": (ok if len(ok) else d).steps_to_target_adam.mean(),
                     "ratio": float(np.exp(ok.delta_speed.mean())) if len(ok) else np.nan,
                     "L_sgd": d.final_train_loss_sgd.mean(), "L_adam": d.final_train_loss_adam.mean(),
                     "acc_sgd": d.best_test_acc_sgd.mean(), "acc_adam": d.best_test_acc_adam.mean()})
    t9 = pd.DataFrame(rows)
    exp9 = {2: ("nan", "5260", "nan", "0.572", "0.224", "0.649", "0.659"),
            5: ("7320", "3920", "1.87", "1.134", "0.424", "0.463", "0.473"),
            10: ("5650", "3070", "1.84", "1.297", "0.589", "0.413", "0.427"),
            20: ("4250", "2430", "1.75", "1.398", "0.744", "0.401", "0.416"),
            50: ("2630", "1440", "1.83", "1.289", "0.850", "0.430", "0.440"),
            100: ("1680", "900", "1.87", "1.042", "0.835", "0.477", "0.479")}
    for r in t9.itertuples():
        for col, e in zip(("steps_sgd", "steps_adam", "ratio", "L_sgd", "L_adam", "acc_sgd", "acc_adam"), exp9[r.C]):
            check("Table S9", f"C={r.C} {col}", getattr(r, col), e, tol=5.01 if col.startswith("steps") else None,
                  known="the report divides the rounded step counts (2630/1440); exp(mean delta_speed) = 1.824 -> 1.82"
                  if (r.C, col) == (50, "ratio") else None)
    save(t9, "tableS9_steps_loss_acc", "Table S9 - steps to target, final train loss, best test accuracy (CIFAR)")

    # In-text numbers of the H2 / H3 results
    h = notes["hypothesis_tests"]
    check("Results H2", "Spearman(delta_speed, -rho) over C", h["H2_spearman_speed_vs_neg_rho"], "-0.30")
    check("Results H2", "its p-value", h["H2_p"], "0.62")
    u = c.dropna(subset=["delta_speed"])
    reg = stats.linregress(np.log(u.C), u.delta_speed)
    check("Results H2", "regression of delta_speed on log C: n", float(len(u)), "23", tol=0)
    check("Results H2", "regression slope", reg.slope, "0.003")
    check("Results H2", "regression slope s.e.", reg.stderr, "0.013")
    dL = t9.L_sgd - t9.L_adam
    for C, e in zip(CS, ("0.35", "0.71", "0.71", "0.65", "0.44", "0.21")):
        check("Appendix ה.4", f"unnormalised L_SGD - L_Adam, C={C}", float(dL[t9.C == C].iloc[0]), e)
    acc = c.groupby("C").delta_acc.mean()
    check_true("Results H2", "|delta_acc| <= 1.5 percentage points (merged CIFAR)", acc.abs().max() <= 0.0155,
               f"max {acc.abs().max():.4f}")
    accs = t9[["acc_sgd", "acc_adam"]]
    check("Appendix ו.3", "test accuracy range: min (%)", 100 * accs.min().min(), "40", tol=0.5)
    check("Appendix ו.3", "test accuracy range: max (%)", 100 * accs.max().max(), "66", tol=0.5)
    sp = g[(g.protocol == CIFAR) & g.superclass]
    check("Appendix ה.4", "superclass mean best test accuracy (%)",
          100 * (sp.best_test_acc_sgd.mean() + sp.best_test_acc_adam.mean()) / 2, "60", tol=0.5)
    r20 = c[c.C == 20]
    check("Appendix ה.4", "random merge C=20 mean best test accuracy (%)",
          100 * (r20.best_test_acc_sgd.mean() + r20.best_test_acc_adam.mean()) / 2, "41", tol=0.5)
    check("Results H3", "Spearman(rho at init, delta_speed)", h["H3_spearman_rho0_vs_speed"], "-0.06")
    check("Results H3", "its p-value", h["H3_p"], "0.78")
    check("Results H3", "number of (C, seed) pairs", float(h["H3_n"]), "23", tol=0)

    # alpha = 0.5: censoring and the smaller step ratio
    g5 = gaps[np.isclose(gaps.alpha, 0.5) & (gaps.protocol == CIFAR) & (~gaps.superclass)]
    cens = g5.groupby("C").steps_to_target_sgd.apply(lambda x: x.isna().sum())
    check_true("Appendix ה.4", "alpha=0.5: SGD censored at C in {2,5,10} except one seed at C=10",
               cens[2] == 5 and cens[5] == 5 and cens[10] == 4 and cens[[20, 50, 100]].sum() == 0, str(cens.to_dict()))
    r5 = g5.dropna(subset=["delta_speed"]).groupby("C").delta_speed.mean().pipe(np.exp)
    r5 = r5[r5.index >= 20]
    check("Appendix ה.4", "alpha=0.5 step ratio, uncensored C: min", r5.min(), "1.4", tol=0.05)
    check("Appendix ה.4", "alpha=0.5 step ratio, uncensored C: max", r5.max(), "1.55")
    fm = g[g.protocol == FMNIST]
    st = pd.concat([fm.steps_to_target_sgd, fm.steps_to_target_adam]).groupby(fm.C.tolist() * 2).mean()
    check_true("Appendix ה.4", "FMNIST target reached after 50-80 steps (inside warmup)", st.min() >= 50 and st.max() <= 80,
               f"{st.min():.0f}-{st.max():.0f}")
    save(pd.DataFrame([{"alpha": 0.5, "C": C, "step_ratio": v} for C, v in r5.items()]), "alpha05_step_ratio",
         "Step ratio at alpha = 0.5 for the uncensored C values")


def lastlayer_control(gaps: pd.DataFrame):
    ll = pd.read_csv(RES / "lastlayer_runs.csv").drop_duplicates("run_id", keep="last")
    ll["t"] = [steps_to_target(pd.read_csv(Path(K.ROOT) / "runs_lastlayer" / rid / "metrics.csv"), C, ALPHA)
               for rid, C in zip(ll.run_id, ll.C)]
    g = gaps[np.isclose(gaps.alpha, ALPHA) & (gaps.protocol == CIFAR) & (~gaps.superclass)]
    ll = ll.merge(g[["C", "seed", "steps_to_target_sgd", "steps_to_target_adam", "final_train_loss_sgd",
                     "final_train_loss_adam"]], on=["C", "seed"])
    ll["delta_speed"] = np.log(ll.t / ll.steps_to_target_adam)
    base0 = g[g.seed == 0].set_index("C")

    best = {}
    for C, d in ll[ll.seed == 0].groupby("C"):  # same rule as lastlayer_control.py: lowest final loss, kappa=1 is the baseline
        cand = {1.0: base0.loc[C, "final_train_loss_sgd"], **dict(zip(d.head_mult, d.final_train_loss))}
        best[C] = min(cand, key=cand.get)
    for C, e in zip((5, 10, 20, 50, 100), ("10", "10", "3", "0.1", "0.3")):
        check("Appendix ה.4 (last-layer control)", f"kappa chosen by final loss, C={C}", best[C], e, tol=1e-9)
    sel = ll[[best[C] == k for C, k in zip(ll.C, ll.head_mult)]].copy()
    sel["gap_closed"] = (sel.final_train_loss_sgd - sel.final_train_loss) / (sel.final_train_loss_sgd - sel.final_train_loss_adam)
    per = sel.groupby("C").agg(kappa=("head_mult", "first"), delta_speed=("delta_speed", "mean"),
                               n=("delta_speed", "count"), gap_closed=("gap_closed", "mean")).reset_index()
    for C, e in zip((5, 10, 20, 50, 100), ("0.50", "0.50", "0.52", "0.66", "0.69")):
        check("Appendix ה.4 (last-layer control)", f"delta_speed vs SGD at chosen kappa, C={C}",
              float(per[per.C == C].delta_speed.iloc[0]), e)
    check("Appendix ה.4 (last-layer control)", "largest share of the loss gap closed (%)", 100 * per.gap_closed.max(), "12")
    u = sel.dropna(subset=["delta_speed"])
    reg = stats.linregress(np.log(u.C), u.delta_speed)
    check("Appendix ה.4 (last-layer control)", "slope of delta_speed on log C: n", float(len(u)), "15", tol=0)
    check("Appendix ה.4 (last-layer control)", "slope", reg.slope, "0.07")
    check("Appendix ה.4 (last-layer control)", "slope s.e.", reg.stderr, "0.02")

    s0 = ll[ll.seed == 0]
    k30 = s0[s0.head_mult == 30].set_index("C").delta_speed
    for C, e in zip((5, 10, 20, 50, 100), ("0.39", "0.48", "0.30", "0.37", "0.49")):
        check("Appendix ה.4 (last-layer control)", f"seed 0, kappa=30: delta_speed C={C}", k30[C], e)
    b = np.log(base0.steps_to_target_sgd / base0.steps_to_target_adam)
    for C, e in zip((10, 20, 50, 100), ("0.61", "0.52", "0.55", "0.64")):
        check("Appendix ה.4 (last-layer control)", f"seed 0, plain SGD: delta_speed C={C}", b[C], e)
    mono = []
    for C, d in s0[s0.C >= 5].groupby("C"):
        t = dict(zip(d.head_mult, d.t))
        if not math.isnan(base0.loc[C, "steps_to_target_sgd"]):
            t[1.0] = base0.loc[C, "steps_to_target_sgd"]
        ks = sorted(t)
        mono.append(all(t[a] >= t[b_] for a, b_ in zip(ks, ks[1:])))
    check_true("Appendix ה.4 (last-layer control)", "seed 0: steps to target fall as kappa grows, every C", all(mono), str(mono))
    tab = s0.pivot(index="C", columns="head_mult", values="delta_speed").reset_index()
    tab.columns = ["C"] + [f"kappa={k:g}" for k in tab.columns[1:]]
    save(tab, "lastlayer_seed0_delta_speed", "Last-layer control - delta_speed per kappa at seed 0", "{:.2f}")
    save(per, "lastlayer_chosen_kappa", "Last-layer control - kappa chosen by final loss, seeds 0-2")


def table_S11():
    e = pd.read_csv(RES / "exact_validation.csv")
    e = e[~is_dry(e.run_id)]
    n_ckpt = e[["run_id", "frac"]].drop_duplicates().shape[0]
    check("Table S11", "FCN checkpoints validated", float(n_ckpt), "188", tol=0)
    rows = []
    for m in ("rho", "r_block", "fro_sq", "e_diag", "e_diag_raw"):
        d = e[e.metric == m]
        rows.append({"metric": m, "pass_rate_pct": 100 * d["pass"].astype(bool).mean(),
                     "median_rel_err_pct": 100 * d.rel_err.median(), "p90_rel_err_pct": 100 * d.rel_err.quantile(0.9)})
    t = pd.DataFrame(rows)
    z = e[e.metric == "e_diag_z"]
    within = 100 * (z.rel_err <= 2).mean()  # rel_err holds |est - exact| / jackknife se for this metric
    exp = {"rho": ("100", "0", "0"), "r_block": ("100", "0", "0"), "fro_sq": (None, "5.1", "17.8"),
           "e_diag": ("15.4", "20.0", "69.4"), "e_diag_raw": (None, "162", "834")}
    for r in t.itertuples():
        for col, ev in zip(("pass_rate_pct", "median_rel_err_pct", "p90_rel_err_pct"), exp[r.metric]):
            if ev is not None:
                check("Table S11", f"{r.metric} {col}", getattr(r, col), ev, tol=0.06 if ev in ("0", "100") else None)
    check("Table S11", "exact e_diag within +-2 jackknife s.e. (%)", within, "95.2")
    ed0 = e[(e.metric == "e_diag") & (e.frac == 0.0)].rel_err.median()
    check("Appendix ה.7", "e_diag median relative error at init (%)", 100 * ed0, "41",
          known="41.6% rounds to 42%")
    naive = e[e.metric == "e_diag_raw"].rel_err.median()
    check("Appendix ה.7", "naive e_diag: median error factor", 1 + naive, "2.6")
    t.loc[len(t)] = {"metric": "e_diag within 2 s.e. (%)", "pass_rate_pct": within, "median_rel_err_pct": np.nan,
                     "p90_rel_err_pct": np.nan}
    save(t, "tableS11_validation", "Table S11 - estimators vs exact FCN Hessian (exact.py --validate-all)", "{:.1f}")


# --------------------------------------------------------------------------- main


def main() -> int:
    runs, best, br, curv, gaps, notes = load()
    if not math.isclose(notes.get("alpha", ALPHA), ALPHA):
        print(f"WARNING: analysis_notes.json was written with alpha={notes.get('alpha')}; the report uses {ALPHA}. "
              f"Rerun `analysis.py --alpha {ALPHA}`.")
    table_S1()
    model_sizes()
    table_S2(runs)
    table_S3(best)
    curvature_tables(curv, notes)
    table_S6()
    gap_tables(gaps, notes)
    lastlayer_control(gaps)
    table_S11()

    chk = pd.DataFrame(CHECKS)
    chk.to_csv(OUT_DIR / "checks.csv", index=False)
    bad = chk[~chk.ok & ~chk.known]
    known = chk[chk.known]
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["# Report numbers, regenerated from results/", "",
             f"Generated by `report_numbers.py`. {int(chk.ok.sum())} of {len(chk)} values match report v10 "
             f"to its printed precision; {len(known)} differ by a known rounding slip in the report; "
             f"{len(bad)} are unexplained mismatches.", ""]
    status = lambda r: "OK" if r.ok else ("KNOWN" if r.known else "MISMATCH")
    view = lambda d: _md_table(d.assign(computed=d.computed.map(fmt), ok=[status(r) for r in d.itertuples()])
                               .drop(columns="known").rename(columns={"ok": "status"}))
    if len(bad):
        lines += ["## Unexplained mismatches", "", view(bad), ""]
    if len(known):
        lines += ["## Known differences (rounding slips in the report)", "", view(known), ""]
    lines += ["## All checks", "", view(chk), ""]
    for title, body in SECTIONS:
        lines += [f"## {title}", "", body, ""]
    MD.write_text("\n".join(lines))
    print(f"{int(chk.ok.sum())}/{len(chk)} values match the report, {len(known)} known rounding differences, "
          f"{len(bad)} unexplained mismatches; tables in {OUT_DIR}, summary in {MD}")
    for d, name in ((known, "known differences"), (bad, "unexplained mismatches")):
        if len(d):
            print(f"\n{name}:\n" + d[["section", "quantity", "computed", "report", "note"]].to_string(index=False))
    return 0 if bad.empty else 1


if __name__ == "__main__":
    sys.exit(main())
