# Reproducing every result of the report

This file explains how to reproduce each figure, table and number in the project report (`project_report` v10).
For setup and the role of each module, see `README.md`.

There are two levels of reproduction:

| level | what it reruns | time | needs |
|---|---|---|---|
| **A. From the saved results** | all figures, all tables, every number in the text | about 1 minute | `results/`, `runs/*/metrics.csv`, `runs_lastlayer/*/metrics.csv`, `probes/` (for figure S1 only) |
| **B. From scratch** | every training run, curvature measurement and validation | about 6 h of training plus the curvature runs, on an Apple Silicon Mac | only the code; the datasets download automatically |

Level A is enough to check that the report matches the experiments. Level B checks the experiments themselves.

---

## Level A: regenerate everything from the saved results

```bash
.venv/bin/python analysis.py --alpha 0.7        # figures 1, 2, S3, S4, S6 + gap / curvature summaries + hypothesis tests
.venv/bin/python plot_training_curves.py        # figure S5
.venv/bin/python plot_merge_example.py          # figure S1
.venv/bin/python plot_architecture.py           # figure S2
.venv/bin/python report_numbers.py              # every table and in-text number, checked against the report
```

`report_numbers.py` writes:

- `results/report_tables/*.csv`: one CSV per report table.
- `results/report_numbers.md`: each of the 362 numbers it checks, with the value it recomputed, the value printed in the report, and OK / KNOWN / MISMATCH.

It exits with code 1 only if a number differs from the report by more than the report's printed precision and the
difference is not explained. On the current `results/` the outcome is **358 OK, 4 KNOWN, 0 MISMATCH**. The 4 known
differences are listed at the end of this file.

---

## Level B: rerun everything from scratch

Run these in order. The time for each step is from the runs in the report.

| # | command | what it does | output | time |
|---|---|---|---|---|
| 0 | `.venv/bin/python -m pytest tests -q` | unit tests | - | seconds |
| 1 | `.venv/bin/python sweep.py --protocol fmnist_merge --phase auto --seeds 3` | Fashion-MNIST LR sweep (5 LRs per optimizer, grid extended at an edge up to twice), then seeds 1-2 at the chosen LR | `runs/fmnist_*`, `results/runs.csv`, `results/sweep_notes.csv` | < 1 min |
| 2 | `.venv/bin/python exact.py --theory` | exact FCN Hessian at init vs. the theory (Table S6) | `results/theory_check.csv` | seconds |
| 3 | `.venv/bin/python exact.py --validate-all` | estimators vs. exact Hessian on all 188 FCN checkpoints (Table S11; the acceptance gate for CIFAR) | `results/exact_validation.csv` | minutes |
| 4 | `.venv/bin/python sweep.py --protocol cifar_merge --phase auto --seeds 3` | CIFAR LR sweep: 7 conditions (C = 2...100 and superclass) x 2 optimizers x 5 LRs at seed 0, then seeds 1-2 at the chosen LR | `runs/cifar_*`, `results/runs.csv` | ~3.4 h (112 s/run on MPS) |
| 5 | `.venv/bin/python analysis.py --alpha 0.7` | gaps and the seed-increase check | `results/analysis_notes.json` (`escalation.escalate`) | seconds |
| 6 | `.venv/bin/python sweep.py --protocol cifar_merge --phase seeds --seeds 5` | seeds 3-4 for every CIFAR condition (the increase was triggered in the report) | `runs/cifar_*_s3`, `_s4` | ~1 h |
| 7 | `.venv/bin/python curvature.py --all-checkpoints --best-lr-only` | curvature at the 4 checkpoints of every best-LR run (18 FCN, 70 CIFAR); refuses CIFAR until step 3 passed for ρ and r_block | `results/curvature.csv` | depends on the machine (Hutchinson, q = 50) |
| 8 | `.venv/bin/python analysis.py --alpha 0.7` | final gaps, curvature summary, hypothesis tests, figures | `results/*.csv`, `results/analysis_notes.json`, `figures/fig1-5` | seconds |
| 9 | `.venv/bin/python lastlayer_control.py` | last-layer LR control: κ ∈ {3, 10, 30} at seed 0 for every C, then seeds 1-2 at the best κ | `runs_lastlayer/`, `results/lastlayer_runs.csv` | steps 9-10 together ~1.7 h (137 s/run) |
| 10 | `.venv/bin/python lastlayer_control.py --mults 0.1 0.3 --Cs 20 50 100` | extends κ below 1, then seeds 1-2 at the new best κ | same | (see step 9) |
| 11 | Level A commands | figures S1, S2, S5 and all tables, checked against the report | see Level A | ~1 min |

The run totals should match report table S2: 126 CIFAR runs (one diverged: SGD, η = 0.3, C = 20), 47 Fashion-MNIST
runs and 34 last-layer runs, for 207 in all.

**Determinism.** Every source of randomness (label merge, initialization, probe batch, batch order and the Hutchinson
vectors) comes from `config.sub_seed(seed, role)`. This makes SGD and Adam paired within a seed, and a rerun uses
the same merges, inits and batches. Training on MPS or GPU is not guaranteed to be bit-identical across PyTorch
versions or hardware. A rerun can therefore differ in the last digits; `report_numbers.py` shows by how much.
Initialization-time quantities (ρ, r_block and the theory checks) depend only on the seed and should match exactly.

**Choice of α.** The pre-registered target fraction was α = 0.5. At 0.5, SGD never reaches the target at
C ∈ {2, 5, 10} (except one seed at C = 10), so the report uses α = 0.7. That is the smallest value at which every
condition with C ≥ 10 is uncensored, and it was chosen after the data was seen. Even at 0.7, C = 2 is fully censored,
so the automatic rule `analysis.choose_alpha` finds no α and raises an error. This is why `--alpha 0.7` is always
passed explicitly. `report_numbers.py` warns if `analysis_notes.json` was written with a different α.

---

## Where each result comes from

"Data" is the file the result is computed from; "check" is the section of `results/report_numbers.md` that compares it with the report.

### Figures

| report | file | command | data |
|---|---|---|---|
| Figure 1: ρ and r_block vs C at 4 checkpoints, with fitted slopes | `figures/fig2_rho_rblock_vs_C.png` | `analysis.py --alpha 0.7` | `results/curvature.csv` |
| Figure 2: Δ_speed, Δ_loss, Δ_acc vs C (CIFAR, superclass, Fashion-MNIST) | `figures/fig1_gaps_vs_C.png` | `analysis.py --alpha 0.7` | `runs/*/metrics.csv` via `results/gaps.csv` |
| Figure S1: random vs. semantic label merge at C = 20 | `figures/figS1_merge_example.png` | `plot_merge_example.py` | `probes/base_cifar_merge.pt`, `probes/labelmap_cifar_merge_C20{,_sc}_s0.npy` |
| Figure S2: SmallCNN and FCN architecture | `figures/figS2_architecture.png` | `plot_architecture.py` | hand-drawn; must match `models.py` |
| Figure S3: LR sweep at seed 0 | `figures/fig5_lr_sweep.png` | `analysis.py --alpha 0.7` | `results/runs.csv` |
| Figure S4: e_diag · P vs C | `figures/fig3_ediag_vs_C.png` | `analysis.py --alpha 0.7` | `results/curvature.csv` |
| Figure S5: training curves at the chosen LR | `figures/fig6_training_curves.png` | `plot_training_curves.py` | `runs/*/metrics.csv`, `results/best_lr.csv` |
| Figure S6: ρ at init vs Δ_speed per (C, seed) | `figures/fig4_init_rho_vs_speed.png` | `analysis.py --alpha 0.7` | `results/curvature.csv`, `results/gaps.csv` |

`figures/hessians/` holds exploratory plots that are not in the report, and no script in this folder produces them.

### Tables

| report | content | produced by | data | check / output |
|---|---|---|---|---|
| Table 1 | ρ at init vs theory, ρ at end, Δ_speed, Δ_loss per C | `analysis.py`, `report_numbers.py` | `curvature.csv`, `gaps.csv` | Table S4 / Table 1, Table S8 / Table 1 |
| Table S1 | fixed hyperparameters | `constants.py` | - | `tableS1_hyperparameters.csv` |
| Table S2 | run budget, divergences, run times | `report_numbers.py` | `runs.csv` (`wall_time_s`), `lastlayer_runs.csv` | `tableS2_run_budget.csv` |
| Table S3 | chosen LR per cell, grid extensions | `analysis.py` (`best_lr_table`), `sweep.py` (extensions) | `best_lr.csv`, `sweep_notes.csv` | `tableS3_best_lr.csv` |
| Table S4 | ρ, r_block at init vs exact theory, and at the end | `report_numbers.py` | `curvature.csv` (10 best-LR runs per C) | `tableS4_blocks_init_end.csv` |
| Table S5 | log-log slopes per checkpoint (fit on the 6 per-C means) | `report_numbers.py` using `analysis.loglog_slope` | `curvature.csv` | `tableS5_slopes_by_checkpoint.csv` |
| Table S6 | exact FCN Hessian at init vs both theory forms | `exact.py --theory` | - | `theory_check.csv`, `tableS6_theory_check.csv` |
| Table S7 | e_diag · P at init, 10%, end | `report_numbers.py` | `curvature.csv` | `tableS7_ediagP.csv` |
| Table S8 | Δ_speed, Δ_loss, Δ_acc (mean ± s.e.) for all 10 conditions | `analysis.py` (`gap_summary`) | `gaps.csv` | `gap_summary.csv`, `tableS8_table1_gaps.csv` |
| Table S9 | steps to target, final train loss, best test accuracy | `report_numbers.py` | `gaps.csv` | `tableS9_steps_loss_acc.csv` |
| Table S10 | cv_block per C and checkpoint | `report_numbers.py` | `curvature.csv` | `tableS10_block_cv.csv` |
| Table S11 | estimators vs exact Hessian (pass rate, median and 90th-percentile error, ±2σ coverage) | `exact.py --validate-all`, `report_numbers.py` | `exact_validation.csv` | `tableS11_validation.csv` |
| Table S12 | notation | - (text only) | - | - |

Notes on how some tables are computed:

- **Table S9 (C = 5).** SGD steps, Adam steps and the ratio use only the 3 seeds where SGD reaches the target. The ratio is exp(mean Δ_speed), i.e. the geometric mean of the per-seed step ratios.
- **Table S8 (C = 2).** Δ_speed is empty because no SGD run at C = 2 reaches the target.
- **Dry runs (`_T50`).** They are excluded everywhere. One CIFAR dry run has curvature rows; including it would change the C = 100 init value of e_diag · P.

### Numbers in the text

All of these are recomputed and checked by `report_numbers.py`.

| claim in the report | value | data / how |
|---|---|---|
| H1: log-log slopes at init over all (C, seed) points | −0.576 ± 0.009 (ρ), −1.151 ± 0.019 (r_block) | `analysis_notes.json` → `hypothesis_tests` |
| H1: exact-form slopes over C = 2...100 | −0.576, −1.152 | fit of 1/√(C−1), 1/(C−1) |
| H1: largest deviation from the exact forms at init | 1.2% (r_block, C = 5) | Table S4 |
| H1: ρ at C = 100 rises during training | 0.101 → 0.355 | Table S4 |
| H1: ρ at the end for C ≥ 10 | 0.31-0.38 | Table S4 |
| H1: FCN ρ at C = 10 by mid-training | 0.34 → 0.69 | `curvature.csv`, Fashion-MNIST |
| H2: Spearman(Δ_speed, −ρ) over the 5 uncensored C | −0.30, p = 0.62 | `analysis_notes.json` |
| H2: slope of Δ_speed on log C over 23 (C, seed) pairs | 0.003 ± 0.013 | `gaps.csv`, OLS |
| H2: unnormalized L_SGD − L_Adam for C = 2...100 | 0.35, 0.71, 0.71, 0.65, 0.44, 0.21 | Table S9 |
| H2: Δ_acc | at most 1.5 percentage points | Table S8 |
| H3: Spearman(ρ at init, Δ_speed) | −0.06, p = 0.78, n = 23 | `analysis_notes.json` |
| Secondary question: e_diag · P slope at init | −0.31 ± 0.06 | `analysis_notes.json` |
| Secondary question: drop of e_diag · P between init and 10% | factor 5-37 | Table S7 |
| Superclass: Δ_speed / Δ_loss / mean accuracy vs random merge at C = 20 | 0.63 / 0.07 / 60% vs 0.56 / 0.22 / 41% | Table S8, `gaps.csv` |
| α = 0.5: censored conditions; step ratio for uncensored C | C ∈ {2, 5, 10} censored; ratio 1.4-1.55 | `gaps.csv` (alpha = 0.5 rows) |
| Fashion-MNIST: target reached within the warmup | 50-80 steps | `gaps.csv` |
| Last-layer control: κ chosen by final loss for C = 5...100 | 10, 10, 3, 0.1, 0.3 | `lastlayer_runs.csv` + the baseline SGD run (κ = 1) |
| Last-layer control: share of the loss gap closed | at most 12% | same |
| Last-layer control: Δ_speed at the chosen κ; slope on log C | 0.50, 0.50, 0.52, 0.66, 0.69; 0.07 ± 0.02 (n = 15) | `runs_lastlayer/*/metrics.csv`, steps to target at α = 0.7 |
| Last-layer control, seed 0: Δ_speed at κ = 30; plain SGD | 0.39, 0.48, 0.30, 0.37, 0.49; 0.61, 0.52, 0.55, 0.64 (C = 10...100) | same |
| Model size; last-layer share of parameters | ≈ 0.3 M (SmallCNN); 6304 + 33C (FCN); 0.1-4.3% | `models.py` (checked with torch if installed) |
| Estimator accuracy: e_diag median error, at init, naive estimator | 20%, 41%, factor 2.6 | `exact_validation.csv` |

The theory in appendix A (equations 1-15) is derivation, not computation. Its finite-C forms are checked
numerically by Table S6 (exact Hessian) and Table S4 (CNN measurement).

---

## Known differences between the saved results and report v10

`report_numbers.py` flags these as KNOWN. They are rounding slips in the report; the experiments are not affected.

| report location | report says | data gives | fix |
|---|---|---|---|
| Table S2, total time | ≈ 6.1 h | 6.05 h (the report adds the rounded rows 4.4 + 1.7) | ≈ 6.0 h |
| Table S9, C = 50 step ratio | 1.83 | 1.824 (1.83 comes from dividing the rounded 2630 / 1440) | 1.82 |
| Table S10, C = 10 at 50% | 0.201 | 0.2005 | 0.200 |
| Appendix ה.7, e_diag error at init | 41% | 41.6% | 42% |

After correcting any of these in the report, also update the matching expected value in `report_numbers.py`, so that
the check continues to compare against the printed report.
