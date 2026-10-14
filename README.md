# Does Adam's advantage grow with the number of classes?

Code for the Deep Learning course project of Yoav Shabtai and Almog Singer (Tel Aviv University).
The final report is `project_report` v10 (Hebrew, October 2026); the design follows `code_plan_v1.md` / `project_plan_v1.md`.

**To reproduce any figure, table or number of the report, see [`REPRODUCE.md`](REPRODUCE.md).**
`report_numbers.py` regenerates every table and in-text number from `results/` and checks each against the report.

Short version of the result: at initialization the last-layer Hessian is block-diagonal by class exactly as
ρ = 1/√(C−1), r_block = 1/(C−1) (H1 confirmed), but the structure erodes during training, and Adam's speed advantage
over SGD is constant in C (Δ_speed ≈ 0.6, step ratio ≈ 1.8 for C = 5…100; H2 and H3 not confirmed).

## Setup

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests -q
```

Python 3.13, PyTorch ≥ 2.1 (`torch.func` for Hessian-vector products and the dense FCN Hessian). All runs of the report
were done on an Apple Silicon Mac: CIFAR on MPS, Fashion-MNIST on CPU, float64 accumulations on CPU.

Datasets are downloaded once by `data.load_base` into `probes/raw/` and cached as `probes/base_{protocol}.pt`.
If the torchvision download fails with an SSL certificate error (python.org builds), fetch the archive manually:

```bash
curl -L -o probes/raw/cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
```

## Execution order (project plan Section 12)

```bash
.venv/bin/python -m pytest tests -q                                        # 1. tests
.venv/bin/python sweep.py --protocol fmnist_merge --phase auto --seeds 3   # 2. FMNIST sweep (CPU, < 1 s/run)
.venv/bin/python exact.py --theory                                         # 3a. exact rho, r_block at init vs theory
.venv/bin/python exact.py --validate-all                                   # 3b. estimator validation on all FCN checkpoints
.venv/bin/python sweep.py --protocol cifar_merge --phase auto --seeds 3    # 4. CIFAR sweep (MPS, ~2 min/run)
.venv/bin/python curvature.py --all-checkpoints --best-lr-only             # 5. curvature (refuses CIFAR until 3b passes)
.venv/bin/python analysis.py --alpha 0.7                                   # 6-7. gaps, tests, escalation check, figures
.venv/bin/python sweep.py --protocol cifar_merge --phase seeds --seeds 5   # 8. if analysis_notes.json says escalate
```

Step 8 did trigger in the report (both escalation conditions held after three seeds), so every CIFAR condition has
five seeds; Fashion-MNIST stayed at three. After step 8, rerun steps 5 and 6 so the new seeds are measured and analysed.

`--alpha 0.7` is required for the report's numbers: at the target fraction α = 0.5 SGD is censored at C ∈ {2, 5, 10},
and even at 0.7 it never reaches the target at C = 2, so the automatic `choose_alpha` finds no fully uncensored α and
raises. α = 0.7 is the smallest value at which every condition with C ≥ 10 is uncensored; it was chosen after seeing
the data (stated as a limitation in the report).

Post-hoc additions (run after the main analysis; not part of the pre-registered plan):

```bash
.venv/bin/python plot_training_curves.py                              # report figure S5 (figures/fig6_training_curves.png)
.venv/bin/python lastlayer_control.py                                 # last-layer learning-rate control, phase 1 + 2
.venv/bin/python lastlayer_control.py --mults 0.1 0.3 --Cs 20 50 100  # extension below κ = 1
```

`lastlayer_control.py` writes to `runs_lastlayer/` and `results/lastlayer_runs.csv` and never touches `runs/` or
`results/runs.csv`. The two calls above produce the report's 34 runs (24 at seed 0, 10 at seeds 1-2 at the best κ),
~1.7 h on MPS. κ (the head-LR multiplier) is called `mult` / `head_mult` in the code. `run_lastlayer.command` is a
double-click wrapper for macOS.

Illustration figures of the report (no training needed; the first needs the CIFAR-100 cache and the seed-0 label maps in `probes/`):

```bash
.venv/bin/python plot_merge_example.py                                # report figure S1 (figures/figS1_merge_example.png)
.venv/bin/python plot_architecture.py                                 # report figure S2 (figures/figS2_architecture.png)
```

All tables and in-text numbers of the report (no training needed, ~1 min):

```bash
.venv/bin/python report_numbers.py      # results/report_tables/*.csv + results/report_numbers.md (each value vs. the report)
```

`train.py --dry-run` (50 steps) writes a real run directory with a `_T50` suffix so it never shadows a real run.
`sweep.py --dry-run` smoke-tests the whole sweep logic the same way.

## Run budget (report table S2)

| protocol | LR sweep (seed 0) | extra seeds | total | diverged | median time/run |
|---|---|---|---|---|---|
| CIFAR-100 merged (6 values of C + superclass) | 70 | 56 | 126 | 1 (SGD, η = 0.3, C = 20) | 112 s |
| Fashion-MNIST merged | 30 + 5 extensions | 12 | 47 | 0 | < 1 s |
| last-layer control (CIFAR) | 24 | 10 | 34 | 0 | 137 s |

Total 207 runs, ≈ 6.1 h.

## Where each report figure and table comes from

| report | file | produced by |
|---|---|---|
| Figure 1 (ρ, r_block vs C) | `figures/fig2_rho_rblock_vs_C.png` | `analysis.py` |
| Figure 2 (Δ_speed, Δ_loss, Δ_acc vs C) | `figures/fig1_gaps_vs_C.png` | `analysis.py` |
| Figure S1 (label merge example) | `figures/figS1_merge_example.png` | `plot_merge_example.py` |
| Figure S2 (architectures) | `figures/figS2_architecture.png` | `plot_architecture.py` |
| Figure S3 (LR sweep) | `figures/fig5_lr_sweep.png` | `analysis.py` |
| Figure S4 (e_diag · P vs C) | `figures/fig3_ediag_vs_C.png` | `analysis.py` |
| Figure S5 (training curves) | `figures/fig6_training_curves.png` | `plot_training_curves.py` |
| Figure S6 (ρ at init vs Δ_speed) | `figures/fig4_init_rho_vs_speed.png` | `analysis.py` |
| Tables 1, S8 (gaps per C) | `results/gap_summary.csv` | `analysis.py` |
| Table S9 (steps to target, losses, accuracy) | `results/gaps.csv` | `analysis.py` |
| Table S3 (chosen LR, grid extensions) | `results/best_lr.csv`, `results/sweep_notes.csv` | `analysis.py`, `sweep.py` |
| Tables S4, S5, S7, S10 (curvature per C and checkpoint) | `results/curvature_summary.csv` (from `results/curvature.csv`) | `curvature.py`, `analysis.py` |
| H1/H2/H3 slopes and Spearman tests, escalation | `results/analysis_notes.json` | `analysis.py` |
| Table S6 (exact FCN Hessian vs theory) | `results/theory_check.csv` | `exact.py --theory` |
| Table S11 (estimators vs exact Hessian, 188 checkpoints) | `results/exact_validation.csv` | `exact.py --validate-all` |
| Table S2 (run budget) | `results/runs.csv`, `results/lastlayer_runs.csv` | `train.py`, `lastlayer_control.py` |
| Last-layer control (appendix ה.4) | `results/lastlayer_runs.csv`, `runs_lastlayer/` | `lastlayer_control.py` |
| Tables S2, S4, S5, S7, S9, S10, S11 and the in-text numbers, as printed | `results/report_tables/*.csv`, `results/report_numbers.md` | `report_numbers.py` |

`figures/hessians/` holds exploratory Hessian plots that are not used in the report and are not produced by any
script in this folder.

## Modules

| file | role | report section |
|---|---|---|
| `constants.py` | every fixed hyperparameter (report table S1) | ג.3 |
| `config.py` | `RunConfig`, `run_id`, `sub_seed` (paired seeds), `default_device` | ג.3 |
| `data.py` | base data cache, label maps (random and superclass merge), probe batch, batch iterator | ג.1 |
| `models.py` | `SmallCNN`, `FCN`, `class_block_indices` | ג.2 |
| `train.py` | `run(cfg) -> RunRecord`, checkpoints, `results/runs.csv` | ג.3 |
| `sweep.py` | LR grid, best-LR rule, edge extension, seeds, escalation rule | ג.3, ה.1 |
| `curvature.py` | closed-form last-layer blocks + Hutchinson, `results/curvature.csv` | ג.5 |
| `exact.py` | dense FCN Hessian, estimator validation, theory check | ג.6, ה.7 |
| `analysis.py` | paired gaps, α, hypothesis tests, figures | ג.4, ה |
| `plot_training_curves.py` | post-hoc: training curves of the best-LR runs (figure S5) | ה.4 |
| `lastlayer_control.py` | post-hoc: SGD with a separate last-layer learning rate κ | ה.4 |
| `plot_merge_example.py` | illustration: random vs. superclass label merge at C = 20 (figure S1) | ג.1 |
| `plot_architecture.py` | illustration: block diagram of `SmallCNN` and `FCN` (figure S2; hand-drawn, keep in sync with `models.py`) | ג.2 |
| `report_numbers.py` | every report table (1, S1-S11) and in-text number recomputed from `results/` and checked against report v10 | all |

## Deviations from the plan worth knowing

- **Bias-corrected `e_diag`.** With K Hutchinson vectors (`--K`, `HUTCH_PROBES` = 50; called q in the report) the naive
  diagonal estimate satisfies E‖d̂‖² = ‖diag H‖² + (‖H‖_F² − ‖diag H‖²)/K, and `e_diag` is ~0.01 during training, so the
  raw ratio is several times too large. `curvature.csv` stores the corrected `e_diag`/`e_diag_P` (U-statistic over
  pairs of distinct Hutchinson vectors, jackknife standard error) and the raw value as `e_diag_raw`. The corrected value
  is unbiased but noisy: median relative error 20% against the exact FCN Hessian, 95% of exact values within ±2 SE.
- **Finite-C theory.** At a uniform softmax A_cc = (C−1)/C², A_cc' = −1/C², so ρ = 1/√(C−1) and r_block = 1/(C−1)
  exactly (ρ = 1 at C = 2). The plan's 1/√C and 1/C are the large-C forms; `theory_check` reports both.
- **`last_layer_blocks`** uses ‖H_cc'‖_F² = (1/m²) a_cc'ᵀ (XXᵀ)^∘2 a_cc' (m = probe-batch size) so no (d+1)² blocks are
  materialised; the Gram matrix is computed once in float64.
- **α choice.** `analysis.choose_alpha` uses 0.5 when every best-LR run reaches it, otherwise the smallest uncensored α
  above 0.5 (the code plan's "largest uncensored alpha" would always return 0.7). On the CIFAR data no α is fully
  uncensored, so the report's analysis was run with `--alpha 0.7` (see above).
- **Dry runs are filtered.** Runs with non-default `steps` (run_id suffix `_T{steps}`) are excluded from
  `sweep.load_runs`, `analysis.load_runs`, and `curvature.all_run_ids`.
- **Import direction.** `sweep.py` owns `best_lr` and `escalation_check` as pure dataframe functions;
  `analysis.py` imports them lazily. Neither imports `train.py` at module level; `curvature.py`/`exact.py` never do.
- **Pooling.** `models.MaxPool2x2` equals `nn.MaxPool2d(2)` in the forward pass but is written as pairwise
  `torch.maximum` so HVPs work on MPS (see the class docstring for the alternatives that were rejected).
- **Devices.** float64 accumulation happens on CPU when the model is on MPS (MPS has no float64).
  FMNIST defaults to CPU (10× faster than MPS for the tiny FCN); CIFAR defaults to cuda > mps > cpu.
