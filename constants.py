"""Every fixed hyperparameter of v1. Pure values, no logic."""
from pathlib import Path

ROOT = Path(__file__).parent
RUNS_DIR, PROBES_DIR, RESULTS_DIR, FIGURES_DIR = (
    ROOT / d for d in ("runs", "probes", "results", "figures")
)

BASE_CLASSES = {"cifar_merge": 100, "fmnist_merge": 10}
C_VALUES = {"cifar_merge": (2, 5, 10, 20, 50, 100), "fmnist_merge": (2, 5, 10)}
STEPS = {"cifar_merge": 8000, "fmnist_merge": 2000}
BATCH_SIZE = 128

SGD_MOMENTUM = 0.9
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 200

LR_GRID = {
    "sgd": (0.3, 0.1, 0.03, 0.01, 0.003),
    "adam": (3e-3, 1e-3, 3e-4, 1e-4, 3e-5),
}
GRID_FACTOR = 10 ** 0.5  # spacing used when extending an edge
MAX_GRID_EXTENSIONS = 2  # per cell

SEEDS, FALLBACK_SEEDS = (0, 1, 2), (3, 4)

CHECKPOINT_FRACS = (0.0, 0.1, 0.5, 1.0)
EVAL_EVERY = 200
FINAL_WINDOW_FRAC = 0.05  # final_train_loss = mean over last 5% of steps
DIVERGE_FACTOR, DIVERGE_GRACE = 2.0, 500  # loss > 2*log C after warmup+500 -> diverged

PROBE_PER_CLASS = 10  # per *original* class -> 1000 (CIFAR) / 100 (FMNIST)
HUTCH_PROBES = 50
SMOOTH_WINDOW = 50
TARGET_ALPHAS = (0.3, 0.5, 0.7)
EXACT_TOL = 0.05

DRY_RUN_STEPS = 50
