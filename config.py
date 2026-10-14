"""RunConfig: the only thing that varies between runs. One frozen config in, files out."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from constants import BASE_CLASSES, BATCH_SIZE, CHECKPOINT_FRACS, RUNS_DIR, STEPS

Protocol = Literal["cifar_merge", "fmnist_merge"]
Optimizer = Literal["sgd", "adam"]


@dataclass(frozen=True)
class RunConfig:
    protocol: Protocol
    C: int
    superclass: bool
    seed: int
    optimizer: Optimizer
    lr: float
    steps: int
    batch_size: int = BATCH_SIZE

    def __post_init__(self):
        assert self.protocol in BASE_CLASSES, f"unknown protocol {self.protocol!r}"
        assert self.optimizer in ("sgd", "adam"), f"unknown optimizer {self.optimizer!r}"
        assert self.C > 0 and BASE_CLASSES[self.protocol] % self.C == 0, (
            f"C={self.C} must divide {BASE_CLASSES[self.protocol]}"
        )
        assert not self.superclass or (self.protocol == "cifar_merge" and self.C == 20), (
            "superclass requires protocol='cifar_merge' and C=20"
        )
        assert self.lr > 0 and self.steps > 0 and self.batch_size > 0

    # ---- derived -------------------------------------------------------
    @property
    def chance_loss(self) -> float:
        return math.log(self.C)

    @property
    def checkpoint_steps(self) -> dict[float, int]:
        return {f: round(f * self.steps) for f in CHECKPOINT_FRACS}

    @property
    def run_id(self) -> str:
        sc = "_sc" if self.superclass else ""
        t = "" if self.steps == STEPS[self.protocol] else f"_T{self.steps}"  # non-default length (e.g. dry runs)
        return f"{self.protocol}_C{self.C}{sc}_{self.optimizer}_lr{self.lr:.1e}_s{self.seed}{t}"

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / self.run_id

    # ---- serialization -------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "RunConfig":
        return cls(**json.loads(s))

    @classmethod
    def load(cls, run_id: str) -> "RunConfig":
        return cls.from_json((RUNS_DIR / run_id / "config.json").read_text())


SUB_SEED_ROLES = ("merge", "init", "probe", "order", "hutchinson")


def sub_seed(seed: int, role: str) -> int:
    """Deterministic child seed. Roles never depend on optimizer or lr -> paired design."""
    assert role in SUB_SEED_ROLES, f"unknown sub_seed role {role!r}"
    digest = hashlib.sha256(f"{seed}:{role}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def default_device(protocol: str | None = None) -> str:
    """cuda > mps > cpu; the tiny FMNIST FCN is ~10x faster on CPU than on MPS, so it defaults to CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if protocol == "fmnist_merge":
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ckpt_path(cfg: RunConfig, frac: float) -> Path:
    return cfg.run_dir / f"ckpt_{frac:.1f}.pt"


def is_done(cfg: RunConfig) -> bool:
    return ckpt_path(cfg, 1.0).exists()
