"""Datasets, label maps, probe batch, batch iterator. Communicates via probes/ on disk."""
from __future__ import annotations

import pickle
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

from config import RunConfig, sub_seed
from constants import BASE_CLASSES, PROBES_DIR, PROBE_PER_CLASS


@dataclass
class Data:
    x_train: Tensor  # [N, C_in, H, W] uint8 on device (normalized per batch)
    y_train: Tensor  # [N] int64, merged labels in [0, C)
    x_test: Tensor
    y_test: Tensor
    probe_x: Tensor  # [n_probe, C_in, H, W] float32, already normalized
    probe_y: Tensor
    C: int
    label_map: np.ndarray  # [base_classes] -> [0, C)
    mean: Tensor  # [C_in,1,1]
    std: Tensor


# --------------------------------------------------------------------------- base data


def _base_cache(protocol: str) -> Path:
    return PROBES_DIR / f"base_{protocol}.pt"


def _download_cifar100(raw_dir: Path) -> dict:
    """Read the CIFAR-100 python pickles (fine + coarse labels) that torchvision downloads."""
    import torchvision  # the only torchvision import in the project

    torchvision.datasets.CIFAR100(str(raw_dir), train=True, download=True)
    folder = raw_dir / "cifar-100-python"
    if not folder.exists():  # some torchvision versions leave only the archive
        with tarfile.open(raw_dir / "cifar-100-python.tar.gz") as tf:
            tf.extractall(raw_dir)

    def load(name):
        with open(folder / name, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        x = np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
        return x, np.asarray(d["fine_labels"], dtype=np.int64), np.asarray(d["coarse_labels"], dtype=np.int64)

    xtr, ytr_f, ytr_c = load("train")
    xte, yte_f, yte_c = load("test")
    return dict(x_train=xtr, y_train_fine=ytr_f, y_train_coarse=ytr_c,
                x_test=xte, y_test_fine=yte_f, y_test_coarse=yte_c)


def _download_fmnist(raw_dir: Path) -> dict:
    import torchvision

    tr = torchvision.datasets.FashionMNIST(str(raw_dir), train=True, download=True)
    te = torchvision.datasets.FashionMNIST(str(raw_dir), train=False, download=True)
    return dict(
        x_train=tr.data.numpy().astype(np.uint8)[:, None],
        y_train_fine=tr.targets.numpy().astype(np.int64),
        y_train_coarse=None,
        x_test=te.data.numpy().astype(np.uint8)[:, None],
        y_test_fine=te.targets.numpy().astype(np.int64),
        y_test_coarse=None,
    )


def load_base(protocol: str) -> dict:
    """Download once via torchvision, cache to probes/base_{protocol}.pt as
    {x_train: uint8, y_train_fine, y_train_coarse|None, x_test: uint8, y_test_fine, mean, std}."""
    assert protocol in BASE_CLASSES
    cache = _base_cache(protocol)
    if cache.exists():
        return torch.load(cache, weights_only=False)
    PROBES_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = PROBES_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)
    d = _download_cifar100(raw_dir) if protocol == "cifar_merge" else _download_fmnist(raw_dir)
    x_train = torch.from_numpy(d["x_train"])
    xf = x_train.double() / 255.0
    mean = xf.mean(dim=(0, 2, 3)).float().view(-1, 1, 1)
    std = xf.std(dim=(0, 2, 3)).float().view(-1, 1, 1)
    base = dict(
        x_train=x_train,
        y_train_fine=torch.from_numpy(d["y_train_fine"]),
        y_train_coarse=None if d["y_train_coarse"] is None else torch.from_numpy(d["y_train_coarse"]),
        x_test=torch.from_numpy(d["x_test"]),
        y_test_fine=torch.from_numpy(d["y_test_fine"]),
        mean=mean,
        std=std,
    )
    assert int(base["y_train_fine"].max()) + 1 == BASE_CLASSES[protocol]
    torch.save(base, cache)
    return base


def coarse_of_fine(protocol: str) -> np.ndarray:
    """CIFAR-100: array [100] mapping each fine class to its coarse (super)class."""
    assert protocol == "cifar_merge"
    base = load_base(protocol)
    yf, yc = base["y_train_fine"].numpy(), base["y_train_coarse"].numpy()
    m = np.full(100, -1, dtype=np.int64)
    m[yf] = yc
    assert (m >= 0).all() and len(np.unique(m)) == 20
    return m


# --------------------------------------------------------------------------- label map / probe


def normalize(x_uint8: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return (x_uint8.float() / 255.0 - mean) / std


def make_label_map(protocol: str, C: int, superclass: bool, seed: int) -> np.ndarray:
    base = BASE_CLASSES[protocol]
    assert base % C == 0
    if superclass:
        assert protocol == "cifar_merge" and C == 20
        return coarse_of_fine(protocol)
    if C == base:
        return np.arange(base, dtype=np.int64)  # identity: no merging
    rng = np.random.default_rng(sub_seed(seed, "merge"))
    perm = rng.permutation(base)  # perm[i] = fine class in slot i
    pos = np.empty(base, dtype=np.int64)
    pos[perm] = np.arange(base)  # slot of each fine class
    return pos // (base // C)


def make_probe_indices(protocol: str, seed: int, y_fine: np.ndarray) -> np.ndarray:
    """PROBE_PER_CLASS training indices per fine class, sorted by class then index."""
    rng = np.random.default_rng(sub_seed(seed, "probe"))
    idx = []
    for k in range(BASE_CLASSES[protocol]):
        members = np.flatnonzero(y_fine == k)
        idx.append(np.sort(rng.choice(members, size=PROBE_PER_CLASS, replace=False)))
    return np.concatenate(idx)


def _labelmap_path(protocol: str, C: int, superclass: bool, seed: int) -> Path:
    sc = "_sc" if superclass else ""
    return PROBES_DIR / f"labelmap_{protocol}_C{C}{sc}_s{seed}.npy"


def _probe_path(protocol: str, seed: int) -> Path:
    return PROBES_DIR / f"{protocol}_s{seed}.pt"


def get_label_map(protocol: str, C: int, superclass: bool, seed: int) -> np.ndarray:
    """Cached make_label_map (probes/labelmap_*.npy)."""
    p = _labelmap_path(protocol, C, superclass, seed)
    if p.exists():
        return np.load(p)
    m = make_label_map(protocol, C, superclass, seed)
    PROBES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(p, m)
    return m


def get_probe(protocol: str, seed: int, base: dict | None = None) -> dict:
    """Cached probe {indices: LongTensor, x: float32 normalized} (probes/{protocol}_s{seed}.pt)."""
    p = _probe_path(protocol, seed)
    if p.exists():
        return torch.load(p, weights_only=True)
    base = base or load_base(protocol)
    idx = torch.from_numpy(make_probe_indices(protocol, seed, base["y_train_fine"].numpy()))
    probe = {"indices": idx, "x": normalize(base["x_train"][idx], base["mean"], base["std"])}
    PROBES_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(probe, p)
    return probe


# --------------------------------------------------------------------------- build / batches


def build(cfg: RunConfig, device: str) -> Data:
    base = load_base(cfg.protocol)
    lm = get_label_map(cfg.protocol, cfg.C, cfg.superclass, cfg.seed)
    lm_t = torch.from_numpy(lm)
    probe = get_probe(cfg.protocol, cfg.seed, base)
    y_train = lm_t[base["y_train_fine"]]
    return Data(
        x_train=base["x_train"].to(device),
        y_train=y_train.to(device),
        x_test=base["x_test"].to(device),
        y_test=lm_t[base["y_test_fine"]].to(device),
        probe_x=probe["x"].to(device),
        probe_y=y_train[probe["indices"]].to(device),
        C=cfg.C,
        label_map=lm,
        mean=base["mean"].to(device),
        std=base["std"].to(device),
    )


def batches(data: Data, batch_size: int, steps: int, seed: int) -> Iterator[tuple[Tensor, Tensor]]:
    """Reshuffle every epoch from sub_seed(seed, 'order'); yield exactly `steps` batches; drop_last."""
    # CPU generator: identical batch order regardless of the training device.
    g = torch.Generator().manual_seed(sub_seed(seed, "order"))
    N, emitted = len(data.y_train), 0
    dev = data.x_train.device
    while emitted < steps:
        order = torch.randperm(N, generator=g).to(dev)
        for i in range(0, N - batch_size + 1, batch_size):
            idx = order[i : i + batch_size]
            yield normalize(data.x_train[idx], data.mean, data.std), data.y_train[idx]
            emitted += 1
            if emitted == steps:
                return
