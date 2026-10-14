import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import data as D
from config import RunConfig
from constants import BASE_CLASSES, C_VALUES, PROBE_PER_CLASS


@pytest.fixture(scope="module", params=["fmnist_merge", "cifar_merge"])
def protocol(request):
    return request.param


@pytest.fixture(scope="module")
def base(protocol):
    return D.load_base(protocol)


def test_label_map_is_partition(protocol):
    base_n = BASE_CLASSES[protocol]
    for C in C_VALUES[protocol]:
        for seed in (0, 1):
            m = D.make_label_map(protocol, C, False, seed)
            assert m.shape == (base_n,)
            assert set(np.unique(m)) == set(range(C))
            assert (np.bincount(m, minlength=C) == base_n // C).all()
    # C = base is identity
    assert (D.make_label_map(protocol, base_n, False, 0) == np.arange(base_n)).all()
    # deterministic in seed, differs across seeds
    assert (D.make_label_map(protocol, 2, False, 3) == D.make_label_map(protocol, 2, False, 3)).all()
    if base_n >= 10:
        assert not (D.make_label_map(protocol, 2, False, 0) == D.make_label_map(protocol, 2, False, 1)).all()


def test_superclass_equals_coarse():
    base = D.load_base("cifar_merge")
    m = D.make_label_map("cifar_merge", 20, True, 0)
    yc = m[base["y_train_fine"].numpy()]
    assert (yc == base["y_train_coarse"].numpy()).all()
    assert (np.bincount(m, minlength=20) == 5).all()


def test_train_counts_per_merged_class(protocol, base):
    yf = base["y_train_fine"].numpy()
    base_n = BASE_CLASSES[protocol]
    per_fine = len(yf) // base_n
    for C in C_VALUES[protocol]:
        m = D.make_label_map(protocol, C, False, 0)
        counts = np.bincount(m[yf], minlength=C)
        assert (counts == per_fine * (base_n // C)).all()


def test_probe_balanced_under_every_C(protocol, base):
    yf = base["y_train_fine"].numpy()
    idx = D.make_probe_indices(protocol, 0, yf)
    base_n = BASE_CLASSES[protocol]
    assert len(idx) == PROBE_PER_CLASS * base_n
    assert len(np.unique(idx)) == len(idx)
    for C in C_VALUES[protocol]:
        m = D.make_label_map(protocol, C, False, 0)
        assert (np.bincount(m[yf[idx]], minlength=C) == PROBE_PER_CLASS * (base_n // C)).all()
    # cached probe file matches
    probe = D.get_probe(protocol, 0, base)
    assert (probe["indices"].numpy() == idx).all()
    assert probe["x"].dtype == torch.float32


def test_build_and_batches(protocol):
    C = C_VALUES[protocol][1]
    cfg = RunConfig(protocol=protocol, C=C, superclass=False, seed=0, optimizer="sgd", lr=0.1, steps=10)
    d = D.build(cfg, "cpu")
    assert d.x_train.dtype == torch.uint8 and d.y_train.dtype == torch.int64
    assert int(d.y_train.max()) == C - 1 and int(d.y_test.max()) == C - 1
    assert d.probe_x.shape[0] == PROBE_PER_CLASS * BASE_CLASSES[protocol]
    assert (torch.bincount(d.probe_y, minlength=C) == PROBE_PER_CLASS * BASE_CLASSES[protocol] // C).all()
    steps, B = 7, 64
    got = list(D.batches(d, B, steps, seed=0))
    assert len(got) == steps
    assert all(x.shape[0] == B and y.shape[0] == B and x.dtype == torch.float32 for x, y in got)
    # deterministic order
    again = list(D.batches(d, B, steps, seed=0))
    assert all(torch.equal(a[1], b[1]) for a, b in zip(got, again))
    # normalized data has ~0 mean / ~1 std over train set
    xs = D.normalize(d.x_train[:5000], d.mean, d.std)
    assert abs(xs.mean().item()) < 0.1 and abs(xs.std().item() - 1) < 0.1


def test_uniform_predictor_loss_is_log_C(protocol):
    for C in C_VALUES[protocol]:
        cfg = RunConfig(protocol=protocol, C=C, superclass=False, seed=0, optimizer="sgd", lr=0.1, steps=10)
        d = D.build(cfg, "cpu")
        logits = torch.zeros(len(d.y_test), C, dtype=torch.float64)  # float32 rounds at ~1e-6 for N=10k
        assert abs(F.cross_entropy(logits, d.y_test).item() - math.log(C)) < 1e-6
