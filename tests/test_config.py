import json

import pytest

from config import RunConfig, ckpt_path, is_done, sub_seed
from constants import BASE_CLASSES, CHECKPOINT_FRACS


def mk(**kw):
    base = dict(protocol="cifar_merge", C=10, superclass=False, seed=0,
                optimizer="adam", lr=1e-3, steps=8000)
    base.update(kw)
    return RunConfig(**base)


def test_bad_C_asserts():
    with pytest.raises(AssertionError):
        mk(C=3)
    with pytest.raises(AssertionError):
        mk(protocol="fmnist_merge", C=4)
    with pytest.raises(AssertionError):
        mk(C=0)


def test_superclass_constraints():
    mk(C=20, superclass=True)  # ok
    with pytest.raises(AssertionError):
        mk(C=10, superclass=True)
    with pytest.raises(AssertionError):
        mk(protocol="fmnist_merge", C=10, superclass=True)


def test_bad_lr_steps():
    with pytest.raises(AssertionError):
        mk(lr=0.0)
    with pytest.raises(AssertionError):
        mk(steps=0)


def test_run_id_stable_across_lr_representations():
    a, b = mk(lr=3e-4), mk(lr=0.0003)
    assert a.run_id == b.run_id == "cifar_merge_C10_adam_lr3.0e-04_s0"
    assert mk(C=20, superclass=True).run_id == "cifar_merge_C20_sc_adam_lr1.0e-03_s0"


def test_run_id_marks_nondefault_steps():
    from constants import STEPS
    assert mk(steps=STEPS["cifar_merge"]).run_id == "cifar_merge_C10_adam_lr1.0e-03_s0"
    assert mk(steps=50).run_id == "cifar_merge_C10_adam_lr1.0e-03_s0_T50"


def test_json_roundtrip():
    cfg = mk(C=20, superclass=True, lr=0.03, optimizer="sgd")
    assert RunConfig.from_json(cfg.to_json()) == cfg
    assert json.loads(cfg.to_json())["batch_size"] == cfg.batch_size


def test_derived():
    cfg = mk(steps=1000)  # non-default length -> _T1000 suffix
    assert cfg.checkpoint_steps == {f: round(f * 1000) for f in CHECKPOINT_FRACS}
    assert cfg.checkpoint_steps[0.0] == 0 and cfg.checkpoint_steps[1.0] == 1000
    assert abs(cfg.chance_loss - __import__("math").log(10)) < 1e-12
    assert ckpt_path(cfg, 0.5).name == "ckpt_0.5.pt"
    assert not is_done(cfg)


def test_sub_seed_deterministic_and_role_distinct():
    assert sub_seed(0, "merge") == sub_seed(0, "merge")
    roles = ["merge", "init", "probe", "order", "hutchinson"]
    vals = [sub_seed(7, r) for r in roles]
    assert len(set(vals)) == len(roles)
    assert sub_seed(0, "init") != sub_seed(1, "init")
    assert all(0 <= v < 2**32 for v in vals)
    with pytest.raises(AssertionError):
        sub_seed(0, "optimizer")
