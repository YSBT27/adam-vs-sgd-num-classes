import json

import pandas as pd
import pytest

import constants
import train as T
from config import RunConfig, ckpt_path, is_done
from constants import CHECKPOINT_FRACS


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    runs, results = tmp_path / "runs", tmp_path / "results"
    monkeypatch.setattr(constants, "RUNS_DIR", runs)
    import config
    monkeypatch.setattr(config, "RUNS_DIR", runs)
    monkeypatch.setattr(T, "RESULTS_DIR", results)
    monkeypatch.setattr(T, "RUNS_CSV", results / "runs.csv")
    return runs, results


def test_dry_run_creates_everything_and_is_idempotent(tmp_dirs):
    runs, results = tmp_dirs
    cfg = RunConfig(protocol="fmnist_merge", C=5, superclass=False, seed=0, optimizer="adam", lr=1e-3, steps=2000)
    rec = T.run(cfg, device="cpu", dry_run=True)
    assert rec.steps == constants.DRY_RUN_STEPS and not rec.diverged
    assert rec.run_id.endswith(f"_T{constants.DRY_RUN_STEPS}") and not is_done(cfg)  # real run untouched
    import dataclasses
    cfg = dataclasses.replace(cfg, steps=constants.DRY_RUN_STEPS)
    assert (cfg.run_dir / "config.json").exists()
    assert RunConfig.from_json((cfg.run_dir / "config.json").read_text()) == cfg
    m = pd.read_csv(cfg.run_dir / "metrics.csv")
    assert len(m) == constants.DRY_RUN_STEPS and list(m.step) == list(range(1, constants.DRY_RUN_STEPS + 1))
    assert m.test_acc.notna().sum() >= 1
    for f in CHECKPOINT_FRACS:
        assert ckpt_path(cfg, f).exists()
    assert is_done(cfg)
    df = pd.read_csv(results / "runs.csv")
    assert len(df) == 1 and df.run_id[0] == cfg.run_id
    # second call is a no-op
    mtime = ckpt_path(cfg, 1.0).stat().st_mtime
    rec2 = T.run(cfg, device="cpu", dry_run=True)
    assert rec2.run_id == rec.run_id and abs(rec2.final_train_loss - rec.final_train_loss) < 1e-9
    assert ckpt_path(cfg, 1.0).stat().st_mtime == mtime
    assert len(pd.read_csv(results / "runs.csv")) == 1


def test_lr_schedule():
    assert T.lr_at(0, 1.0) == 0.0
    assert abs(T.lr_at(constants.WARMUP_STEPS // 2, 1.0) - 0.5) < 1e-12
    assert T.lr_at(constants.WARMUP_STEPS, 1.0) == 1.0
    assert T.lr_at(10 * constants.WARMUP_STEPS, 0.3) == 0.3
