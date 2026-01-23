from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.settings import PROJECT_ROOT_PATH
from src.data.generate.base import WindowsDataset
from src.data.factory import save_dataset, make_dataset


def _data_conf_dir() -> Path:
    """Find the Hydra data config root that contains config.yaml."""
    root = Path(PROJECT_ROOT_PATH)
    candidates = [
        root / "experiment" / "conf" / "data",
        root / "src" / "experiment" / "conf" / "data",
    ]
    for c in candidates:
        if (c / "config.yaml").exists():
            return c
    raise FileNotFoundError(f"Could not find data config root (config.yaml) under {candidates}.")


def test_save_dataset_npz_unit(tmp_path: Path):
    """UNIT: test save_dataset() without touching Hydra/generators."""
    N, w, d, L = 8, 5, 3, 2
    X = np.random.default_rng(0).normal(size=(N, w, d))
    y = np.random.default_rng(1).normal(size=(L, N))
    ds = WindowsDataset(X=X, y=y, label_functionals=[], meta={"hello": "world"})

    cfg = OmegaConf.create(
        {
            "seed": 123,
            "sampling": {"N": N, "w": w, "d": d, "s": 10},
            "process": {"kind": "varma"},
            "functionals": {"kind": "unit"},
            "output": {
                "save_dir": str(tmp_path),  # absolute => bypass PROJECT_ROOT_PATH
                "name": "pytest_ds",
                "format": "npz",
                "overwrite": True,
                "save_meta": True,
                "save_config": True,
            },
        }
    )

    art = save_dataset(ds, cfg)
    assert art.data_path.exists()

    loaded = np.load(art.data_path)
    assert loaded["X"].shape == (N, w, d)
    assert loaded["y"].shape == (L, N)
    assert np.allclose(loaded["X"], X)
    assert np.allclose(loaded["y"], y)

    assert art.meta_path is not None and art.meta_path.exists()
    meta = json.loads(art.meta_path.read_text(encoding="utf-8"))
    assert meta["hello"] == "world"

    assert art.config_path is not None and art.config_path.exists()


def test_make_dataset_hydra_integration():
    GlobalHydra.instance().clear()
    conf_dir = _data_conf_dir()

    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "sampling=tiny",
                "process=varma",
                "functionals=one_step",
                "noise=none",
                "output.name=pytest_integration",
                "output.format=npz",
                "output.overwrite=true",
                "output.save_meta=false",
                "output.save_config=false",
                "output.save_dir=/tmp",
            ],
        )

    ds = make_dataset(cfg)
    assert ds.X.shape == (int(cfg.sampling.N), int(cfg.sampling.w), int(cfg.sampling.d))
    assert ds.y.shape == (1, int(cfg.sampling.N))
    assert ds.meta["process_kind"] == "varma"
    assert ds.meta["functionals_kind"] == "one_step"
