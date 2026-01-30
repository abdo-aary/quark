"""
Pytest fixtures for model configuration wiring tests.

Provides:
- in-memory DictConfig builders (no disk),
- optional monkeypatch to replace the real QRCFeaturizer with DummyFeaturizer,
- helpers to locate your Hydra config root for disk-based tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
from omegaconf import DictConfig, OmegaConf

from hydra.core.global_hydra import GlobalHydra
from src.settings import PROJECT_ROOT_PATH

from .dummies import DummyFeaturizer
import tests.tests_models.tests_model_configuring.dummies as _dummies


def make_in_memory_experiment_cfg(
        *,
        angle_positioning: str = "linear",
        pubs_family: str = "ising_ring_swap",
        include_runner_kwargs: bool = True,
        include_fmp_kwargs: bool = True,
        standardize: bool = True,
        test_ratio: float = 0.2,
        split_seed: int = 0,
) -> DictConfig:
    """
    Build a minimal *full* experiment config in-memory for `.from_config()`.

    This avoids reading any YAMLs from disk and is meant to be fast.

    Parameters
    ----------
    angle_positioning : str, default="linear"
        Angle positioning name to pass through wiring.
    pubs_family : str, default="ising_ring_swap"
        PUBS family name to pass through wiring.
    include_runner_kwargs : bool, default=True
        Whether to include `qrc.runner.runner_kwargs`.
    include_fmp_kwargs : bool, default=True
        Whether to include `qrc.features.kwargs`.
    standardize : bool, default=True
        Model preprocess flag.
    test_ratio : float, default=0.2
        Train/test split ratio.
    split_seed : int, default=0
        Split RNG seed.

    Returns
    -------
    omegaconf.DictConfig
        Config with structure `{seed, model: {...}}`.
    """
    dummy_mod = _dummies.__name__  # e.g. "tests_model_configuring.dummies"

    cfg_dict: Dict[str, Any] = {
        "seed": 12345,
        "model": {
            "qrc": {
                "cfg": {
                    "_target_": f"{dummy_mod}.DummyQRConfig",
                    "input_dim": 2,
                    "num_qubits": 3,
                    "seed": 12345,
                },
                "runner": {
                    "_target_": f"{dummy_mod}.DummyRunner",
                },
                "features": {
                    "observables": {
                        "_target_": f"{dummy_mod}.make_dummy_observables",
                        "locality": 2,
                        "num_qubits": 3,
                    },
                    "retriever": {
                        "_target_": f"{dummy_mod}.DummyRetriever",
                    },
                },
                "pubs": {
                    "family": pubs_family,
                    "angle_positioning": angle_positioning,
                    "lam_0": 0.1,
                    "num_reservoirs": 2,
                    "eps": 1e-8,
                },
            },
            "training": {
                "split": {"test_ratio": float(test_ratio), "seed": int(split_seed)},
                "preprocess": {"standardize": bool(standardize)},
            },
            "tuning": {
                "strategy": "grid",
                "val_ratio": 0.2,
                "seed": 0,
                "reg": 1e-6,
                "xi_bounds": [1e-3, 1e3],
                "nu_grid": [0.5, 1.5],
                "xi_maxiter": 5,
            },
        },
    }

    if include_runner_kwargs:
        cfg_dict["model"]["qrc"]["runner"]["runner_kwargs"] = {"device": "CPU", "shots": 128}
    if include_fmp_kwargs:
        cfg_dict["model"]["qrc"]["features"]["kwargs"] = {"batch_size": 16}

    return OmegaConf.create(cfg_dict)


@pytest.fixture()
def patch_qrc_featurizer(monkeypatch: pytest.MonkeyPatch):
    """
    Patch the regressor module to use DummyFeaturizer instead of the real QRCFeaturizer.

    This ensures `.from_config()` stays lightweight (no Qiskit runner constraints)
    and enables asserting exact forwarded parameters.

    Returns
    -------
    None
    """
    import src.models.qrc_matern_krr as mod

    monkeypatch.setattr(mod, "QRCFeaturizer", DummyFeaturizer)
    yield


def find_conf_root() -> Path:
    """
    Locate the Hydra config root directory on disk.

    Returns
    -------
    pathlib.Path
        Path to `.../src/experiment/conf` or `.../experiment/conf`.

    Raises
    ------
    FileNotFoundError
        If no config root is found.
    """
    root = Path(PROJECT_ROOT_PATH)
    candidates = [
        root / "src" / "experiment" / "conf",
        root / "experiment" / "conf",
    ]
    for c in candidates:
        if c.exists():
            # heuristic: must contain `data/` and `model/` directories
            if (c / "data").exists() and (c / "model").exists():
                return c
    raise FileNotFoundError(f"Could not find conf root under {candidates}")


@pytest.fixture()
def hydra_conf_root() -> Path:
    """
    Provide the Hydra config root directory used for disk-based configuration tests.

    Returns
    -------
    pathlib.Path
        The discovered config root.
    """
    return find_conf_root()


@pytest.fixture(autouse=True)
def _reset_hydra_global():
    """
    Ensure Hydra global state is cleared between tests.

    Returns
    -------
    None
    """
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()
