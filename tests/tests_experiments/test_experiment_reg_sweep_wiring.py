from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

import src.models.qrc_matern_krr as mk
from src.data.factory import save_dataset
from src.data.generate.base import WindowsDataset
from src.experiment.experiment import Experiment


class CountingFeaturizer:
    """Deterministic featurizer with a call counter (fast tests; no Qiskit)."""

    def __init__(self, D: int = 6):
        self.D = int(D)
        self.calls = 0

    def transform(self, X: np.ndarray) -> np.ndarray:
        self.calls += 1
        X = np.asarray(X)
        m = X.mean(axis=(1, 2))
        s = X.std(axis=(1, 2))
        feats = np.stack([m, s, m**2, s**2, np.sin(m), np.cos(m), np.tanh(s)], axis=1).astype(float)
        return feats[:, : self.D]


def _fast_grid_tuner(Phi: np.ndarray, y: np.ndarray, **kwargs):
    """Fast deterministic tuner stub to avoid slow xi/nu search in tests."""
    reg = float(kwargs.get("reg", 1e-6))
    xi = 0.8 + abs(float(np.mean(y)))
    nu = 1.5
    return {"xi": xi, "nu": nu, "reg": reg}, 0.0


def _cfg_for_saving_dataset(tmp_path: Path):
    """Minimal config structure required by save_dataset."""
    return OmegaConf.create(
        {
            "seed": 0,
            "sampling": {"N": 40, "w": 5, "d": 3, "s": 10},
            "process": {"kind": "unit"},
            "functionals": {"kind": "unit", "items": []},
            "output": {
                "save_dir": str(tmp_path),
                "name": "exp_reg_sweep",
                "format": "npz",
                "overwrite": True,
                "save_meta": True,
                "save_config": False,
            },
        }
    )


def _mse_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return per-output MSE (shape (L,) or scalar-as-0d array)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        return np.asarray(np.mean((y_true - y_pred) ** 2, dtype=float))

    # (L, N)
    return np.mean((y_true - y_pred) ** 2, axis=1, dtype=float)


def test_experiment_run_reg_sweep_delegates_and_stores_unit():
    """Pure unit test: ensure Experiment forwards reg_grid and stores output."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3, 2)).astype(float)
    y = rng.normal(size=(8,)).astype(float)
    ds = WindowsDataset(X=X, y=y, label_functionals=[], meta={"functionals_kind": "unit"})

    class DummyModel:
        def __init__(self):
            self.called_with = None

        def sweep_regularization(self, reg_grid, **kwargs):
            self.called_with = (list(reg_grid), dict(kwargs))
            return {
                "reg_grid": np.asarray(list(reg_grid), dtype=float),
                "mse_train": np.zeros(len(list(reg_grid))),
                "mse_test": np.ones(len(list(reg_grid))),
            }

    m = DummyModel()
    exp = Experiment(dataset=ds, model=m)  # type: ignore[arg-type]

    grid = [1e-6, 1e-4, 1e-2]
    out = exp.run_reg_sweep(grid)

    assert m.called_with is not None
    assert m.called_with[0] == grid
    assert isinstance(out, dict)
    assert exp.reg_sweep_ is out


def test_experiment_run_reg_sweep_from_cfg_respects_enabled_unit():
    """Unit test: disabled config should not call the model."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 3, 2)).astype(float)
    y = rng.normal(size=(8,)).astype(float)
    ds = WindowsDataset(X=X, y=y, label_functionals=[], meta={"functionals_kind": "unit"})

    class DummyModel:
        def sweep_regularization(self, *args, **kwargs):
            raise AssertionError("Should not be called when reg_sweep.enabled=false")

    exp = Experiment(dataset=ds, model=DummyModel())  # type: ignore[arg-type]

    cfg = OmegaConf.create({"reg_sweep": {"enabled": False, "reg_grid": [1e-6, 1e-4]}})
    assert exp.run_reg_sweep_from_cfg(cfg) is None


def test_experiment_reg_sweep_integration_matches_fit_mse_and_no_extra_featurizer_calls(tmp_path: Path, monkeypatch):
    """Integration-style test: fit then sweep through Experiment.

    Validates:
    - Experiment wiring calls model.sweep_regularization exactly once.
    - sweep includes the tuning regularization and reproduces the test MSE from .fit().
    - no additional featurizer.transform calls occur during the sweep.
    """
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    # Multi-output target (L=2) to cover the vector MSE case.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 5, 3)).astype(float)
    z = X.mean(axis=(1, 2))
    y = np.stack([np.sin(z), np.cos(1.5 * z)], axis=0).astype(float)  # (L, N)

    ds = WindowsDataset(X=X, y=y, label_functionals=[], meta={"functionals_kind": "unit"})
    art = save_dataset(ds, _cfg_for_saving_dataset(tmp_path))

    featurizer = CountingFeaturizer(D=6)
    model = mk.QRCMaternKRRRegressor(
        featurizer,
        standardize=True,
        test_ratio=0.25,
        split_seed=123,
        tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6, "nu_grid": (1.5,), "xi_maxiter": 3},
    )

    exp = Experiment.from_paths(dataset_path=art.root, model=model, instantiate_functionals=False)
    exp.fit(num_workers=1)

    assert featurizer.calls == 1  # featurizer used during fit

    # Wrap sweep_regularization to verify wiring calls it exactly once.
    call_counter = {"n": 0}
    original = exp.model.sweep_regularization

    def wrapped(reg_grid, **kwargs):
        call_counter["n"] += 1
        return original(reg_grid, **kwargs)

    monkeypatch.setattr(exp.model, "sweep_regularization", wrapped, raising=True)

    reg_grid = [1e-9, 1e-6, 1e-3]
    out = exp.run_reg_sweep(reg_grid)

    assert call_counter["n"] == 1
    assert exp.reg_sweep_ is out
    assert featurizer.calls == 1  # sweep should NOT recompute features

    # Must at least expose test MSE per reg.
    assert "mse_test" in out

    mse_test = np.asarray(out["mse_test"])
    G = len(reg_grid)
    idx = reg_grid.index(1e-6)

    expected = _mse_per_output(exp.model.y_test_, exp.model.y_pred_test_)
    expected = np.asarray(expected)

    if mse_test.ndim == 1:
        # single-output: (R,)
        assert mse_test.shape == (G,)
        got = np.asarray(mse_test[idx])
        assert expected.ndim == 0
        assert np.allclose(got, expected, atol=1e-10, rtol=0.0)

    elif mse_test.ndim == 2:
        # multi-output: accept either (L, R) [your current convention] or (R, L)
        if mse_test.shape[1] == G:
            # (L, R)
            got = mse_test[:, idx]
            assert expected.ndim == 1
            assert got.shape == expected.shape
            assert np.allclose(got, expected, atol=1e-10, rtol=0.0)
        elif mse_test.shape[0] == G:
            # (R, L)
            got = mse_test[idx, :]
            assert expected.ndim == 1
            assert got.shape == expected.shape
            assert np.allclose(got, expected, atol=1e-10, rtol=0.0)
        else:
            raise AssertionError(f"Unexpected mse_test shape {mse_test.shape}; expected (L,{G}) or ({G},L).")

    else:
        raise AssertionError(f"Unexpected mse_test ndim={mse_test.ndim} with shape {mse_test.shape}.")


def test_experiment_run_reg_sweep_from_cfg_happy_path(tmp_path: Path, monkeypatch):
    """Small integration: config-driven sweep triggers run_reg_sweep."""
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 4, 2)).astype(float)
    y = np.sin(X.mean(axis=(1, 2))).astype(float)
    ds = WindowsDataset(X=X, y=y, label_functionals=[], meta={"functionals_kind": "unit"})
    art = save_dataset(ds, _cfg_for_saving_dataset(tmp_path))

    featurizer = CountingFeaturizer(D=5)
    model = mk.QRCMaternKRRRegressor(
        featurizer,
        standardize=True,
        test_ratio=0.2,
        split_seed=0,
        tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6, "nu_grid": (1.5,), "xi_maxiter": 3},
    )

    exp = Experiment.from_paths(dataset_path=art.root, model=model, instantiate_functionals=False)
    exp.fit(num_workers=1)

    cfg = OmegaConf.create({"reg_sweep": {"enabled": True, "reg_grid": [1e-6, 1e-4]}})
    out = exp.run_reg_sweep_from_cfg(cfg)

    assert isinstance(out, dict)
    assert exp.reg_sweep_ is out
    assert np.asarray(out["mse_test"]).shape[0] == 2
