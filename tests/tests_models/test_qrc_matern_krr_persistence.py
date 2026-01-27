import json

import numpy as np
import pytest

import src.models.qrc_matern_krr as mk


class CountingFeaturizer:
    """Deterministic featurizer with a call counter.

    This keeps persistence tests fast and avoids any Qiskit/Aer dependency.
    """

    def __init__(self, D: int = 6):
        self.D = int(D)
        self.calls = 0

    def transform(self, X: np.ndarray) -> np.ndarray:
        self.calls += 1
        X = np.asarray(X)
        m = X.mean(axis=(1, 2))
        s = X.std(axis=(1, 2))
        feats = np.stack(
            [m, s, m**2, s**2, np.sin(m), np.cos(m), np.tanh(s)],
            axis=1,
        ).astype(float)
        return feats[:, : self.D]


def _fast_grid_tuner(Phi: np.ndarray, y: np.ndarray, **kwargs):
    """Fast deterministic tuner for unit tests."""
    reg = float(kwargs.get("reg", 1e-6))
    xi = 0.8 + abs(float(np.mean(y)))
    nu = 1.5
    return {"xi": xi, "nu": nu, "reg": reg}, 0.0


def _rehydrate_train_features(model: mk.QRCMaternKRRRegressor) -> np.ndarray:
    """Rebuild X_train_features_ from Phi_full_ + train_idx_ (+ scaler_)."""
    Phi_full = np.asarray(model.Phi_full_)
    tr_idx = np.asarray(model.train_idx_)
    Phi_tr = Phi_full[tr_idx]
    if getattr(model, "scaler_", None) is not None:
        Phi_tr = model.scaler_.transform(Phi_tr)
    model.X_train_features_ = Phi_tr
    return Phi_tr


def _raw_test_features(model: mk.QRCMaternKRRRegressor) -> np.ndarray:
    Phi_full = np.asarray(model.Phi_full_)
    te_idx = np.asarray(model.test_idx_)
    return Phi_full[te_idx]


def _make_singleoutput_dataset(seed: int = 0, *, N: int = 48, w: int = 5, d: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, w, d)).astype(float)
    z = X.mean(axis=(1, 2))
    y = np.sin(2.0 * z) + 0.01 * rng.standard_normal(N)
    return X, y


def _make_multioutput_dataset(seed: int = 0, *, N: int = 60, w: int = 4, d: int = 2, L: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, w, d)).astype(float)
    z = X.mean(axis=(1, 2))
    outs = [
        np.sin(1.5 * z),
        np.cos(2.0 * z),
        z**2 - 0.1 * z,
        np.tanh(z),
    ]
    Y = np.stack(outs[:L], axis=0)
    return X, Y


def test_persistence_roundtrip_singleoutput_standardize_true(monkeypatch, tmp_path):
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)
    monkeypatch.setattr(
        mk,
        "tune_matern_continuous_train_val",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("continuous tuner should not be called")),
        raising=True,
    )

    X, y = _make_singleoutput_dataset(seed=1, N=64, w=6, d=3)
    featurizer = CountingFeaturizer(D=6)

    model = mk.QRCMaternKRRRegressor(
        featurizer,
        standardize=True,
        test_ratio=0.25,
        split_seed=123,
        tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6},
    )
    model.fit(X, y, num_workers=1)

    # Sanity: cached test predictions exist and match recomputation from features
    yhat_ref = model.predict()
    yhat_from_phi = model.predict_from_features(_raw_test_features(model), apply_scaler=True)
    assert np.allclose(yhat_ref, yhat_from_phi, atol=1e-12)

    # Save artifact
    artifact_dir = tmp_path / "qrc_krr_full_std_true"
    model.save(artifact_dir)

    assert (artifact_dir / "arrays.npz").exists()
    assert (artifact_dir / "meta.json").exists()

    # Inspect stored keys
    npz = np.load(artifact_dir / "arrays.npz", allow_pickle=False)
    assert set(npz.files) >= {"Phi_full", "train_idx", "test_idx", "alpha", "scaler_mean", "scaler_scale"}

    meta = json.loads((artifact_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["format_version"] == mk._SAVE_FORMAT_VERSION
    assert meta["artifact"] == "QRCMaternKRRRegressor.full"
    assert meta["standardize"] is True
    assert meta["Phi_shape"][0] == int(model.Phi_full_.shape[0])

    # Load (no featurizer needed for predict_from_features)
    loaded = mk.QRCMaternKRRRegressor.load(artifact_dir, featurizer=None)

    assert loaded.standardize is True
    assert loaded.scaler_ is not None
    assert loaded.best_params_ == model.best_params_
    assert np.array_equal(loaded.train_idx_, model.train_idx_)
    assert np.array_equal(loaded.test_idx_, model.test_idx_)
    assert np.allclose(loaded.Phi_full_, model.Phi_full_, atol=0.0, rtol=0.0)
    assert np.allclose(loaded.alpha_, model.alpha_, atol=0.0, rtol=0.0)

    # After rehydrating the train features, we can predict without re-featurizing.
    _rehydrate_train_features(loaded)
    yhat_loaded = loaded.predict_from_features(_raw_test_features(loaded), apply_scaler=True)
    assert np.allclose(yhat_loaded, yhat_ref, atol=1e-12)

    # Predicting with no stored X_test_ after load should raise
    with pytest.raises(ValueError):
        loaded.predict()


def test_persistence_roundtrip_singleoutput_standardize_false(monkeypatch, tmp_path):
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    X, y = _make_singleoutput_dataset(seed=2, N=50, w=5, d=2)
    model = mk.QRCMaternKRRRegressor(
        CountingFeaturizer(D=5),
        standardize=False,
        test_ratio=0.2,
        split_seed=7,
        tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6},
    )
    model.fit(X, y, num_workers=1)
    yhat_ref = model.predict()

    artifact_dir = tmp_path / "qrc_krr_full_std_false"
    model.save(artifact_dir)

    npz = np.load(artifact_dir / "arrays.npz", allow_pickle=False)
    assert set(npz.files) >= {"Phi_full", "train_idx", "test_idx", "alpha"}
    assert "scaler_mean" not in npz.files
    assert "scaler_scale" not in npz.files

    loaded = mk.QRCMaternKRRRegressor.load(artifact_dir, featurizer=None)
    assert loaded.standardize is False
    assert loaded.scaler_ is None

    _rehydrate_train_features(loaded)
    yhat_loaded = loaded.predict_from_features(_raw_test_features(loaded), apply_scaler=True)
    assert np.allclose(yhat_loaded, yhat_ref, atol=1e-12)


def test_persistence_roundtrip_multioutput(monkeypatch, tmp_path):
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    X, Y = _make_multioutput_dataset(seed=3, N=72, w=4, d=3, L=3)
    model = mk.QRCMaternKRRRegressor(
        CountingFeaturizer(D=6),
        standardize=True,
        test_ratio=0.25,
        split_seed=99,
        tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6},
    )
    model.fit(X, Y, num_workers=1)
    yhat_ref = model.predict()
    assert yhat_ref.shape == model.y_test_.shape and yhat_ref.ndim == 2

    artifact_dir = tmp_path / "qrc_krr_full_multi"
    model.save(artifact_dir)

    loaded = mk.QRCMaternKRRRegressor.load(artifact_dir, featurizer=None)
    assert isinstance(loaded.kernel_, list) and len(loaded.kernel_) == 3
    assert loaded.alpha_ is not None and loaded.alpha_.shape[0] == 3

    _rehydrate_train_features(loaded)
    yhat_loaded = loaded.predict_from_features(_raw_test_features(loaded), apply_scaler=True)

    assert yhat_loaded.shape == yhat_ref.shape
    assert np.allclose(yhat_loaded, yhat_ref, atol=1e-12)


def test_loaded_model_can_predict_new_X_if_featurizer_provided(monkeypatch, tmp_path):
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    X, y = _make_singleoutput_dataset(seed=4, N=60, w=5, d=2)
    model = mk.QRCMaternKRRRegressor(
        CountingFeaturizer(D=6),
        standardize=True,
        test_ratio=0.25,
        split_seed=0,
        tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6},
    )
    model.fit(X, y, num_workers=1)

    artifact_dir = tmp_path / "qrc_krr_full_predict_newX"
    model.save(artifact_dir)

    new_featurizer = CountingFeaturizer(D=6)
    loaded = mk.QRCMaternKRRRegressor.load(artifact_dir, featurizer=new_featurizer)

    # Rehydrate training features so prediction can run.
    _rehydrate_train_features(loaded)

    rng = np.random.default_rng(0)
    X_new = rng.normal(size=(10, 5, 2)).astype(float)
    yhat = loaded.predict(X_new)

    assert new_featurizer.calls == 1
    assert yhat.shape == (10,)
    assert np.all(np.isfinite(yhat))


def test_save_raises_before_fit(tmp_path):
    model = mk.QRCMaternKRRRegressor(CountingFeaturizer(D=4), tuning={"strategy": "grid"})
    with pytest.raises(ValueError):
        model.save(tmp_path / "should_fail")  # missing best_params_ / alpha_


def test_save_raises_when_standardize_true_but_scaler_none(tmp_path):
    # Create a "partially" fitted object to trigger the scaler guard.
    model = mk.QRCMaternKRRRegressor(CountingFeaturizer(D=4), standardize=True, tuning={"strategy": "grid"})
    model.best_params_ = {"xi": 1.0, "nu": 1.5, "reg": 1e-6}
    model.alpha_ = np.ones(5, dtype=float)
    model.Phi_full_ = np.zeros((6, 4), dtype=float)
    model.train_idx_ = np.array([0, 1, 2, 3, 4])
    model.test_idx_ = np.array([5])
    model.scaler_ = None  # inconsistent with standardize=True

    with pytest.raises(ValueError):
        model.save(tmp_path / "bad_scaler")


def test_load_raises_on_missing_files(tmp_path):
    d = tmp_path / "broken_artifact"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        mk.QRCMaternKRRRegressor.load(d)


def test_load_raises_on_version_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    X, y = _make_singleoutput_dataset(seed=5, N=40, w=4, d=2)
    model = mk.QRCMaternKRRRegressor(
        CountingFeaturizer(D=5),
        standardize=False,
        test_ratio=0.2,
        split_seed=0,
        tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6},
    )
    model.fit(X, y, num_workers=1)

    artifact_dir = tmp_path / "qrc_krr_full_version_test"
    model.save(artifact_dir)

    meta_path = artifact_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["format_version"] = int(mk._SAVE_FORMAT_VERSION) + 999
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with pytest.raises(ValueError):
        mk.QRCMaternKRRRegressor.load(artifact_dir)
