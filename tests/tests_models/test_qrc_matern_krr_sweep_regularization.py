import numpy as np
import pytest

import src.models.qrc_matern_krr as mk


class CountingFeaturizer:
    """Deterministic featurizer with call counter (keeps tests fast)."""

    def __init__(self, D: int = 6):
        self.D = int(D)
        self.calls = 0

    def transform(self, X: np.ndarray) -> np.ndarray:
        self.calls += 1
        X = np.asarray(X)
        m = X.mean(axis=(1, 2))
        s = X.std(axis=(1, 2))
        feats = np.stack([m, s, m ** 2, s ** 2, np.sin(m), np.cos(m)], axis=1).astype(float)
        return feats[:, : self.D]


def _fast_grid_tuner(Phi_tr: np.ndarray, y_tr: np.ndarray, **kwargs):
    reg = float(kwargs.get("reg", 1e-6))
    return {"xi": 1.23, "nu": 1.5, "reg": reg}, 0.0


def _make_dataset(seed=0, N=80, w=5, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, w, d)).astype(float)
    z = X.mean(axis=(1, 2))
    y = np.sin(2.0 * z) + 0.01 * rng.normal(size=N)
    return X, y


def _make_multioutput_dataset(seed=0, N=90, w=4, d=2, L=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, w, d)).astype(float)
    z = X.mean(axis=(1, 2))
    Y = np.stack([np.sin(z), np.cos(2 * z), z ** 2 - 0.1 * z][:L], axis=0)  # (L,N)
    return X, Y


def test_sweep_regularization_singleoutput_basic(monkeypatch):
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)
    monkeypatch.setattr(
        mk,
        "tune_matern_continuous_train_val",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("continuous tuner should not be called")),
        raising=True,
    )

    X, y = _make_dataset(seed=1)
    f = CountingFeaturizer(D=6)
    model = mk.QRCMaternKRRRegressor(
        f,
        standardize=True,
        test_ratio=0.25,
        split_seed=123,
        tuning={"strategy": "grid", "reg": 1e-6, "val_ratio": 0.2, "seed": 0},
    )
    model.fit(X, y, num_workers=1)
    assert f.calls == 1

    alpha0 = np.asarray(model.alpha_).copy()
    mse_test0 = float(np.mean((np.asarray(model.y_pred_test_) - np.asarray(model.y_test_)) ** 2))

    regs = np.array([1e-8, 1e-6, 1e-3], dtype=float)
    out = model.sweep_regularization(regs)

    # no extra featurization
    assert f.calls == 1

    # shapes
    assert out["reg_grid"].shape == (3,)
    assert out["alpha_grid"].ndim == 2
    assert out["alpha_grid"].shape[0] == 3
    assert out["mse_train"].shape == (3,)
    assert out["mse_test"].shape == (3,)

    # fitted attributes must not be overwritten
    assert np.allclose(model.alpha_, alpha0, atol=1e-12, rtol=1e-12)

    # at reg == reg0 (1e-6), alpha should match fitted alpha (up to numerical tol)
    idx = int(np.where(np.isclose(out["reg_grid"], 1e-6))[0][0])
    assert np.allclose(out["alpha_grid"][idx], alpha0, atol=1e-10, rtol=1e-10)

    # and test MSE should match the model's stored test prediction MSE
    assert out["mse_test"][idx] == pytest.approx(mse_test0, rel=1e-10, abs=1e-10)


def test_sweep_regularization_works_without_X_train_features(monkeypatch):
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    X, y = _make_dataset(seed=2, N=60)
    f = CountingFeaturizer(D=6)
    model = mk.QRCMaternKRRRegressor(
        f,
        standardize=True,
        test_ratio=0.2,
        split_seed=0,
        tuning={"strategy": "grid", "reg": 1e-6, "val_ratio": 0.2, "seed": 0},
    )
    model.fit(X, y, num_workers=1)
    assert f.calls == 1

    # simulate a loaded artifact: X_train_features_ may be absent
    model.X_train_features_ = None

    out = model.sweep_regularization([1e-6, 1e-4])
    assert out["mse_test"].shape == (2,)
    assert f.calls == 1


def test_sweep_regularization_multioutput(monkeypatch):
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    X, Y = _make_multioutput_dataset(seed=3, L=3)
    f = CountingFeaturizer(D=6)
    model = mk.QRCMaternKRRRegressor(
        f,
        standardize=True,
        test_ratio=0.2,
        split_seed=7,
        tuning={"strategy": "grid", "reg": 1e-6, "val_ratio": 0.2, "seed": 0},
    )
    model.fit(X, Y, num_workers=1)
    assert f.calls == 1

    alpha0 = np.asarray(model.alpha_).copy()
    regs = np.array([1e-6, 1e-3], dtype=float)

    out = model.sweep_regularization(regs)

    assert f.calls == 1
    assert out["alpha_grid"].shape[0] == 3  # L
    assert out["alpha_grid"].shape[1] == 2  # R
    assert out["alpha_grid"].shape[2] == alpha0.shape[1]  # n_train
    assert out["mse_train"].shape == (3, 2)
    assert out["mse_test"].shape == (3, 2)

    # alpha at reg0 must match fitted alpha for each output
    idx = int(np.where(np.isclose(out["reg_grid"], 1e-6))[0][0])
    for l in range(3):
        assert np.allclose(out["alpha_grid"][l, idx], alpha0[l], atol=1e-10, rtol=1e-10)

    # fitted attributes unchanged
    assert np.allclose(model.alpha_, alpha0, atol=1e-12, rtol=1e-12)


def test_sweep_regularization_raises_before_fit():
    model = mk.QRCMaternKRRRegressor(CountingFeaturizer(D=4), tuning={"strategy": "grid"})
    with pytest.raises((RuntimeError, ValueError)):
        model.sweep_regularization([1e-6, 1e-3])


@pytest.mark.parametrize("bad_regs", [[0.0, 1e-6], [-1e-6, 1e-6], [np.nan, 1e-6], []])
def test_sweep_regularization_rejects_bad_regs(monkeypatch, bad_regs):
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    X, y = _make_dataset(seed=4, N=40)
    model = mk.QRCMaternKRRRegressor(
        CountingFeaturizer(D=6),
        standardize=False,
        test_ratio=0.25,
        split_seed=0,
        tuning={"strategy": "grid", "reg": 1e-6, "val_ratio": 0.2, "seed": 0},
    )
    model.fit(X, y, num_workers=1)

    with pytest.raises(ValueError):
        model.sweep_regularization(bad_regs)


def test_sweep_regularization_persistence_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    X, y = _make_dataset(seed=5, N=70)
    f = CountingFeaturizer(D=6)
    model = mk.QRCMaternKRRRegressor(
        f,
        standardize=True,
        test_ratio=0.2,
        split_seed=0,
        tuning={"strategy": "grid", "reg": 1e-6, "val_ratio": 0.2, "seed": 0},
    )
    model.fit(X, y, num_workers=1)

    # Save & load; loaded object typically lacks y_test_, so we reattach it for mse_test checks.
    save_dir = tmp_path / "qrc_krr"
    model.save(save_dir)

    loaded = mk.QRCMaternKRRRegressor.load(save_dir, featurizer=None)
    loaded.y_test_ = np.asarray(model.y_test_).copy()

    out = loaded.sweep_regularization([1e-6, 1e-3])

    # alpha at reg0 should match loaded.alpha_
    idx = int(np.where(np.isclose(out["reg_grid"], 1e-6))[0][0])
    assert np.allclose(out["alpha_grid"][idx], np.asarray(loaded.alpha_), atol=1e-10, rtol=1e-10)


def test_fit_solution_is_exact_dual_krr_system(monkeypatch):
    """
    Guardrail test (single + multi-output): ensures the fitted solution is the exact
    dual KRR solution for the Gram matrix Ktt and the ridge reg0 used during fit:

        (Ktt + reg0 I) alpha0 == y_train

    This is exactly the condition needed for the sweep reconstruction step:
        y_train = (Ktt + reg0 I) @ alpha0

    For multi-output, the check is done independently per output l using kernel_[l].
    """
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)
    monkeypatch.setattr(
        mk,
        "tune_matern_continuous_train_val",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("continuous tuner should not be called")),
        raising=True,
    )

    # ---------- helper to build standardized Phi_tr ----------
    def _get_standardized_Phi_tr(model):
        Phi_full = np.asarray(model.Phi_full_)
        tr_idx = np.asarray(model.train_idx_, dtype=int)
        Phi_tr = Phi_full[tr_idx]
        if model.scaler_ is not None:
            Phi_tr = model.scaler_.transform(Phi_tr)
        return Phi_tr, tr_idx

    # =========================================================
    # 1) Single-output
    # =========================================================
    X, y = _make_dataset(seed=9, N=80)
    f = CountingFeaturizer(D=6)

    model = mk.QRCMaternKRRRegressor(
        f,
        standardize=True,
        test_ratio=0.25,
        split_seed=123,
        tuning={"strategy": "grid", "reg": 1e-6, "val_ratio": 0.2, "seed": 0},
    )
    model.fit(X, y, num_workers=1)

    Phi_tr, tr_idx = _get_standardized_Phi_tr(model)
    Ktt = model.kernel_(Phi_tr, Phi_tr)

    reg0 = float(model.best_params_.get("reg", model.tuning.get("reg", 1e-6)))
    y_train_true = np.asarray(y, dtype=float)[tr_idx].reshape(-1)
    alpha0 = np.asarray(model.alpha_, dtype=float).reshape(-1)

    assert alpha0.shape == (Ktt.shape[0],)

    A = Ktt + reg0 * np.eye(Ktt.shape[0], dtype=Ktt.dtype)
    lhs = A @ alpha0
    assert np.allclose(lhs, y_train_true, atol=1e-10, rtol=1e-10)

    alpha_ref = np.linalg.solve(A, y_train_true)
    assert np.allclose(alpha0, alpha_ref, atol=1e-10, rtol=1e-10)

    # =========================================================
    # 2) Multi-output
    # =========================================================
    X2, Y2 = _make_multioutput_dataset(seed=10, N=90, L=3)  # Y2 shape (L, N)
    f2 = CountingFeaturizer(D=6)

    model2 = mk.QRCMaternKRRRegressor(
        f2,
        standardize=True,
        test_ratio=0.2,
        split_seed=7,
        tuning={"strategy": "grid", "reg": 1e-6, "val_ratio": 0.2, "seed": 0},
    )
    model2.fit(X2, Y2, num_workers=1)

    Phi_tr2, tr_idx2 = _get_standardized_Phi_tr(model2)
    alpha0_2 = np.asarray(model2.alpha_, dtype=float)
    assert alpha0_2.ndim == 2, f"Expected alpha_ shape (L, n_train). Got {alpha0_2.shape}."

    kernels = model2.kernel_
    assert isinstance(kernels, list), "Expected kernel_ to be a list in multi-output mode."

    L = alpha0_2.shape[0]

    # best_params_ might be a list[dict] or a dict (broadcasted); normalize.
    bp = model2.best_params_
    if isinstance(bp, dict):
        bp_list = [bp for _ in range(L)]
    else:
        bp_list = bp
    assert isinstance(bp_list, list) and len(bp_list) == L

    Y2 = np.asarray(Y2, dtype=float)
    assert Y2.shape[0] == L

    for l in range(L):
        Ktt_l = kernels[l](Phi_tr2, Phi_tr2)
        reg0_l = float(bp_list[l].get("reg", model2.tuning.get("reg", 1e-6)))

        y_train_true_l = Y2[l, tr_idx2].reshape(-1)
        alpha0_l = alpha0_2[l].reshape(-1)

        assert alpha0_l.shape == (Ktt_l.shape[0],)

        A_l = Ktt_l + reg0_l * np.eye(Ktt_l.shape[0], dtype=Ktt_l.dtype)
        lhs_l = A_l @ alpha0_l
        assert np.allclose(lhs_l, y_train_true_l, atol=1e-10, rtol=1e-10)

        alpha_ref_l = np.linalg.solve(A_l, y_train_true_l)
        assert np.allclose(alpha0_l, alpha_ref_l, atol=1e-10, rtol=1e-10)
