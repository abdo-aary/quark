import numpy as np

import src.models.qrc_matern_krr as mk


class DummyFeaturizer:
    @staticmethod
    def transform(X):
        X = np.asarray(X)
        m = X.mean(axis=(1, 2))
        s = X.std(axis=(1, 2))
        return np.stack([m, s, np.sin(m), np.cos(m)], axis=1).astype(float)


def test_end_to_end_grid_tuning_beats_baseline():
    rng = np.random.default_rng(0)

    # windows X(N,w,d)
    N, w, d = 80, 6, 3
    X = rng.normal(size=(N, w, d))

    # smooth target from hidden feature + noise
    z = X.mean(axis=(1, 2))
    y = np.sin(4.0 * z) + 0.05 * rng.standard_normal(N)

    model = mk.QRCMaternKRRRegressor(
        DummyFeaturizer(),
        standardize=True,
        test_ratio=0.25,
        split_seed=123,
        tuning={
            "strategy": "grid",
            "val_ratio": 0.25,
            "seed": 999,
            "reg": 1e-6,
            "xi_bounds": (1e-2, 1e2),
            "nu_grid": (0.5, 1.5, 2.5),
            "xi_maxiter": 25,  # keep integration test fast
        },
    )

    model.fit(X, y)
    yhat = model.predict()  # predicts on internal test set

    assert yhat.shape == model.y_test_.shape
    assert np.all(np.isfinite(yhat))

    # Baseline: mean predictor
    y_mean = float(np.mean(model.y_test_))
    mse_baseline = float(np.mean((model.y_test_ - y_mean) ** 2))
    mse_model = float(np.mean((model.y_test_ - yhat) ** 2))

    # Require meaningful improvement (not too strict to avoid flakiness)
    assert mse_model < 0.95 * mse_baseline


# def test_from_yaml_smoke(tmp_path):
#     # This is just a smoke test to ensure YAML wiring works.
#     # It doesn't run circuits.
#     yaml_text = """
# qrc:
#   cfg:
#     kind: ring
#     input_dim: 4
#     num_qubits: 2
#     seed: 0
#
#   pubs:
#     family: ising_ring_swap
#     angle_positioning: tanh
#     num_reservoirs: 2
#     lam_0: 0.1
#     seed: 0
#     eps: 1e-8
#
#   runner:
#     kind: exact_aer
#     kwargs: {device: "GPU"}
#
#   features:
#     kind: exact
#     observables:
#       kind: explicit_pauli
#       labels: ["Z", "X"]
#     kwargs: {}
#
# model:
#   preprocess:
#     standardize: true
#   split:
#     test_ratio: 0.2
#     seed: 0
#   tuning:
#     strategy: grid
#     val_ratio: 0.2
#     seed: 0
#     reg: 1e-6
#     xi_bounds: [1e-3, 1e3]
#     nu_grid: [0.5, 1.5]
#     xi_maxiter: 10
# """
#     p = tmp_path / "cfg.yaml"
#     p.write_text(yaml_text, encoding="utf-8")
#
#     mdl = mk.QRCMaternKRRRegressor.from_yaml(str(p))
#     assert isinstance(mdl, mk.QRCMaternKRRRegressor)
#     assert mdl.tuning["strategy"] == "grid"
