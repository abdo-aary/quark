"""
Smoke tests for QRCMaternKRRRegressor.from_config().

These tests replicate the old from_yaml smoke tests, but now build a DictConfig
in-memory and call `.from_config(cfg)`.

They validate:
- End-to-end fit/predict doesn't crash (with Aer execution patched out).
- `runner_kwargs` are forwarded to ExactAerCircuitsRunner.run_pubs.
- `features.kwargs` are forwarded to CSFeatureMapsRetriever.get_feature_maps.
- (Optional) If Aer GPU is available, device='GPU' is effectively configured.

Notes
-----
- We patch the Aer runner to return deterministic density matrices quickly.
- The config uses Hydra-style `_target_` so `.from_config()` can instantiate.
"""
from __future__ import annotations
import os
import inspect
import numpy as np
import pytest
from omegaconf import OmegaConf

from src.models.qrc_matern_krr import QRCMaternKRRRegressor
from src.qrc.run.circuit_run import ExactAerCircuitsRunner, ExactResults


def _observables_cfg(num_qubits: int, locality: int = 1) -> dict:
    """
    Build an observables instantiation config robust to function signature changes.
    """
    fn = __import__("src.qrc.circuits.utils", fromlist=["generate_k_local_paulis"]).generate_k_local_paulis
    sig = inspect.signature(fn)
    cfg = {
        "_target_": "src.qrc.circuits.utils.generate_k_local_paulis",
        "locality": int(locality),
        "num_qubits": int(num_qubits),
    }
    if "basis" in sig.parameters:
        cfg["basis"] = ["X", "Y", "Z"]
    return cfg


def _cfg_dict(
        *,
        num_qubits: int = 2,
        input_dim: int = 4,
        runner_kwargs: dict,
        fmp_kind: str = "exact",
        fmp_kwargs: dict | None = None,
        angle_positioning: str = "tanh",
        tuning_strategy: str = "grid",
        standardize: bool = True,
) -> dict:
    """
    Build an in-memory Hydra config dict for `.from_config()`.

    Parameters
    ----------
    num_qubits : int, default=2
        Number of reservoir qubits.
    input_dim : int, default=4
        Input dimension `d` (must match X.shape[-1]).
    runner_kwargs : dict
        Arguments forwarded to `ExactAerCircuitsRunner.run_pubs(**runner_kwargs)`.
    fmp_kind : {"exact","cs"}, default="exact"
        Feature map retriever kind.
    fmp_kwargs : dict or None, default=None
        Arguments forwarded to `retriever.get_feature_maps(..., **fmp_kwargs)`.
    angle_positioning : str, default="tanh"
        Injection nonlinearity name.
    tuning_strategy : {"grid","powell"}, default="grid"
        Matérn hyperparameter tuning strategy.
    standardize : bool, default=True
        Standardize X/y before fitting.

    Returns
    -------
    dict
        A config dict suitable for `OmegaConf.create(...)`.
    """
    if fmp_kwargs is None:
        fmp_kwargs = {}

    # Observables: keep small (1-local) to make feature maps cheap.
    # Uses existing helper that returns list[SparsePauliOp].
    observables_cfg = _observables_cfg(num_qubits=num_qubits, locality=1)

    if fmp_kind == "exact":
        retriever_cfg = {"_target_": "src.qrc.run.fmp_retriever.ExactFeatureMapsRetriever"}
    elif fmp_kind == "cs":
        retriever_cfg = {"_target_": "src.qrc.run.cs_fmp_retriever.CSFeatureMapsRetriever"}
    else:
        raise ValueError(f"Unknown fmp_kind={fmp_kind!r}")

    cfg = {
        "seed": 0,
        "model": {
            "qrc": {
                "cfg": {
                    "_target_": "src.qrc.circuits.qrc_configs.RingQRConfig",
                    "input_dim": int(input_dim),
                    "num_qubits": int(num_qubits),
                    "seed": 0,
                },
                "pubs": {
                    "family": "ising_ring_swap",
                    "angle_positioning": str(angle_positioning),
                    "num_reservoirs": 2,
                    "lam_0": 0.1,
                    "eps": 1e-8,
                },
                "runner": {
                    "_target_": "src.qrc.run.circuit_run.ExactAerCircuitsRunner",
                    "runner_kwargs": dict(runner_kwargs),
                },
                "features": {
                    "observables": observables_cfg,
                    "retriever": retriever_cfg,
                    "kwargs": dict(fmp_kwargs),
                },
            },
            "training": {
                "split": {"test_ratio": 0.25, "seed": 0},
                "preprocess": {"standardize": bool(standardize)},
            },
            "tuning": {},  # filled below
        },
    }

    # Match your regressor's tuning schema under `model.tuning`.
    if tuning_strategy == "grid":
        cfg["model"]["tuning"] = {
            "strategy": "grid",
            "val_ratio": 0.25,
            "seed": 0,
            "reg": 1e-6,
            "xi_bounds": [1e-3, 1e3],
            "nu_grid": [1.5],
            "xi_maxiter": 8,
        }
    elif tuning_strategy == "powell":
        cfg["model"]["tuning"] = {
            "strategy": "powell",
            "val_ratio": 0.25,
            "seed": 0,
            "reg": 1e-6,
            "xi_bounds": [1e-3, 1e3],
            "nu_bounds": [0.2, 5.0],
            "n_restarts": 1,
        }
    else:
        raise ValueError(f"Unknown tuning_strategy={tuning_strategy!r}")

    return cfg


def _patch_runner_to_fake_states(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch ExactAerCircuitsRunner.run_pubs so it:
    - records kwargs
    - returns valid ExactResults(states=..., qrc_cfg=...)
    - avoids Aer simulation entirely (fast + GPU-agnostic)

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.

    Returns
    -------
    None
    """

    def fake_run_pubs(self, pubs, **kwargs):
        self._last_run_kwargs = dict(kwargs)

        if not pubs:
            raise ValueError("pubs must be non-empty")

        vals0 = np.asarray(pubs[0][1], dtype=float)

        # Support both runner attribute names
        cfg_obj = getattr(self, "qrc_cfg", getattr(self, "cfg", None))
        if cfg_obj is None:
            raise AttributeError("Runner has neither 'qrc_cfg' nor 'cfg' attribute")

        n = int(cfg_obj.num_qubits)
        dim = 1 << n

        # Detect template PUBs: one pub, vals shape (N, R, P)
        template_mode = (len(pubs) == 1) and (vals0.ndim == 3)
        if template_mode:
            N = int(vals0.shape[0])
            R = int(vals0.shape[1])
        else:
            # Legacy PUBs: pubs length N, each vals shape (R, P)
            N = len(pubs)
            R = int(vals0.shape[0])

        def sigmoid(x: float) -> float:
            x = float(x)
            if x >= 0:
                z = np.exp(-x)
                return 1.0 / (1.0 + z)
            z = np.exp(x)
            return z / (1.0 + z)

        states = np.zeros((N, R, dim, dim), dtype=complex)

        if template_mode:
            # vals0: (N, R, P_total)
            for i in range(N):
                for r in range(R):
                    s = float(np.mean(vals0[i, r]))
                    p = sigmoid(s)
                    p = float(np.clip(p, 1e-6, 1 - 1e-6))
                    diag = np.full(dim, (1.0 - p) / (dim - 1), dtype=float)
                    diag[0] = p
                    states[i, r] = np.diag(diag).astype(complex)
        else:
            # legacy: iterate pubs
            for i, (_qc, vals) in enumerate(pubs):
                vals = np.asarray(vals, dtype=float)  # (R, P_res)
                for r in range(R):
                    s = float(np.mean(vals[r]))
                    p = sigmoid(s)
                    p = float(np.clip(p, 1e-6, 1 - 1e-6))
                    diag = np.full(dim, (1.0 - p) / (dim - 1), dtype=float)
                    diag[0] = p
                    states[i, r] = np.diag(diag).astype(complex)

        # Robust constructor for ExactResults: supports field name qrc_cfg or cfg
        fields = getattr(ExactResults, "__dataclass_fields__", {})
        cfg_field = "qrc_cfg" if "qrc_cfg" in fields else "cfg"
        return ExactResults(states=states, **{cfg_field: cfg_obj})

    # Patch the *real* class used by Hydra instantiation.
    monkeypatch.setattr(ExactAerCircuitsRunner, "run_pubs", fake_run_pubs, raising=True)


@pytest.mark.parametrize(
    "runner_kwargs",
    [
        {"device": "CPU"},
        {"device": "GPU"},  # forwarding only (no real GPU required here)
        {
            "device": "GPU",
            "max_parallel_threads": 2,
            "max_parallel_experiments": 2,
            "max_parallel_shots": 2,
            "seed_simulator": 123,
            "optimization_level": 0,
        },
    ],
)
@pytest.mark.parametrize("tuning_strategy", ["grid", "powell"])
def test_from_config_end_to_end_smoke_runner_kwargs_forwarded(
        monkeypatch: pytest.MonkeyPatch, runner_kwargs: dict, tuning_strategy: str
) -> None:
    """
    End-to-end smoke test for `.from_config()` with patched Aer execution.
    Verifies runner kwargs are forwarded to `runner.run_pubs`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    runner_kwargs : dict
        Runner kwargs variant to test.
    tuning_strategy : str
        Tuning strategy ("grid" or "powell").

    Returns
    -------
    None
    """
    _patch_runner_to_fake_states(monkeypatch)

    cfg = OmegaConf.create(
        _cfg_dict(
            num_qubits=2,
            input_dim=4,
            runner_kwargs=runner_kwargs,
            fmp_kind="exact",
            fmp_kwargs={},
            tuning_strategy=tuning_strategy,
            standardize=True,
        )
    )

    mdl = QRCMaternKRRRegressor.from_config(cfg)

    rng = np.random.default_rng(0)
    N, w, d = 24, 5, 4
    X = rng.normal(size=(N, w, d))
    y = np.sin(2.0 * X.mean(axis=(1, 2))) + 0.01 * rng.standard_normal(N)

    mdl.fit(X, y)
    yhat = mdl.predict()

    assert yhat.shape == mdl.y_test_.shape
    assert np.all(np.isfinite(yhat))

    last = getattr(mdl.featurizer.runner, "_last_run_kwargs", None)
    assert isinstance(last, dict)
    for k, v in runner_kwargs.items():
        assert last.get(k) == v


def test_from_config_end_to_end_smoke_cs_feature_maps_kwargs_forwarded(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    End-to-end smoke test (patched Aer execution) verifying that `features.kwargs`
    are forwarded to `CSFeatureMapsRetriever.get_feature_maps`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.

    Returns
    -------
    None
    """
    _patch_runner_to_fake_states(monkeypatch)

    # Patch the *real* CSFeatureMapsRetriever class used by Hydra instantiation.
    from src.qrc.run.cs_fmp_retriever import CSFeatureMapsRetriever

    orig_get = CSFeatureMapsRetriever.get_feature_maps

    def wrapped_get(self, results, **kwargs):
        self._last_fmp_kwargs = dict(kwargs)
        return orig_get(self, results, **kwargs)

    monkeypatch.setattr(CSFeatureMapsRetriever, "get_feature_maps", wrapped_get, raising=True)

    fmp_kwargs = {"shots": 32, "seed": 0, "n_groups": 4}

    cfg = OmegaConf.create(
        _cfg_dict(
            num_qubits=2,
            input_dim=4,
            runner_kwargs={"device": "GPU"},  # forwarding only
            fmp_kind="cs",
            fmp_kwargs=fmp_kwargs,
            tuning_strategy="grid",
            standardize=False,
        )
    )

    mdl = QRCMaternKRRRegressor.from_config(cfg)

    rng = np.random.default_rng(1)
    N, w, d = 20, 4, 4
    X = rng.normal(size=(N, w, d))
    y = (X[:, -1, 0] - 0.5 * X[:, -1, 1]) + 0.01 * rng.standard_normal(N)

    mdl.fit(X, y)
    yhat = mdl.predict()
    assert np.all(np.isfinite(yhat))

    last_fmp = getattr(mdl.featurizer.fmp_retriever, "_last_fmp_kwargs", None)
    assert isinstance(last_fmp, dict)
    for k, v in fmp_kwargs.items():
        assert last_fmp.get(k) == v


def _aer_available_devices(backend) -> list[str]:
    """
    Return Aer backend available_devices() if present.

    Parameters
    ----------
    backend : object
        Aer backend instance.

    Returns
    -------
    list[str]
        List of available devices (e.g. ["CPU","GPU"]) or empty.
    """
    if hasattr(backend, "available_devices"):
        try:
            return list(backend.available_devices())
        except Exception:
            return []
    return []


def _backend_device_option(backend):
    """
    Read backend.options.device if present.

    Parameters
    ----------
    backend : object
        Aer backend instance.

    Returns
    -------
    object
        backend.options.device or None.
    """
    opts = getattr(backend, "options", None)
    if opts is None:
        return None
    return getattr(opts, "device", None)


def test_from_config_real_aer_gpu_device_sets_backend_option_or_skips() -> None:
    """
    End-to-end (config -> model -> featurizer.transform -> ExactAerCircuitsRunner.run_pubs):
    If Aer GPU is available, ensure the backend is actually configured with device='GPU'.
    Skips cleanly otherwise.

    Returns
    -------
    None
    """
    pytest.importorskip("qiskit_aer")

    rng = np.random.default_rng(0)
    N, w, d = 2, 2, 3
    X = rng.normal(size=(N, w, d)).astype(float)

    cfg = OmegaConf.create(
        _cfg_dict(
            num_qubits=3,
            input_dim=d,
            runner_kwargs={"device": "GPU", "optimization_level": 0, "seed_simulator": 0},
            fmp_kind="exact",
            fmp_kwargs={},
            tuning_strategy="grid",
            standardize=False,
        )
    )

    mdl = QRCMaternKRRRegressor.from_config(cfg)

    backend = mdl.featurizer.runner.backend
    available = _aer_available_devices(backend)
    if "GPU" not in available:
        pytest.skip(f"Aer GPU not available (available_devices={available})")

    Phi = mdl.featurizer.transform(X)
    assert Phi.shape[0] == N
    assert np.all(np.isfinite(Phi))

    # Strong check: Aer backend option should reflect GPU.
    assert _backend_device_option(backend) == "GPU"


def test_from_config_gpu0_only_via_cuda_visible_devices() -> None:
    """
    End-to-end GPU test. Forces GPU0 via CUDA_VISIBLE_DEVICES=0.

    Notes
    -----
    This only works if CUDA was not initialized before this test runs.
    """
    # Force only GPU0 visible *before importing/initializing Aer backend*
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    pytest.importorskip("qiskit_aer")

    rng = np.random.default_rng(0)
    N, w, d = 2, 2, 3
    X = rng.normal(size=(N, w, d)).astype(float)

    cfg = OmegaConf.create(
        _cfg_dict(
            num_qubits=3,
            input_dim=d,
            runner_kwargs={"device": "GPU", "optimization_level": 0, "seed_simulator": 0},
            fmp_kind="exact",
            fmp_kwargs={},
            tuning_strategy="grid",
            standardize=False,
        )
    )

    mdl = QRCMaternKRRRegressor.from_config(cfg)

    backend = mdl.featurizer.runner.backend
    available = _aer_available_devices(backend)
    if "GPU" not in available:
        pytest.skip(f"Aer GPU not available (available_devices={available})")

    Phi = mdl.featurizer.transform(X)
    assert Phi.shape[0] == N
    assert np.all(np.isfinite(Phi))

    assert _backend_device_option(backend) == "GPU"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
