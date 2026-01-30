"""End-to-end regression: QRC features + Matérn kernel ridge regression.

This module defines :class:`src.models.qrc_matern_krr.QRCMaternKRRRegressor`, an sklearn-like estimator that:

- computes quantum features ``Phi`` once from inputs ``X`` via :class:`src.models.qrc_featurizer.QRCFeaturizer`,
- tunes Matérn hyperparameters on a train/validation split of the training set,
- fits kernel ridge regression (KRR) in dual form,
- predicts by evaluating ``K(Phi_test, Phi_train) @ alpha``.

Multi-output labels
-------------------
Labels are accepted as ``(N,)`` (single output), ``(L, N)`` or ``(N, L)`` (multi-output).
When ``L > 1``, tuning and KRR fitting can be parallelized across outputs with multiprocessing using
the ``num_workers`` argument of :meth:`QRCMaternKRRRegressor.fit` (feature extraction remains a single call).
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import json
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from typing import Any, Dict, List, Optional, Tuple, Union

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import Matern as SkMatern, ConstantKernel

from src.models.kernel import tune_matern_grid_train_val, tune_matern_continuous_train_val
from src.models.qrc_featurizer import QRCFeaturizer

from qiskit.quantum_info import SparsePauliOp

_SAVE_FORMAT_VERSION = 1


def _to_builtin(obj):
    """Convert numpy scalars to builtin Python scalars for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _to_plain_dict(node) -> dict:
    """
    Convert an OmegaConf node to a plain dict safely.

    Parameters
    ----------
    node : Any
        OmegaConf node (DictConfig/ListConfig) or a Python object.

    Returns
    -------
    dict
        Plain resolved dict, or {} if node is missing/None.
    """
    if node is None:
        return {}
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True) or {}
    if isinstance(node, dict):
        return dict(node)
    return {}


def build_sparse_pauli_ops(labels: List[str]) -> List[SparsePauliOp]:
    """
    Build a list of Qiskit `SparsePauliOp` from string labels.

    Parameters
    ----------
    labels : list of str
        Pauli labels, e.g. ["XIZ", "ZZI"].

    Returns
    -------
    list of SparsePauliOp
        Qiskit Pauli operators.
    """
    return [SparsePauliOp(lab) for lab in labels]


def _train_test_split_indices(N: int, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a random train/test split.

    Parameters
    ----------
    N : int
        Number of samples.
    test_ratio : float
        Fraction assigned to the test set.
    seed : int
        RNG seed.

    Returns
    -------
    train_idx, test_idx : numpy.ndarray
        Index arrays defining the split.

    Notes
    -----
    Guarantees at least one test sample and at least one train sample.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_test = max(1, int(test_ratio * N))
    if n_test >= N:
        n_test = N - 1
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def _build_kernel(xi: float, nu: float) -> Any:
    """
    Build a fixed-amplitude Matérn kernel.

    Parameters
    ----------
    xi : float
        Matérn length-scale.
    nu : float
        Matérn smoothness parameter.

    Returns
    -------
    sklearn.gaussian_process.kernels.Kernel
        Kernel object usable as ``K = kernel(X, X')``.
    """
    # fixed amplitude=1.0; we can expose this if we want
    return ConstantKernel(1.0, constant_value_bounds="fixed") * SkMatern(
        length_scale=float(xi),
        length_scale_bounds="fixed",
        nu=float(nu),
    )


# -----------------------------------------------------------------------------
# Multiprocessing helpers (module-level for pickling)
# -----------------------------------------------------------------------------

_MP_PHI_TR: Optional[np.ndarray] = None
_MP_TUNING: Optional[Dict[str, Any]] = None


def _mp_init(phi_tr: np.ndarray, tuning: Dict[str, Any]) -> None:
    """Initializer that stores read-only data in each worker process."""
    global _MP_PHI_TR, _MP_TUNING
    _MP_PHI_TR = phi_tr
    _MP_TUNING = tuning


def _tune_and_fit_one_output(args: Tuple[int, np.ndarray]) -> Tuple[int, Dict[str, float], np.ndarray]:
    """Tune Matérn params and solve KRR for one output.

    Notes
    -----
    This runs in a worker process. It relies on `_MP_PHI_TR` and `_MP_TUNING` being
    set via `_mp_init`.
    """
    out_idx, y_tr = args
    if _MP_PHI_TR is None or _MP_TUNING is None:
        raise RuntimeError("Multiprocessing globals not initialized.")

    Phi_tr = _MP_PHI_TR
    tuning = _MP_TUNING

    # tune Matérn hyperparameters on train_val
    strategy = tuning.get("strategy", "grid")
    val_ratio = float(tuning.get("val_ratio", 0.2))
    tune_seed = int(tuning.get("seed", 0))
    reg = float(tuning.get("reg", 1e-6))
    xi_bounds = tuple(tuning.get("xi_bounds", (1e-3, 1e3)))

    if strategy == "grid":
        nu_grid = tuning.get("nu_grid", (0.5, 1.5, 2.5, 5.0))
        xi_maxiter = int(tuning.get("xi_maxiter", 80))
        best_params, _ = tune_matern_grid_train_val(
            Phi_tr,
            y_tr,
            val_ratio=val_ratio,
            seed=tune_seed,
            reg=reg,
            xi_bounds=xi_bounds,
            nu_grid=nu_grid,
            xi_maxiter=xi_maxiter,
        )
    elif strategy == "powell":
        nu_bounds = tuple(tuning.get("nu_bounds", (0.2, 5.0)))
        n_restarts = int(tuning.get("n_restarts", 8))
        best_params, _ = tune_matern_continuous_train_val(
            Phi_tr,
            y_tr,
            val_ratio=val_ratio,
            seed=tune_seed,
            reg=reg,
            xi_bounds=xi_bounds,
            nu_bounds=nu_bounds,
            n_restarts=n_restarts,
        )
    else:
        raise ValueError(f"Unknown tuning.strategy={strategy!r}")

    xi = float(best_params["xi"])
    nu = float(best_params["nu"])
    reg = float(best_params["reg"])

    # fit KRR (dual) on all train_val
    kernel = _build_kernel(xi=xi, nu=nu)
    Ktt = kernel(Phi_tr, Phi_tr)
    A = Ktt + reg * np.eye(Ktt.shape[0])
    alpha = np.linalg.solve(A, y_tr)

    return out_idx, dict(best_params), alpha


class QRCMaternKRRRegressor(BaseEstimator, RegressorMixin):
    """
    Quantum feature + Matérn KRR regressor (single- or multi-output).

    Parameters
    ----------
    featurizer : src.models.qrc_featurizer.QRCFeaturizer
        Object used to compute quantum features ``Phi`` from window datasets.
    standardize : bool, default=True
        If True, standardize ``Phi`` before tuning and fitting.
    test_ratio : float, default=0.2
        Fraction of samples used as held-out test set.
    split_seed : int, default=0
        RNG seed controlling the train/test split.
    tuning : dict, optional
        Matérn tuner configuration. Common keys:

        - ``strategy``: ``"grid"`` or ``"powell"``
        - ``val_ratio``: validation ratio inside the training set
        - ``seed``: tuning split seed
        - ``reg``: ridge regularization parameter
        - ``xi_bounds``: bounds for length-scale

        Grid strategy keys:
        - ``nu_grid``, ``xi_maxiter``

        Powell strategy keys:
        - ``nu_bounds``, ``n_restarts``

    Notes
    -----
    For multi-output labels (``L > 1``), this estimator fits one kernel+KRR model per output while reusing
    the same feature matrix. Per-output tuning+fit can be parallelized with multiprocessing using
    ``num_workers`` passed to :meth:`fit`.
    """

    def __init__(
            self,
            featurizer: QRCFeaturizer,
            *,
            standardize: bool = True,
            test_ratio: float = 0.2,
            split_seed: int = 0,
            tuning: Optional[Dict[str, Any]] = None,
    ):
        self.featurizer = featurizer
        self.standardize = bool(standardize)
        self.test_ratio = float(test_ratio)
        self.split_seed = int(split_seed)
        self.tuning = tuning or {}

        # learned attrs
        self.scaler_: Optional[StandardScaler] = None
        # If single-output: kernel_ is a sklearn kernel, alpha_ is (n_train,).
        # If multi-output: kernel_ is List[kernel], alpha_ is (L, n_train).
        self.kernel_: Any = None
        self.alpha_: Optional[np.ndarray] = None
        self.X_train_features_: Optional[np.ndarray] = None

        self.X_test_: Optional[np.ndarray] = None
        self.y_test_: Optional[np.ndarray] = None
        self.y_pred_test_: Optional[np.ndarray] = None

        # If single-output: Dict[str,float]. If multi-output: List[Dict[str,float]].
        self.best_params_: Optional[Union[Dict[str, float], List[Dict[str, float]]]] = None
        self.n_outputs_: Optional[int] = None

        # Stored values to be reused
        self.Phi_full_ = None
        self.train_idx_, self.test_idx_ = None, None

    @staticmethod
    def from_config(cfg: DictConfig) -> "QRCMaternKRRRegressor":
        """
        Construct a fully-wired regressor from an OmegaConf/Hydra config.

        All objects are instantiated via Hydra using
        `_target_` entries in the config.

        Parameters
        ----------
        cfg : omegaconf.DictConfig
            Model configuration. This can be either:
            - the model node itself (recommended): `cfg = composed_cfg.model`, or
            - a full experiment config that contains a `model` section.

            Expected structure (high level):

            - `qrc.cfg`: Hydra instantiable circuit config (e.g., RingQRConfig)
            - `qrc.runner`: Hydra instantiable runner class (constructed with qrc_cfg)
            - `qrc.runner.runner_kwargs`: dict of kwargs forwarded to runner execution (not ctor)
            - `qrc.features.retriever`: Hydra instantiable FMP retriever (qrc_cfg, observables)
            - `qrc.features.observables`: Hydra instantiable builder returning list[SparsePauliOp]
            - `qrc.features.kwargs`: dict of kwargs forwarded to retriever execution (not ctor)
            - `qrc.pubs`: contains `family`, `angle_positioning`, plus other PUBS kwargs
            - `preprocess.standardize`, `split.test_ratio`, `split.seed`, `tuning` dict

        Returns
        -------
        QRCMaternKRRRegressor
            A ready-to-fit estimator.

        Notes
        -----
        We intentionally separate:
        - constructor-time objects (runner, retriever) handled via `_target_`, and
        - execution-time kwargs (`runner_kwargs`, `features.kwargs`) stored as plain dicts,
          because these are not necessarily accepted by the constructors.
        """
        # Allow passing either the full cfg or cfg.model
        model_cfg = cfg.model if "model" in cfg else cfg
        qrc_node = model_cfg.qrc


        observables = instantiate(qrc_node.features.observables)

        qrc_cfg = instantiate(qrc_node.cfg)

        # 1) Extract runtime kwargs (stay in config, but not passed to __init__)
        runner_kwargs = _to_plain_dict(qrc_node.runner.get("runner_kwargs"))

        # 2) Build a clean config node for instantiation (remove runner_kwargs)
        runner_cfg_dict = OmegaConf.to_container(qrc_node.runner, resolve=True)
        runner_cfg_dict.pop("runner_kwargs", None)
        runner_cfg_clean = OmegaConf.create(runner_cfg_dict)

        # 3) Instantiate the runner (only ctor kwargs)
        runner = instantiate(runner_cfg_clean, qrc_cfg=qrc_cfg)

        fmp = instantiate(qrc_node.features.retriever, qrc_cfg=qrc_cfg, observables=observables)
        fmp_kwargs = _to_plain_dict(qrc_node.features.get("kwargs"))

        pubs_container = OmegaConf.to_container(qrc_node.pubs, resolve=True)
        pubs_family = pubs_container["family"]
        angle_positioning = pubs_container["angle_positioning"]
        pubs_kwargs = {k: v for k, v in pubs_container.items() if k not in ("family", "angle_positioning")}

        # --- model-level params
        training = model_cfg.get("training", {})
        split = training.get("split", {})
        preprocess = training.get("preprocess", {})

        test_ratio = float(split.get("test_ratio", 0.2))
        split_seed = int(split.get("seed", 0))
        standardize = bool(preprocess.get("standardize", True))

        tuning = OmegaConf.to_container(model_cfg.get("tuning", {}), resolve=True) or {}

        featurizer = QRCFeaturizer(
            qrc_cfg=qrc_cfg,
            runner=runner,
            fmp_retriever=fmp,
            pubs_family=pubs_family,
            angle_positioning_name=angle_positioning,
            pubs_kwargs=pubs_kwargs,
            runner_kwargs=runner_kwargs,
            fmp_kwargs=fmp_kwargs,
        )

        return QRCMaternKRRRegressor(
            featurizer,
            standardize=standardize,
            test_ratio=test_ratio,
            split_seed=split_seed,
            tuning=dict(tuning),
        )

    def fit(self, X: np.ndarray, y: np.ndarray, *, num_workers: int = 1):
        """
        Fit the model.

        Parameters
        ----------
        X : numpy.ndarray
            Window dataset with shape ``(N, w, d)``.
        y : numpy.ndarray
            Labels as ``(N,)`` (single output), ``(L, N)`` (multi-output), or ``(N, L)`` (multi-output).
        num_workers : int, default=1
            Reserved for multi-output parallelism. If num_workers=1, outputs are fit sequentially.

        Returns
        -------
        self : QRCMaternKRRRegressor
            Fitted estimator.

        Notes
        -----
        Quantum feature extraction ``Phi = featurizer.transform(X)`` is executed exactly once, and
        the resulting feature matrix is reused for all outputs.
        """
        X = np.asarray(X)
        y_arr = np.asarray(y)

        if X.ndim != 3:
            raise ValueError(f"X must be (N,w,d). Got {X.shape}.")

        N = int(X.shape[0])

        # Accept labels in any of:
        #   - (N,) (single output)
        #   - (L,N) (preferred internal layout)
        #   - (N,L) (sklearn-like); we transpose into (L,N)
        if y_arr.ndim == 1:
            y2d = y_arr.reshape(1, -1)
        elif y_arr.ndim == 2:
            if y_arr.shape[1] == N:
                y2d = y_arr
            elif y_arr.shape[0] == N and y_arr.shape[1] != N:
                y2d = y_arr.T
            else:
                raise ValueError(
                    f"y must have shape (N,), (L,N) or (N,L). Got {y_arr.shape} for X {X.shape}."
                )
        else:
            raise ValueError(
                f"y must have shape (N,), (L,N) or (N,L). Got {y_arr.shape} for X {X.shape}."
            )

        if y2d.shape[1] != N:
            raise ValueError(f"y must have N={N} samples (second axis). Got y {y2d.shape}.")

        L = int(y2d.shape[0])
        self.n_outputs_ = L

        # quantum featurization (single expensive call)
        Phi = self.featurizer.transform(X)  # (N,D)

        # split train_val vs test
        if not self.train_idx_ or not self.test_idx_:
            tr_idx, te_idx = _train_test_split_indices(N, self.test_ratio, self.split_seed)
        else:
            tr_idx, te_idx = self.train_idx_, self.test_idx_

        Phi_tr = Phi[tr_idx]
        Phi_te = Phi[te_idx]

        # (L, n_train) / (L, n_test)
        y_tr = y2d[:, tr_idx]
        y_te = y2d[:, te_idx]

        # optional standardization (recommended for kernel lengthscale stability)
        if self.standardize:
            self.scaler_ = StandardScaler()
            Phi_tr = self.scaler_.fit_transform(Phi_tr)
            Phi_te = self.scaler_.transform(Phi_te)
        else:
            self.scaler_ = None

        # -----------------------------------------------------------------------
        # Tune + fit KRR per output label, reusing the *same* Phi_tr.
        # This avoids re-running circuits when the same X is used for many tasks.
        # -----------------------------------------------------------------------

        # Normalize num_workers
        if num_workers is None:
            num_workers = 1
        num_workers = int(num_workers)
        if num_workers <= 0:
            num_workers = max(1, mp.cpu_count())

        tasks: List[Tuple[int, np.ndarray]] = [(l, y_tr[l].reshape(-1)) for l in range(L)]

        # Cap workers to number of outputs
        requested = int(num_workers) if num_workers is not None else 1
        effective_workers = max(1, min(requested, L))

        if effective_workers == 1:
            _mp_init(Phi_tr, dict(self.tuning))
            tuned = [_tune_and_fit_one_output(task) for task in tasks]
        else:
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                ctx = mp.get_context()

            with ctx.Pool(
                    processes=effective_workers,
                    initializer=_mp_init,
                    initargs=(Phi_tr, dict(self.tuning)),
            ) as pool:
                tuned = pool.map(_tune_and_fit_one_output, tasks)

        # Reorder by output index
        tuned = sorted(tuned, key=lambda t: t[0])

        best_params_list: List[Dict[str, float]] = []
        alpha_list: List[np.ndarray] = []
        kernel_list: List[Any] = []

        for _, bp, alpha in tuned:
            best_params_list.append(dict(bp))
            alpha = np.asarray(alpha, dtype=float).reshape(-1)
            alpha_list.append(alpha)
            kernel_list.append(_build_kernel(xi=float(bp["xi"]), nu=float(bp["nu"])))

        self.X_train_features_ = Phi_tr

        if L == 1:
            self.best_params_ = best_params_list[0]
            self.alpha_ = alpha_list[0]
            self.kernel_ = kernel_list[0]
        else:
            self.best_params_ = best_params_list
            self.alpha_ = np.stack(alpha_list, axis=0)  # (L, n_train)
            self.kernel_ = kernel_list

        # store test set to allow predict() with X=None
        self.X_test_ = X[te_idx]
        # keep y_test_ as 1D for single-output (backward compat), else (L,n_test)
        self.y_test_ = y_te[0] if L == 1 else y_te

        # optional: compute & store test predictions immediately
        y_pred_te = self._predict_from_features(Phi_te)
        self.y_pred_test_ = y_pred_te

        # Store for persistance
        self.Phi_full_ = Phi
        self.train_idx_, self.test_idx_ = tr_idx, te_idx
        return self

    def _predict_from_features(self, Phi: np.ndarray) -> np.ndarray:
        """
        Predict from (already standardized) feature matrix.

        Parameters
        ----------
        Phi : numpy.ndarray
            Feature matrix of shape ``(N, D)``.

        Returns
        -------
        numpy.ndarray
            Predictions with shape ``(N,)`` for single output or ``(L, N)`` for multi-output.
        """
        if self.alpha_ is None or self.kernel_ is None or self.X_train_features_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # single-output
        if not isinstance(self.kernel_, list):
            Kxt = self.kernel_(Phi, self.X_train_features_)
            return Kxt @ self.alpha_

        # multi-output
        kernels: List[Any] = self.kernel_
        alphas = np.asarray(self.alpha_)
        if alphas.ndim != 2:
            raise RuntimeError(f"Expected alpha_ to be 2D for multi-output, got {alphas.shape}.")

        L = alphas.shape[0]
        yhat = np.empty((L, Phi.shape[0]), dtype=float)
        for l in range(L):
            Kxt = kernels[l](Phi, self.X_train_features_)
            yhat[l] = Kxt @ alphas[l]
        return yhat

    def predict_from_features(self, Phi: np.ndarray, *, apply_scaler: bool = True) -> np.ndarray:
        """Predict from precomputed feature matrix.

        Parameters
        ----------
        Phi:
            Feature matrix of shape (N, D).
        apply_scaler:
            If True and the model was trained with standardization, apply the
            stored ``StandardScaler`` to ``Phi`` before prediction.
        """
        Phi = np.asarray(Phi)
        if Phi.ndim != 2:
            raise ValueError(f"Phi must be 2D (N,D). Got {Phi.shape}.")
        if apply_scaler and self.scaler_ is not None:
            Phi = self.scaler_.transform(Phi)
        return self._predict_from_features(Phi)

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict labels for new inputs or for the stored test set.

        Parameters
        ----------
        X : numpy.ndarray, optional
            If provided, windows with shape ``(N, w, d)``. If None, predicts on the test set stored during ``fit``.

        Returns
        -------
        numpy.ndarray
            Predictions with shape ``(N_test,)`` for single output or ``(L, N_test)`` for multi-output.
        """
        # sklearn-like: predict(X). Convenience: predict() predicts on stored test set.
        if X is None:
            if self.X_test_ is None:
                raise ValueError("No X provided and no stored test set. Call fit(...) first or pass X.")

            # Return the test predictions previously computed in .fit().
            return self.y_pred_test_

        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"X must be (N,w,d). Got {X.shape}.")

        Phi = self.featurizer.transform(X)
        return self.predict_from_features(Phi, apply_scaler=True)

    #######################################################################################
    ##################################### PERSISTANCE #####################################
    #######################################################################################

    def save(self, path: str | Path) -> None:
        """
        Save FULL model artifact (no featurizer rerun required).
        Stores: Phi_full_, train/test indices, best_params_, alpha_, scaler params.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # --- sanity checks (fail fast) ---
        required = ["best_params_", "alpha_"]
        for name in required:
            if getattr(self, name, None) is None:
                raise ValueError(f"Cannot save: missing attribute `{name}` (did you fit the model?).")

        if getattr(self, "Phi_full_", None) is None:
            raise ValueError(
                "Cannot save: missing `Phi_full_`. "
                "Recommendation: in fit(), store the full features before splitting, e.g. `self.Phi_full_ = Phi`."
            )
        if getattr(self, "train_idx_", None) is None or getattr(self, "test_idx_", None) is None:
            raise ValueError(
                "Cannot save: missing `train_idx_`/`test_idx_`. "
                "Recommendation: in fit(), store indices from train_test_split."
            )

        Phi_full = np.asarray(self.Phi_full_)
        train_idx = np.asarray(self.train_idx_)
        test_idx = np.asarray(self.test_idx_)
        alpha = np.asarray(self.alpha_)

        arrays: dict[str, np.ndarray] = {
            "Phi_full": Phi_full,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "alpha": alpha,
        }

        if bool(getattr(self, "standardize", False)):
            if getattr(self, "scaler_", None) is None:
                raise ValueError("standardize=True but `scaler_` is None; cannot save a consistent artifact.")
            scaler_mean = np.asarray(self.scaler_.mean_)
            scaler_scale = np.asarray(self.scaler_.scale_)
            arrays["scaler_mean"] = scaler_mean
            arrays["scaler_scale"] = scaler_scale

        meta = {
            "format_version": _SAVE_FORMAT_VERSION,
            "artifact": "QRCMaternKRRRegressor.full",
            "standardize": bool(getattr(self, "standardize", False)),
            "test_ratio": float(getattr(self, "test_ratio", 0.0)),
            "split_seed": int(getattr(self, "split_seed", 0)),
            "n_outputs_": None if getattr(self, "n_outputs_", None) is None else int(self.n_outputs_),
            "Phi_shape": [int(Phi_full.shape[0]), int(Phi_full.shape[1])],
            "alpha_shape": list(map(int, alpha.shape)),
            "best_params_": getattr(self, "best_params_", None),
        }

        # Write
        np.savez_compressed(path / "arrays.npz", **arrays)

        with (path / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=_to_builtin)

    @classmethod
    def load(cls, path: str | Path, *, featurizer=None, **init_kwargs) -> "QRCMaternKRRRegressor":
        """
        Load FULL model artifact.
        - `featurizer` is optional. If provided, you can still call predict(X) (it will re-featurize).
          If you want to avoid re-featurizing, use predict_from_features with cached Phi_full.
        - `init_kwargs` lets you pass required constructor args (e.g., tuning config) if your __init__ needs them.
        """
        path = Path(path)
        meta_path = path / "meta.json"
        arrays_path = path / "arrays.npz"

        if not meta_path.exists() or not arrays_path.exists():
            raise FileNotFoundError(f"Invalid artifact: expected {meta_path.name} and {arrays_path.name} in {path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = np.load(arrays_path, allow_pickle=False)

        if meta.get("format_version", None) != _SAVE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported artifact version: {meta.get('format_version')}, expected {_SAVE_FORMAT_VERSION}"
            )

        # Create instance (inject featurizer if you want)
        obj = cls(featurizer=featurizer, **init_kwargs)

        # Restore core state
        obj.standardize = bool(meta["standardize"])
        obj.test_ratio = float(meta["test_ratio"])
        obj.split_seed = int(meta["split_seed"])
        obj.n_outputs_ = meta.get("n_outputs_", None)

        obj.best_params_ = meta.get("best_params_", None)

        obj.Phi_full_ = data["Phi_full"]
        obj.train_idx_ = data["train_idx"]
        obj.test_idx_ = data["test_idx"]
        obj.alpha_ = data["alpha"]

        # Rebuild scaler (store only params, not the full sklearn object)
        obj.scaler_ = None
        if obj.standardize:
            sc = StandardScaler()
            sc.mean_ = data["scaler_mean"]
            sc.scale_ = data["scaler_scale"]
            sc.var_ = sc.scale_ ** 2
            sc.n_features_in_ = int(sc.mean_.shape[0])
            obj.scaler_ = sc

        # (Optional but recommended) rebuild kernels from best_params_ so you can
        # do predict_from_features immediately.
        # NOTE: replace `_build_kernel(...)` with your actual internal kernel builder.
        if obj.best_params_ is not None:
            if isinstance(obj.best_params_, list):
                obj.kernel_ = [_build_kernel(xi=bp["xi"], nu=bp["nu"]) for bp in obj.best_params_]
            else:
                bp = obj.best_params_
                obj.kernel_ = _build_kernel(xi=bp["xi"], nu=bp["nu"])

        return obj

    ####################################################################
    ################# Utility for regularization Sweep #################
    ####################################################################

    def sweep_regularization(self, reg_grid, *, store: bool = True) -> Dict[str, Any]:
        """
        Sweep ridge regularization values *after* fit().

        This method assumes the Matérn hyperparameters (xi, nu) are already fixed by fit().
        It does NOT re-featurize and does NOT modify the fitted solution (alpha_, kernel_, best_params_).

        It reconstructs the training targets used in KRR from the stored fitted dual weights:
            y_tr = (Ktt + reg0 I) @ alpha0
        where reg0 is the regularization used during fit (from best_params_).

        Parameters
        ----------
        reg_grid : array-like
            List/array of ridge regularization values (must be finite and > 0).
        store : bool, default=True
            If True, store results in `self.reg_sweep_`.

        Returns
        -------
        Dict[str, Any]
            Dictionary with:
              - "reg_grid": np.ndarray shape (R,)
              - "alpha_grid": (R, n_train) for single-output, or (L, R, n_train) for multi-output
              - "mse_train": (R,) or (L, R)
              - "mse_test":  (R,) or (L, R)
              - "rkhs_norm": (R,) or (L, R)
        """
        # --- basic fitted-state checks ---
        if getattr(self, "Phi_full_", None) is None:
            raise RuntimeError("sweep_regularization requires `Phi_full_` (fit() must have been called).")
        if getattr(self, "train_idx_", None) is None or getattr(self, "test_idx_", None) is None:
            raise RuntimeError(
                "sweep_regularization requires `train_idx_` and `test_idx_` (fit() must have been called).")
        if getattr(self, "alpha_", None) is None or getattr(self, "kernel_", None) is None:
            raise RuntimeError("sweep_regularization requires a fitted `alpha_` and `kernel_` (call fit() first).")
        if getattr(self, "best_params_", None) is None:
            raise RuntimeError("sweep_regularization requires `best_params_` (call fit() first).")
        if getattr(self, "y_test_", None) is None:
            raise RuntimeError("sweep_regularization requires `y_test_` to compute test MSE (call fit() first).")

        reg_arr = np.asarray(reg_grid, dtype=float).reshape(-1)
        if reg_arr.size == 0:
            raise ValueError("reg_grid must be non-empty.")
        if np.any(~np.isfinite(reg_arr)) or np.any(reg_arr <= 0.0):
            raise ValueError(f"All reg values must be finite and > 0. Got: {reg_arr}")

        # Rebuild train/test features from cached full features (works also after load()).
        Phi_full = np.asarray(self.Phi_full_)
        tr_idx = np.asarray(self.train_idx_, dtype=int)
        te_idx = np.asarray(self.test_idx_, dtype=int)

        Phi_tr = Phi_full[tr_idx]
        Phi_te = Phi_full[te_idx]

        if self.scaler_ is not None:
            Phi_tr = self.scaler_.transform(Phi_tr)
            Phi_te = self.scaler_.transform(Phi_te)

        # Prefer the kernel utilities if present; otherwise fall back to local formulas.
        try:
            from src.models.kernel import solve_krr_dual_weights as _solve_krr_dual_weights
            from src.models.kernel import rkhs_norm_from_dual_weights as _rkhs_norm_from_dual_weights
        except Exception:
            def _solve_krr_dual_weights(Ktt: np.ndarray, y: np.ndarray, *, reg: float) -> np.ndarray:
                A = Ktt + float(reg) * np.eye(Ktt.shape[0], dtype=Ktt.dtype)
                return np.linalg.solve(A, y)

            def _rkhs_norm_from_dual_weights(alpha: np.ndarray, Ktt: np.ndarray):
                alpha = np.asarray(alpha)
                if alpha.ndim == 1:
                    val = float(alpha.T @ Ktt @ alpha)
                    return float(np.sqrt(max(val, 0.0)))
                # (n,m): return (m,)
                KA = Ktt @ alpha
                vals = np.sum(alpha * KA, axis=0)
                vals = np.maximum(vals, 0.0)
                return np.sqrt(vals)

        R = int(reg_arr.size)

        # -----------------------
        # Single-output case
        # -----------------------
        if not isinstance(self.kernel_, list):
            kernel = self.kernel_
            alpha0 = np.asarray(self.alpha_, dtype=float).reshape(-1)

            bp = self.best_params_
            if not isinstance(bp, dict):
                raise RuntimeError(f"Expected best_params_ to be dict for single-output, got {type(bp)}")

            reg0 = float(bp.get("reg", self.tuning.get("reg", 1e-6)))

            Ktt = kernel(Phi_tr, Phi_tr)
            Kvt = kernel(Phi_te, Phi_tr)

            # Reconstruct training targets from fitted solution
            y_tr = (Ktt + reg0 * np.eye(Ktt.shape[0], dtype=Ktt.dtype)) @ alpha0
            y_te = np.asarray(self.y_test_, dtype=float).reshape(-1)

            alpha_grid = np.empty((R, Ktt.shape[0]), dtype=float)
            mse_train = np.empty((R,), dtype=float)
            mse_test = np.empty((R,), dtype=float)
            rkhs_norm = np.empty((R,), dtype=float)

            for i, reg in enumerate(reg_arr):
                a = _solve_krr_dual_weights(Ktt, y_tr, reg=float(reg))
                a = np.asarray(a, dtype=float).reshape(-1)

                alpha_grid[i] = a
                yhat_tr = Ktt @ a
                yhat_te = Kvt @ a

                mse_train[i] = float(np.mean((yhat_tr - y_tr) ** 2))
                mse_test[i] = float(np.mean((yhat_te - y_te) ** 2))
                rkhs_norm[i] = float(_rkhs_norm_from_dual_weights(a, Ktt))

            out = {
                "reg_grid": reg_arr,
                "alpha_grid": alpha_grid,
                "mse_train": mse_train,
                "mse_test": mse_test,
                "rkhs_norm": rkhs_norm,
            }

            if store:
                self.reg_sweep_ = out
            return out

        # -----------------------
        # Multi-output case
        # -----------------------
        kernels: List[Any] = self.kernel_
        alpha0 = np.asarray(self.alpha_, dtype=float)
        if alpha0.ndim != 2:
            raise RuntimeError(f"Expected alpha_ to be (L, n_train) for multi-output. Got {alpha0.shape}.")

        L, n_train = alpha0.shape

        bp_list = self.best_params_
        if isinstance(bp_list, dict):
            bp_list = [bp_list for _ in range(L)]
        if not isinstance(bp_list, list) or len(bp_list) != L:
            raise RuntimeError(f"Expected best_params_ to be list[dict] of length {L}. Got {type(self.best_params_)}.")

        y_te = np.asarray(self.y_test_, dtype=float)
        if y_te.ndim == 1:
            y_te = y_te.reshape(1, -1)
        if y_te.shape[0] != L:
            raise RuntimeError(f"y_test_ first dim must match L={L}. Got y_test_ shape {y_te.shape}.")

        alpha_grid = np.empty((L, R, n_train), dtype=float)
        mse_train = np.empty((L, R), dtype=float)
        mse_test = np.empty((L, R), dtype=float)
        rkhs_norm = np.empty((L, R), dtype=float)

        for l in range(L):
            kernel = kernels[l]
            bp = bp_list[l]
            reg0 = float(bp.get("reg", self.tuning.get("reg", 1e-6)))

            Ktt = kernel(Phi_tr, Phi_tr)
            Kvt = kernel(Phi_te, Phi_tr)

            # Reconstruct y_tr for this output
            y_tr = (Ktt + reg0 * np.eye(n_train, dtype=Ktt.dtype)) @ alpha0[l].reshape(-1)

            for i, reg in enumerate(reg_arr):
                a = _solve_krr_dual_weights(Ktt, y_tr, reg=float(reg))
                a = np.asarray(a, dtype=float).reshape(-1)

                alpha_grid[l, i] = a
                yhat_tr = Ktt @ a
                yhat_te = Kvt @ a

                mse_train[l, i] = float(np.mean((yhat_tr - y_tr) ** 2))
                mse_test[l, i] = float(np.mean((yhat_te - y_te[l].reshape(-1)) ** 2))
                rkhs_norm[l, i] = float(_rkhs_norm_from_dual_weights(a, Ktt))

        out = {
            "reg_grid": reg_arr,
            "alpha_grid": alpha_grid,
            "mse_train": mse_train,
            "mse_test": mse_test,
            "rkhs_norm": rkhs_norm,
        }

        if store:
            self.reg_sweep_ = out
        return out
