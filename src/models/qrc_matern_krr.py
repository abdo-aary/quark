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
            - `qrc.runner_kwargs`: dict of kwargs forwarded to runner execution (not ctor)
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

        qrc_cfg = instantiate(qrc_node.cfg)

        observables = instantiate(qrc_node.features.observables)
        runner = instantiate(qrc_node.runner, qrc_cfg=qrc_cfg)
        fmp = instantiate(qrc_node.features.retriever, qrc_cfg=qrc_cfg, observables=observables)

        runner_kwargs = _to_plain_dict(qrc_node.get("runner_kwargs"))
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
        tr_idx, te_idx = _train_test_split_indices(N, self.test_ratio, self.split_seed)
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
