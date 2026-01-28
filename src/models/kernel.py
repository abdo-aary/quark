"""Kernel utilities for QRC models.

This module provides helper routines used by :class:`src.models.qrc_matern_krr.QRCMaternKRRRegressor`
to tune Matérn kernel hyperparameters and evaluate kernel ridge regression (KRR) validation error.

The tuners are intentionally lightweight: a single train/validation split and simple optimizers.
"""

import numpy as np
from typing import Tuple, Dict, Sequence
from scipy.optimize import minimize, minimize_scalar

from sklearn.gaussian_process.kernels import Matern as SkMatern
from sklearn.gaussian_process.kernels import ConstantKernel


def krr_val_mse_for_params(
        Xtr: np.ndarray,
        ytr: np.ndarray,
        Xva: np.ndarray,
        yva: np.ndarray,
        log_xi: float,
        nu: float,
        reg: float,
) -> float:
    """
    Compute the validation MSE for kernel ridge regression with a fixed Matérn kernel.

    Parameters
    ----------
    Xtr, ytr : numpy.ndarray
        Training features and targets. ``Xtr`` has shape ``(N_tr, D)`` and ``ytr`` has shape ``(N_tr,)``.
    Xva, yva : numpy.ndarray
        Validation features and targets. ``Xva`` has shape ``(N_va, D)`` and ``yva`` has shape ``(N_va,)``.
    log_xi : float
        Log length-scale, i.e. ``log_xi = log(xi)``.
    nu : float
        Matérn smoothness parameter.
    reg : float
        Ridge regularization parameter added to the Gram matrix diagonal.

    Returns
    -------
    float
        Mean squared error on the validation set.
    """
    xi = float(np.exp(log_xi))
    # Build kernel with fixed params
    ker = ConstantKernel(1.0, constant_value_bounds="fixed") * SkMatern(
        length_scale=xi, length_scale_bounds="fixed", nu=float(nu)
    )

    Ktt = ker(Xtr, Xtr)
    Kvt = ker(Xva, Xtr)

    A = Ktt + reg * np.eye(Ktt.shape[0])
    alpha = np.linalg.solve(A, ytr)
    yhat = Kvt @ alpha
    return float(np.mean((yhat - yva) ** 2))


def tune_matern_continuous_train_val(
        X: np.ndarray,
        y: np.ndarray,
        *,
        val_ratio: float = 0.2,
        seed: int = 0,
        reg: float = 1e-6,
        # bounds for xi and nu
        xi_bounds: Tuple[float, float] = (1e-3, 1e3),
        nu_bounds: Tuple[float, float] = (0.2, 5.0),
        n_restarts: int = 8,
) -> Tuple[Dict, float]:
    """
    Tune Matérn hyperparameters with a continuous 2D search over ``(xi, nu)``.

    This routine performs a single train/validation split and then runs multiple random restarts
    of a bounded Powell optimization over ``(log(xi), nu)``.

    Parameters
    ----------
    X, y : numpy.ndarray
        Features and targets. ``X`` has shape ``(N, D)`` and ``y`` has shape ``(N,)``.
    val_ratio : float, default=0.2
        Fraction of samples used for validation.
    seed : int, default=0
        Seed controlling the split and random restarts.
    reg : float, default=1e-6
        Ridge regularization parameter.
    xi_bounds : tuple[float, float], default=(1e-3, 1e3)
        Bounds for the Matérn length-scale ``xi``.
    nu_bounds : tuple[float, float], default=(0.2, 5.0)
        Bounds for the Matérn smoothness ``nu``.
    n_restarts : int, default=8
        Number of random restarts.

    Returns
    -------
    best_params : dict
        Best parameters with keys ``{"xi", "nu", "reg"}``.
    best_mse : float
        Best validation MSE achieved.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = rng.permutation(N)
    n_val = max(1, int(val_ratio * N))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[val_idx], y[val_idx]

    log_xi_lo, log_xi_hi = np.log(xi_bounds[0]), np.log(xi_bounds[1])
    nu_lo, nu_hi = nu_bounds

    def obj(theta):
        log_xi, nu = theta
        # keep within bounds (optimizer respects bounds, but safety is fine)
        if not (log_xi_lo <= log_xi <= log_xi_hi and nu_lo <= nu <= nu_hi):
            return 1e30
        return krr_val_mse_for_params(Xtr, ytr, Xva, yva, log_xi, nu, reg=reg)

    bounds = [(log_xi_lo, log_xi_hi), (nu_lo, nu_hi)]

    best_val = np.inf
    best_theta = None

    for _ in range(n_restarts):
        # random init in bounds
        init = np.array([
            rng.uniform(log_xi_lo, log_xi_hi),
            rng.uniform(nu_lo, nu_hi),
        ])
        res = minimize(obj, init, method="Powell", bounds=bounds, options={"maxiter": 200})
        if res.fun < best_val:
            best_val = float(res.fun)
            best_theta = res.x

    best_log_xi, best_nu = best_theta
    best = {"xi": float(np.exp(best_log_xi)), "nu": float(best_nu), "reg": float(reg)}
    return best, best_val


def tune_matern_grid_train_val(
        X: np.ndarray,
        y: np.ndarray,
        *,
        val_ratio: float = 0.2,
        seed: int = 0,
        reg: float = 1e-6,
        xi_bounds: Tuple[float, float] = (1e-3, 1e3),
        nu_grid: Sequence[float] = (0.5, 1.5, 2.5, 5.0),
        xi_maxiter: int = 80,
) -> Tuple[Dict, float]:
    """
    Tune Matérn hyperparameters with a grid over ``nu`` and a 1D bounded search over ``xi``.

    For each ``nu`` in ``nu_grid``, this routine minimizes the validation MSE over
    ``xi in [xi_bounds[0], xi_bounds[1]]`` using bounded optimization on ``log(xi)``.
    The best pair is returned.

    Parameters
    ----------
    X, y : numpy.ndarray
        Features and targets. ``X`` has shape ``(N, D)`` and ``y`` has shape ``(N,)``.
    val_ratio : float, default=0.2
        Fraction of samples used for validation.
    seed : int, default=0
        Seed controlling the split.
    reg : float, default=1e-6
        Ridge regularization parameter.
    xi_bounds : tuple[float, float], default=(1e-3, 1e3)
        Bounds for the Matérn length-scale ``xi``.
    nu_grid : Sequence[float], default=(0.5, 1.5, 2.5, 5.0)
        Candidate values for Matérn smoothness ``nu``.
    xi_maxiter : int, default=80
        Maximum iterations for the 1D optimizer.

    Returns
    -------
    best_params : dict
        Best parameters with keys ``{"xi", "nu", "reg"}``.
    best_mse : float
        Best validation MSE achieved.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,d). Got shape {X.shape}.")
    N = X.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 samples to do a train/val split.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    n_val = int(val_ratio * N)
    n_val = max(1, n_val)
    if n_val >= N:
        n_val = N - 1  # ensure non-empty train set

    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[val_idx], y[val_idx]

    if xi_bounds[0] <= 0 or xi_bounds[1] <= 0 or xi_bounds[0] >= xi_bounds[1]:
        raise ValueError(f"xi_bounds must be positive and increasing. Got {xi_bounds}.")

    log_xi_lo, log_xi_hi = float(np.log(xi_bounds[0])), float(np.log(xi_bounds[1]))

    best_mse = float("inf")
    best_log_xi = None
    best_nu = None

    for nu in nu_grid:
        nu = float(nu)

        res = minimize_scalar(
            lambda log_xi: krr_val_mse_for_params(
                Xtr, ytr, Xva, yva, float(log_xi), nu, reg=reg
            ),
            bounds=(log_xi_lo, log_xi_hi),
            method="bounded",
            options={"maxiter": int(xi_maxiter)},
        )

        if float(res.fun) < best_mse:
            best_mse = float(res.fun)
            best_log_xi = float(res.x)
            best_nu = nu

    best = {
        "xi": float(np.exp(best_log_xi)),
        "nu": float(best_nu),
        "reg": float(reg),
    }
    return best, best_mse


####################################################################
################ Utilities for regularization Sweep ################
####################################################################

def solve_krr_dual_weights(Ktt: np.ndarray, y: np.ndarray, *, reg: float) -> np.ndarray:
    """
    Solve kernel ridge regression dual weights:

        (Ktt + reg * I) alpha = y

    Parameters
    ----------
    Ktt : np.ndarray
        Train Gram matrix, shape (n, n), must be square.
    y : np.ndarray
        Targets, shape (n,) or (n, m).
    reg : float
        Ridge regularization (> 0).

    Returns
    -------
    alpha : np.ndarray
        Dual weights, same shape as y.
    """
    Ktt = np.asarray(Ktt)
    y = np.asarray(y)

    if Ktt.ndim != 2 or Ktt.shape[0] != Ktt.shape[1]:
        raise ValueError(f"Ktt must be a square 2D matrix. Got shape {Ktt.shape}.")

    n = Ktt.shape[0]
    if y.ndim == 1:
        if y.shape[0] != n:
            raise ValueError(f"y must have length {n}. Got shape {y.shape}.")
    elif y.ndim == 2:
        if y.shape[0] != n:
            raise ValueError(f"y must have shape (n, m) with n={n}. Got shape {y.shape}.")
    else:
        raise ValueError(f"y must be 1D or 2D. Got shape {y.shape}.")

    reg = float(reg)
    if not np.isfinite(reg) or reg <= 0.0:
        raise ValueError(f"reg must be a finite positive float. Got {reg}.")

    A = Ktt + reg * np.eye(n, dtype=Ktt.dtype)
    alpha = np.linalg.solve(A, y)
    return alpha


def rkhs_norm_from_dual_weights(alpha: np.ndarray, Ktt: np.ndarray) -> float | np.ndarray:
    r"""
    Compute RKHS norm induced by the kernel:

        ||f||_H = sqrt(alpha^T Ktt alpha)

    Supports:
      - alpha shape (n,)  -> returns float
      - alpha shape (n,m) -> returns np.ndarray shape (m,) (column-wise norms)

    Parameters
    ----------
    alpha : np.ndarray
        Dual weights, shape (n,) or (n, m).
    Ktt : np.ndarray
        Train Gram matrix, shape (n, n).

    Returns
    -------
    float or np.ndarray
        RKHS norm(s).
    """
    Ktt = np.asarray(Ktt)
    alpha = np.asarray(alpha)

    if Ktt.ndim != 2 or Ktt.shape[0] != Ktt.shape[1]:
        raise ValueError(f"Ktt must be a square 2D matrix. Got shape {Ktt.shape}.")

    n = Ktt.shape[0]

    if alpha.ndim == 1:
        if alpha.shape[0] != n:
            raise ValueError(f"alpha must have length {n}. Got shape {alpha.shape}.")
        val = float(alpha.T @ Ktt @ alpha)
        return float(np.sqrt(max(val, 0.0)))

    if alpha.ndim == 2:
        if alpha.shape[0] != n:
            raise ValueError(f"alpha must have shape (n, m) with n={n}. Got shape {alpha.shape}.")
        # column-wise: alpha[:,j]^T K alpha[:,j]
        KA = Ktt @ alpha  # (n, m)
        vals = np.sum(alpha * KA, axis=0)  # (m,)
        vals = np.maximum(vals, 0.0)
        return np.sqrt(vals)

    raise ValueError(f"alpha must be 1D or 2D. Got shape {alpha.shape}.")
