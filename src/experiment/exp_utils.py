import numpy as np
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union
from tqdm.auto import tqdm


def _as_LN(y: np.ndarray, N: int) -> np.ndarray:
    """
    Return y as shape (L, N), where L is the number of tasks.

    Accepts:
      - (N,) -> (1, N)
      - (L, N) -> unchanged
      - (N, L) -> transposed to (L, N)
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        if y.shape[0] != N:
            raise ValueError(f"y has shape {y.shape}, expected (N,) with N={N}")
        return y.reshape(1, N)
    if y.ndim == 2:
        if y.shape[1] == N:
            return y
        if y.shape[0] == N:
            return y.T
        raise ValueError(f"y has shape {y.shape}, cannot align with N={N}")
    raise ValueError(f"y must be 1D/2D, got {y.shape}")


def select_lambda_via_train_val_per_task(
    exp,
    reg_grid: Sequence[float],
    *,
    val_ratio: float = 0.2,
    seed: int = 0,
    split_strategy: Literal["random", "chronological"] = "random",
    refit_on_full_train: bool = True,
) -> Dict[str, Any]:
    """
    Select the best regularization λ PER TASK using a train/validation split inside the
    outer training set (no test peeking), and optionally refit alpha_ on the full outer
    training set using those per-task λ's.

    This function assumes:
      - exp.model.Phi_full_ exists (cached features, no circuit reruns)
      - exp.model.train_idx_ exists (outer train indices)
      - exp.model.kernel_ exists (callable kernel or list of kernels)
      - exp.model.scaler_ may exist (StandardScaler-like with .transform)

    Parameters
    ----------
    exp:
        Experiment containing dataset + loaded model.
    reg_grid:
        Sequence of positive λ values to evaluate (e.g., np.logspace(-10, 0, 41)).
    val_ratio:
        Fraction of the outer train set to hold out as validation.
    seed:
        RNG seed (used only when split_strategy="random").
    split_strategy:
        - "random": random split of outer train indices into inner-train/val
        - "chronological": val is the last chunk of train indices in time/order
    refit_on_full_train:
        If True, refit model.alpha_ using full outer train indices, with one λ per task.

    Returns
    -------
    out : dict
        Contains:
          - best_lambda_per_task: (L,) array
          - best_idx_per_task: (L,) argmin indices into reg_grid
          - reg_grid: (R,) array
          - mse_val: (L, R) validation MSE curves
          - val_idx: (n_val,) global indices used for validation
          - train_inner_idx: (n_tr_in,) global indices used for inner training
    """
    m = exp.model
    Phi_full = np.asarray(m.Phi_full_)
    N = int(Phi_full.shape[0])

    y2d = _as_LN(np.asarray(exp.dataset.y), N)  # (L, N)
    L = int(y2d.shape[0])

    tr_idx = np.asarray(m.train_idx_, dtype=int).reshape(-1)
    if tr_idx.size < 3:
        raise ValueError("train_idx_ too small for train/val split.")

    reg_arr = np.asarray(reg_grid, dtype=float).reshape(-1)
    if reg_arr.size == 0 or np.any(~np.isfinite(reg_arr)) or np.any(reg_arr <= 0):
        raise ValueError("reg_grid must be non-empty with finite positive values.")

    # --- build train/val split inside outer train ---
    n_val = max(1, int(round(val_ratio * tr_idx.size)))
    if n_val >= tr_idx.size:
        n_val = tr_idx.size - 1

    if split_strategy == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(tr_idx.size)
        val_local = perm[:n_val]
        tr_local = perm[n_val:]
    else:  # chronological
        order = np.argsort(tr_idx)
        val_local = order[-n_val:]
        tr_local = order[:-n_val]

    val_idx = tr_idx[val_local]
    tr_in_idx = tr_idx[tr_local]

    # --- features for inner train and val (apply scaler if present) ---
    Phi_tr_in = Phi_full[tr_in_idx]
    Phi_val = Phi_full[val_idx]
    if m.scaler_ is not None:
        Phi_tr_in = m.scaler_.transform(Phi_tr_in)
        Phi_val = m.scaler_.transform(Phi_val)

    y_tr_in = y2d[:, tr_in_idx]  # (L, n_tr_in)
    y_val = y2d[:, val_idx]      # (L, n_val)

    R = int(reg_arr.size)
    mse_val = np.empty((L, R), dtype=float)

    # Kernel can be single callable or list-of-callables.
    kernels = m.kernel_ if isinstance(m.kernel_, list) else [m.kernel_]

    # --- sweep λ per task on validation ---
    for l in range(L):
        ker = kernels[l] if l < len(kernels) else kernels[-1]

        Ktt = ker(Phi_tr_in, Phi_tr_in)          # (n_tr_in, n_tr_in)
        Kvt = ker(Phi_val, Phi_tr_in)            # (n_val, n_tr_in)

        y_l_tr = y_tr_in[l].reshape(-1)
        y_l_val = y_val[l].reshape(-1)

        I = np.eye(Ktt.shape[0], dtype=Ktt.dtype)
        for j, lam in enumerate(reg_arr):
            alpha = np.linalg.solve(Ktt + float(lam) * I, y_l_tr)
            yhat = Kvt @ alpha
            mse_val[l, j] = float(np.mean((y_l_val - yhat) ** 2))

    best_idx_per_task = np.argmin(mse_val, axis=1)          # (L,)
    best_lambda_per_task = reg_arr[best_idx_per_task]       # (L,)

    out: Dict[str, Any] = dict(
        best_lambda_per_task=best_lambda_per_task,
        best_idx_per_task=best_idx_per_task,
        reg_grid=reg_arr,
        mse_val=mse_val,
        val_idx=val_idx,
        train_inner_idx=tr_in_idx,
    )

    # --- refit alpha_ on the FULL outer train set using per-task λ's ---
    if refit_on_full_train:
        Phi_tr = Phi_full[tr_idx]
        if m.scaler_ is not None:
            Phi_tr = m.scaler_.transform(Phi_tr)
        y_tr = y2d[:, tr_idx]  # (L, n_train)

        alpha_list = []
        for l in range(L):
            ker = kernels[l] if l < len(kernels) else kernels[-1]
            Ktt = ker(Phi_tr, Phi_tr)
            I = np.eye(Ktt.shape[0], dtype=Ktt.dtype)
            lam_l = float(best_lambda_per_task[l])
            alpha_list.append(np.linalg.solve(Ktt + lam_l * I, y_tr[l].reshape(-1)))

        alpha_stack = np.stack(alpha_list, axis=0)  # (L, n_train)

        # Keep backward-compatibility: if single-task, store 1D alpha_
        m.alpha_ = alpha_stack[0] if L == 1 else alpha_stack

        # Preserve existing scalar attribute if your code expects it,
        # and add the per-task vector explicitly.
        m.lambda_reg_per_task_ = best_lambda_per_task
        m.lambda_reg_ = float(best_lambda_per_task[0]) if L == 1 else float(np.mean(best_lambda_per_task))

    return out


def _get_reg_sweep_container(exp) -> Dict[str, Any]:
    """Return the reg_sweep dict stored on exp or exp.model, else raise."""
    rs = getattr(exp, "reg_sweep_", None)
    if rs is None:
        rs = getattr(getattr(exp, "model", None), "reg_sweep_", None)
    if rs is None:
        raise RuntimeError(
            "No regularization sweep found on `exp.reg_sweep_` nor `exp.model.reg_sweep_`. "
            "Attach it first (e.g., load reg_sweep.npz into exp.reg_sweep_)."
        )
    if not isinstance(rs, dict):
        raise TypeError(f"reg_sweep_ must be a dict, got {type(rs)}")
    return rs


def _best_lambda_per_task_from_reg_sweep(
    exp,
    *,
    prefer: Tuple[str, ...] = ("mse_val", "mse_test"),
) -> np.ndarray:
    """
    Extract best lambda per task from an existing reg_sweep dict.

    Looks for reg_grid and one of mse_val/mse_test (prefers mse_val if present).
    Returns lambdas of shape (L,).
    """
    rs = _get_reg_sweep_container(exp)

    reg_grid = rs.get("reg_grid", None)
    if reg_grid is None:
        # tolerate alternative key name
        reg_grid = rs.get("lambda_grid", None)
    if reg_grid is None:
        raise KeyError("reg_sweep_ must contain 'reg_grid' (or 'lambda_grid').")

    reg_grid = np.asarray(reg_grid, dtype=float).reshape(-1)
    if reg_grid.size == 0:
        raise ValueError("reg_grid is empty.")
    if np.any(reg_grid <= 0) or np.any(~np.isfinite(reg_grid)):
        raise ValueError("reg_grid must contain finite positive values.")

    mse = None
    used_key = None
    for k in prefer:
        if k in rs:
            mse = np.asarray(rs[k], dtype=float)
            used_key = k
            break
    if mse is None:
        raise KeyError(f"reg_sweep_ must contain one of {prefer} to pick best lambdas.")

    # Determine number of tasks L from dataset y
    m = exp.model
    Phi_full = np.asarray(m.Phi_full_)
    N = int(Phi_full.shape[0])
    y2d = _as_LN(np.asarray(exp.dataset.y), N)  # (L,N)
    L = int(y2d.shape[0])

    # Canonicalize mse to (L, R)
    R = int(reg_grid.size)
    if mse.ndim == 1:
        # (R,) single-task
        if L != 1 or mse.shape[0] != R:
            raise ValueError(f"{used_key} has shape {mse.shape}, expected (R,) with R={R} for L=1.")
        mse_lr = mse.reshape(1, R)
    elif mse.ndim == 2:
        if mse.shape == (L, R):
            mse_lr = mse
        elif mse.shape == (R, L):
            mse_lr = mse.T
        else:
            raise ValueError(f"{used_key} has shape {mse.shape}, expected (L,R)=({L},{R}) or (R,L)=({R},{L}).")
    else:
        raise ValueError(f"{used_key} must be 1D or 2D, got {mse.shape}.")

    best_idx = np.argmin(mse_lr, axis=1)              # (L,)
    best_lams = reg_grid[best_idx].astype(float)      # (L,)
    return best_lams


def sweep_training_sizes_fixed_test(
    exp,
    train_sizes: Sequence[int],
    *,
    lambda_regs: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    order: Literal["chronological", "as_is"] = "chronological",
    apply_scaler: bool = True,
    jitter: float = 0.0,
    max_jitter_tries: int = 6,
    sweep_metric_prefer: Tuple[str, ...] = ("mse_val", "mse_test"),
    progress: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Nested training-size sweep with fixed test split, using cached features (no circuit reruns).

    - Test set is fixed to exp.model.test_idx_.
    - For each N_tr, the training set is the prefix of ordered train indices, ensuring
      D_{n1} ⊂ D_{n2} ⊂ ... (nested datasets).

    Regularization:
    - If lambda_regs is provided: use it (scalar or per-task vector).
    - Else: extract best per-task lambdas from existing reg_sweep_ attached to the experiment.
            If no reg_sweep exists -> raise.

    Returns
    -------
    results_by_size : dict
        N_tr -> {
          "train_idx", "test_idx",
          "lambda_regs",                 # (L,)
          "alpha",                       # list length L, alpha[l] shape (N_tr,)
          "y_pred_train", "y_pred_test", # (L,N_tr), (L,N_te)
          "mse_train", "mse_test",       # (L,), (L,)
          "y_true_train", "y_true_test"  # (L,N_tr), (L,N_te)
        }
    """
    m = exp.model
    if getattr(m, "Phi_full_", None) is None:
        raise RuntimeError("exp.model.Phi_full_ is None. Load/fit a model with cached features first.")

    Phi_full = np.asarray(m.Phi_full_)
    N = int(Phi_full.shape[0])

    y2d = _as_LN(np.asarray(exp.dataset.y), N)  # (L,N)
    L = int(y2d.shape[0])

    train_idx_all = np.asarray(getattr(m, "train_idx_", None), dtype=int).reshape(-1)
    test_idx = np.asarray(getattr(m, "test_idx_", None), dtype=int).reshape(-1)
    if train_idx_all.size == 0 or test_idx.size == 0:
        raise ValueError("Model must expose non-empty train_idx_ and test_idx_.")

    # Order train indices for chronological nested prefixes
    train_idx_ord = np.sort(train_idx_all) if order == "chronological" else train_idx_all.copy()
    n_tr_max = int(train_idx_ord.size)

    # Normalize train_sizes
    sizes = sorted(set(int(s) for s in train_sizes))
    if any(s <= 0 for s in sizes):
        raise ValueError(f"train_sizes must be positive. Got: {train_sizes}")
    if any(s > n_tr_max for s in sizes):
        raise ValueError(f"Some train_sizes exceed available train size {n_tr_max}: {train_sizes}")

    # Resolve lambda_regs (per task)
    if lambda_regs is None:
        lam_vec = _best_lambda_per_task_from_reg_sweep(exp, prefer=sweep_metric_prefer)
    else:
        arr = np.asarray(lambda_regs, dtype=float).reshape(-1)
        if arr.size == 1:
            lam_vec = np.full((L,), float(arr[0]), dtype=float)
        elif arr.size == L:
            lam_vec = arr.astype(float)
        else:
            raise ValueError(f"lambda_regs must be scalar or length L={L}; got len={arr.size}.")

    if np.any(~np.isfinite(lam_vec)) or np.any(lam_vec <= 0):
        raise ValueError(f"Invalid lambda_regs: {lam_vec}")

    # Pre-slice features
    Phi_tr_full = Phi_full[train_idx_ord]
    Phi_te = Phi_full[test_idx]
    if apply_scaler and getattr(m, "scaler_", None) is not None:
        Phi_tr_full = m.scaler_.transform(Phi_tr_full)
        Phi_te = m.scaler_.transform(Phi_te)

    # Kernels: single callable or list per task
    kernels = m.kernel_ if isinstance(m.kernel_, list) else [m.kernel_]
    if kernels is None or len(kernels) == 0:
        raise RuntimeError("Model kernel_ is missing/empty.")

    # Initialize outputs
    n_te = int(test_idx.size)
    results_by_size: Dict[int, Dict[str, Any]] = {}
    for ntr in sizes:
        results_by_size[ntr] = dict(
            train_idx=train_idx_ord[:ntr].copy(),
            test_idx=test_idx.copy(),
            lambda_regs=lam_vec.copy(),
            alpha=[None] * L,
            y_pred_train=np.empty((L, ntr), dtype=float),
            y_pred_test=np.empty((L, n_te), dtype=float),
            mse_train=np.empty((L,), dtype=float),
            mse_test=np.empty((L,), dtype=float),
            y_true_train=y2d[:, train_idx_ord[:ntr]].astype(float, copy=False),
            y_true_test=y2d[:, test_idx].astype(float, copy=False),
        )

    def _chol_spd(A: np.ndarray, base_jitter: float) -> np.ndarray:
        jit = float(base_jitter)
        for _ in range(max_jitter_tries):
            try:
                if jit > 0:
                    return np.linalg.cholesky(A + jit * np.eye(A.shape[0], dtype=A.dtype))
                return np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                jit = (1e-12 if jit == 0 else jit * 10.0)
        raise np.linalg.LinAlgError("Cholesky failed even after jitter escalation.")

    # Sweep per task, reusing full Gram/Cholesky and taking leading principal blocks
    task_iter = range(L)
    if progress:
        task_iter = tqdm(task_iter, desc="tasks", leave=True)
    for l in task_iter:
        ker = kernels[l] if l < len(kernels) else kernels[-1]
        lam = float(lam_vec[l])

        # full outer-train Gram and test-cross Gram
        Ktt = ker(Phi_tr_full, Phi_tr_full)   # (n_tr_max, n_tr_max)
        Kxt = ker(Phi_te, Phi_tr_full)        # (n_te, n_tr_max)

        y_tr_full_l = y2d[l, train_idx_ord].astype(float, copy=False).reshape(-1)
        y_te_l = y2d[l, test_idx].astype(float, copy=False).reshape(-1)

        A = Ktt + lam * np.eye(n_tr_max, dtype=Ktt.dtype)
        Lchol = _chol_spd(A, jitter)

        size_iter = sizes
        if progress:
            size_iter = tqdm(size_iter, desc=f"sizes (task {l})", leave=False)
        for ntr in size_iter:
            Lsub = Lchol[:ntr, :ntr]
            ysub = y_tr_full_l[:ntr]

            z = np.linalg.solve(Lsub, ysub)
            alpha = np.linalg.solve(Lsub.T, z)

            yhat_tr = Ktt[:ntr, :ntr] @ alpha
            yhat_te = Kxt[:, :ntr] @ alpha

            out = results_by_size[ntr]
            out["alpha"][l] = alpha
            out["y_pred_train"][l, :] = yhat_tr
            out["y_pred_test"][l, :] = yhat_te
            out["mse_train"][l] = float(np.mean((ysub - yhat_tr) ** 2))
            out["mse_test"][l] = float(np.mean((y_te_l - yhat_te) ** 2))

        # free large matrices before next task
        del Ktt, Kxt, Lchol

    return results_by_size
