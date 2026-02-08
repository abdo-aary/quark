import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt
from typing import Sequence, Optional, Tuple, Any, Dict, Literal

from src.experiment.experiment import Experiment


def _as_KN(y: np.ndarray) -> np.ndarray:
    """
    Canonicalize an array of labels/predictions to shape (K, N).

    This helper accepts either a 1D array (interpreted as a single task) or a
    2D array in one of the common conventions: (K, N) or (N, K). It returns a
    float array with shape (K, N), where:
      - K = number of tasks / functionals
      - N = number of samples / windows

    Heuristic for 2D input:
        If the first dimension is larger than the second (y.shape[0] > y.shape[1]),
        it assumes input is (N, K) and transposes to (K, N). Otherwise it assumes
        input is already (K, N).

    Parameters
    ----------
    y:
        Labels or predictions. Allowed shapes:
          - (N,)        -> returned as (1, N)
          - (K, N)      -> returned as-is
          - (N, K)      -> returned as (K, N) (via transpose heuristic)

    Returns
    -------
    y_kn:
        Array of dtype float with shape (K, N).

    Raises
    ------
    ValueError
        If `y` is not 1D or 2D.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y[None, :]
    if y.ndim != 2:
        raise ValueError(f"y must be 1D/2D, got {y.shape}")
    # heuristic: N usually larger than K
    return y.T if y.shape[0] > y.shape[1] else y


def _get_task_names(exp) -> list[str]:
    """
    Return a best-effort list of task/functional names for an Experiment.

    If the dataset exposes `label_functionals`, this returns each functional's class
    name (e.g., "OneStepForecastFunctional"). Otherwise it falls back to generic
    names "y0", "y1", ..., inferred from the number of label rows in `exp.dataset.y`.

    Parameters
    ----------
    exp:
        Experiment-like object with `.dataset` attribute. The dataset is expected
        to expose either:
          - `label_functionals` (optional), or
          - `y` (labels), which can be 1D or 2D.

    Returns
    -------
    names:
        List of length K, where K is the number of tasks/functionals.
    """
    ds = exp.dataset
    if getattr(ds, "label_functionals", None):
        return [type(f).__name__ for f in ds.label_functionals]
    # fallback
    y = _as_KN(ds.y)
    return [f"y{k}" for k in range(y.shape[0])]


def plot_true_vs_pred_grid(
    exp,
    task_indices=None,
    *,
    split="test",
    x_axis="relative",
    max_points=300,
    sharex=True,
    sharey=False,
    figsize: Optional[Tuple[float, float]] = None,
    panel_size: Tuple[float, float] = (5.0, 3.2),
    panel_titles: Optional[Sequence[str]] = None,
    show_mse_in_title: bool = True,
    title_fontsize: Optional[float] = None,
    label_fontsize: Optional[float] = None,
    tick_fontsize: Optional[float] = None,
    legend_fontsize: Optional[float] = 9.0,
) -> plt.Figure:
    """
    Multi-panel plot of true vs predicted labels (one subplot per task).

    The function uses cached feature vectors `exp.model.Phi_full_` and calls
    `exp.model.predict_from_features(...)` to obtain predictions, then plots the
    selected split (train or test). This is purely a visualization helper; it does
    not fit the model and does not rerun any featurizer/quantum circuits.

    Parameters
    ----------
    exp:
        Experiment instance with:
          - `exp.dataset` providing labels `y`
          - `exp.model` providing cached features and a prediction method:
              * Phi_full_
              * train_idx_ / test_idx_
              * predict_from_features(Phi, apply_scaler=...)
    task_indices:
        Which tasks to plot:
          - None: plot all tasks
          - int: plot a single task index
          - sequence[int]: plot selected task indices
    split:
        Which split to visualize: "test" (default) or "train".
        Uses `exp.model.test_idx_` or `exp.model.train_idx_`.
    x_axis:
        X-axis indexing mode:
          - "relative": x = 0..(n_sub-1) after optional thinning
          - "original": x = the original dataset window indices (then sorted)
    max_points:
        If not None, thin the split indices to at most `max_points` evenly spaced
        points (for readability).
    sharex, sharey:
        Forwarded to `plt.subplots(..., sharex=..., sharey=...)`.
    figsize:
        Overall figure size. If None, computed from `panel_size` and the grid layout.
    panel_size:
        Base size (width, height) of each subplot when `figsize` is None.
    panel_titles:
        Optional list of subplot titles (must match the number of plotted tasks,
        in the same order).
    show_mse_in_title:
        If True, appends the task MSE on the plotted subset to each subplot title.
    title_fontsize:
        Font size for subplot titles (passed to `ax.set_title`).
    label_fontsize:
        Font size for x/y axis labels (passed to `ax.set_xlabel`/`ax.set_ylabel`).
    tick_fontsize:
        Font size for tick labels (applied via `ax.tick_params`).
    legend_fontsize:
        Font size for legend entries.

    Returns
    -------
    fig:
        Matplotlib Figure containing a squarish grid of subplots (one per task).

    Raises
    ------
    ValueError
        If split indices are empty or task indices are out of range.
    RuntimeError
        If cached features `Phi_full_` are missing.
    """
    model = exp.model
    ds = exp.dataset

    y = _as_KN(ds.y)  # (K, N)
    K, N = y.shape

    idx = np.asarray(model.test_idx_ if split == "test" else model.train_idx_, dtype=int)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError(f"Empty or invalid {split}_idx_")

    # thin for readability
    if max_points is not None and idx.size > max_points:
        keep = np.linspace(0, idx.size - 1, num=max_points).round().astype(int)
        idx = idx[keep]

    # resolve task list
    if task_indices is None:
        tasks = list(range(K))
    elif isinstance(task_indices, int):
        tasks = [int(task_indices)]
    else:
        tasks = [int(k) for k in task_indices]

    # validate & unique preserve order
    seen = set()
    tasks_clean = []
    for k in tasks:
        if k in seen:
            continue
        if not (0 <= k < K):
            raise ValueError(f"task index {k} out of range [0,{K-1}]")
        tasks_clean.append(k)
        seen.add(k)
    tasks = tasks_clean
    if panel_titles is not None:
        if len(panel_titles) != len(tasks):
            raise ValueError(
                f"panel_titles must have length {len(tasks)} (one per subplot), got {len(panel_titles)}"
            )

    # get predictions on those indices using cached features
    if model.Phi_full_ is None:
        raise RuntimeError("model.Phi_full_ is None; did you load/fit the model?")
    Phi = model.Phi_full_[idx]
    y_pred = _as_KN(model.predict_from_features(Phi, apply_scaler=True))  # (K, nsub)

    # true labels
    y_true = y[:, idx]  # (K, nsub)

    # x axis
    x = np.arange(idx.size) if x_axis == "relative" else idx

    if x_axis == "original":
        order = np.argsort(x)
        x = x[order]
        y_true = y_true[:, order]
        y_pred = y_pred[:, order]

    # layout: squarish grid
    nplots = len(tasks)
    ncols = int(ceil(sqrt(nplots)))
    nrows = int(ceil(nplots / ncols))

    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
    )
    axes = np.atleast_1d(axes).ravel()

    names = _get_task_names(exp)

    for i, k in enumerate(tasks):
        ax = axes[i]
        mse = float(np.mean((y_true[k] - y_pred[k]) ** 2))

        ax.plot(x, y_true[k], linestyle="-", label="true")
        ax.plot(x, y_pred[k], linestyle="--", label="pred")

        base_title = panel_titles[i] if panel_titles is not None else names[k]
        if show_mse_in_title:
            ax.set_title(f"{base_title} ({split})  MSE={mse:.1g}", fontsize=title_fontsize)
        else:
            ax.set_title(f"{base_title} ({split})", fontsize=title_fontsize)

        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=legend_fontsize)

        if i % ncols == 0:
            ax.set_ylabel("y", fontsize=label_fontsize)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("window index" if x_axis == "relative" else "window index", fontsize=label_fontsize)

        if tick_fontsize is not None:
            ax.tick_params(axis="both", labelsize=tick_fontsize)

    # turn off unused axes
    for j in range(nplots, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    return fig


def plot_mse_vs_train_size_grid(
    results_by_size: Dict[int, Dict[str, Any]],
    exp: Optional[Experiment] = None,
    task_indices=None,
    *,
    split: Literal["test", "train", "both"] = "test",
    x_scale: Literal["linear", "log"] = "linear",
    y_scale: Literal["linear", "log"] = "log",
    sharex: bool = True,
    sharey: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    panel_size: Tuple[float, float] = (5.0, 3.2),
    panel_titles: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    show_lambda_in_title: bool = False,
    title_fontsize: Optional[float] = None,
    label_fontsize: Optional[float] = None,
    tick_fontsize: Optional[float] = None,
    legend_fontsize: Optional[float] = 9.0,
) -> plt.Figure:
    """
    Plot learning curves: MSE versus training size (one subplot per task).

    This function consumes the nested dictionary returned by
    `sweep_training_sizes_fixed_test(...)`, i.e., a mapping:
        N_tr -> record
    where each record typically contains arrays:
        "mse_train": (K,), "mse_test": (K,)
    and optionally "lambda_regs": (K,).

    The plot shows, for each task k, the evolution of the selected MSE metric as
    the training size N_tr increases.

    Parameters
    ----------
    results_by_size:
        Dictionary produced by `sweep_training_sizes_fixed_test`.
        Keys are training sizes (ints). Values are dicts containing at least one of:
          - "mse_test": array-like shape (K,)
          - "mse_train": array-like shape (K,)
        and optionally:
          - "lambda_regs": array-like shape (K,) or scalar
    exp:
        Optional Experiment used only to infer nice task names for titles. If None,
        generic titles "y0", "y1", ... are used.
    task_indices:
        Which tasks to plot:
          - None: plot all tasks
          - int: plot a single task index
          - sequence[int]: plot selected task indices
    split:
        Which curve(s) to plot:
          - "test": plot only test MSE
          - "train": plot only train MSE
          - "both": plot both (train solid, test dashed)
    x_scale, y_scale:
        Axis scaling for x and y ("linear" or "log").
    sharex, sharey:
        Forwarded to `plt.subplots(..., sharex=..., sharey=...)`.
    figsize:
        Overall figure size. If None, computed from `panel_size` and the grid layout.
    panel_size:
        Base size (width, height) of each subplot when `figsize` is None.
    panel_titles:
        Optional list of subplot titles (must match the number of plotted tasks,
        in the same order).
    title:
        Optional figure-level title (suptitle).
    show_lambda_in_title:
        If True and lambda values are present in `results_by_size[*]["lambda_regs"]`,
        append the per-task lambda to each subplot title.
    title_fontsize:
        Font size for subplot titles and optional suptitle.
    label_fontsize:
        Font size for axis labels ("MSE" and "$N_{\\mathrm{tr}}$").
    tick_fontsize:
        Font size for tick labels (applied to both major/minor via `tick_params`).
    legend_fontsize:
        Font size for legend entries.

    Returns
    -------
    fig:
        Matplotlib Figure with a squarish grid of subplots (one per task).

    Raises
    ------
    ValueError
        If `results_by_size` is empty or `task_indices` are invalid.
    KeyError
        If the requested `split` requires missing MSE entries (e.g., split="test"
        but "mse_test" is absent).
    """
    if not results_by_size:
        raise ValueError("results_by_size is empty")

    Ns = np.array(sorted(int(k) for k in results_by_size.keys()), dtype=int)
    first = results_by_size[int(Ns[0])]

    # infer K from mse arrays
    mse_key = "mse_test" if "mse_test" in first else ("mse_train" if "mse_train" in first else None)
    if mse_key is None:
        raise KeyError("results entries must contain at least one of {'mse_train','mse_test'}")

    K = int(np.asarray(first[mse_key]).reshape(-1).shape[0])

    # resolve task list
    if task_indices is None:
        tasks = list(range(K))
    elif isinstance(task_indices, int):
        tasks = [int(task_indices)]
    else:
        tasks = [int(k) for k in task_indices]

    # validate & unique preserve order
    seen = set()
    tasks_clean = []
    for k in tasks:
        if k in seen:
            continue
        if not (0 <= k < K):
            raise ValueError(f"task index {k} out of range [0,{K-1}]")
        tasks_clean.append(k)
        seen.add(k)
    tasks = tasks_clean

    if panel_titles is not None and len(panel_titles) != len(tasks):
        raise ValueError(
            f"panel_titles must have length {len(tasks)} (one per subplot), got {len(panel_titles)}"
        )

    # task names
    if exp is not None:
        names = _get_task_names(exp)
    else:
        names = [f"y{k}" for k in range(K)]

    # collect curves: shape (K, len(Ns))
    mse_train = None
    mse_test = None
    if split in ("train", "both"):
        mse_train = np.vstack([np.asarray(results_by_size[int(n)]["mse_train"]).reshape(-1) for n in Ns])
        mse_train = mse_train.T  # (K, T)
    if split in ("test", "both"):
        mse_test = np.vstack([np.asarray(results_by_size[int(n)]["mse_test"]).reshape(-1) for n in Ns])
        mse_test = mse_test.T  # (K, T)

    # optional lambdas (assumed fixed across Ns)
    lam_vec = None
    if "lambda_regs" in first:
        lam_vec = np.asarray(first["lambda_regs"], dtype=float).reshape(-1)
        if lam_vec.size == 1:
            lam_vec = np.full((K,), float(lam_vec[0]))
        elif lam_vec.size != K:
            lam_vec = None

    # layout: squarish grid
    nplots = len(tasks)
    ncols = int(ceil(sqrt(nplots)))
    nrows = int(ceil(nplots / ncols))
    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
    )
    axes = np.atleast_1d(axes).ravel()

    for i, k in enumerate(tasks):
        ax = axes[i]

        if split in ("train", "both"):
            ax.plot(Ns, mse_train[k], marker="o", linestyle="-", label="train")
        if split in ("test", "both"):
            ax.plot(Ns, mse_test[k], marker="o", linestyle="--", label="test")

        base_title = panel_titles[i] if panel_titles is not None else names[k]
        if show_lambda_in_title and lam_vec is not None:
            base_title = f"{base_title}  (λ={lam_vec[k]:.1g})"
        ax.set_title(base_title, fontsize=title_fontsize)

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=legend_fontsize)

        ax.set_ylabel("MSE", fontsize=label_fontsize)

        if i >= (nrows - 1) * ncols:
            ax.set_xlabel(r"$N_{\mathrm{tr}}$", fontsize=label_fontsize)

        if tick_fontsize is not None:
            ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)

    # turn off unused axes
    for j in range(nplots, len(axes)):
        axes[j].axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()

    return fig
