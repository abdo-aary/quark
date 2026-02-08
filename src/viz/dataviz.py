# src/data/dataviz.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Dict, Any, Union

import numpy as np
import matplotlib.pyplot as plt

from src.data.factory import load_windows_dataset, DatasetArtifact
from src.data.generate.base import WindowsDataset


# ---------------------------
# Plotting utilities
# ---------------------------

def _as_2d_labels(y: np.ndarray) -> np.ndarray:
    """
    Ensure labels have shape (K, N) where K = number of tasks/labels.
    Accepts y shaped (K,N) or (N,K) or (N,).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y[None, :]
    if y.ndim != 2:
        raise ValueError(f"y must be 1D or 2D, got shape={y.shape}")
    # Heuristic: N usually larger than K
    if y.shape[0] > y.shape[1]:
        return y.T
    return y


def _zscore_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise z-score normalization for plotting."""
    A = np.asarray(A, dtype=float)
    mu = A.mean(axis=1, keepdims=True)
    sd = A.std(axis=1, keepdims=True)
    sd = np.maximum(sd, eps)
    return (A - mu) / sd


def window_summary(
    X: np.ndarray,
    *,
    summary: Literal["last", "mean", "energy", "pca1"] = "last",
) -> np.ndarray:
    """
    Summarize windows X of shape (N, w, d) into a per-window representation.

    Returns:
      - "last":   (N, d) last vector in each window
      - "mean":   (N, d) mean over time in window
      - "energy": (N, d) mean squared value per dimension
      - "pca1":   (N,)   first PCA component of flattened windows
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"X must have shape (N,w,d), got shape={X.shape}")
    N, w, d = X.shape

    if summary == "last":
        return X[:, -1, :]
    if summary == "mean":
        return X.mean(axis=1)
    if summary == "energy":
        return (X**2).mean(axis=1)
    if summary == "pca1":
        Z = X.reshape(N, w * d)
        Z = Z - Z.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(Z, full_matrices=False)
        return U[:, 0] * S[0]
    raise ValueError(f"Unknown summary={summary!r}")


def plot_labels_vs_window_index(
    X: np.ndarray,
    y: np.ndarray,
    *,
    t: Optional[np.ndarray] = None,
    input_summary: Literal["none", "last", "mean", "energy", "pca1"] = "last",
    normalize: Literal["none", "zscore"] = "zscore",
    title: Optional[str] = None,
    labels_names: Optional[Sequence[str]] = None,
    inputs_names: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Single-axes plot: overlay label curves with a per-window summary of inputs.

    - If normalize="zscore", all curves are z-scored so they are comparable.
    - If input_summary="last"/"mean"/"energy", you see d curves (d=3).
    - If input_summary="pca1", you see one summary curve.

    X: (N,w,d), y: (K,N) or (N,K)
    """
    X = np.asarray(X, dtype=float)
    Y = _as_2d_labels(y)  # (K,N)
    K, N = Y.shape
    if X.shape[0] != N:
        raise ValueError(f"X has N={X.shape[0]} windows but y has N={N}")

    if t is None:
        t = np.arange(N)
    else:
        t = np.asarray(t)
        if t.shape[0] != N:
            raise ValueError("t must have length N")

    curves = []
    curve_names = []

    # Labels
    for k in range(K):
        curves.append(Y[k])
        if labels_names is not None and k < len(labels_names):
            curve_names.append(str(labels_names[k]))
        else:
            curve_names.append(f"y{k}")

    # Input summaries
    if input_summary != "none":
        S = window_summary(X, summary=input_summary)
        if S.ndim == 1:
            curves.append(S)
            curve_names.append(f"X:{input_summary}")
        else:
            d = S.shape[1]
            for j in range(d):
                curves.append(S[:, j])
                if inputs_names is not None and j < len(inputs_names):
                    curve_names.append(f"X:{input_summary}[{inputs_names[j]}]")
                else:
                    curve_names.append(f"X:{input_summary}[dim{j}]")

    C = np.vstack([np.asarray(c, dtype=float) for c in curves])  # (M,N)
    if normalize == "zscore":
        C = _zscore_rows(C)

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(C.shape[0]):
        ls = "-" if i < K else "--"
        ax.plot(t, C[i], linestyle=ls, label=curve_names[i], alpha=0.9)

    ax.set_xlabel("window index")
    ax.set_ylabel("value (z-scored)" if normalize == "zscore" else "value")
    if title is not None:
        ax.set_title(title)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_windows_heatmap_with_labels(
    X: np.ndarray,
    y: np.ndarray,
    *,
    t: Optional[np.ndarray] = None,
    normalize_heatmap: bool = True,
    overlay_normalize: Literal["none", "zscore"] = "zscore",
    title: Optional[str] = None,
    labels_names: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Single-figure visualization: heatmap of *flattened windows* + overlay label curves.

    Heatmap:
      rows = flattened window features (w*d)
      cols = windows (N)

    Labels are overlaid on a twin y-axis.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"X must have shape (N,w,d), got {X.shape}")
    N, w, d = X.shape

    Y = _as_2d_labels(y)  # (K,N)
    K, Ny = Y.shape
    if Ny != N:
        raise ValueError(f"X has N={N} windows but y has N={Ny}")

    if t is None:
        t = np.arange(N)
    else:
        t = np.asarray(t)
        if t.shape[0] != N:
            raise ValueError("t must have length N")

    # Heatmap data: (w*d, N)
    H = X.reshape(N, w * d).T

    if normalize_heatmap:
        mu = H.mean(axis=1, keepdims=True)
        sd = H.std(axis=1, keepdims=True)
        sd = np.maximum(sd, 1e-12)
        H = (H - mu) / sd

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        H,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[t[0], t[-1], 0, H.shape[0]],
    )
    ax.set_xlabel("window index")
    ax.set_ylabel("flattened window feature index")
    if title is not None:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("window (normalized)" if normalize_heatmap else "window value")

    ax2 = ax.twinx()
    Yp = Y.copy()
    if overlay_normalize == "zscore":
        Yp = _zscore_rows(Yp)

    for k in range(K):
        name = str(labels_names[k]) if (labels_names is not None and k < len(labels_names)) else f"y{k}"
        ax2.plot(t, Yp[k], label=name, linewidth=2.0)

    ax2.set_ylabel("labels (z-scored)" if overlay_normalize == "zscore" else "labels")
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def _reconstruct_time_series_from_windows(
    X: np.ndarray,
    *,
    burn_in: int,
    s: int,
    include_gaps: bool = True,
    start_window: int = 0,
    max_windows: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a chronological (t, X_t) view from windowed data X of shape (N,w,d).

    We use the exact indexing implied by BetaMixingGenerator:
      window i covers times [burn_in + i*(w+s), ..., burn_in + i*(w+s) + (w-1)].

    If include_gaps=True and s>0, we insert NaNs for the s missing time steps
    between consecutive windows so matplotlib draws breaks in the curves.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"X must be (N,w,d), got {X.shape}")

    N, w, d = X.shape
    if start_window < 0 or start_window >= N:
        raise ValueError(f"start_window={start_window} out of range [0,{N-1}]")

    end_window = N if max_windows is None else min(N, start_window + int(max_windows))
    stride = w + int(s)

    t_blocks = []
    x_blocks = []

    for i in range(start_window, end_window):
        t0 = int(burn_in) + i * stride
        t_block = t0 + np.arange(w, dtype=int)           # (w,)
        x_block = X[i]                                   # (w,d)
        t_blocks.append(t_block)
        x_blocks.append(x_block)

        # insert a NaN gap so the line breaks visually (optional)
        if include_gaps and s > 0 and i < end_window - 1:
            t_gap = t0 + w + np.arange(s, dtype=int)     # (s,)
            x_gap = np.full((s, d), np.nan, dtype=float)
            t_blocks.append(t_gap)
            x_blocks.append(x_gap)

    t = np.concatenate(t_blocks, axis=0)
    x = np.concatenate(x_blocks, axis=0)
    return t, x


# ---------------------------
# DataVisualizer class
# ---------------------------

@dataclass
class DataVisualizer:
    """
    Utility class to load a saved DatasetArtifact and generate standard plots.

    Parameters
    ----------
    dataset_path : str | Path
        Path to a dataset directory or dataset file (npz or *.X.npy).
    instantiate_functionals : bool
        Whether to instantiate label functionals from the sidecar config yaml (if present).
        This is used only to auto-name the label curves; plots work either way.
    """
    dataset_path: str | Path
    instantiate_functionals: bool = True

    ds: Optional[WindowsDataset] = None
    art: Optional[DatasetArtifact] = None

    def load(self) -> "DataVisualizer":
        ds, art = load_windows_dataset(self.dataset_path, instantiate_functionals=self.instantiate_functionals)
        self.ds = ds
        self.art = art
        return self

    def label_names(self) -> Sequence[str]:
        """
        Try to infer label names from instantiated functionals.
        Falls back to y0,y1,... if not available.
        """
        if self.ds is None:
            raise RuntimeError("Call .load() first")

        if getattr(self.ds, "label_functionals", None):
            names = []
            for f in self.ds.label_functionals:
                names.append(type(f).__name__)
            return names

        # Fallback
        K = _as_2d_labels(self.ds.y).shape[0]
        return [f"y{k}" for k in range(K)]

    def _subset(
            self,
            *,
            start: int = 0,
            stop: Optional[int] = None,
            step: int = 1,
            max_windows: Optional[int] = None,
            select: Literal["first", "even"] = "first",
            x_axis: Literal["original", "relative"] = "original",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.ds is None:
            raise RuntimeError("Call .load() first")
        X = np.asarray(self.ds.X)
        y = np.asarray(self.ds.y)

        N = X.shape[0]
        if stop is None:
            stop = N

        idx = np.arange(start, stop, step)
        if idx.size == 0:
            raise ValueError("Empty subset selection; check start/stop/step")

        if max_windows is not None and idx.size > max_windows:
            if select == "first":
                idx = idx[:max_windows]
            elif select == "even":
                keep = np.linspace(0, idx.size - 1, num=max_windows).round().astype(int)
                idx = idx[keep]
            else:
                raise ValueError(f"Unknown select={select!r}")

        Xsub = X[idx]
        y2 = _as_2d_labels(y)
        ysub = y2[:, idx]  # (K, nsub)

        tsub = np.arange(idx.size) if x_axis == "relative" else idx
        return Xsub, ysub, tsub

    def plot_tasks(
            self,
            task_indices: Optional[Union[int, Sequence[int]]] = None,
            *,
            start: int = 0,
            stop: Optional[int] = None,
            step: int = 1,
            max_windows: Optional[int] = None,
            select: Literal["first", "even"] = "first",
            x_axis: Literal["original", "relative"] = "relative",
            input_summary: Literal["none", "last", "mean", "energy", "pca1"] = "none",
            normalize: Literal["none", "zscore"] = "zscore",
            title: Optional[str] = None,
            figsize: Tuple[int, int] = (12, 5),
    ) -> plt.Figure:
        """
        Plot the evolution of one or more label functionals across a subset of windows,
        optionally overlaying a simple summary of the input windows.

        This function unifies the previous "plot overlay" and "plot single label"
        behaviors:
          - Plot all tasks (default)
          - Plot a single task (pass an int)
          - Plot a subset of tasks (pass a list/tuple of ints)

        Parameters
        ----------
        task_indices : int | Sequence[int] | None, optional
            Which task(s) to plot.

            - If None (default), plot all tasks available in the dataset.
            - If an int k, plot only task k.
            - If a sequence of ints, plot only the selected tasks, in the provided order.

            Task indices refer to the first axis of the stored labels after canonicalization
            to shape (K, N), where K is the number of label functionals and N is the number
            of windows.

        start : int, default=0
            Start index (inclusive) of the window range to consider, expressed in the
            dataset's native window indexing.

        stop : int | None, default=None
            Stop index (exclusive) of the window range to consider. If None, uses the
            end of the dataset.

        step : int, default=1
            Step size used to form the initial index range `idx = arange(start, stop, step)`
            before optional down-selection via `max_windows`.

        max_windows : int | None, default=None
            Maximum number of windows to include in the plot.

            If None, all windows in the selected [start:stop:step] range are used.
            If an int, the index range is reduced according to `select`:

            - select="first": keep the first `max_windows` indices.
            - select="even": keep `max_windows` indices evenly spaced over the index range.

        select : {"first", "even"}, default="first"
            Strategy to reduce the number of plotted windows when `max_windows` is set.

            - "first": take the first `max_windows` windows from the selected range.
                      This preserves the early part of the trajectory.
            - "even":  take `max_windows` windows approximately evenly spaced across the
                      selected range, providing a global overview.

        x_axis : {"original", "relative"}, default="relative"
            How to label the x-axis.

            - "original": use the dataset's native window indices (e.g., 0..1023), even if
                          you subsample with `max_windows`.
            - "relative": relabel the selected windows as 0..(n_sub-1), where n_sub is the
                          number of plotted windows.

            Note: if you use select="first" with start=0, then "original" and "relative"
            coincide (0..max_windows-1).

        input_summary : {"none","last","mean","energy","pca1"}, default="none"
            Whether and how to overlay a summary of the input windows `X` (plotted as dashed
            curves) on the same axes as the labels.

            - "none": do not plot any input summary (labels only).
            - "last": plot the last vector in each window, producing d curves for d-dimensional
                      inputs (for VARMA with d=3, you will see 3 dashed curves).
            - "mean": plot the mean vector over time within each window (d dashed curves).
            - "energy": plot mean squared values per input dimension within each window
                        (d dashed curves).
            - "pca1": plot a single curve equal to the first PCA component score of the
                      flattened windows (1 dashed curve).

        normalize : {"none", "zscore"}, default="zscore"
            Optional normalization applied to curves for visualization.

            - "none": plot curves on their original scale.
            - "zscore": z-score each plotted curve independently (subtract mean, divide by
                        standard deviation) so labels and input summaries can be compared
                        visually on one axis.

        title : str | None, default=None
            Figure title. If None, a title is autogenerated depending on the selected tasks.

        figsize : (int, int), default=(12, 5)
            Matplotlib figure size in inches, passed to `plt.subplots(figsize=...)`.

        Returns
        -------
        matplotlib.figure.Figure
            The created matplotlib Figure. The caller can display it (e.g., `fig.show()`)
            or save it (e.g., `fig.savefig(...)`).

        Raises
        ------
        RuntimeError
            If called before `DataVisualizer.load()` (i.e., no dataset is loaded).
        ValueError
            If task indices are out of range, or if the window range selection is empty.
        """
        if self.ds is None:
            raise RuntimeError("Call .load() first")

        Xsub, ysub, tsub = self._subset(
            start=start,
            stop=stop,
            step=step,
            max_windows=max_windows,
            select=select,
            x_axis=x_axis,
        )  # ysub is (K, nsub)

        K = ysub.shape[0]

        # Resolve which tasks to plot
        if task_indices is None:
            tasks = list(range(K))
        elif isinstance(task_indices, int):
            tasks = [int(task_indices)]
        else:
            tasks = [int(k) for k in task_indices]

        # Validate & unique-preserve order
        seen = set()
        tasks_clean = []
        for k in tasks:
            if k in seen:
                continue
            if not (0 <= k < K):
                raise ValueError(f"task index {k} out of range [0,{K - 1}]")
            tasks_clean.append(k)
            seen.add(k)
        tasks = tasks_clean

        # Slice labels and names
        all_names = list(self.label_names()) if self.label_names() else [f"y{i}" for i in range(K)]
        yplot = ysub[tasks, :]  # (len(tasks), nsub)
        names = [all_names[k] for k in tasks]

        if title is None:
            if len(tasks) == K:
                title = "Label evolution (all tasks)"
            elif len(tasks) == 1:
                title = f"Label evolution: {names[0]}"
            else:
                title = f"Label evolution: {', '.join(names)}"

        return plot_labels_vs_window_index(
            Xsub,
            yplot,
            t=tsub,
            input_summary=input_summary,  # "none" -> labels only
            normalize=normalize,
            title=title,
            labels_names=names,
            inputs_names=("x1", "x2", "x3"),
            figsize=figsize,
        )

    def plot_heatmap(
        self,
        *,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 1,
        max_windows: int = 400,
        normalize_heatmap: bool = True,
        overlay_normalize: Literal["none", "zscore"] = "zscore",
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Heatmap of flattened windows + label curves overlay.

        Default max_windows=400 to keep the heatmap readable and fast.
        """
        Xsub, ysub, tsub = self._subset(start=start, stop=stop, step=step, max_windows=max_windows)
        if title is None and self.art is not None:
            title = f"Heatmap: {self.art.root.name}"
        return plot_windows_heatmap_with_labels(
            Xsub,
            ysub,
            t=tsub,
            normalize_heatmap=normalize_heatmap,
            overlay_normalize=overlay_normalize,
            title=title,
            labels_names=self.label_names(),
            figsize=figsize,
        )

@dataclass
class ProcessVisualizer:
    """
    Load a saved windows dataset (DatasetArtifact) and plot the underlying process view.

    Note: the full latent path is not stored; we reconstruct a chronological view from
    the stored windows. This is exact on the window segments, and shows gaps if s>0.
    """
    dataset_path: Union[str, Path]
    instantiate_functionals: bool = False

    ds: Optional[WindowsDataset] = None

    def load(self) -> "ProcessVisualizer":
        ds, _ = load_windows_dataset(self.dataset_path, instantiate_functionals=self.instantiate_functionals)
        self.ds = ds
        return self

    def plot_process(
            self,
            *,
            dims: Optional[Sequence[int]] = None,
            start_window: int = 0,
            max_windows: Optional[int] = 50,
            include_gaps: bool = True,
            x_axis: Literal["time", "relative"] = "time",
            title: Optional[str] = None,
            figsize: Tuple[int, int] = (12, 4),
    ) -> plt.Figure:
        """
        Plot X_t (d curves) in chronological order.

        Parameters
        ----------
        dims:
            Which dimensions to plot (e.g., [0,1,2]). If None, plot all d dims.
        start_window:
            First window index to use.
        max_windows:
            Number of windows to include (None -> all windows).
        include_gaps:
            If True and s>0, insert NaNs for the s missing time steps between windows.
        x_axis:
            - "time": use true time indices t (burn_in + i*(w+s) + k)
            - "relative": relabel points 0..T-1 (after concatenation)
        """
        if self.ds is None:
            raise RuntimeError("Call .load() first")

        X = np.asarray(self.ds.X, dtype=float)
        N, w, d = X.shape

        burn_in = int(self.ds.meta.get("burn_in", 0))
        s = int(self.ds.meta.get("s", 0))

        t, x = _reconstruct_time_series_from_windows(
            X,
            burn_in=burn_in,
            s=s,
            include_gaps=include_gaps,
            start_window=start_window,
            max_windows=max_windows,
        )

        if x_axis == "relative":
            t_plot = np.arange(len(t), dtype=int)
        else:
            t_plot = t

        if dims is None:
            dims = list(range(d))
        dims = [int(j) for j in dims]
        for j in dims:
            if not (0 <= j < d):
                raise ValueError(f"dim {j} out of range [0,{d - 1}]")

        fig, ax = plt.subplots(figsize=figsize)
        for j in dims:
            ax.plot(t_plot, x[:, j], label=f"x[{j}]")

        ax.set_xlabel("t" if x_axis == "time" else "index")
        ax.set_ylabel("X_t")
        if title is None:
            title = f"VARMA process view from windows (start_window={start_window}, max_windows={max_windows})"
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=min(len(dims), 3), fontsize=9)
        fig.tight_layout()
        return fig
