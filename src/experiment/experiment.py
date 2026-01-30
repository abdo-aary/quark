"""High-level experiment wrapper.

The goal of :class:`~src.experiment.experiment.Experiment` is to provide a thin,
test-friendly orchestration layer for the numerical validation section.

Responsibilities (initial version)
---------------------------------
1) Load a :class:`~src.data.generate.base.WindowsDataset` from disk.
2) Instantiate a :class:`~src.models.qrc_matern_krr.QRCMaternKRRRegressor` either
   from an OmegaConf/Hydra :class:`~omegaconf.DictConfig` or from an already
   constructed regressor.
3) Train the regressor via :meth:`~src.models.qrc_matern_krr.QRCMaternKRRRegressor.fit`.
4) (Placeholder) Compute per-functional metrics.
5) Persist the model via the regressor's persistence interface.

This class is intentionally minimalist: it does not aim to replace Hydra scripts;
instead it provides an object-oriented API that can be composed from notebooks,
unit tests, and future CLI entrypoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig

from src.data.factory import DatasetArtifact, load_windows_dataset
from src.data.generate.base import WindowsDataset
from src.models.qrc_matern_krr import QRCMaternKRRRegressor

_FRIENDLY_FUNCTIONAL_NAMES_BY_CLASS = {
    "OneStepForecastFunctional": "Single step forecasting",
    "ExpFadingLinearFunctional": "Exponential fading",
    "VolteraFunctional": "Voltera",
}

_FRIENDLY_FUNCTIONAL_NAMES_BY_KIND = {
    # Order is the same as your e2_three functional list
    "e2_three": ["Single step forecasting", "Exponential fading", "Voltera"],
}

def _default_functional_names(ds: WindowsDataset) -> list[str]:
    """Best-effort functional names for reporting.

    Priority order:
    1) friendly names from instantiated label functionals (if available)
    2) explicit meta["functionals_names"] (if present)
    3) friendly names from meta["functionals_kind"] (if known)
    4) meta["functionals_kind"] + index
    5) fallback "functional_{i}"
    """
    L = int(ds.y.shape[0]) if getattr(ds.y, "ndim", 0) >= 2 else 1

    # 1) If we have instantiated functionals, name by class (then map to friendly)
    if ds.label_functionals and len(ds.label_functionals) == L:
        raw = [lf.__class__.__name__ for lf in ds.label_functionals]
        return [_FRIENDLY_FUNCTIONAL_NAMES_BY_CLASS.get(r, r) for r in raw]

    kind = None
    names = None
    if isinstance(ds.meta, dict):
        kind = ds.meta.get("functionals_kind")
        # 2) Optional explicit names stored in meta (future-proof)
        names = ds.meta.get("functionals_names")

    if isinstance(names, (list, tuple)) and len(names) == L and all(isinstance(x, str) for x in names):
        return list(names)

    if kind is None:
        return [f"functional_{i}" for i in range(L)]

    # 3) Known kind -> friendly names
    if kind in _FRIENDLY_FUNCTIONAL_NAMES_BY_KIND:
        friendly = _FRIENDLY_FUNCTIONAL_NAMES_BY_KIND[kind]
        if len(friendly) == L:
            return list(friendly)

    # 4) Default
    return [f"{kind}[{i}]" for i in range(L)]


@dataclass
class Experiment:
    """Orchestrate dataset loading, model training, and persistence."""

    dataset: WindowsDataset
    model: QRCMaternKRRRegressor
    dataset_artifact: Optional[DatasetArtifact] = None

    # Results of the last regularization sweep executed via :meth:`run_reg_sweep`.
    # Stored here to make plotting and downstream reporting easy without re-running
    # the sweep.
    reg_sweep_: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)

    @classmethod
    def from_paths(
            cls,
            *,
            dataset_path: str | Path,
            model_cfg: DictConfig | None = None,
            model: QRCMaternKRRRegressor | None = None,
            instantiate_functionals: bool = True,
    ) -> "Experiment":
        """Build an experiment from a dataset path plus either a model cfg or model instance."""

        ds, art = load_windows_dataset(dataset_path, instantiate_functionals=instantiate_functionals)

        if model is None:
            if model_cfg is None:
                raise ValueError("You must provide either model_cfg or model")
            model = QRCMaternKRRRegressor.from_config(model_cfg)

        return cls(dataset=ds, model=model, dataset_artifact=art)

    def fit(self, *, num_workers: int = 1, **fit_kwargs: Any) -> "Experiment":
        """Train the wrapped model on the loaded dataset."""
        self.model.fit(self.dataset.X, self.dataset.y, num_workers=int(num_workers), **fit_kwargs)
        return self

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics per functional.

        Placeholder: returns an empty dict per functional.
        """
        names = _default_functional_names(self.dataset)
        return {name: {} for name in names}

    def save_model(self, path: str | Path) -> None:
        """Persist the trained model.

        This delegates to :meth:`src.models.qrc_matern_krr.QRCMaternKRRRegressor.save`.
        """
        self.model.save(path)

    def load_model(self, path: str | Path, *, featurizer=None, **init_kwargs: Any) -> QRCMaternKRRRegressor:
        """Load a persisted model artifact and attach it to this experiment.

        This delegates to :meth:`src.models.qrc_matern_krr.QRCMaternKRRRegressor.load` and then
        *rehydrates* ``X_train_features_`` from the cached ``Phi_full_`` and stored train indices,
        so that :meth:`~src.models.qrc_matern_krr.QRCMaternKRRRegressor.predict_from_features` works
        immediately without re-running the featurizer.

        Parameters
        ----------
        path:
            Directory containing ``arrays.npz`` and ``meta.json`` created by ``model.save``.
        featurizer:
            Optional featurizer to inject into the loaded model (only needed if you plan to call
            ``model.predict(X)`` on new windows).
        init_kwargs:
            Forwarded to the regressor constructor in ``QRCMaternKRRRegressor.load``.

        Returns
        -------
        QRCMaternKRRRegressor
            The loaded model instance (also stored in ``self.model``).
        """
        loaded = QRCMaternKRRRegressor.load(path, featurizer=featurizer, **init_kwargs)

        # Rehydrate training features needed for kernel prediction.
        Phi_tr = loaded.Phi_full_[loaded.train_idx_]
        Phi_tr = loaded.scaler_.transform(Phi_tr) if loaded.scaler_ is not None else Phi_tr
        loaded.X_train_features_ = Phi_tr

        self.model = loaded
        return loaded

    def run_reg_sweep(self, reg_grid, **sweep_kwargs: Any) -> Dict[str, Any]:
        """Run a post-fit sweep over ridge regularization values.
        Thin wrapper around
        `QRCMaternKRRRegressor.sweep_regularization`. Assumes `.fit()` was called
        already (so xi/nu are fixed and cached features are available).
        """

        out = self.model.sweep_regularization(reg_grid, **sweep_kwargs)
        self.reg_sweep_ = out
        return out

    def run_reg_sweep_from_cfg(self, cfg: DictConfig | dict) -> Optional[Dict[str, Any]]:
        """Run the sweep if enabled in a Hydra/OmegaConf config.
        Expected:
            cfg.reg_sweep.enabled: bool
            cfg.reg_sweep.reg_grid: list[float]
        """
        reg_sweep = None

        if isinstance(cfg, DictConfig):
            reg_sweep = cfg.get("reg_sweep", None)
        elif isinstance(cfg, dict):
            reg_sweep = cfg.get("reg_sweep", None)

        if reg_sweep is None:
            return None
        enabled = bool(reg_sweep.get("enabled", True))

        if not enabled:
            return None

        reg_grid = reg_sweep.get("reg_grid")

        if reg_grid is None:
            raise ValueError("cfg.reg_sweep.reg_grid is required when reg_sweep.enabled=true")

        return self.run_reg_sweep(reg_grid)

    def build_reg_sweep_metrics_table(self) -> list[dict[str, Any]]:
        """Build a long-form metrics table from the latest regularization sweep.

        The returned structure is intentionally simple (a list of row dicts) so it can be
        serialized to CSV/JSON and re-plotted without re-running the sweep.

        Returns
        -------
        list[dict[str, Any]]
            Rows with keys: ``functional``, ``split``, ``lambda_reg``, ``mse``.
        """
        if self.reg_sweep_ is None:
            raise RuntimeError("No sweep results found. Run `run_reg_sweep(...)` first.")

        import numpy as np

        reg = np.asarray(self.reg_sweep_.get("reg_grid"), dtype=float).reshape(-1)
        if reg.size == 0:
            raise RuntimeError("reg_sweep_ contains an empty reg_grid.")

        names = _default_functional_names(self.dataset)
        L = len(names)
        R = int(reg.size)

        def _as_LR(arr, *, arr_name: str) -> np.ndarray:
            a = np.asarray(arr, dtype=float)
            if a.ndim == 1:
                if L != 1:
                    raise ValueError(f"{arr_name} is 1D but dataset has L={L} functionals.")
                if a.shape != (R,):
                    raise ValueError(f"{arr_name} expected shape (R,) with R={R}, got {a.shape}.")
                return a.reshape(1, R)
            if a.ndim == 2:
                if a.shape == (L, R):
                    return a
                if a.shape == (R, L):
                    return a.T
                raise ValueError(
                    f"{arr_name} expected shape (L,R)=({L},{R}) or (R,L)=({R},{L}), got {a.shape}."
                )
            raise ValueError(f"{arr_name} expected 1D or 2D array, got ndim={a.ndim} with shape {a.shape}.")

        mse_train = _as_LR(self.reg_sweep_.get("mse_train"), arr_name="mse_train")
        mse_test = _as_LR(self.reg_sweep_.get("mse_test"), arr_name="mse_test")

        rows: list[dict[str, Any]] = []
        for li, fname in enumerate(names):
            for ri, lam in enumerate(reg.tolist()):
                rows.append(
                    {"functional": fname, "split": "train", "lambda_reg": float(lam), "mse": float(mse_train[li, ri])}
                )
                rows.append(
                    {"functional": fname, "split": "test", "lambda_reg": float(lam), "mse": float(mse_test[li, ri])}
                )
        return rows

    def save_reg_sweep_artifacts(
            self,
            out_dir: str | Path,
            *,
            formats: tuple[str, ...] = ("pdf", "png"),
            csv_name: str = "reg_sweep_metrics.csv",
            train_plot_name: str = "reg_sweep_train",
            test_plot_name: str = "reg_sweep_test",
    ) -> Dict[str, Path]:
        """Save CSV + train/test plots for the latest regularization sweep.

        Parameters
        ----------
        out_dir:
            Output directory (created if missing).
        formats:
            Image formats to save for plots, e.g. ("pdf", "png").
        csv_name:
            File name for the CSV metrics table.
        train_plot_name:
            Base name (without extension) for the train plot.
        test_plot_name:
            Base name (without extension) for the test plot.

        Returns
        -------
        Dict[str, Path]
            Paths to written artifacts. Keys: ``csv``, ``train_plot_<fmt>``, ``test_plot_<fmt>``.
        """
        if self.reg_sweep_ is None:
            raise RuntimeError("No sweep results found. Run `run_reg_sweep(...)` first.")

        import csv
        import numpy as np

        # Ensure headless plotting in CI.
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        reg = np.asarray(self.reg_sweep_.get("reg_grid"), dtype=float).reshape(-1)
        if reg.size == 0:
            raise RuntimeError("reg_sweep_ contains an empty reg_grid.")

        names = _default_functional_names(self.dataset)
        L = len(names)
        R = int(reg.size)

        def _as_LR(arr, *, arr_name: str) -> np.ndarray:
            a = np.asarray(arr, dtype=float)
            if a.ndim == 1:
                if L != 1:
                    raise ValueError(f"{arr_name} is 1D but dataset has L={L} functionals.")
                if a.shape != (R,):
                    raise ValueError(f"{arr_name} expected shape (R,) with R={R}, got {a.shape}.")
                return a.reshape(1, R)
            if a.ndim == 2:
                if a.shape == (L, R):
                    return a
                if a.shape == (R, L):
                    return a.T
                raise ValueError(
                    f"{arr_name} expected shape (L,R)=({L},{R}) or (R,L)=({R},{L}), got {a.shape}."
                )
            raise ValueError(f"{arr_name} expected 1D or 2D array, got ndim={a.ndim} with shape {a.shape}.")

        mse_train = _as_LR(self.reg_sweep_.get("mse_train"), arr_name="mse_train")
        mse_test = _as_LR(self.reg_sweep_.get("mse_test"), arr_name="mse_test")

        # 1) CSV
        rows = self.build_reg_sweep_metrics_table()
        csv_path = out_dir / csv_name
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["functional", "split", "lambda_reg", "mse"])
            writer.writeheader()
            writer.writerows(rows)

        artifacts: Dict[str, Path] = {"csv": csv_path}

        def _plot(split: str, y: np.ndarray, base_name: str) -> None:
            fig, ax = plt.subplots()
            for li, fname in enumerate(names):
                ax.plot(reg, y[li], label=fname)
            ax.set_xscale("log")
            ax.set_xlabel(r"$\lambda_{\mathrm{reg}}$")
            ax.set_ylabel("MSE")
            ax.set_title(f"Regularization sweep ({split})")
            ax.legend()
            fig.tight_layout()
            for fmt in formats:
                fmt = fmt.lower().lstrip(".")
                p = out_dir / f"{base_name}.{fmt}"
                fig.savefig(p)
                artifacts[f"{split}_plot_{fmt}"] = p
            plt.close(fig)

        _plot("train", mse_train, train_plot_name)
        _plot("test", mse_test, test_plot_name)

        return artifacts
