from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from src.settings import PROJECT_ROOT_PATH

from src.experiment.experiment import Experiment

os.environ["PROJECT_ROOT"] = str(Path(PROJECT_ROOT_PATH))
log = logging.getLogger(__name__)


def _save_sweep_npz(sweep: Dict[str, Any], path: Path) -> None:
    """Persist sweep results to a compressed NPZ (numpy-friendly artifact)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {}

    # Required / expected fields
    for k in ("reg_grid", "mse_train", "mse_test"):
        if k in sweep and sweep[k] is not None:
            payload[k] = np.asarray(sweep[k])

    # Optional fields (save if present)
    for k in ("alpha_grid", "rkhs_norm", "extra"):
        if k in sweep and sweep[k] is not None:
            try:
                payload[k] = np.asarray(sweep[k])
            except Exception:
                # keep it JSON-ish if not array-like
                payload[k] = np.array([sweep[k]], dtype=object)

    np.savez_compressed(path, **payload)


@hydra.main(version_base=None, config_path="../conf", config_name="reg_sweep_experiment")
def main(cfg: DictConfig) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # ---- 0) Validate inputs ----
    dataset_path = cfg.get("dataset_path", None)
    if dataset_path in (None, "???", ""):
        raise ValueError(
            "cfg.dataset_path is required. Provide it via CLI: "
            "dataset_path=/path/to/dataset_artifact"
        )

    instantiate_functionals = bool(cfg.get("instantiate_functionals", True))
    num_workers = int(cfg.get("num_workers", 1))

    # ---- 1) Load dataset + 2) Build model ----
    exp = Experiment.from_paths(
        dataset_path=dataset_path,
        model_cfg=cfg.model,
        instantiate_functionals=instantiate_functionals,
    )

    # ---- 3) Fit once (fixes kernel hyperparams, caches features, etc.) ----
    exp.fit(num_workers=num_workers)

    # ---- 4) (Optional) persist fitted model ----
    output_cfg = cfg.get("output", {})
    model_dir = output_cfg.get("model_dir", None)
    if model_dir:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        exp.save_model(model_dir)
        log.info("Saved fitted model to: %s", model_dir)

    # ---- 5) Regularization sweep (post-fit) ----
    reg_sweep_cfg = cfg.get("reg_sweep", {})
    if not bool(reg_sweep_cfg.get("enabled", True)):
        log.info("reg_sweep.enabled=false, stopping after fit.")
        return

    reg_grid = reg_sweep_cfg.get("reg_grid", None)
    if reg_grid is None:
        raise ValueError("cfg.reg_sweep.reg_grid is required when reg_sweep.enabled=true")

    # Prefer the Experiment wrapper if available, else fall back to the model method.
    if hasattr(exp, "run_reg_sweep"):
        sweep_out = exp.run_reg_sweep(reg_grid)
    else:
        sweep_out = exp.model.sweep_regularization(reg_grid)
        # keep compatibility with artifact methods
        setattr(exp, "reg_sweep_", sweep_out)

    log.info("Sweep completed. Keys: %s", list(sweep_out.keys()))

    # Save sweep arrays (npz)
    sweep_path = output_cfg.get("sweep_path", None)
    if sweep_path:
        sweep_path = Path(sweep_path)
        _save_sweep_npz(sweep_out, sweep_path)
        log.info("Saved sweep arrays to: %s", sweep_path)

    # ---- 6) Save plots + CSV artifacts ----
    artifacts_dir = output_cfg.get("artifacts_dir", None)
    if artifacts_dir is None:
        # default inside Hydra run directory
        artifacts_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "artifacts"
    else:
        artifacts_dir = Path(artifacts_dir)

    if hasattr(exp, "save_reg_sweep_artifacts"):
        formats = tuple(output_cfg.get("plot_formats", ("pdf", "png")))
        artifacts = exp.save_reg_sweep_artifacts(artifacts_dir, formats=formats)
        for k, p in artifacts.items():
            log.info("Wrote artifact %s: %s", k, p)
    else:
        log.warning(
            "Experiment.save_reg_sweep_artifacts not found; "
            "skipping plot/CSV generation. (Did you apply the artifacts patch?)"
        )


if __name__ == "__main__":
    main()
