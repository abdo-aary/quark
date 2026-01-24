"""
Disk-based wiring test: compose actual Hydra configs and call `.from_config()`.

This test reads your real YAML config tree under src/experiment/conf (or experiment/conf),
but overrides heavy components to lightweight dummy targets.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from src.models.qrc_matern_krr import QRCMaternKRRRegressor

import tests.tests_models.tests_model_configuring.dummies as dummies


def _pick_top_config_name(conf_root: Path) -> str:
    """
    Pick the top-level experiment config name.

    Parameters
    ----------
    conf_root : pathlib.Path
        Hydra config root directory.

    Returns
    -------
    str
        Config name (without extension).
    """
    if (conf_root / "experiment.yaml").exists():
        return "experiment"
    if (conf_root / "config.yaml").exists():
        return "config"
    raise FileNotFoundError(f"No experiment.yaml or config.yaml found under {conf_root}")


def test_from_config_with_disk_yaml_wiring(hydra_conf_root: Path):
    """
    Compose disk configs and verify `.from_config()` succeeds.

    Parameters
    ----------
    hydra_conf_root : pathlib.Path
        Provided by fixture; points to config root.

    Returns
    -------
    None
    """
    top = _pick_top_config_name(hydra_conf_root)
    dummy_mod = dummies.__name__  # e.g. "tests_model_configuring.dummies"

    with initialize_config_dir(version_base=None, config_dir=str(hydra_conf_root)):
        cfg = compose(
            config_name=top,
            overrides=[
                # override heavy objects to dummies (instantiable without Qiskit Aer)
                f"model.qrc.cfg._target_={dummy_mod}.DummyQRConfig",
                "model.qrc.cfg.input_dim=2",
                "model.qrc.cfg.num_qubits=3",
                "model.qrc.cfg.seed=12345",
                f"model.qrc.runner._target_={dummy_mod}.DummyRunner",
                f"model.qrc.features.retriever._target_={dummy_mod}.DummyRetriever",
                f"model.qrc.features.observables._target_={dummy_mod}.make_dummy_observables",
                "model.qrc.features.observables.locality=2",
                "model.qrc.features.observables.num_qubits=3",
            ],
        )

    model = QRCMaternKRRRegressor.from_config(cfg)
    assert model.featurizer is not None
