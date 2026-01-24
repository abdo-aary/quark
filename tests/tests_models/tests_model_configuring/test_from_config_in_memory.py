"""
Fast in-memory wiring tests for QRCMaternKRRRegressor.from_config.

These tests do NOT read YAML files from disk and are meant to validate:
- config shape expectations,
- forwarding of PUBS parameters and runtime kwargs,
- robustness to optional sections being missing.
"""

from __future__ import annotations

import pytest

from src.models.qrc_matern_krr import QRCMaternKRRRegressor
from .conftest import make_in_memory_experiment_cfg


@pytest.mark.parametrize("angle_positioning", ["linear", "tanh"])
def test_from_config_accepts_various_angle_positioning_fast(
        patch_qrc_featurizer, angle_positioning: str
):
    """
    Ensure `.from_config()` does not crash when `angle_positioning` varies.

    Parameters
    ----------
    patch_qrc_featurizer : fixture
        Patches QRCFeaturizer to a dummy to keep test lightweight.
    angle_positioning : str
        Angle positioning option under test.

    Returns
    -------
    None
    """
    cfg = make_in_memory_experiment_cfg(angle_positioning=angle_positioning)
    model = QRCMaternKRRRegressor.from_config(cfg)

    assert model.featurizer.angle_positioning_name == angle_positioning


def test_from_config_forwards_runner_and_fmp_kwargs(patch_qrc_featurizer):
    """
    Validate that runtime kwargs are forwarded into the featurizer object.

    Returns
    -------
    None
    """
    cfg = make_in_memory_experiment_cfg(include_runner_kwargs=True, include_fmp_kwargs=True)
    model = QRCMaternKRRRegressor.from_config(cfg)

    assert model.featurizer.runner_kwargs["device"] == "CPU"
    assert model.featurizer.fmp_kwargs["batch_size"] == 16


def test_from_config_handles_missing_optional_kwargs(patch_qrc_featurizer):
    """
    Validate that missing `runner_kwargs` and `features.kwargs` do not crash.

    Returns
    -------
    None
    """
    cfg = make_in_memory_experiment_cfg(include_runner_kwargs=False, include_fmp_kwargs=False)
    model = QRCMaternKRRRegressor.from_config(cfg)

    assert model.featurizer.runner_kwargs == {}
    assert model.featurizer.fmp_kwargs == {}


def test_from_config_accepts_model_node_directly(patch_qrc_featurizer):
    """
    Validate that passing `cfg.model` (instead of full cfg) works.

    Returns
    -------
    None
    """
    cfg = make_in_memory_experiment_cfg()
    model_cfg = cfg.model
    model = QRCMaternKRRRegressor.from_config(model_cfg)

    assert model.standardize is True
    assert model.test_ratio == pytest.approx(0.2)


def test_from_config_reads_training_block(patch_qrc_featurizer):
    """
    Validate that training.split + training.preprocess are consumed properly.

    Returns
    -------
    None
    """
    cfg = make_in_memory_experiment_cfg(standardize=False, test_ratio=0.33, split_seed=7)
    model = QRCMaternKRRRegressor.from_config(cfg)

    assert model.standardize is False
    assert model.test_ratio == pytest.approx(0.33)
    assert model.split_seed == 7
