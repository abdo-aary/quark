"""
Hydra entrypoint to generate and persist a synthetic dataset.

This script composes a dataset configuration from `experiment/conf/data/`
and calls :func:`src.data.factory.generate_and_save_dataset`.

Usage
-----
From the project root:

    python -m src.experiment.scripts.generate_data sampling=tiny process=varma functionals=one_step noise=none

You can also run Hydra multiruns (sweeps):

    python -m src.experiment.scripts.generate_data -m sampling.N=64,256 sampling.w=10,25

Notes
-----
- We set `hydra.job.chdir=false` in the config to keep the working directory stable.
- This module defines :func:`run` as a pure entrypoint for unit tests.
"""
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
from src.settings import PROJECT_ROOT_PATH
from src.data.factory import generate_and_save_dataset
import logging

log = logging.getLogger(__name__)

# Two levels above the directory you launched from
os.environ["PROJECT_ROOT"] = str(Path(PROJECT_ROOT_PATH))


def run(cfg: DictConfig):
    """
    Generate and save a dataset from a composed Hydra config.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Fully composed dataset config (typically from Hydra composition).

    Returns
    -------
    (src.data.generate.base.WindowsDataset, src.data.factory.DatasetArtifact)
        The in-memory dataset and the saved artifact descriptor.
    """
    return generate_and_save_dataset(cfg)


@hydra.main(version_base=None, config_path="../conf/data", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    CLI entrypoint (Hydra main).

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra-composed config provided by the decorator.

    Returns
    -------
    None
        Writes dataset files to disk and prints a short confirmation message.
    """
    ds, art = run(cfg)
    log.info("Saved X%s, y%s -> %s", ds.X.shape, ds.y.shape, art.data_path)


if __name__ == "__main__":
    main()
