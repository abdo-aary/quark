"""Quantum reservoir featurization.

This module defines :class:`src.models.qrc_featurizer.QRCFeaturizer`, a lightweight transformer that turns
a window dataset ``X`` of shape ``(N, w, d)`` into a feature matrix ``Phi`` of shape ``(N, D)`` by:

1. Building a pubs dataset (circuits + parameter binds) using :class:`src.qrc.circuits.circuit_factory.CircuitFactory`.
2. Executing pubs with a :class:`src.qrc.run.circuit_run.BaseCircuitsRunner`.
3. Converting runner outputs (:class:`src.qrc.run.circuit_run.Results`) into real-valued features using a
   :class:`src.qrc.run.fmp_retriever.BaseFeatureMapsRetriever`.

The featurizer is intentionally *stateless*, so the same ``Phi`` can be reused for many downstream supervised
tasks (including multi-output labels) without rerunning circuits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from src.qrc.circuits.qrc_configs import BaseQRConfig
from src.qrc.circuits.circuit_factory import CircuitFactory
from src.qrc.circuits.utils import angle_positioning_linear, angle_positioning_tanh
from src.qrc.run.circuit_run import BaseCircuitsRunner
from src.qrc.run.fmp_retriever import BaseFeatureMapsRetriever

_ANGLE_POS_REGISTRY = {
    "linear": angle_positioning_linear,
    "tanh": angle_positioning_tanh,
}


@dataclass
class QRCFeaturizer:
    """
    Stateless quantum featurizer for window datasets.

    Parameters
    ----------
    qrc_cfg : src.qrc.circuits.qrc_configs.BaseQRConfig
        Quantum reservoir configuration (number of qubits, input dimension, seed, ...).
    runner : src.qrc.run.circuit_run.BaseCircuitsRunner
        Runner used to execute pubs and return :class:`src.qrc.run.circuit_run.Results`.
    fmp_retriever : src.qrc.run.fmp_retriever.BaseFeatureMapsRetriever
        Retriever that converts runner results into a feature matrix ``Phi``.
    pubs_family : str
        Name of the pubs family used to build circuits (e.g. ``"ising_ring_swap"``).
    angle_positioning_name : str
        Name of the angle positioning map to be applied to window values before they are used as circuit angles.
        Small numerical constant forwarded to the pubs builder (used for stability in some encodings).
    runner_kwargs : dict
        Keyword arguments forwarded to ``runner.run_pubs(...)`` (e.g. ``device="GPU"``).
    fmp_kwargs : dict
        Keyword arguments forwarded to ``fmp_retriever.get_feature_maps(...)`` (e.g. ``shots`` for CS retriever).

    Notes
    -----
    This class is conceptually similar to an ``sklearn`` transformer, but kept minimal on purpose.
    """
    qrc_cfg: BaseQRConfig
    runner: BaseCircuitsRunner
    fmp_retriever: BaseFeatureMapsRetriever
    pubs_family: str
    angle_positioning_name: str
    pubs_kwargs: Dict[str, Any]
    runner_kwargs: Dict[str, Any]
    fmp_kwargs: Dict[str, Any]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute a feature matrix for a batch of windows.

        Parameters
        ----------
        X : numpy.ndarray
            Window dataset with shape ``(N, w, d)``.

        Returns
        -------
        numpy.ndarray
            Feature matrix ``Phi`` with shape ``(N, D)``.

        Raises
        ------
        ValueError
            If the input shape is not ``(N, w, d)``.
        """
        if X.ndim != 3:
            raise ValueError(f"X must be (N,w,d). Got {X.shape}.")

        # Right now we have one concrete family in CircuitFactory
        if self.pubs_family != "ising_ring_swap":
            raise ValueError(f"Unknown pubs_family={self.pubs_family!r}. Add it to the featurizer.")

        angle_positioning = _ANGLE_POS_REGISTRY.get(self.angle_positioning_name)
        if angle_positioning is None:
            raise ValueError(f"Unknown angle_positioning={self.angle_positioning_name!r}")

        pubs = CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP(
            qrc_cfg=self.qrc_cfg,
            angle_positioning=angle_positioning,
            X=X,
            **self.pubs_kwargs,
        )

        results = self.runner.run_pubs(pubs=pubs, **self.runner_kwargs)

        # CSFeatureMapsRetriever expects shots/seed/n_groups in get_feature_maps
        Phi = self.fmp_retriever.get_feature_maps(results, **self.fmp_kwargs)
        if Phi.ndim != 2 or Phi.shape[0] != X.shape[0]:
            raise ValueError(f"Feature maps must be (N,D). Got {Phi.shape}.")
        return Phi
