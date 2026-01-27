"""src.qrc.run.cs_fmp_retriever

Noisy (classical-shadow-like) feature maps from exact density matrices.

This module provides :class:`CSFeatureMapsRetriever`, which wraps an exact expectation
retriever and adds a simple **shot noise + median-of-means** (MoM) aggregation model.

It is intended for *simulation-based* experiments where you want to mimic shot noise
without implementing full classical-shadow tomography.

Model:
- For each observable expectation μ = Tr(ρ O) with μ ∈ [-1, 1], simulate ±1 outcomes:
  P(X=+1) = (1+μ)/2.
- Aggregate ``shots`` samples using median-of-means by splitting into ``n_groups`` blocks.

The returned feature matrix has the same shape and ordering as
:class:`~src.qrc.run.fmp_retriever.ExactFeatureMapsRetriever`.
"""

import numpy as np
from typing import Optional, Sequence

from qiskit.quantum_info import Operator, SparsePauliOp

from .fmp_retriever import BaseFeatureMapsRetriever, ExactFeatureMapsRetriever
from .circuit_run import ExactResults


class CSFeatureMapsRetriever(BaseFeatureMapsRetriever):
    """Classical-shadow-like (shot-noisy) feature maps from exact density matrices.

    Parameters
    ----------
    qrc_cfg : BaseQRConfig
        Circuit configuration; used to validate state dimension.
    observables : Sequence[Operator | SparsePauliOp]
        Observables defining features.
    default_shots : int, optional
        Default number of shots if not provided to :meth:`get_feature_maps`.
    default_n_groups : int, optional
        Default number of MoM groups. If not provided, a heuristic based on ``shots`` is used.

    Notes
    -----
    Input results must be :class:`~src.qrc.run.circuit_run.ExactResults` with
    ``results.states`` of shape ``(N, R, 2**n, 2**n)``.

    Output feature matrix has shape ``(N, R*K)`` where ``K=len(observables)``.

    Noise model:
    - Assumes each observable is bounded in [-1, 1] (Pauli strings satisfy this).
    - Simulates independent ±1 measurements using a binomial model.
    - Applies median-of-means to improve robustness.

    The exact expectation values are computed using an internal
    :class:`~src.qrc.run.fmp_retriever.ExactFeatureMapsRetriever` to ensure identical
    ordering and Pauli caching behavior.
    """

    def __init__(
        self,
        qrc_cfg,
        observables: Sequence[Operator | SparsePauliOp],
        *,
        default_shots: Optional[int] = None,
        default_n_groups: Optional[int] = None,
    ):
        self.qrc_cfg = qrc_cfg
        self.observables = list(observables)
        if len(self.observables) == 0:
            raise ValueError("observables must be non-empty.")

        self.default_shots = default_shots
        self.default_n_groups = default_n_groups

        # Reuse the exact retriever to compute μ = Tr(ρ O) with the same ordering/cache.
        self._exact = ExactFeatureMapsRetriever(qrc_cfg, self.observables)

    @staticmethod
    def _pick_n_groups(shots: int) -> int:
        """Heuristic choice for the number of MoM groups.

        Parameters
        ----------
        shots : int
            Total number of shots.

        Returns
        -------
        int
            Number of groups (capped to keep compute manageable).
        """
        return max(1, min(16, int(np.sqrt(shots))))

    def get_feature_maps(
        self,
        results: "ExactResults",
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        n_groups: Optional[int] = None,
    ) -> np.ndarray:
        """Compute shot-noisy feature maps using median-of-means.

        Parameters
        ----------
        results : ExactResults
            Exact density-matrix results.
        shots : int, optional
            Number of simulated shots per observable. If omitted, ``default_shots`` is used.
        seed : int, optional
            RNG seed for reproducible noise.
        n_groups : int, optional
            Number of MoM groups. If omitted, uses ``default_n_groups`` or a heuristic.

        Returns
        -------
        np.ndarray
            Feature matrix of shape ``(N, R*K)``.

        Raises
        ------
        ValueError
            If ``shots`` is missing/non-positive, or if ``results.states`` has incompatible shape.
        """
        # ------------------------
        # validate / defaults
        # ------------------------
        if shots is None:
            shots = self.default_shots
        if shots is None:
            raise ValueError("shots must be provided (or set default_shots in constructor).")
        shots = int(shots)
        if shots <= 0:
            raise ValueError(f"shots must be positive, got {shots}.")

        # Basic qrc_cfg compatibility check
        n = int(self.qrc_cfg.num_qubits)
        dim = 1 << n
        states = np.asarray(results.states)
        if states.ndim != 4:
            raise ValueError(f"Expected results.states shape (N,R,dim,dim), got {states.shape}")
        if states.shape[2:] != (dim, dim):
            raise ValueError(f"State dim mismatch: expected {(dim, dim)}, got {states.shape[2:]}")

        N, R = states.shape[0], states.shape[1]
        K = len(self.observables)

        # ------------------------
        # exact expectations μ
        # ------------------------
        mu_flat = self._exact.get_feature_maps(results)   # (N, R*K)
        mu = mu_flat.reshape(N, R, K)

        # For Pauli observables, μ ∈ [-1,1]. Clip for numerical drift.
        mu = np.clip(mu, -1.0, 1.0)

        # ------------------------
        # MoM setup
        # ------------------------
        if n_groups is None:
            n_groups = self.default_n_groups
        if n_groups is None:
            n_groups = self._pick_n_groups(shots)

        n_groups = int(n_groups)
        n_groups = max(1, min(n_groups, shots))
        batch_size = max(shots // n_groups, 1)

        # ------------------------
        # simulate group means using Binomial model (exact for ±1 outcomes)
        #
        # If X ∈ {±1} with E[X]=μ, then P(X=+1) = (1+μ)/2.
        # For a batch of size b, count(+1) ~ Binomial(b, p),
        # and batch mean = (2*count - b) / b.
        # ------------------------
        rng = np.random.default_rng(seed)
        p = (1.0 + mu) / 2.0
        p = np.clip(p, 0.0, 1.0)

        counts = rng.binomial(
            n=batch_size,
            p=p[..., None],
            size=mu.shape + (n_groups,),
        )
        group_means = (2.0 * counts - batch_size) / float(batch_size)  # (N,R,K,G)

        # Median-of-Means aggregation
        cs_est = np.median(group_means, axis=-1)  # (N,R,K)

        fmps = cs_est.reshape(N, R * K).astype(float)
        self.fmps = fmps
        return fmps
