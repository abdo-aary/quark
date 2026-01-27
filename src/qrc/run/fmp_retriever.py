"""src.qrc.run.fmp_retriever

Feature-map retrievers for QRC outputs.

A *feature-map retriever* converts circuit execution results into a classical
feature matrix suitable for kernel methods / regression.

Two common cases are supported:

- **Exact** feature maps from density matrices:
  ``phi(i,r,k) = Tr(rho_{i,r} O_k)``
- **Approximate / noisy** feature maps (e.g., classical shadows) can be built on top
  of exact expectations; see :mod:`src.qrc.run.cs_fmp_retriever`.

This module defines a persistence API (`save` / `load`) to make retrievers
easy to cache and reuse across experiments.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

from src.qrc.circuits.qrc_configs import BaseQRConfig
from .circuit_run import Results


class BaseFeatureMapsRetriever(ABC):
    """Abstract base class for mapping :class:`~src.qrc.run.circuit_run.Results` to features.

    Attributes
    ----------
    qrc_cfg : BaseQRConfig
        Circuit configuration (num qubits, projection, topology).
    observables : Sequence[Operator]
        Observables used to compute features. Each feature is an expectation value
        ``Tr(rho O)`` for a state ``rho``.
    fmps : np.ndarray or None
        Cached feature matrix produced by the last call to :meth:`get_feature_maps`.
    """

    qrc_cfg: BaseQRConfig
    fmps: np.ndarray = None
    observables: Sequence[Operator]

    @abstractmethod
    def get_feature_maps(self, results: Results, **kwargs) -> np.ndarray:
        """Compute feature maps from execution results.

        Parameters
        ----------
        results : Results
            Circuit execution results.
        **kwargs
            Implementation-specific options (e.g., number of shots for noisy methods).

        Returns
        -------
        np.ndarray
            Feature matrix, typically shape ``(N, D)``.
        """
        ...

    # ------------------------------------------------------------------
    # Persistence API
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialize this retriever to disk.

        Parameters
        ----------
        path : str
            Destination path for the pickle file.

        Notes
        -----
        This is intended for experiment reproducibility and caching.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BaseFeatureMapsRetriever":
        """Load a previously saved retriever from disk.

        Parameters
        ----------
        path : str
            Path to the pickle file created by :meth:`save`.

        Returns
        -------
        BaseFeatureMapsRetriever
            The reconstructed retriever instance.

        Raises
        ------
        TypeError
            If the loaded object is not a :class:`BaseFeatureMapsRetriever`.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, BaseFeatureMapsRetriever):
            raise TypeError(f"Expected a BaseFeatureMapsRetriever instance in {path!r}, got {type(obj)!r}.")
        return obj


class ExactFeatureMapsRetriever(BaseFeatureMapsRetriever):
    """Compute *exact* expectation-value feature maps from density matrices.

    Given results with ``states`` of shape ``(N, R, 2**n, 2**n)``, this retriever computes:

    ``phi[i, r*K + k] = Tr( rho[i, r] * O_k )``

    where:
    - ``K = len(observables)``
    - ``R`` is the number of reservoir parameterizations per window (spatial multiplexing)
    - ordering is: ``[obs_0..obs_{K-1} for r=0]`` then ``[obs_0..obs_{K-1} for r=1]`` etc.

    Two evaluation paths are supported:

    1. **Fast Pauli path**: if all observables are single-term :class:`SparsePauliOp`,
       expectations are computed using bitmask tricks without forming dense matrices.
    2. **Generic fallback**: computes ``Tr(rho O)`` using dense matrices (fine for small ``n``).

    Parameters
    ----------
    qrc_cfg : BaseQRConfig
        Circuit configuration. Only ``qrc_cfg.num_qubits`` is required for shape validation.
    observables : Sequence[Operator | SparsePauliOp]
        Observables to measure.

    Raises
    ------
    ValueError
        If the observable labels do not match ``num_qubits`` (Pauli fast path).
    """

    def __init__(self, qrc_cfg: BaseQRConfig, observables: Sequence[Operator | SparsePauliOp]):
        self.qrc_cfg = qrc_cfg
        self.observables = observables
        self._pauli_cache = None  # list[(xmask:int, zmask:int, ny:int)] or None
        self._dense_cache = None  # obs_mat (K,dim,dim) if needed

    @staticmethod
    def _pauli_label(op: Operator | SparsePauliOp) -> str | None:
        """Return a Pauli label like ``'IXYZ'`` if `op` is a single-term SparsePauliOp.

        Parameters
        ----------
        op : Operator or SparsePauliOp
            Observable.

        Returns
        -------
        str or None
            Pauli string label if supported by the fast path; otherwise ``None``.
        """
        if isinstance(op, SparsePauliOp):
            if len(op.paulis) == 1:
                return op.paulis[0].to_label()
        return None

    @staticmethod
    def _masks_from_label(label: str) -> tuple[int, int, int]:
        """Convert a Pauli label into bitmasks for fast expectation evaluation.

        The label follows Qiskit's convention:

        - leftmost character corresponds to qubit ``n-1``
        - rightmost character corresponds to qubit ``0``

        We convert this to (xmask, zmask, ny) such that the Pauli operator acts as:

        ``P |x⟩ = i^{ny} (-1)^{popcount(zmask & x)} |x ⊕ xmask⟩``

        Parameters
        ----------
        label : str
            Pauli string composed of characters in {'I','X','Y','Z'}.

        Returns
        -------
        tuple[int, int, int]
            ``(xmask, zmask, ny)``, where:
            - ``xmask`` has 1s on qubits with X or Y
            - ``zmask`` has 1s on qubits with Z or Y
            - ``ny`` is the number of Y's (phase factor ``i^{ny}``)
        """
        n = len(label)
        xmask = 0
        zmask = 0
        ny = 0
        for q in range(n):
            ch = label[n - 1 - q]  # qubit 0 is rightmost character
            if ch == "X":
                xmask |= (1 << q)
            elif ch == "Z":
                zmask |= (1 << q)
            elif ch == "Y":
                xmask |= (1 << q)
                zmask |= (1 << q)
                ny += 1
        return xmask, zmask, ny

    @staticmethod
    def _bitcount_parity(arr: np.ndarray) -> np.ndarray:
        """Compute parity (0/1) of the population count for each element in `arr`.

        Parameters
        ----------
        arr : np.ndarray
            Unsigned integer array.

        Returns
        -------
        np.ndarray
            Integer array with entries in {0,1} giving parity of bit-counts.
        """
        if hasattr(np, "bit_count"):
            return (np.bit_count(arr) & 1).astype(np.int8)
        v = np.vectorize(lambda x: int(x).bit_count() & 1, otypes=[np.int8])
        return v(arr)

    def _ensure_pauli_cache(self, n: int) -> None:
        """Build cached bitmask representations for Pauli observables.

        Parameters
        ----------
        n : int
            Number of qubits.

        Raises
        ------
        ValueError
            If any label length differs from ``n``.
        """
        labels = []
        for op in self.observables:
            lab = self._pauli_label(op)
            if lab is None:
                self._pauli_cache = None
                return
            if len(lab) != n:
                raise ValueError(f"Observable label length {len(lab)} != num_qubits {n}: {lab!r}")
            labels.append(lab)
        self._pauli_cache = [self._masks_from_label(lab) for lab in labels]

    def _ensure_dense_cache(self) -> None:
        """Materialize dense observable matrices for the generic fallback."""
        if self._dense_cache is None:
            self._dense_cache = np.stack([op.to_matrix() for op in self.observables], axis=0)

    def get_feature_maps(self, results: Results, **kwargs) -> np.ndarray:
        """Compute exact feature maps from results.

        Parameters
        ----------
        results : Results
            Results object with a ``states`` attribute of shape ``(N, R, 2**n, 2**n)``.

        Returns
        -------
        np.ndarray
            Feature matrix of shape ``(N, R*K)`` where ``K=len(observables)``.

        Raises
        ------
        ValueError
            If ``results.states`` has an unexpected shape or does not match ``qrc_cfg.num_qubits``.
        """
        states = np.asarray(results.states)
        if states.ndim != 4:
            raise ValueError(f"Expected states shape (N,R,dim,dim), got {states.shape}")

        N, R, dim1, dim2 = states.shape
        if dim1 != dim2:
            raise ValueError(f"Density matrices must be square; got {dim1}x{dim2}")
        n = int(self.qrc_cfg.num_qubits)
        dim = 1 << n
        if dim1 != dim:
            raise ValueError(f"Expected dim=2**n={dim}, got {dim1}. Check qrc_cfg.num_qubits vs results.states.")

        K = len(self.observables)
        if K == 0:
            out = np.zeros((N, 0), dtype=np.float64)
            self.fmps = out
            return out

        rho = states.reshape(N * R, dim, dim)  # (B,dim,dim), B=N*R

        # --- Fast path for Pauli strings ---
        if self._pauli_cache is None and all(isinstance(op, SparsePauliOp) for op in self.observables):
            self._ensure_pauli_cache(n)

        if self._pauli_cache is not None:
            rows = np.arange(dim, dtype=np.uint32)
            feats = np.empty((N * R, K), dtype=np.float64)

            for k, (xmask, zmask, ny) in enumerate(self._pauli_cache):
                cols = rows ^ np.uint32(xmask)
                vals = rho[:, rows, cols]  # (B,dim)

                parity = self._bitcount_parity(np.bitwise_and(rows, np.uint32(zmask)))
                sign = 1.0 - 2.0 * parity.astype(np.float64)
                phase = (1j ** ny) * sign  # (dim,)

                exp = np.sum(vals * phase[None, :], axis=1)  # (B,)
                feats[:, k] = exp.real

            fmps = feats.reshape(N, R, K).reshape(N, R * K)
            self.fmps = fmps
            return fmps

        # --- Generic fallback ---
        self._ensure_dense_cache()
        obs_mat = self._dense_cache  # (K,dim,dim)
        exp = np.einsum("bij,kji->bk", rho, obs_mat).real  # (B,K)
        fmps = exp.reshape(N, R, K).reshape(N, R * K)

        self.fmps = fmps
        return fmps
