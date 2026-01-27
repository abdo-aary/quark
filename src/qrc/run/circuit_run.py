"""
src.qrc.run.circuit_run
======================

Runners and result containers for executing QRC circuits.

This module defines:

- :class:`Results` and concrete subclasses such as :class:`ExactResults` that store
  simulator outputs in a consistent shape.
- :class:`BaseCircuitsRunner`, an interface for executing a list of PUBs.
- :class:`ExactAerCircuitsRunner`, an Aer-based runner that executes PUBs using
  ``AerSimulator(method="density_matrix")`` and returns reduced density matrices
  on the reservoir subsystem.

PUB format
----------
A **PUB** is represented as a ``(qc, param_values)`` tuple where:

- ``qc`` is a Qiskit :class:`~qiskit.QuantumCircuit`.
- ``param_values`` is a NumPy array of numeric parameter bindings.

Two PUB layouts are supported:

1) **Legacy PUBs** (one circuit per window)
   ``pubs`` has length ``N`` and each PUB is ``(qc_i, vals_i)`` with:

   - ``vals_i.shape == (R, P_res)``
   - ``P_res = |J| + |h_x| + |h_z| + 1``

   Here, the injected inputs are already numeric in ``qc_i`` (only reservoir
   parameters are bound).

2) **Template PUB** (recommended; one circuit for all windows)
   ``pubs`` has length ``1`` and the single PUB is ``(qc_template, vals)`` with:

   - ``vals.shape == (N, R, P_total)``
   - ``P_total = (w·n) + (|J| + |h_x| + |h_z| + 1)``

   Here, *both* injected inputs (per time-step vectors ``z_0, …, z_{w-1}``) and
   reservoir parameters are bound numerically from ``vals``.

Parameter alignment contract
----------------------------
To avoid relying on ``qc.parameters`` (which can change after transpilation),
bindings must follow a deterministic parameter order:

- If available, use ``qc.metadata["param_order"]`` (flat list of Parameters).
- Otherwise, runners fall back to rebuilding the order from metadata:
  ``list(J) + list(h_x) + list(h_z) + [lam]`` (legacy circuits).

These metadata keys are produced by the circuit factory in :mod:`src.qrc.circuits`.

Performance note
----------------
In template-PUB mode, the same parameterized circuit can be transpiled once and
executed over many parameter bindings (``N·R`` experiments). The runner supports
optional chunking to reduce peak memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict

import numpy as np
import pickle
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from src.qrc.circuits.qrc_configs import BaseQRConfig

PUB = Tuple[QuantumCircuit, np.ndarray]


def _to_array(dm_obj: Any) -> np.ndarray:
    """Convert a Qiskit density-matrix-like object to a complex NumPy array.

    Parameters
    ----------
    dm_obj : Any
        A Qiskit object that may expose a ``.data`` attribute (e.g., Aer density matrix),
        or an array-like object.

    Returns
    -------
    np.ndarray
        Complex-valued array representation of the density matrix.
    """
    if hasattr(dm_obj, "data"):
        return np.asarray(dm_obj.data, dtype=complex)
    return np.asarray(dm_obj, dtype=complex)


@dataclass
class Results(ABC):
    """Abstract container for circuit execution outputs.

    Attributes
    ----------
    qrc_cfg : BaseQRConfig
        Configuration used to build/run the circuits (topology, number of qubits, etc.).
    states : np.ndarray
        Array storing simulator outputs. The exact meaning/shape depends on the subclass.
    """

    qrc_cfg: BaseQRConfig
    states: np.ndarray

    @abstractmethod
    def save(self, file: str | Path) -> None:
        """Serialize this results object to disk.

        Parameters
        ----------
        file : str or pathlib.Path
            Destination path.

        Notes
        -----
        Subclasses should define a forward-compatible serialization format.
        """
        ...

    @staticmethod
    @abstractmethod
    def load(file: str | Path) -> "Results":
        """Load a results object from disk.

        Parameters
        ----------
        file : str or pathlib.Path
            Path to a file created by :meth:`save`.

        Returns
        -------
        Results
            A reconstructed results object.
        """
        ...


@dataclass
class ExactResults(Results):
    """Exact (density-matrix) results for a PUBS dataset.

    This class stores **reduced** density matrices on the reservoir subsystem
    for each input window and each reservoir draw.

    Let ``pubs = [(qc_i, vals_i)]_{i=1..N}``, where:
    - each ``vals_i`` has shape ``(R, P)`` with *R* reservoir parameterizations,
    - ``qc_i`` is parameterized by reservoir parameters.

    Then this class stores:

    - ``states`` of shape ``(N, R, 2**n, 2**n)``, where ``n = qrc_cfg.num_qubits``.

    Parameters
    ----------
    states : np.ndarray
        Complex array of reduced density matrices, shape ``(N, R, 2**n, 2**n)``.
    qrc_cfg : BaseQRConfig
        Configuration associated with these results.
    """

    states: np.ndarray
    qrc_cfg: BaseQRConfig

    def save(self, file: str | Path) -> None:
        """Serialize :class:`ExactResults` to disk.

        Parameters
        ----------
        file : str or pathlib.Path
            Output file path.

        Notes
        -----
        The serialization uses a dict payload (class name, states, qrc_cfg)
        to remain more forward-compatible if the dataclass layout changes.
        """
        path = Path(file)
        with path.open("wb") as f:
            pickle.dump(
                {"cls": "ExactResults", "states": self.states, "qrc_cfg": self.qrc_cfg},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @staticmethod
    def load(file: str | Path) -> "ExactResults":
        """Load :class:`ExactResults` from disk.

        Parameters
        ----------
        file : str or pathlib.Path
            Path created by :meth:`save`.

        Returns
        -------
        ExactResults
            Reconstructed results instance.

        Raises
        ------
        TypeError
            If the file contents are not recognized as ``ExactResults``.
        """
        path = Path(file)
        with path.open("rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, ExactResults):
            return obj

        if isinstance(obj, dict) and obj.get("cls") == "ExactResults":
            return ExactResults(states=obj["states"], qrc_cfg=obj["qrc_cfg"])

        raise TypeError(f"File {file} does not contain a valid ExactResults object.")


class BaseCircuitsRunner(ABC):
    """Interface for executing PUBS datasets.

    A runner takes a list of PUBs (parameterized circuits + parameter matrices)
    and returns a :class:`Results` object.
    """

    @abstractmethod
    def run_pubs(self, **kwargs) -> Results:
        """Execute a list of PUBs and return results.

        Returns
        -------
        Results
            Concrete results object (e.g., :class:`ExactResults`).
        """
        ...


class ExactAerCircuitsRunner(BaseCircuitsRunner):
    """Execute PUBS datasets using Qiskit Aer in density-matrix mode.

    The runner:

    1. Configures an :class:`~qiskit_aer.AerSimulator` backend with ``method="density_matrix"``.
    2. For each PUB, copies the circuit, ensures a density-matrix save instruction exists,
       and prepares a parameter-binds mapping.
    3. Transpiles all circuits in a single batched call to reduce Python overhead.
    4. Executes all circuits in a single Aer job using ``parameter_binds``.
    5. Extracts reduced density matrices (reservoir qubits only) into an array
       of shape ``(N, R, 2**n, 2**n)``.

    Notes
    -----
    **Parameter alignment**:

- Preferred: bind using the explicit column order in ``qc.metadata["param_order"]``.
  This is required for the template-PUB workflow, where injected inputs and reservoir
  parameters are concatenated into one vector.
- Fallback (legacy circuits): if ``param_order`` is absent, the runner assumes the
  reservoir-parameter column order:

  ``list(J) + list(h_x) + list(h_z) + [lam]``

  where these parameters are retrieved from ``qc.metadata``.

**GPU execution**:


    If Aer is built with GPU support and exposes ``available_devices()``,
    passing ``device="GPU"`` to :meth:`run_pubs` will request GPU execution.
    """

    def __init__(self, qrc_cfg: BaseQRConfig):
        """Construct the runner.

        Parameters
        ----------
        qrc_cfg : BaseQRConfig
            Circuit configuration (number of reservoir qubits, etc.).
        """
        self.qrc_cfg = qrc_cfg
        self.backend = AerSimulator(method="density_matrix")

    def run_pubs(
            self,
            pubs: List[PUB],
            max_threads: Optional[int] = None,
            seed_simulator: int = 0,
            optimization_level: int = 1,
            device: str = "CPU",
            max_parallel_threads: Optional[int] = None,
            max_parallel_experiments: Optional[int] = None,
            max_parallel_shots: Optional[int] = None,
            chunk_size: Optional[int] = None,
    ) -> ExactResults:
        """Run a list of PUBs using Aer (density-matrix) and return reduced DMs.

        Parameters
        ----------
        pubs : list[tuple[qiskit.QuantumCircuit, numpy.ndarray]]
            PUBs collection.

            **Legacy mode**:
                ``pubs`` has length ``N`` and each ``vals`` is shape ``(R, P)``.

            **Batched-template mode** (the “big fix”):
                ``pubs`` has length ``1`` and the single ``vals`` is shape ``(N, R, P_total)``.
                The same parameterized circuit template is transpiled once and executed with
                vectorized parameter binds.

        max_threads : int, optional
            Requested maximum threads for Aer. If ``None`` or < 1, uses Aer convention
            ``0`` meaning “use all available”.
        seed_simulator : int, default=0
            Seed passed to Aer simulator.
        optimization_level : int, default=1
            Transpilation optimization level.
        device : {"CPU","GPU"}, default="CPU"
            Aer execution device. GPU requires Aer GPU support.
        max_parallel_threads : int, optional
            Aer parallelism option.
        max_parallel_experiments : int, optional
            Aer parallelism option.
        max_parallel_shots : int, optional
            Aer parallelism option.
        chunk_size : int, optional
            Only used in **batched-template mode**. If provided, splits the total
            number of experiments ``N*R`` into chunks of at most ``chunk_size`` to
            reduce peak memory and job sizes.

        Returns
        -------
        ExactResults
            Reduced reservoir density matrices, shape ``(N, R, 2**n, 2**n)``.

        Notes
        -----
        Parameter binds follow ``qc.metadata["param_order"]`` when present.

        Raises
        ------
        ValueError
            If PUB formatting is inconsistent, metadata is missing, or dimensions mismatch.
        KeyError
            If Aer result objects do not contain a density matrix payload.
        """
        if not pubs:
            raise ValueError("pubs must be non-empty")

        n_res = int(getattr(self.qrc_cfg, "num_qubits"))
        dim_res = 1 << n_res

        # Aer parallelism options
        if max_threads is None or max_threads < 1:
            max_threads = 0  # Aer convention: 0 => use all available
        if max_parallel_threads is None:
            max_parallel_threads = max_threads
        if max_parallel_experiments is None:
            max_parallel_experiments = max_parallel_threads
        if max_parallel_shots is None:
            max_parallel_shots = max_parallel_threads

        self.backend.set_options(
            device=device,
            max_parallel_threads=max_parallel_threads,
            max_parallel_experiments=max_parallel_experiments,
            max_parallel_shots=max_parallel_shots,
        )

        def _param_order_from_metadata(qc: QuantumCircuit) -> List[Parameter]:
            """Resolve the expected parameter ordering from qc.metadata."""
            if qc.metadata is None:
                raise ValueError("Circuit has no metadata; cannot align parameter columns.")
            md = qc.metadata

            # Recommended new way
            if "param_order" in md:
                order = list(md["param_order"])
                if not order:
                    raise ValueError("metadata['param_order'] is empty.")
                return order

            # Legacy fallback: J, h_x, h_z, lam
            try:
                J = list(md["J"])
                hx = list(md["h_x"])
                hz = list(md["h_z"])
                lam = md["lam"]
            except KeyError as e:
                raise ValueError(
                    f"Missing metadata key {e}. Need 'param_order' or (J,h_x,h_z,lam)."
                ) from e
            return J + hx + hz + [lam]

        def _ensure_save_density_matrix(qc: QuantumCircuit) -> QuantumCircuit:
            """Copy qc and add save_density_matrix(qubits=reservoir) if missing."""
            qc_work = qc.copy()
            has_save_dm = any(
                getattr(inst.operation, "name", "") == "save_density_matrix"
                for inst in qc_work.data
            )
            if not has_save_dm:
                qc_work.save_density_matrix(qubits=list(range(n_res)))
            return qc_work

        # ------------------------------------------------------------------
        # Mode detection
        # ------------------------------------------------------------------
        qc0, vals0 = pubs[0]
        vals0 = np.asarray(vals0)

        batched_template_mode = (len(pubs) == 1) and (vals0.ndim == 3)

        if batched_template_mode:
            # ==============================================================
            # Batched-template mode: one circuit, vals shape (N, R, P)
            # ==============================================================
            N, R, P = map(int, vals0.shape)

            param_order = _param_order_from_metadata(qc0)
            if len(param_order) != P:
                raise ValueError(
                    f"param cols mismatch: template vals has {P} cols but expected {len(param_order)} "
                    f"(from circuit metadata)."
                )

            qc_work = _ensure_save_density_matrix(qc0)
            tqc = transpile(qc_work, backend=self.backend, optimization_level=optimization_level)

            # Map parameters by NAME (robust if Parameter objects differ post-transpile)
            tparam_by_name: Dict[str, Parameter] = {p.name: p for p in list(tqc.parameters)}
            order_names = [p.name for p in param_order]
            missing = [nm for nm in order_names if nm not in tparam_by_name]
            if missing:
                raise ValueError(f"Transpiled circuit missing parameters {missing}")

            states = np.empty((N, R, dim_res, dim_res), dtype=complex)

            B = N * R  # total number of experiments for the single template circuit
            if chunk_size is None or chunk_size < 1:
                chunk_size = B

            # Pre-flatten each parameter column once; slice per chunk.
            flat_cols: List[np.ndarray] = [vals0[:, :, j].reshape(B) for j in range(P)]

            offset = 0
            while offset < B:
                end = min(B, offset + chunk_size)
                chunk_len = end - offset

                bind = {
                    tparam_by_name[nm]: flat_cols[j][offset:end].tolist()
                    for j, nm in enumerate(order_names)
                }
                job = self.backend.run(
                    [tqc],
                    parameter_binds=[bind],
                    shots=1,
                    seed_simulator=seed_simulator,
                )
                result = job.result()

                for kk in range(chunk_len):
                    data_k = result.data(kk)

                    dm = data_k.get("density_matrix", None)
                    if dm is None:
                        # older Aer sometimes stores the key with a prefix
                        for key, val in data_k.items():
                            if "density_matrix" in key:
                                dm = val
                                break
                    if dm is None:
                        raise KeyError(
                            f"No density_matrix found in result.data({kk}). Keys={list(data_k.keys())}"
                        )

                    dm_arr = _to_array(dm)
                    if dm_arr.shape != (dim_res, dim_res):
                        raise ValueError(
                            f"Got DM shape {dm_arr.shape}, expected {(dim_res, dim_res)} at global_k={offset + kk}"
                        )

                    global_k = offset + kk
                    i = global_k // R
                    r = global_k % R
                    states[i, r] = dm_arr

                offset = end

            return ExactResults(states=states, qrc_cfg=self.qrc_cfg)

        # ------------------------------------------------------------------
        # Legacy mode: pubs length N, each vals shape (R, P)
        # ------------------------------------------------------------------
        N = len(pubs)
        if vals0.ndim != 2:
            raise ValueError(
                f"Legacy PUB mode expects vals.ndim==2 for each pub, got pubs[0] shape {vals0.shape}. "
                f"To use batched-template mode pass a single pub with vals shape (N,R,P)."
            )

        R = int(vals0.shape[0])

        circuits: List[QuantumCircuit] = []
        binds_list: List[Dict[Parameter, List[float]]] = []

        for i, (qc, vals) in enumerate(pubs):
            vals = np.asarray(vals)
            if vals.ndim != 2:
                raise ValueError(f"pub[{i}] params must be 2D (R,P). Got {vals.shape}")
            if vals.shape[0] != R:
                raise ValueError(f"pub[{i}] has R={vals.shape[0]} but expected R={R}")

            param_order = _param_order_from_metadata(qc)
            P = len(param_order)
            if vals.shape[1] != P:
                raise ValueError(
                    f"pub[{i}] param cols mismatch: vals has {vals.shape[1]} cols but expected {P} "
                    f"(from circuit metadata)."
                )

            qc_work = _ensure_save_density_matrix(qc)
            tqc = transpile(qc_work, backend=self.backend, optimization_level=optimization_level)
            circuits.append(tqc)

            tparam_by_name: Dict[str, Parameter] = {p.name: p for p in list(tqc.parameters)}
            order_names = [p.name for p in param_order]
            missing = [nm for nm in order_names if nm not in tparam_by_name]
            if missing:
                raise ValueError(f"pub[{i}] transpiled circuit missing parameters {missing}")

            bind = {tparam_by_name[nm]: vals[:, j].tolist() for j, nm in enumerate(order_names)}
            binds_list.append(bind)

        job = self.backend.run(
            circuits,
            parameter_binds=binds_list,
            shots=1,
            seed_simulator=seed_simulator,
        )
        result = job.result()

        # Aer flattens: circuit0 param0..paramR-1, circuit1 param0..paramR-1, ...
        states = np.empty((N, R, dim_res, dim_res), dtype=complex)

        for i in range(N):
            for r in range(R):
                k = i * R + r
                data_k = result.data(k)

                dm = data_k.get("density_matrix", None)
                if dm is None:
                    for key, val in data_k.items():
                        if "density_matrix" in key:
                            dm = val
                            break
                if dm is None:
                    raise KeyError(
                        f"No density_matrix found in result.data({k}). Keys={list(data_k.keys())}"
                    )

                dm_arr = _to_array(dm)
                if dm_arr.shape != (dim_res, dim_res):
                    raise ValueError(
                        f"Got DM shape {dm_arr.shape}, expected {(dim_res, dim_res)} at (i={i}, r={r})"
                    )

                states[i, r] = dm_arr

        return ExactResults(states=states, qrc_cfg=self.qrc_cfg)
