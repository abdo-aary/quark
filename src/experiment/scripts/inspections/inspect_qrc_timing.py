"""
scripts/profile_run_pubs_overhead.py

Profile the staging overhead inside ExactAerCircuitsRunner.run_pubs-like logic.

Run (CPU):
    python scripts/profile_run_pubs_overhead.py --device CPU

Run (GPU 0 only):
    CUDA_VISIBLE_DEVICES=0 python scripts/profile_run_pubs_overhead.py --device GPU

Notes
-----
- CUDA_VISIBLE_DEVICES must be set *before* importing qiskit_aer to reliably pin GPU=0.
- First run often includes one-time overhead (CUDA context init, JIT, cache warmup).
  Run twice to see steady-state timings.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# IMPORTANT: keep qiskit_aer import after CUDA_VISIBLE_DEVICES is set externally.
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator  # will raise if not installed


def _make_parametric_qc(num_qubits: int) -> QuantumCircuit:
    """
    Create a small parametric circuit with metadata keys expected by the runner.

    Parameters
    ----------
    num_qubits : int
        Number of reservoir qubits.

    Returns
    -------
    qiskit.QuantumCircuit
        Circuit containing parameters and metadata fields:
        'J', 'h_x', 'h_z', 'lam'.
    """
    qc = QuantumCircuit(num_qubits)

    J = ParameterVector("J", length=num_qubits)
    hx = ParameterVector("h_x", length=num_qubits)
    hz = ParameterVector("h_z", length=num_qubits)
    lam = Parameter("lam")

    # simple param structure (enough for binding + transpile)
    for q in range(num_qubits):
        qc.rx(hx[q], q)
        qc.rz(hz[q], q)

    for q in range(num_qubits - 1):
        qc.rzz(J[q] * lam, q, q + 1)
    if num_qubits > 2:
        qc.rzz(J[num_qubits - 1] * lam, num_qubits - 1, 0)

    qc.metadata = {"J": list(J), "h_x": list(hx), "h_z": list(hz), "lam": lam}
    return qc


def _ensure_save_density_matrix(qc: QuantumCircuit, qubits: List[int]) -> QuantumCircuit:
    """
    Append save_density_matrix if not already present.

    Parameters
    ----------
    qc : qiskit.QuantumCircuit
        Circuit to modify (copied before modification).
    qubits : list[int]
        Qubits to save density matrix for.

    Returns
    -------
    qiskit.QuantumCircuit
        Modified circuit.
    """
    qc2 = qc.copy()
    has_save = any(inst.operation.name == "save_density_matrix" for inst in qc2.data)
    if not has_save:
        qc2.save_density_matrix(qubits=qubits)
    return qc2


@dataclass
class ProfileTimings:
    """Container for per-stage timings (seconds)."""
    build_work_circuits: float
    transpile: float
    build_binds: float
    aer_run: float
    extract: float
    total: float


def profile_like_run_pubs(
    *,
    pubs: List[Tuple[QuantumCircuit, np.ndarray]],
    num_qubits: int,
    device: str,
    seed_simulator: int = 0,
    optimization_level: int = 0,
) -> ProfileTimings:
    """
    Profile stages analogous to ExactAerCircuitsRunner.run_pubs().

    Parameters
    ----------
    pubs : list[tuple[QuantumCircuit, numpy.ndarray]]
        Each item is (qc, vals) where vals has shape (R, P).
    num_qubits : int
        Reservoir qubits count (density matrix is saved on these).
    device : {"CPU","GPU"}
        Aer device option.
    seed_simulator : int
        Seed for AerSimulator.
    optimization_level : int
        Transpile optimization level.

    Returns
    -------
    ProfileTimings
        Timing breakdown.
    """
    t0 = time.perf_counter()

    # --- Stage A: prepare circuits (copy + save_density_matrix) + collect param order
    ta0 = time.perf_counter()
    work_circuits: List[QuantumCircuit] = []
    param_order: List[Any] = []

    for qc, _vals in pubs:
        qc_dm = _ensure_save_density_matrix(qc, qubits=list(range(num_qubits)))
        work_circuits.append(qc_dm)

        md = qc_dm.metadata or {}
        if not param_order:
            param_order = list(md["J"]) + list(md["h_x"]) + list(md["h_z"]) + [md["lam"]]

    ta1 = time.perf_counter()

    # --- Stage B: transpile (batch)
    tb0 = time.perf_counter()
    backend = AerSimulator(method="density_matrix")
    backend.set_options(device=device)
    transpiled = transpile(work_circuits, backend=backend, optimization_level=optimization_level)
    if isinstance(transpiled, QuantumCircuit):
        transpiled = [transpiled]
    tb1 = time.perf_counter()

    # --- Stage C: build parameter_binds
    tc0 = time.perf_counter()
    # map transpiled parameters by name (safe if names unique)
    tparams = list(transpiled[0].parameters)
    tmap = {p.name: p for p in tparams}
    ordered_tparams = [tmap[p.name] for p in param_order]

    binds_list: List[Dict[Any, List[float]]] = []
    for (_qc, vals) in pubs:
        # vals: (R, P)
        R = vals.shape[0]
        bind: Dict[Any, List[float]] = {}
        for j, p in enumerate(ordered_tparams):
            bind[p] = vals[:, j].tolist()
        binds_list.append(bind)
    tc1 = time.perf_counter()

    # --- Stage D: run Aer
    td0 = time.perf_counter()
    job = backend.run(transpiled, parameter_binds=binds_list, seed_simulator=seed_simulator)
    res = job.result()
    td1 = time.perf_counter()

    # --- Stage E: extract density matrices (forces materialization)
    te0 = time.perf_counter()
    # Flattened order: for each circuit i, for each bind index r -> experiment index increases.
    # We just touch the data to force extraction cost.
    R = pubs[0][1].shape[0]
    _ = res.data(0)["density_matrix"]  # touch one
    for i in (0, len(pubs) - 1):
        for r in (0, R - 1):
            _ = res.data(i * R + r)["density_matrix"]
    te1 = time.perf_counter()

    t1 = time.perf_counter()
    return ProfileTimings(
        build_work_circuits=ta1 - ta0,
        transpile=tb1 - tb0,
        build_binds=tc1 - tc0,
        aer_run=td1 - td0,
        extract=te1 - te0,
        total=t1 - t0,
    )


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="CPU", choices=["CPU", "GPU"])
    ap.add_argument("--n_pubs", type=int, default=2000)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--num_qubits", type=int, default=3)
    ap.add_argument("--input_dim", type=int, default=4)
    ap.add_argument("--opt", type=int, default=0)
    args = ap.parse_args()

    qc = _make_parametric_qc(args.num_qubits)

    P = 3 * args.num_qubits + 1
    rng = np.random.default_rng(0)
    pubs: List[Tuple[QuantumCircuit, np.ndarray]] = []
    for _ in range(args.n_pubs):
        vals = rng.normal(size=(args.R, P)).astype(float)
        pubs.append((qc, vals))

    # Warmup (helps separate “first-call GPU init” from steady-state)
    _ = profile_like_run_pubs(
        pubs=pubs[:5],
        num_qubits=args.num_qubits,
        device=args.device,
        seed_simulator=0,
        optimization_level=args.opt,
    )

    t = profile_like_run_pubs(
        pubs=pubs,
        num_qubits=args.num_qubits,
        device=args.device,
        seed_simulator=0,
        optimization_level=args.opt,
    )

    print("\n=== Timing breakdown (seconds) ===")
    print(f"build_work_circuits : {t.build_work_circuits:8.3f}")
    print(f"transpile           : {t.transpile:8.3f}")
    print(f"build_binds         : {t.build_binds:8.3f}")
    print(f"aer_run             : {t.aer_run:8.3f}")
    print(f"extract             : {t.extract:8.3f}")
    print(f"TOTAL               : {t.total:8.3f}\n")


if __name__ == "__main__":
    main()
