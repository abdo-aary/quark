"""src.qrc.circuits.circuit_factory
================================

Factory utilities to build the *quantum reservoir circuits* used throughout the
project.

The main entry points are:

- :meth:`CircuitFactory.createIsingRingCircuitSWAP` and
  :meth:`CircuitFactory.createIsingRingCircuitDynamic`, which build a single
  *per-time-step* reservoir circuit implementing

  ``data injection -> Ising unitary W -> contraction E_λ``

- :meth:`CircuitFactory.instantiateFullIsingRingEvolution`, which composes the
  per-time-step circuit over a full input window to obtain a circuit whose
  *input angles are bound*, while *reservoir parameters* (Ising couplings/fields
  and the contraction strength) remain free.

- :meth:`CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP`, which
  prepares a PUBS dataset compatible with :class:`src.qrc.run.circuit_run.ExactAerCircuitsRunner`.

Notes
-----
Metadata contract
~~~~~~~~~~~~~~~~~
Most downstream code (notably the Aer runner) expects the circuits returned by
this module to carry the following metadata keys:

- ``"z"``: :class:`qiskit.circuit.ParameterVector` for the injected coordinates
- ``"J"``: :class:`qiskit.circuit.ParameterVector` for ZZ couplings (one per edge)
- ``"h_x"``: :class:`qiskit.circuit.ParameterVector` for local X fields (one per qubit)
- ``"h_z"``: :class:`qiskit.circuit.ParameterVector` for local Z fields (one per qubit)
- ``"lam"``: :class:`qiskit.circuit.Parameter` for the contraction strength λ

The runner uses this metadata to reconstruct the expected parameter column order
for the PUBS arrays, which is *not* necessarily the same as ``qc.parameters``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter

from src.qrc.circuits.qrc_configs import RingQRConfig


class CircuitFactory:
    """Factory for building QRC circuits and PUBS datasets.

    This class centralizes the construction of the reservoir circuit family used
    in the experiments/pipeline. All methods are `@staticmethod` to keep usage
    lightweight and stateless.

    See Also
    --------
    src.qrc.run.circuit_run.ExactAerCircuitsRunner
        Runner expecting the metadata contract described in this module docstring.
    """

    @staticmethod
    def createIsingRingCircuitSWAP(
        qrc_cfg: RingQRConfig,
        angle_positioning: callable,
    ) -> QuantumCircuit:
        r"""Create the *deterministic* per-step Ising ring reservoir circuit (SWAP dilation).

        This circuit implements the same high-level transformation as
        :meth:`createIsingRingCircuitDynamic`, but without mid-circuit measurement.
        The contraction channel is realized via a Stinespring dilation:

        .. math::

            E_\lambda(\rho) = \lambda\,\rho + (1-\lambda)\,|+\rangle\langle+|^{\otimes n}.

        Construction:

        - Reservoir register: qubits ``0..n-1``
        - Environment register: auxiliary qubits ``n..2n-1`` prepared in ``|+>^{⊗n}``
        - Coin qubit: ``2n`` prepared so that ``P(coin=1)=1-λ``
        - Controlled-SWAP between reservoir and environment conditioned on coin
        - Reset the environment to trace it out deterministically

        Parameters
        ----------
        qrc_cfg : RingQRConfig
            Reservoir configuration, including topology.
        angle_positioning : callable
            Map from injected coordinate vector ``z`` to per-qubit ``Ry`` angles.

        Returns
        -------
        QuantumCircuit
            Parameterized circuit with ``2*qrc_cfg.num_qubits + 1`` qubits. Metadata keys
            ``"z"``, ``"J"``, ``"h_x"``, ``"h_z"``, ``"lam"`` are set.

        Notes
        -----
        This version is simulator-friendly and backend-agnostic because it avoids
        mid-circuit measurement/conditional control.
        """

        n = qrc_cfg.num_qubits
        topology = qrc_cfg.topology

        z = ParameterVector("z", n)
        angles = angle_positioning(z)

        zz_params = ParameterVector("J", len(topology.edges))
        rx_params = ParameterVector("h_x", n)
        rz_params = ParameterVector("h_z", n)
        lam_param = Parameter("lam")

        # Qubit layout:
        #   reservoir: 0..n-1
        #   aux (+ state): n..2n-1
        #   coin: 2n
        qc = QuantumCircuit(2 * n + 1, name="SMC_det")
        res = list(range(n))
        aux = list(range(n, 2 * n))
        coin = 2 * n

        # 1) Data injection on reservoir
        for i in range(n):
            qc.ry(angles[i], res[i])

        # 2) Ising unitary W on reservoir
        for e, (q1, q2) in enumerate(topology.edges):
            qc.rzz(zz_params[e], q1, q2)

        for i in range(n):
            qc.rz(rz_params[i], res[i])
        for i in range(n):
            qc.rx(rx_params[i], res[i])

        # 3) Deterministic contraction via unitary dilation + env reset
        qc.reset(aux)
        qc.reset(coin)
        qc.h(aux)

        theta = 2 * ((1 - lam_param) ** 0.5).arcsin()
        qc.ry(theta, coin)

        for i in range(n):
            qc.cswap(coin, res[i], aux[i])

        # qc.reset(coin)
        # qc.reset(aux)

        qc.metadata["z"] = z
        qc.metadata["J"] = zz_params
        qc.metadata["h_x"] = rx_params
        qc.metadata["h_z"] = rz_params
        qc.metadata["lam"] = lam_param

        return qc

    @staticmethod
    def instantiateFullIsingRingEvolution(
        qrc_cfg: RingQRConfig,
        angle_positioning: callable,
        x_window: np.ndarray,
    ) -> QuantumCircuit:
        """Instantiate (bind) the input window and return a circuit free in reservoir parameters.

        Given a time window ``x_window`` of shape ``(w, d)`` (with ``d=qrc_cfg.input_dim``),
        this routine:

        1) Builds the per-step circuit (SWAP contraction version).
        2) Initializes the reservoir to ``|+>^{⊗n}``.
        3) For each time step ``t``, projects ``x_t`` to ``z_t = x_t @ qrc_cfg.projection`` and
           binds the circuit's input parameters ``z`` to ``z_t``.
        4) Composes the bound step circuit into a single window circuit.

        The returned circuit has *no remaining input parameters* (``z`` is bound),
        but keeps reservoir parameters ``J, h_x, h_z, lam`` free.

        Parameters
        ----------
        qrc_cfg : RingQRConfig
            Reservoir configuration. Must satisfy ``qrc_cfg.input_dim == x_window.shape[-1]``.
        angle_positioning : callable
            Injection angle map used by the per-step circuit.
        x_window : numpy.ndarray
            Input window of shape ``(w, d)``. By convention, ``x_window[0]`` corresponds
            to the oldest sample in the window.

        Returns
        -------
        QuantumCircuit
            A composed circuit representing the full evolution over the window.
            Metadata keys ``"J"``, ``"h_x"``, ``"h_z"``, ``"lam"`` are copied from the
            per-step circuit for downstream runners.

        Raises
        ------
        AssertionError
            If `x_window` does not have shape ``(w, qrc_cfg.input_dim)``.
        """
        assert x_window.shape[-1] == qrc_cfg.input_dim, (
            "Mismatch between the window dimension and input_dim in the provided config. "
            f"Got x_window.shape[-1]={x_window.shape[-1]} and qrc_cfg.input_dim={qrc_cfg.input_dim}."
        )
        assert x_window.ndim == 2, (
            f"x_window should be a 2D array of shape (w, d), got x_window.ndim={x_window.ndim}."
        )

        qc_reservoir = CircuitFactory.createIsingRingCircuitSWAP(
            qrc_cfg=qrc_cfg, angle_positioning=angle_positioning
        )

        n = qrc_cfg.num_qubits
        qubits = list(range(n))
        z_params = qc_reservoir.metadata["z"]

        qc = QuantumCircuit(qc_reservoir.num_qubits, name="QRC")
        qc.h(qubits[:n])

        for x_t in x_window:
            z_t = x_t @ qrc_cfg.projection
            bind_map_input = dict(zip(z_params, z_t))
            qc_step_bound = qc_reservoir.assign_parameters(bind_map_input)
            qc.compose(qc_step_bound, inplace=True)

        qc.metadata["J"] = qc_reservoir.metadata["J"]
        qc.metadata["h_x"] = qc_reservoir.metadata["h_x"]
        qc.metadata["h_z"] = qc_reservoir.metadata["h_z"]
        qc.metadata["lam"] = qc_reservoir.metadata["lam"]
        return qc

    @staticmethod
    def set_reservoirs_parameterizationSWAP(
        qrc_cfg: RingQRConfig,
        angle_positioning: callable,
        num_reservoirs: int,
        lam_0: float,
        seed: float = 12345,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """Sample reservoir parameter sets for ``R`` reservoirs (SWAP circuit variant).

        Reservoir parameters are sampled as follows:

        - ZZ couplings ``J``: iid Uniform(-π, π) for each edge.
        - Local fields ``h_x`` and ``h_z``: iid Uniform(-π, π) per qubit.
        - Contraction strengths ``λ``:
            - reservoir 0 uses ``lam_0`` (user-chosen)
            - reservoirs 1..R-1 use iid Uniform(eps, 1-eps)

        Parameters
        ----------
        qrc_cfg : RingQRConfig
            Reservoir configuration.
        angle_positioning : callable
            Injection angle map (only used to build the parameter vectors; it does
            not affect the sampled values).
        num_reservoirs : int
            Number of reservoirs ``R`` to sample.
        lam_0 : float
            Fixed contraction strength for the first reservoir. Must satisfy
            ``eps < lam_0 < 1-eps``.
        seed : float, default=12345
            RNG seed for deterministic sampling.
        eps : float, default=1e-8
            Margin to avoid sampling exactly 0 or 1 for λ.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(R, P)`` where ``P = |J| + |h_x| + |h_z| + 1`` is the
            number of free reservoir parameters. Columns follow the *metadata order*
            expected by downstream runners:

            ``param_order = list(J) + list(h_x) + list(h_z) + [lam]``.

        Raises
        ------
        AssertionError
            If inputs violate basic constraints (e.g. `num_reservoirs` < 1).
        """
        R = num_reservoirs
        assert R >= 1, f"num_reservoirs must be >= 1, got {num_reservoirs}"
        assert eps < lam_0 < 1 - eps, f"lam_0 must be in (eps, 1-eps); got lam_0={lam_0}, eps={eps}"

        qc = CircuitFactory.createIsingRingCircuitSWAP(qrc_cfg=qrc_cfg, angle_positioning=angle_positioning)
        rng = np.random.default_rng(seed)

        rzz_params = qc.metadata["J"]
        rx_params = qc.metadata["h_x"]
        rz_params = qc.metadata["h_z"]
        lam_param = qc.metadata["lam"]

        rzz_values = rng.uniform(-np.pi, np.pi, size=(R, len(rzz_params)))
        rx_values = rng.uniform(-np.pi, np.pi, size=(R, len(rx_params)))
        rz_values = rng.uniform(-np.pi, np.pi, size=(R, len(rz_params)))

        lam_values = np.empty(R, dtype=float)
        lam_values[0] = float(lam_0)
        if R > 1:
            lam_values[1:] = rng.uniform(eps, 1 - eps, size=R - 1)

        param_order = list(rzz_params) + list(rx_params) + list(rz_params) + [lam_param]
        P = len(param_order)
        param_values = np.empty((R, P), dtype=float)

        for r in range(R):
            bind = {}
            bind.update(dict(zip(rzz_params, rzz_values[r])))
            bind.update(dict(zip(rx_params, rx_values[r])))
            bind.update(dict(zip(rz_params, rz_values[r])))
            bind[lam_param] = float(lam_values[r])
            param_values[r, :] = [bind[p] for p in param_order]

        return param_values

    @staticmethod
    def instantiateFullIsingRingEvolutionTemplate(
            qrc_cfg: RingQRConfig,
            angle_positioning: callable,
            w: int,
    ) -> Tuple[QuantumCircuit, List[ParameterVector]]:
        """
        Build a *single* parameterized window circuit (template) with *unbound* input injections.

        This differs from `instantiateFullIsingRingEvolution(...)`:

        - Here we do NOT bind the injected inputs.
        - Instead, each time step t uses its own ParameterVector z_t, so the full
          window circuit remains parameterized by all injected inputs.
        - Reservoir parameters (J, h_x, h_z, lam) are shared across all steps.

        The goal is to allow downstream runners (Aer) to transpile ONCE and run
        many shots/experiments via batched parameter binding.

        Parameters
        ----------
        qrc_cfg : RingQRConfig
            Reservoir configuration.
        angle_positioning : callable
            Function mapping a length-n ParameterVector z_t to per-qubit angles.
        w : int
            Window length (number of time steps in the composed circuit).

        Returns
        -------
        qc_window : QuantumCircuit
            Parameterized circuit representing the full w-step evolution.
            Metadata contains shared reservoir params ("J", "h_x", "h_z", "lam")
            and also "z_steps" with the per-step ParameterVectors.
        z_steps : list[ParameterVector]
            The list [z_0, ..., z_{w-1}] used in the template circuit.
        """
        n = qrc_cfg.num_qubits

        # Step template with shared reservoir params + a generic z vector.
        qc_step = CircuitFactory.createIsingRingCircuitSWAP(qrc_cfg=qrc_cfg, angle_positioning=angle_positioning)
        z_base = qc_step.metadata["z"]  # ParameterVector("z", n)

        # Start full window circuit with the same width as step circuit.
        qc_window = QuantumCircuit(qc_step.num_qubits, name="QRC_template")
        qc_window.h(list(range(n)))  # initial reservoir |+>^n

        z_steps: List[ParameterVector] = []

        for t in range(w):
            z_t = ParameterVector(f"z_{t}", n)
            z_steps.append(z_t)

            # Rename step's base z -> time-specific z_t (Parameter -> Parameter mapping).
            rename_map = dict(zip(z_base, z_t))
            qc_step_t = qc_step.assign_parameters(rename_map, inplace=False)

            qc_window.compose(qc_step_t, inplace=True)

        # Keep shared reservoir params in metadata for downstream ordering/binding.
        qc_window.metadata["z_steps"] = z_steps
        qc_window.metadata["J"] = qc_step.metadata["J"]
        qc_window.metadata["h_x"] = qc_step.metadata["h_x"]
        qc_window.metadata["h_z"] = qc_step.metadata["h_z"]
        qc_window.metadata["lam"] = qc_step.metadata["lam"]

        return qc_window, z_steps

    @staticmethod
    def create_pubs_dataset_reservoirs_IsingRingSWAP(
            qrc_cfg: RingQRConfig,
            angle_positioning: callable,
            X: np.ndarray,
            num_reservoirs: int,
            lam_0: float,
            seed: float = 12345,
            eps: float = 1e-8,
    ):
        """
        Create PUBs for a dataset X across multiple reservoirs (SWAP dilation variant).

        Returns a *single* template PUB:
            pubs = [(qc_template, vals)]
        where:
            - qc_template is one parameterized circuit for the whole window (w steps)
            - vals has shape (N, R, P_total), with columns matching qc_template.metadata["param_order"]

        Parameters
        ----------
        qrc_cfg : RingQRConfig
            Reservoir config.
        angle_positioning : callable
            Injection mapping used inside the circuit.
        X : numpy.ndarray
            Input windows of shape (N, w, d).
        num_reservoirs : int
            Number of reservoirs R.
        lam_0 : float
            Contraction λ for reservoir 0.
        seed : float
            RNG seed for reservoir parameter sampling (J, h_x, h_z, λ for r>=1).
        eps : float
            Margin to avoid sampling exactly 0 or 1 for λ.

        Returns
        -------
        pubs : list
            Always: length-1 list [(qc_template, vals_3d)]
        """
        # ---- Backward-compat shim (optional, but saves you from positional-call breakage) ----
        if isinstance(num_reservoirs, (float, np.floating)) and isinstance(lam_0, (int, np.integer)):
            num_reservoirs, lam_0 = int(lam_0), float(num_reservoirs)

        X = np.asarray(X, dtype=float)
        assert X.ndim == 3, f"Expected X shape (N,w,d), got {X.shape}"
        N, w, d = X.shape
        assert d == qrc_cfg.input_dim, f"X last dim {d} != qrc_cfg.input_dim {qrc_cfg.input_dim}"

        # ---- Template mode: one window circuit, all inputs unbound ----
        qc_template, z_steps = CircuitFactory.instantiateFullIsingRingEvolutionTemplate(
            qrc_cfg=qrc_cfg, angle_positioning=angle_positioning, w=w
        )

        # ---- Generate reservoir parameter table internally ----
        R = int(num_reservoirs)
        parameters_reservoirs = CircuitFactory.set_reservoirs_parameterizationSWAP(
            qrc_cfg=qrc_cfg,
            angle_positioning=angle_positioning,
            num_reservoirs=R,
            lam_0=lam_0,
            seed=seed,
            eps=eps,
        )
        parameters_reservoirs = np.asarray(parameters_reservoirs, dtype=float)
        assert parameters_reservoirs.shape[0] == R, (
            f"Expected parameters_reservoirs shape (R,P), got {parameters_reservoirs.shape} with R={R}"
        )

        # ---- Compute injected z-values for every (N,w,n) ----
        Z = X @ qrc_cfg.projection  # (N,w,d) @ (d,n) -> (N,w,n)
        Z = np.asarray(Z, dtype=float)

        # Flatten injected params in the same order as we created z_steps:
        # [z_0[0..n-1], z_1[0..n-1], ..., z_{w-1}[0..n-1]]
        inj = Z.reshape(N, -1)  # (N, w*n)
        P_inj = inj.shape[1]

        # Reservoir params are (R, P_res)
        P_res = parameters_reservoirs.shape[1]

        # Broadcast and concatenate -> (N, R, P_total)
        inj_b = np.broadcast_to(inj[:, None, :], (N, R, P_inj))
        res_b = np.broadcast_to(parameters_reservoirs[None, :, :], (N, R, P_res))
        vals = np.concatenate([inj_b, res_b], axis=2)

        # ---- Deterministic parameter column order for downstream binding ----
        inj_params = [p for z_t in z_steps for p in z_t]
        param_order = (
                inj_params
                + list(qc_template.metadata["J"])
                + list(qc_template.metadata["h_x"])
                + list(qc_template.metadata["h_z"])
                + [qc_template.metadata["lam"]]
        )
        qc_template.metadata["param_order"] = param_order

        return [(qc_template, vals)]

