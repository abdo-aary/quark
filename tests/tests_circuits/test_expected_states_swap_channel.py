import numpy as np
import pytest

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, partial_trace

from src.qrc.circuits.qrc_configs import RingQRConfig as Config

from src.qrc.circuits.circuit_factory import CircuitFactory
from src.qrc.circuits.utils import angle_positioning_linear

qiskit_aer = pytest.importorskip("qiskit_aer")
from qiskit_aer import AerSimulator


# -------------------------
# Helpers
# -------------------------

def run_dm(qc: QuantumCircuit, seed: int = 0) -> DensityMatrix:
    backend = AerSimulator(method="density_matrix")
    qc2 = qc.copy()
    qc2.save_density_matrix()
    tqc = transpile(qc2, backend)
    res = backend.run(tqc, shots=1, seed_simulator=seed).result()
    return DensityMatrix(res.data(0)["density_matrix"])


def reduce_reservoir(dm_full: DensityMatrix, n: int) -> DensityMatrix:
    """Keep reservoir qubits 0..n-1, trace out env n..2n (aux+coin)."""
    env = list(range(n, 2 * n + 1))
    return DensityMatrix(partial_trace(dm_full, env))


def dm_plus_1q() -> DensityMatrix:
    qc = QuantumCircuit(1)
    qc.h(0)
    return DensityMatrix.from_instruction(qc)


def dm_minus_1q() -> DensityMatrix:
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.z(0)
    return DensityMatrix.from_instruction(qc)


def make_param_row_by_name(qc: QuantumCircuit, *, hz0: float, hx0: float, lam: float) -> np.ndarray:
    """
    Build a single parameter row aligned with qc.parameters using parameter NAME matching.
    This avoids the 'Parameter identity' issue (UUID differs across circuit instances).
    """
    row = np.zeros((len(list(qc.parameters)),), dtype=float)
    for i, p in enumerate(list(qc.parameters)):
        if p.name.startswith("h_z"):
            row[i] = hz0
        elif p.name.startswith("h_x"):
            row[i] = hx0
        elif p.name == "lam":
            row[i] = lam
        elif p.name.startswith("J"):
            # for n=1, there should be no J, but keep safe default
            row[i] = 0.0
        else:
            raise AssertionError(f"Unexpected free parameter in window circuit: {p} (name={p.name})")
    return row


@pytest.fixture
def cfg1():
    # Use n=1 for fully analytic targets; edges are empty in ring_topology(1)
    return Config(input_dim=1, num_qubits=1, seed=0)


@pytest.fixture
def x_window_zero(cfg1):
    # Window length 1, input_dim 1, ensures z_t = 0 => injection Ry(0)=I
    return np.zeros((1, cfg1.input_dim), dtype=float)


# -------------------------
# Expected-state tests
# -------------------------

def test_swap_channel_lambda0_outputs_plus(cfg1, x_window_zero):
    """
    For any unitary part, E_{lam=0}(rho) = |+><+| exactly.
    """
    qc = CircuitFactory.instantiateFullIsingRingEvolution(
        qrc_cfg=cfg1, angle_positioning=angle_positioning_linear, x_window=x_window_zero
    )

    # Make unitary nontrivial (hz=pi makes |+> -> |->), but lambda=0 should override to |+>
    row = make_param_row_by_name(qc, hz0=np.pi, hx0=0.0, lam=0.0)

    bind = dict(zip(list(qc.parameters), row))
    dm_full = run_dm(qc.assign_parameters(bind, inplace=False), seed=0)
    dm_res = reduce_reservoir(dm_full, n=cfg1.num_qubits)

    assert np.allclose(dm_res.data, dm_plus_1q().data, atol=1e-12, rtol=0.0)


def test_swap_channel_lambda1_matches_unitary_output(cfg1, x_window_zero):
    """
    With z=0 (no injection) and hz=pi, hx=0, the unitary maps |+> -> |-> (up to global phase).
    For lambda=1, contraction does nothing => output should be |->.
    """
    qc = CircuitFactory.instantiateFullIsingRingEvolution(
        qrc_cfg=cfg1, angle_positioning=angle_positioning_linear, x_window=x_window_zero
    )

    row = make_param_row_by_name(qc, hz0=np.pi, hx0=0.0, lam=1.0)
    bind = dict(zip(list(qc.parameters), row))

    dm_full = run_dm(qc.assign_parameters(bind, inplace=False), seed=0)
    dm_res = reduce_reservoir(dm_full, n=cfg1.num_qubits)

    assert np.allclose(dm_res.data, dm_minus_1q().data, atol=1e-12, rtol=0.0)


def test_swap_channel_intermediate_lambda_gives_exact_mixture(cfg1, x_window_zero):
    """
    With z=0 and hz=pi, hx=0: unitary output is |->.
    Channel should yield:
        rho_out = lam * |-><-| + (1-lam) * |+><+|.
    Deterministic with shots=1 under density_matrix because SWAP+reset is CPTP (no measurement).
    """
    qc = CircuitFactory.instantiateFullIsingRingEvolution(
        qrc_cfg=cfg1, angle_positioning=angle_positioning_linear, x_window=x_window_zero
    )

    lam = 0.23
    row = make_param_row_by_name(qc, hz0=np.pi, hx0=0.0, lam=lam)
    bind = dict(zip(list(qc.parameters), row))

    dm_full = run_dm(qc.assign_parameters(bind, inplace=False), seed=0)
    dm_res = reduce_reservoir(dm_full, n=cfg1.num_qubits)

    expected = lam * dm_minus_1q().data + (1.0 - lam) * dm_plus_1q().data
    assert np.allclose(dm_res.data, expected, atol=1e-12, rtol=0.0)


# def test_env_is_reset_and_factorizes_for_n1(cfg1, x_window_zero):
#     """
#     Because your circuit does:
#         ... cswap ... ; reset(coin); reset(aux)
#     the final global state should factorize as:
#         rho_full = rho_res ⊗ |0...0><0...0|_env
#     (for n=1, env has 2 qubits).
#     This is a great runtime check that reset really “traces out” and reinitializes the env.
#     """
#     qc = CircuitFactory.instantiateFullIsingRingEvolution(
#         qrc_cfg=cfg1, angle_positioning=angle_positioning_linear, x_window=x_window_zero
#     )
#
#     lam = 0.37
#     row = make_param_row_by_name(qc, hz0=np.pi, hx0=0.0, lam=lam)
#     bind = dict(zip(list(qc.parameters), row))
#
#     dm_full = run_dm(qc.assign_parameters(bind, inplace=False), seed=0)
#     dm_res = reduce_reservoir(dm_full, n=cfg1.num_qubits)
#
#     # env is aux(1 qubit) + coin(1 qubit) => 2 qubits in |00>
#     dm_env = DensityMatrix.from_label("00")
#
#     expected_full = np.kron(dm_env.data, dm_res.data)
#
#     assert np.allclose(dm_full.data, expected_full, atol=1e-12, rtol=0.0)
