# tests/tests_run/test_runner_spec_swap_dataset_dm.py
import inspect
import numpy as np
import pytest

pytest.importorskip("qiskit_aer")

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

from src.qrc.circuits.circuit_factory import CircuitFactory
from src.qrc.circuits.qrc_configs import RingQRConfig
from src.qrc.circuits.utils import angle_positioning_linear
from src.qrc.run.circuit_run import ExactAerCircuitsRunner


# ----------------------------
# Helpers (template-PUB + signature robust)
# ----------------------------
def get_pub_dataset_fn():
    fn = getattr(CircuitFactory, "create_pubs_dataset_reservoirs_IsingRingSWAP", None)
    if fn is None:
        fn = getattr(CircuitFactory, "create_pubs_dataset_reservoir_IsingRingSWAP", None)
    if fn is None:
        raise AttributeError(
            "Could not find create_pubs_dataset_reservoirs_IsingRingSWAP "
            "or create_pubs_dataset_reservoir_IsingRingSWAP"
        )
    return fn


def _pub_qc_vals(pub):
    if isinstance(pub, (tuple, list)) and len(pub) >= 2:
        return pub[0], pub[1]
    raise TypeError(f"Unsupported PUB container type: {type(pub)}")


def pubs_is_template(pubs) -> bool:
    if not isinstance(pubs, list) or len(pubs) == 0:
        return False
    _, vals0 = _pub_qc_vals(pubs[0])
    return (len(pubs) == 1) and isinstance(vals0, np.ndarray) and (vals0.ndim == 3)


def pubs_N_R(pubs):
    if pubs_is_template(pubs):
        _, vals = _pub_qc_vals(pubs[0])
        return int(vals.shape[0]), int(vals.shape[1])
    else:
        N = len(pubs)
        _, vals0 = _pub_qc_vals(pubs[0])
        return int(N), int(vals0.shape[0])


def pubs_get_qc_vals_for_window(pubs, i: int):
    """
    Returns (qc, vals_window) where vals_window has shape (R,P)
    in both legacy and template modes.
    """
    if pubs_is_template(pubs):
        qc, vals3 = _pub_qc_vals(pubs[0])  # (N,R,P)
        return qc, vals3[i]
    else:
        return _pub_qc_vals(pubs[i])  # (qc_i, vals_i) where vals_i is (R,P)


def call_create_pubs_dataset(
    *,
    qrc_cfg,
    angle_positioning,
    X,
    lam_0: float,
    num_reservoirs: int,
    seed: int = 0,
    eps: float = 1e-8,
    template_pub: bool = True,
):
    """
    Dispatch across legacy vs optimized CircuitFactory signatures.

    - Optimized signature: (qrc_cfg, angle_positioning, X, parameters_reservoirs, template_pub=...)
    - Legacy signature: (qrc_cfg, angle_positioning, X, lam_0, num_reservoirs, seed, eps, ...)
    """
    fn = get_pub_dataset_fn()
    sig = inspect.signature(fn)
    params = sig.parameters

    kwargs = dict(qrc_cfg=qrc_cfg, angle_positioning=angle_positioning, X=X)

    if "parameters_reservoirs" in params:
        parameters_reservoirs = CircuitFactory.set_reservoirs_parameterizationSWAP(
            qrc_cfg=qrc_cfg,
            angle_positioning=angle_positioning,
            num_reservoirs=num_reservoirs,
            lam_0=lam_0,
            seed=seed,
            eps=eps,
        )
        kwargs["parameters_reservoirs"] = parameters_reservoirs
        if "template_pub" in params:
            kwargs["template_pub"] = template_pub
    else:
        kwargs.update(lam_0=lam_0, num_reservoirs=num_reservoirs, seed=seed, eps=eps)
        if "template_pub" in params:
            kwargs["template_pub"] = template_pub

    return fn(**kwargs)


def param_order_from_metadata(qc):
    """
    Must match how vals columns are built.

    Prefer qc.metadata["param_order"] (new).
    Fallback to (J,h_x,h_z,lam) (legacy).
    """
    md = getattr(qc, "metadata", None) or {}
    if "param_order" in md:
        return list(md["param_order"])

    # legacy fallback
    J = list(md["J"])
    hx = list(md["h_x"])
    hz = list(md["h_z"])
    lam = md["lam"]
    return J + hx + hz + [lam]


def lam_col_index(qc):
    """
    Robustly find which column corresponds to lambda.
    """
    order = param_order_from_metadata(qc)
    md = getattr(qc, "metadata", None) or {}
    if "lam" in md:
        lam_param = md["lam"]
        try:
            return order.index(lam_param)
        except ValueError:
            pass
    # Fallback (historical): last
    return len(order) - 1


def plus_density(n):
    sv = Statevector.from_label("+" * n)
    return DensityMatrix(sv)


def run_reservoir_dm_direct(qc, qrc_cfg, row, seed=0, label="dm_res"):
    """
    Direct Aer run for a single bound parameter row.
    Saves ONLY the reservoir DM (qubits 0..n-1), returns DensityMatrix.
    """
    n = int(qrc_cfg.num_qubits)
    backend = AerSimulator(method="density_matrix")

    qc2 = qc.copy()
    qc2.save_density_matrix(qubits=list(range(n)), label=label)

    order = param_order_from_metadata(qc2)
    if len(order) != len(row):
        raise ValueError(
            f"param cols mismatch: row has {len(row)} but expected {len(order)} (from metadata)."
        )

    bind = {p: float(row[j]) for j, p in enumerate(order)}
    bound = qc2.assign_parameters(bind, inplace=False)

    tqc = transpile(bound, backend=backend, optimization_level=0)
    result = backend.run(tqc, shots=1, seed_simulator=seed).result()
    data = result.data(0)
    dm = data[label]
    return dm if isinstance(dm, DensityMatrix) else DensityMatrix(dm)


def run_full_and_reduced_dm_direct(qc, qrc_cfg, row, seed=0):
    """
    Direct Aer run saving:
      - full density matrix
      - reduced reservoir density matrix
    and returning both DensityMatrix objects.
    """
    n = int(qrc_cfg.num_qubits)
    backend = AerSimulator(method="density_matrix")

    qc2 = qc.copy()
    qc2.save_density_matrix(label="dm_full")
    qc2.save_density_matrix(qubits=list(range(n)), label="dm_res")

    order = param_order_from_metadata(qc2)
    if len(order) != len(row):
        raise ValueError(
            f"param cols mismatch: row has {len(row)} but expected {len(order)} (from metadata)."
        )

    bind = {p: float(row[j]) for j, p in enumerate(order)}
    bound = qc2.assign_parameters(bind, inplace=False)

    tqc = transpile(bound, backend=backend, optimization_level=0)
    result = backend.run(tqc, shots=1, seed_simulator=seed).result()
    data = result.data(0)

    dm_full = data["dm_full"]
    dm_res = data["dm_res"]
    dm_full = dm_full if isinstance(dm_full, DensityMatrix) else DensityMatrix(dm_full)
    dm_res = dm_res if isinstance(dm_res, DensityMatrix) else DensityMatrix(dm_res)
    return dm_full, dm_res


def run_runner(runner, pubs, seed=0, opt=0):
    """
    Call run_pubs with common kwargs, but tolerate differing signatures.
    """
    try:
        return runner.run_pubs(
            pubs=pubs,
            seed_simulator=seed,
            optimization_level=opt,
            device="CPU",
            max_parallel_threads=1,
            max_parallel_experiments=1,
            max_parallel_shots=1,
        )
    except TypeError:
        return runner.run_pubs(pubs=pubs, seed_simulator=seed, optimization_level=opt)


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def cfg_small():
    # Keep it tiny/fast but non-trivial
    input_dim = 5
    n = 2
    seed = 123
    return RingQRConfig(input_dim=input_dim, num_qubits=n, seed=seed)


@pytest.fixture
def random_X(cfg_small):
    eps = 1e-6
    N, w, d = 2, 1, cfg_small.input_dim  # IMPORTANT: w=1 for exact mixture spec test
    rng = np.random.default_rng(2026)
    return rng.uniform(-1 + eps, 1 - eps, size=(N, w, d))


@pytest.fixture
def pubs(cfg_small, random_X):
    return call_create_pubs_dataset(
        qrc_cfg=cfg_small,
        angle_positioning=angle_positioning_linear,
        X=random_X,
        lam_0=0.05,
        num_reservoirs=3,
        seed=999,   # deterministic parameterization draw
        eps=1e-8,
        template_pub=True,
    )


# ============================================================
# 1) Spec test: one-step SWAP channel matches exact mixture
# ============================================================
def test_one_step_swap_channel_matches_mixture_formula(cfg_small, pubs):
    """
    For w=1, the circuit implements:
        rho_out = lam * rho_unitary + (1-lam) * rho_plus
    where rho_unitary is the output with lam=1 (i.e., no replacement).
    """
    runner = ExactAerCircuitsRunner(cfg_small)
    res = run_runner(runner, pubs, seed=0, opt=0)

    assert res.states.ndim == 4  # (N,R,2^n,2^n)
    N, R, D1, D2 = res.states.shape
    N_pub, _ = pubs_N_R(pubs)
    assert N == N_pub
    assert D1 == D2 == 2 ** cfg_small.num_qubits

    dm_plus = plus_density(cfg_small.num_qubits)

    # Use the first window only (w=1, spec is exact per window)
    qc, vals = pubs_get_qc_vals_for_window(pubs, 0)
    lam_idx = lam_col_index(qc)

    for r in range(R):
        lam = float(vals[r, lam_idx])
        dm_out = DensityMatrix(res.states[0, r])

        # Reference unitary output: same row but with lam=1
        row_unitary = vals[r].copy()
        row_unitary[lam_idx] = 1.0
        dm_unitary = run_reservoir_dm_direct(qc, cfg_small, row_unitary, seed=0, label="dm_res")

        expected = lam * dm_unitary.data + (1.0 - lam) * dm_plus.data

        assert np.allclose(dm_out.data, expected, atol=1e-10, rtol=0.0), (
            f"Mismatch at reservoir r={r}, lam={lam}"
        )


# ============================================================
# 2) Spec test: saved reservoir DM equals partial trace of full DM
# ============================================================
def test_saved_reservoir_dm_matches_partial_trace(cfg_small, pubs):
    """
    Verifies 'save_density_matrix(qubits=reservoir)' is consistent with full DM.
    This catches qubit-order / subset mistakes.
    """
    qc, vals = pubs_get_qc_vals_for_window(pubs, 0)
    row = vals[0]  # any reservoir row

    dm_full, dm_res_saved = run_full_and_reduced_dm_direct(qc, cfg_small, row, seed=0)

    n = int(cfg_small.num_qubits)
    total = qc.num_qubits
    env = list(range(n, total))  # env qubits are everything except reservoir [0..n-1]

    dm_res_traced = partial_trace(dm_full, env)

    assert dm_res_traced.dim == dm_res_saved.dim
    assert np.allclose(dm_res_traced.data, dm_res_saved.data, atol=1e-10, rtol=0.0)


# ============================================================
# 3) Dataset-level invariants: parameter reuse + outputs vary + determinism
# ============================================================
def test_dataset_parameterization_reuse_and_output_variation(cfg_small, pubs):
    """
    Intention: reuse the same reservoir parameterization across all windows.

    - Legacy mode: vals matrices (R,P) are identical across windows.
    - Template mode: vals include injection params varying with window; we instead
      check that the *constant columns across windows* (reservoir params) are identical.

    Also verifies:
      - outputs differ across windows for same reservoir (with high prob)
      - outputs differ across reservoirs for same window (with high prob)
      - determinism: same seed => same states
    """
    N, R = pubs_N_R(pubs)
    assert N >= 2, "Need at least 2 windows in pubs fixture"

    if pubs_is_template(pubs):
        qc, vals3 = _pub_qc_vals(pubs[0])  # (N,R,P)
        order = param_order_from_metadata(qc)

        # Identify reservoir params as columns that are exactly constant across windows+reservoirs
        base = vals3[0:1, :, :]  # (1,R,P)
        mask_const = np.all(vals3 == base, axis=(0, 1))  # (P,)

        # Optional name-based refinement: injection often named "z..."
        mask_name = np.array([not str(p.name).startswith("z") for p in order], dtype=bool)
        mask_res = mask_const & mask_name if np.any(mask_const & mask_name) else mask_const

        assert np.any(mask_res), "Could not isolate any constant (reservoir) parameter columns in template mode"

        ref = vals3[0, :, mask_res]  # (R,Pres)
        for i in range(1, N):
            assert np.allclose(vals3[i, :, mask_res], ref, atol=0.0, rtol=0.0)
    else:
        qc0, vals0 = _pub_qc_vals(pubs[0])
        qc1, vals1 = _pub_qc_vals(pubs[1])
        assert vals0.shape == vals1.shape
        assert np.allclose(vals0, vals1, atol=0.0, rtol=0.0)

    runner = ExactAerCircuitsRunner(cfg_small)
    res1 = run_runner(runner, pubs, seed=0, opt=0)
    res2 = run_runner(runner, pubs, seed=0, opt=0)

    assert np.allclose(res1.states, res2.states, atol=0.0, rtol=0.0)

    states = res1.states
    N2, R2, d1, d2 = states.shape
    assert N2 == N
    assert R2 == R
    assert d1 == d2 == 2 ** cfg_small.num_qubits

    diffs_win = [np.linalg.norm(states[0, r] - states[1, r]) for r in range(R)]
    assert max(diffs_win) > 1e-6

    diffs_res = [np.linalg.norm(states[0, 0] - states[0, r]) for r in range(1, R)]
    assert max(diffs_res) > 1e-6
