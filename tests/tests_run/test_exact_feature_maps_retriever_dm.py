import numpy as np
import pytest

import inspect


def _get_pub_dataset_fn():
    fn = getattr(CircuitFactory, "create_pubs_dataset_reservoirs_IsingRingSWAP", None)
    if fn is None:
        fn = getattr(CircuitFactory, "create_pubs_dataset_reservoir_IsingRingSWAP", None)
    if fn is None:
        raise AttributeError(
            "Could not find create_pubs_dataset_reservoirs_IsingRingSWAP or create_pubs_dataset_reservoir_IsingRingSWAP"
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
    if pubs_is_template(pubs):
        qc, vals = _pub_qc_vals(pubs[0])
        # Help runner / reference binding
        qc.metadata = qc.metadata or {}
        qc.metadata["param_order"] = param_order_from_metadata(qc)
        return qc, vals[i]
    else:
        return _pub_qc_vals(pubs[i])


def pubs_get_qc_row(pubs, i: int, r: int):
    qc, vals = pubs_get_qc_vals_for_window(pubs, i)
    if vals.ndim == 2:
        return qc, vals[r]
    raise ValueError(f"Unexpected vals.ndim={vals.ndim} for pubs_get_qc_row")


def call_create_pubs_dataset(*, qrc_cfg, angle_positioning, X, lam_0, num_reservoirs, seed=0, eps=1e-8, template_pub=True):
    """Dispatcher across legacy vs optimized CircuitFactory signatures."""
    fn = _get_pub_dataset_fn()
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
    md = getattr(qc, "metadata", None) or {}
    if "param_order" in md:
        return list(md["param_order"])
    J = list(md["J"])
    hx = list(md["h_x"])
    hz = list(md["h_z"])
    lam = md["lam"]
    return J + hx + hz + [lam]


from qiskit.quantum_info import SparsePauliOp, Operator

from src.qrc.run.circuit_run import ExactAerCircuitsRunner, ExactResults
from src.qrc.run.fmp_retriever import ExactFeatureMapsRetriever
from src.qrc.circuits.circuit_factory import CircuitFactory
from src.qrc.circuits.qrc_configs import RingQRConfig
from src.qrc.circuits.utils import angle_positioning_linear


# ----------------------------- Helpers ----------------------------------------

def random_density_matrices(N: int, R: int, dim: int, seed: int = 0) -> np.ndarray:
    """Generate valid random density matrices of shape (N,R,dim,dim)."""
    rng = np.random.default_rng(seed)
    out = np.empty((N, R, dim, dim), dtype=complex)
    for i in range(N):
        for r in range(R):
            A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
            rho = A @ A.conj().T
            rho = rho / np.trace(rho)
            out[i, r] = rho
    return out


def plus_density(n: int) -> np.ndarray:
    """|+><+|^{⊗n} as a dense matrix."""
    dim = 1 << n
    psi = np.ones((dim,), dtype=complex) / np.sqrt(dim)
    return np.outer(psi, psi.conj())


def pauli_observables_1local(n: int):
    """
    Return a small deterministic observable set as SparsePauliOp:
      [X on each qubit] + [Z on each qubit]
    """
    obs = []
    for q in range(n):
        # rightmost label char is qubit 0
        lab = ["I"] * n
        lab[n - 1 - q] = "X"
        obs.append(SparsePauliOp.from_list([("".join(lab), 1.0)]))
    for q in range(n):
        lab = ["I"] * n
        lab[n - 1 - q] = "Z"
        obs.append(SparsePauliOp.from_list([("".join(lab), 1.0)]))
    return obs


def trace_expectations(rho: np.ndarray, observables) -> np.ndarray:
    """Compute expectations Tr(rho O) for dense rho, return real parts."""
    exps = []
    for op in observables:
        if isinstance(op, SparsePauliOp):
            O = op.to_matrix()
        else:
            O = np.asarray(op.data, dtype=complex)
        exps.append(np.trace(rho @ O).real)
    return np.asarray(exps, dtype=float)


def make_pub_all_lam_zero(qrc_cfg, x_window: np.ndarray, R: int, seed: int = 0):
    """
    Create a single pub (qc, param_values) where lam=0 for all reservoirs.
    This bypasses any eps constraints in higher-level builders and is ideal for a spec test:
        E_{lam=0} forces the final reservoir state to |+><+|^{⊗n}.
    """
    qc = CircuitFactory.instantiateFullIsingRingEvolution(
        qrc_cfg=qrc_cfg, angle_positioning=angle_positioning_linear, x_window=x_window
    )

    J = list(qc.metadata["J"])
    hx = list(qc.metadata["h_x"])
    hz = list(qc.metadata["h_z"])
    P = len(J) + len(hx) + len(hz) + 1

    rng = np.random.default_rng(seed)
    vals = np.empty((R, P), dtype=float)

    # same column order used by runner: J + h_x + h_z + [lam]
    for r in range(R):
        col = 0
        vals[r, col:col + len(J)] = rng.uniform(-np.pi, np.pi, size=len(J))
        col += len(J)
        vals[r, col:col + len(hx)] = rng.uniform(-np.pi, np.pi, size=len(hx))
        col += len(hx)
        vals[r, col:col + len(hz)] = rng.uniform(-np.pi, np.pi, size=len(hz))
        col += len(hz)
        vals[r, col] = 0.0  # lam=0

    return qc, vals


# ----------------------------- Fixtures ---------------------------------------

@pytest.fixture(scope="module")
def cfg_small():
    # keep this small so Aer runs fast
    return RingQRConfig(input_dim=4, num_qubits=2, seed=11)


@pytest.fixture(scope="module")
def X_small(cfg_small):
    rng = np.random.default_rng(123)
    N, w, d = 3, 2, cfg_small.input_dim
    eps = 1e-6
    return rng.uniform(-1 + eps, 1 - eps, size=(N, w, d))


@pytest.fixture(scope="module")
def pubs_small(cfg_small, X_small):
    lam_0 = 0.05
    R = 3

    pubs = call_create_pubs_dataset(
        qrc_cfg=cfg_small,
        angle_positioning=angle_positioning_linear,
        X=X_small,
        lam_0=lam_0,
        num_reservoirs=R,
        seed=999,
        eps=1e-8,
        template_pub=True,
    )
    N, _ = pubs_N_R(pubs)
    assert N == X_small.shape[0]
    return pubs


@pytest.fixture(scope="module")
def exact_results_small(cfg_small, pubs_small):
    runner = ExactAerCircuitsRunner(cfg_small)
    return runner.run_pubs(pubs=pubs_small, seed_simulator=0, optimization_level=1)


# ----------------------------- Unit tests (no Aer) ----------------------------

def test_fmps_shape_and_order_unit(cfg_small):
    n = cfg_small.num_qubits
    dim = 1 << n
    N, R = 4, 3
    states = random_density_matrices(N, R, dim, seed=7)
    results = ExactResults(states=states, qrc_cfg=cfg_small)

    obs = pauli_observables_1local(n)  # K=2n
    K = len(obs)

    retr = ExactFeatureMapsRetriever(cfg_small, obs)
    fmps = retr.get_feature_maps(results)

    assert fmps.shape == (N, R * K)

    # Check ordering for a single (i,r): columns r*K..r*K+K-1 must match direct trace
    i, r = 1, 2
    expected = trace_expectations(states[i, r], obs)
    got = fmps[i, r * K:(r + 1) * K]
    assert np.allclose(got, expected, atol=1e-12, rtol=0.0)


def test_fast_pauli_path_matches_dense_trace_unit(cfg_small):
    n = cfg_small.num_qubits
    dim = 1 << n
    N, R = 2, 2
    states = random_density_matrices(N, R, dim, seed=9)
    results = ExactResults(states=states, qrc_cfg=cfg_small)

    obs_sparse = pauli_observables_1local(n)
    # Force dense path by converting to Operator
    obs_dense = [Operator(op.to_matrix()) for op in obs_sparse]

    fm_fast = ExactFeatureMapsRetriever(cfg_small, obs_sparse).get_feature_maps(results)
    fm_dense = ExactFeatureMapsRetriever(cfg_small, obs_dense).get_feature_maps(results)

    assert fm_fast.shape == fm_dense.shape
    assert np.allclose(fm_fast, fm_dense, atol=1e-12, rtol=0.0)


# ----------------------------- Integration tests (Aer run_pubs) ----------------

def test_fmps_from_run_pubs_shapes_and_determinism(cfg_small, pubs_small):
    runner = ExactAerCircuitsRunner(cfg_small)
    res1 = runner.run_pubs(pubs=pubs_small, seed_simulator=0, optimization_level=1)
    res2 = runner.run_pubs(pubs=pubs_small, seed_simulator=0, optimization_level=1)

    n = cfg_small.num_qubits
    obs = pauli_observables_1local(n)
    K = len(obs)

    fm1 = ExactFeatureMapsRetriever(cfg_small, obs).get_feature_maps(res1)
    fm2 = ExactFeatureMapsRetriever(cfg_small, obs).get_feature_maps(res2)

    N, R = pubs_N_R(pubs_small)
    assert fm1.shape == (N, R * K)
    assert np.allclose(fm1, fm2, atol=0.0, rtol=0.0)  # exact determinism expected


def test_fmps_matches_direct_trace_on_run_pubs(cfg_small, exact_results_small):
    res = exact_results_small
    states = res.states  # (N,R,dim,dim)
    N, R, dim, _ = states.shape
    n = cfg_small.num_qubits

    # test a slightly richer observable set
    obs = [
        SparsePauliOp.from_list([("I" * n, 1.0)]),  # identity
        SparsePauliOp.from_list([("X" + "I" * (n - 1), 1.0)]),  # X on top qubit
        SparsePauliOp.from_list([("I" * (n - 1) + "Z", 1.0)]),  # Z on qubit 0
    ]
    K = len(obs)

    fmps = ExactFeatureMapsRetriever(cfg_small, obs).get_feature_maps(res)

    # check a few random entries (i,r) quantitatively
    for (i, r) in [(0, 0), (1, 2), (2, 1)]:
        expected = trace_expectations(states[i, r], obs)
        got = fmps[i, r * K:(r + 1) * K]
        assert np.allclose(got, expected, atol=1e-12, rtol=0.0)


def test_lam_zero_forces_plus_state_and_expected_observable_values(cfg_small):
    """
    Strong spec test:
      If lam=0 for the final step(s), the reservoir state must be |+><+|^{⊗n}.
    Then:
      <X_i> = 1, <Z_i> = 0 for each qubit i.
    """
    n = cfg_small.num_qubits
    R = 3

    # small window (w=1) of zeros keeps z=0, but this test should pass regardless because lam=0 overwrites state
    x_window = np.zeros((1, cfg_small.input_dim), dtype=float)

    pub = make_pub_all_lam_zero(cfg_small, x_window=x_window, R=R, seed=0)
    runner = ExactAerCircuitsRunner(cfg_small)
    res = runner.run_pubs(pubs=[pub], seed_simulator=0, optimization_level=1)

    rho_plus = plus_density(n)

    # 1) states themselves are plus
    for r in range(R):
        assert np.allclose(res.states[0, r], rho_plus, atol=1e-12, rtol=0.0)

    # 2) feature maps yield <X_i>=1 and <Z_i>=0
    obs = pauli_observables_1local(n)  # X on each qubit then Z on each qubit
    K = len(obs)
    fmps = ExactFeatureMapsRetriever(cfg_small, obs).get_feature_maps(res)

    # fmps has shape (1, R*K)
    for r in range(R):
        vec = fmps[0, r * K:(r + 1) * K]
        x_vals = vec[:n]
        z_vals = vec[n:2 * n]
        assert np.allclose(x_vals, np.ones(n), atol=1e-12, rtol=0.0)
        assert np.allclose(z_vals, np.zeros(n), atol=1e-12, rtol=0.0)
