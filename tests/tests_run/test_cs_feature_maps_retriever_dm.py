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


from src.qrc.run.circuit_run import ExactAerCircuitsRunner, ExactResults
from src.qrc.run.fmp_retriever import ExactFeatureMapsRetriever
from src.qrc.run.cs_fmp_retriever import CSFeatureMapsRetriever
from src.qrc.circuits.circuit_factory import CircuitFactory
from src.qrc.circuits.qrc_configs import RingQRConfig
from src.qrc.circuits.utils import angle_positioning_linear, generate_k_local_paulis, get_theoretical_shots


# ----------------------------- Helpers ---------------------------------

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


def sup_norm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


# ----------------------------- Fixtures --------------------------------

@pytest.fixture(scope="module")
def cfg_small():
    # keep small so Aer is fast
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

    fn = getattr(CircuitFactory, "create_pubs_dataset_reservoirs_IsingRingSWAP", None)
    if fn is None:
        pytest.skip("CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP not found.")

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


@pytest.fixture(scope="module")
def obs_k1(cfg_small):
    # k-local Pauli set (excluding identity) generated by your utils
    locality = 1
    obs = generate_k_local_paulis(locality=locality, num_qubits=cfg_small.num_qubits)
    assert len(obs) > 0
    return obs


# ======================================================================
# 1) UNIT TESTS for CSFeatureMapsRetriever
# ======================================================================

def test_cs_requires_shots_if_no_default(cfg_small, obs_k1):
    dim = 1 << cfg_small.num_qubits
    states = random_density_matrices(N=2, R=2, dim=dim, seed=0)
    results = ExactResults(states=states, qrc_cfg=cfg_small)

    cs = CSFeatureMapsRetriever(cfg_small, obs_k1, default_shots=None)
    with pytest.raises(ValueError):
        _ = cs.get_feature_maps(results)  # must pass shots if no default


def test_cs_validates_states_shape(cfg_small, obs_k1):
    # wrong shape -> should raise
    bad_states = np.zeros((2, 2, 4), dtype=complex)
    results = ExactResults(states=bad_states, qrc_cfg=cfg_small)

    cs = CSFeatureMapsRetriever(cfg_small, obs_k1, default_shots=100)
    with pytest.raises(ValueError):
        _ = cs.get_feature_maps(results)


def test_cs_output_shape_and_range_unit(cfg_small, obs_k1):
    n = cfg_small.num_qubits
    dim = 1 << n
    N, R = 4, 3
    states = random_density_matrices(N=N, R=R, dim=dim, seed=7)
    results = ExactResults(states=states, qrc_cfg=cfg_small)

    shots = 200
    cs = CSFeatureMapsRetriever(cfg_small, obs_k1, default_shots=shots)
    fmps = cs.get_feature_maps(results, seed=0)

    K = len(obs_k1)
    assert fmps.shape == (N, R * K)

    # Since it's a MoM of ±1 measurements, outputs should lie in [-1, 1]
    assert np.max(fmps) <= 1.0 + 1e-12
    assert np.min(fmps) >= -1.0 - 1e-12


def test_cs_deterministic_given_seed_unit(cfg_small, obs_k1):
    n = cfg_small.num_qubits
    dim = 1 << n
    states = random_density_matrices(N=2, R=2, dim=dim, seed=9)
    results = ExactResults(states=states, qrc_cfg=cfg_small)

    cs = CSFeatureMapsRetriever(cfg_small, obs_k1, default_shots=500)

    a = cs.get_feature_maps(results, seed=123)
    b = cs.get_feature_maps(results, seed=123)
    assert np.allclose(a, b, atol=0.0, rtol=0.0)

    c = cs.get_feature_maps(results, seed=124)
    # Typically differs; we only require "not identical everywhere"
    assert not np.allclose(a, c, atol=0.0, rtol=0.0)


# ======================================================================
# 2) INTEGRATION TESTS with ExactAerCircuitsRunner.run_pubs
# ======================================================================

def test_cs_integration_shapes_and_sane_outputs(cfg_small, exact_results_small, obs_k1):
    res = exact_results_small
    N, R, dim1, dim2 = res.states.shape
    assert dim1 == dim2 == (1 << cfg_small.num_qubits)

    cs = CSFeatureMapsRetriever(cfg_small, obs_k1, default_shots=300)
    fmps = cs.get_feature_maps(res, seed=0)

    K = len(obs_k1)
    assert fmps.shape == (N, R * K)
    assert np.isfinite(fmps).all()


def test_cs_matches_exact_on_average_for_large_shots(cfg_small, exact_results_small, obs_k1):
    """
    With large shots, CS fmps should be close to Exact fmps.
    We use a moderate tolerance on sup-norm for stability.
    """
    res = exact_results_small
    exact = ExactFeatureMapsRetriever(cfg_small, obs_k1).get_feature_maps(res)

    cs = CSFeatureMapsRetriever(cfg_small, obs_k1)
    approx = cs.get_feature_maps(res, shots=20000, seed=0)

    err = sup_norm(approx, exact)
    # This is conservative for n=2 and k=1; adjust if you expand obs / n.
    assert err < 0.15


# ======================================================================
# 3) COMPARATIVE TEST: more shots => smaller sup-norm error (in expectation)
# ======================================================================

def test_error_decreases_with_more_shots_sup_norm(cfg_small, exact_results_small, obs_k1):
    res = exact_results_small
    exact = ExactFeatureMapsRetriever(cfg_small, obs_k1).get_feature_maps(res)

    cs = CSFeatureMapsRetriever(cfg_small, obs_k1)

    shots_small = 200
    shots_large = 20000

    errs_small = []
    errs_large = []
    for seed in range(20):
        a = cs.get_feature_maps(res, shots=shots_small, seed=seed)
        b = cs.get_feature_maps(res, shots=shots_large, seed=seed)
        errs_small.append(sup_norm(a, exact))
        errs_large.append(sup_norm(b, exact))

    mean_small = float(np.mean(errs_small))
    mean_large = float(np.mean(errs_large))

    # With 100x more shots, error should drop substantially.
    assert mean_large < 0.6 * mean_small


# ======================================================================
# 4) THEORETICAL SHOTS TEST via get_theoretical_shots
# ======================================================================

def test_theoretical_shots_give_eps_sup_norm_high_prob(cfg_small, exact_results_small, obs_k1):
    """
    Validate that using get_theoretical_shots yields a sup-norm error <= eps
    "with high probability" (approximated by checking multiple RNG seeds).

    This test is designed to be stable by choosing a small eps and a tiny delta,
    making shots sufficiently large.
    """
    res = exact_results_small
    exact = ExactFeatureMapsRetriever(cfg_small, obs_k1).get_feature_maps(res)

    N = res.states.shape[0]  # num data points
    R = res.states.shape[1]  # num reservoirs
    K = len(obs_k1)  # num observables
    locality = 1

    eps = 0.20
    delta = 1e-6

    shots = get_theoretical_shots(
        eps=eps,
        delta=delta,
        locality=locality,
        num_data_pts=N,
        num_obs=K,
        num_draws=R,
    )

    cs = CSFeatureMapsRetriever(cfg_small, obs_k1)

    # "High probability" check: across a handful of seeds, all should pass comfortably.
    # (Shots is large enough that failure here would indicate a bug, not randomness.)
    for seed in range(10):
        approx = cs.get_feature_maps(res, shots=shots, seed=seed)
        err = sup_norm(approx, exact)
        assert err <= 1.10 * eps
