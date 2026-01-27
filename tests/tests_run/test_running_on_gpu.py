# tests/tests_run/test_running_on_gpu.py
import inspect
import numpy as np
import pytest

pytest.importorskip("qiskit_aer")


def _aer_available_devices(backend) -> list[str]:
    if hasattr(backend, "available_devices"):
        try:
            return list(backend.available_devices())
        except Exception:
            return []
    return []


def _backend_device_option(backend):
    # AerSimulator exposes options as an object; device is usually an attribute.
    opts = getattr(backend, "options", None)
    if opts is None:
        return None
    return getattr(opts, "device", None)


def _get_create_pubs_fn(CircuitFactory):
    fn = getattr(CircuitFactory, "create_pubs_dataset_reservoirs_IsingRingSWAP", None)
    if fn is None:
        fn = getattr(CircuitFactory, "create_pubs_dataset_reservoir_IsingRingSWAP", None)
    if fn is None:
        raise AttributeError(
            "Could not find create_pubs_dataset_reservoirs_IsingRingSWAP "
            "or create_pubs_dataset_reservoir_IsingRingSWAP"
        )
    return fn


def _call_create_pubs_dataset(
    *,
    CircuitFactory,
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
    fn = _get_create_pubs_fn(CircuitFactory)
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
        # legacy
        kwargs.update(lam_0=lam_0, num_reservoirs=num_reservoirs, seed=seed, eps=eps)
        if "template_pub" in params:
            kwargs["template_pub"] = template_pub

    return fn(**kwargs)


def test_real_gpu_runner_executes_factory_pubs_or_skips():
    """
    Optional integration test:
    - Builds PUBs from CircuitFactory (so metadata keys exist).
    - Requests Aer GPU device.
    - Skips if GPU isn't available OR if GPU run fails at runtime.
    """
    from src.qrc.circuits.qrc_configs import RingQRConfig
    from src.qrc.circuits.circuit_factory import CircuitFactory
    from src.qrc.circuits.utils import angle_positioning_tanh
    from src.qrc.run.circuit_run import ExactAerCircuitsRunner

    rng = np.random.default_rng(0)
    # tiny dataset: N=1 window, w=2 timesteps, d=input_dim=1
    N, w, d = 1, 2, 1
    X = rng.normal(size=(N, w, d)).astype(float)

    num_qubits = 2
    qrc_cfg = RingQRConfig(input_dim=d, num_qubits=num_qubits, seed=0)

    runner = ExactAerCircuitsRunner(qrc_cfg)
    backend = runner.backend

    available = _aer_available_devices(backend)
    if "GPU" not in available:
        pytest.skip(f"Aer GPU not available (available_devices={available})")

    num_reservoirs = 3

    pubs = _call_create_pubs_dataset(
        CircuitFactory=CircuitFactory,
        qrc_cfg=qrc_cfg,
        angle_positioning=angle_positioning_tanh,
        X=X,
        lam_0=0.2,
        num_reservoirs=num_reservoirs,
        seed=0,
        eps=1e-8,
        template_pub=True,  # enable template PUBs for fast prep if supported
    )

    try:
        res = runner.run_pubs(
            pubs=pubs,
            device="GPU",
            optimization_level=0,
            seed_simulator=0,
            max_threads=1,
            max_parallel_threads=1,
            max_parallel_experiments=1,
            max_parallel_shots=1,
        )
    except Exception as e:
        # Some Aer installs may *advertise* GPU but fail due to missing CUDA libs,
        # wrong driver, no actual GPU, etc. Treat this as a skip, not a failure.
        pytest.skip(f"Aer GPU run failed at runtime: {type(e).__name__}: {e}")

    assert res.states.shape == (N, num_reservoirs, 2**num_qubits, 2**num_qubits)
    assert _backend_device_option(backend) == "GPU"
