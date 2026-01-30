import numpy as np
import pytest

from src.data.generate.beta_mixing import BetaMixingGenerator
from src.data.label.base import LabelNoise
from src.data.label.functionals import (
    OneStepForecastFunctional,
    ExpFadingLinearFunctional,
    VolteraFunctional,
)


class DeterministicBetaMixingGen(BetaMixingGenerator):
    """Deterministic path to make stability tests non-flaky.

    path[t, k] = scale * (k+1) * t
    """

    def __init__(self, *, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = float(scale)

    def simulate_path(self, *, T: int, d: int, rng: np.random.Generator) -> np.ndarray:
        t = np.arange(T, dtype=float)  # (T,)
        path = np.stack([(k + 1) * t for k in range(d)], axis=1)  # (T,d)
        return self.scale * path


@pytest.mark.parametrize(
    "functional",
    [
        OneStepForecastFunctional(u=None, nonlinearity="none", noise=LabelNoise(0.0)),
        ExpFadingLinearFunctional(alpha=0.7, u=None, nonlinearity="none", noise=LabelNoise(0.0)),
    ],
)
def test_u_is_materialized_once_and_reused(functional):
    """If u is not provided, the functional must not resample it per window."""
    rng_X = np.random.default_rng(0)
    X_win = rng_X.normal(size=(8, 3))

    # First call materializes u (using rng1) then caches it.
    rng1 = np.random.default_rng(123)
    y1 = functional(X_win, rng1)

    # Second call should *not* depend on rng2 anymore (since u is cached).
    rng2 = np.random.default_rng(999)
    y2 = functional(X_win, rng2)

    assert np.isfinite(y1) and np.isfinite(y2)
    assert y1 == pytest.approx(y2, rel=0.0, abs=1e-12)

    # White-box guardrail: ensure the cache is actually populated.
    assert getattr(functional, "_u_cached") is not None


def test_volterra_u_v_materialized_once_and_v_orthogonal_to_u_when_requested():
    rng_X = np.random.default_rng(0)
    X_win = rng_X.normal(size=(10, 5))

    f = VolteraFunctional(
        alpha=0.85,
        u=None,
        v=None,
        orthogonalize_v=True,
        nonlinearity="none",
        noise=LabelNoise(0.0),
    )

    rng1 = np.random.default_rng(123)
    y1 = f(X_win, rng1)

    rng2 = np.random.default_rng(999)
    y2 = f(X_win, rng2)

    assert y1 == pytest.approx(y2, rel=0.0, abs=1e-12)

    u = getattr(f, "_u_cached")
    v = getattr(f, "_v_cached")
    assert u is not None and v is not None

    u = np.asarray(u, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)

    assert np.linalg.norm(u) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert np.linalg.norm(v) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert abs(float(np.dot(u, v))) < 1e-10


def test_make_windows_dataset_labels_are_reproducible_with_materialized_functionals():
    """Guardrail against the pathology: 'functional changes per window'.

    After generating a dataset, re-evaluating the *same* label_functionals on ds.X
    should reproduce ds.y exactly (with noise=0).
    """
    gen = DeterministicBetaMixingGen(
        scale=0.1,
        s=3,
        burn_in=5,
        bounded="none",
        seed=7,
    )

    f1 = OneStepForecastFunctional(u=None, nonlinearity="none", noise=LabelNoise(0.0))
    f2 = ExpFadingLinearFunctional(alpha=0.7, u=None, nonlinearity="none", noise=LabelNoise(0.0))
    f3 = VolteraFunctional(alpha=0.85, u=None, v=None, orthogonalize_v=True, nonlinearity="none", noise=LabelNoise(0.0))

    ds = gen.make_windows_dataset(N=25, w=8, d=4, label_functionals=[f1, f2, f3])
    assert ds.y.shape == (3, 25)
    assert np.all(np.isfinite(ds.y))

    # Re-evaluate using a fresh RNG. If functionals were resampling u/v per call,
    # these would NOT match ds.y.
    rng_re = np.random.default_rng(999)
    y_re = np.empty_like(ds.y)
    for i in range(ds.X.shape[0]):
        X_win = ds.X[i]
        y_re[0, i] = f1(X_win, rng_re)
        y_re[1, i] = f2(X_win, rng_re)
        y_re[2, i] = f3(X_win, rng_re)

    assert np.allclose(y_re, ds.y, atol=1e-12, rtol=0.0)

    # Extra sanity: labels should have non-trivial variance on this deterministic path.
    for row in ds.y:
        assert float(np.std(row)) > 1e-10
