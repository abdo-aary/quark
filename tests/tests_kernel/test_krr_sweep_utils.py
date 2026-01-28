import numpy as np
import pytest

from src.models.kernel import solve_krr_dual_weights, rkhs_norm_from_dual_weights


def _make_spd(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    K = A @ A.T
    return K + 1e-3 * np.eye(n)


@pytest.mark.parametrize("n,reg", [(5, 1e-8), (8, 1e-3), (12, 1e-1)])
def test_solve_krr_dual_weights_matches_numpy_solve_vector(n, reg):
    rng = np.random.default_rng(0)
    Ktt = _make_spd(n, seed=1)
    y = rng.normal(size=n)

    alpha = solve_krr_dual_weights(Ktt, y, reg=reg)

    alpha_ref = np.linalg.solve(Ktt + reg * np.eye(n), y)
    assert alpha.shape == (n,)
    assert np.allclose(alpha, alpha_ref, atol=1e-12, rtol=1e-12)


def test_solve_krr_dual_weights_matches_numpy_solve_matrix_rhs():
    n, m = 7, 3
    rng = np.random.default_rng(2)
    Ktt = _make_spd(n, seed=3)
    Y = rng.normal(size=(n, m))
    reg = 1e-6

    alpha = solve_krr_dual_weights(Ktt, Y, reg=reg)
    alpha_ref = np.linalg.solve(Ktt + reg * np.eye(n), Y)

    assert alpha.shape == (n, m)
    assert np.allclose(alpha, alpha_ref, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("bad_reg", [0.0, -1e-12, -1.0, np.nan, np.inf])
def test_solve_krr_dual_weights_rejects_bad_reg(bad_reg):
    Ktt = _make_spd(5, seed=0)
    y = np.ones(5)
    with pytest.raises(ValueError):
        solve_krr_dual_weights(Ktt, y, reg=bad_reg)


def test_solve_krr_dual_weights_rejects_shape_mismatch():
    Ktt = _make_spd(6, seed=0)
    y = np.ones(5)
    with pytest.raises(ValueError):
        solve_krr_dual_weights(Ktt, y, reg=1e-6)


def test_solve_krr_dual_weights_rejects_nonsquare_Ktt():
    Ktt = np.ones((4, 5))
    y = np.ones(4)
    with pytest.raises(ValueError):
        solve_krr_dual_weights(Ktt, y, reg=1e-6)


def test_rkhs_norm_from_dual_weights_matches_definition_vector():
    n = 9
    rng = np.random.default_rng(4)
    Ktt = _make_spd(n, seed=5)
    y = rng.normal(size=n)
    reg = 1e-4

    alpha = solve_krr_dual_weights(Ktt, y, reg=reg)

    norm = rkhs_norm_from_dual_weights(alpha, Ktt)
    norm_ref = float(np.sqrt(alpha.T @ Ktt @ alpha))

    assert isinstance(norm, float)
    assert norm >= 0.0
    assert np.allclose(norm, norm_ref, atol=1e-12, rtol=1e-12)


def test_rkhs_norm_from_dual_weights_matrix_returns_columnwise_norms():
    n, m = 10, 4
    rng = np.random.default_rng(0)
    Ktt = _make_spd(n, seed=1)
    alpha = rng.normal(size=(n, m))

    norms = rkhs_norm_from_dual_weights(alpha, Ktt)
    assert norms.shape == (m,)

    # check each column matches scalar formula
    for j in range(m):
        col = alpha[:, j]
        ref = float(np.sqrt(col.T @ Ktt @ col))
        assert np.allclose(norms[j], ref, atol=1e-12, rtol=1e-12)


def test_rkhs_norm_from_dual_weights_handles_zero_alpha():
    Ktt = _make_spd(4, seed=0)
    alpha = np.zeros(4)
    assert rkhs_norm_from_dual_weights(alpha, Ktt) == 0.0


def test_effective_rkhs_norm_decreases_when_reg_increases_typically():
    """
    Sanity test (not a theorem): for a typical SPD K and generic y,
    increasing ridge reg tends to shrink the RKHS norm of the solution.
    """
    n = 15
    rng = np.random.default_rng(0)
    Ktt = _make_spd(n, seed=0)
    y = rng.normal(size=n)

    regs = [1e-10, 1e-6, 1e-2, 1e0]
    norms = []
    for reg in regs:
        alpha = solve_krr_dual_weights(Ktt, y, reg=reg)
        norms.append(rkhs_norm_from_dual_weights(alpha, Ktt))

    for a, b in zip(norms, norms[1:]):
        assert b <= a + 1e-10


def test_rkhs_norm_from_dual_weights_rejects_shape_mismatch():
    Ktt = _make_spd(6, seed=0)
    alpha = np.ones(5)
    with pytest.raises(ValueError):
        rkhs_norm_from_dual_weights(alpha, Ktt)
