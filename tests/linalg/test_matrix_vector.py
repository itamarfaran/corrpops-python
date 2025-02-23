import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.moment_helpers import cov2corr

from corrpops.linalg import matrix, vector
from linalg.matrix import cov_to_corr
from tests.model.test_covariance_of_correlation import N_P_VALUES


def test_matrix_power():
    mat = np.random.random(size=(5, 10))
    mat = np.cov(mat)

    mat_sqr = matrix.matrix_power(mat, 2)
    np.testing.assert_allclose(
        mat_sqr,
        mat @ mat,
    )
    mat_sqrt = matrix.matrix_power(mat, 1 / 2)
    np.testing.assert_allclose(
        mat_sqrt @ mat_sqrt,
        mat,
    )


def test_force_symmetry():
    a = np.arange(5 * 4 * 4).reshape(5, 4, 4)
    a_sym = matrix.force_symmetry(a)

    for a_ in a_sym:
        np.testing.assert_allclose(a_, a_.T)

    with pytest.raises(ValueError):
        a = np.arange(5 * 4 * 5).reshape(5, 4, 5)
        matrix.force_symmetry(a)


def test_is_positive_definite():
    v = stats.ortho_group.rvs(5, random_state=7549)
    w = np.arange(5)

    assert matrix.is_positive_definite(v @ np.diag(w + 1) @ v.T)
    assert not matrix.is_positive_definite(v @ np.diag(w - 1) @ v.T)
    assert not matrix.is_positive_definite(v @ np.diag(w + 1) @ v)


@pytest.mark.parametrize("n, p", N_P_VALUES)
def test_cov_to_corr(n: int, p: int):
    matrices = stats.wishart.rvs(
        df=5 * p,
        scale=np.eye(p),
        size=n,
    ).reshape((n, p, p))

    for i in np.random.choice(n, min(n, 5), replace=False):
        np.testing.assert_allclose(
            cov_to_corr(matrices)[i],
            cov2corr(matrices[i]),
        )


def test_fill_other_triangle():
    a = 1 + np.arange(5 * 4 * 4).reshape(5, 4, 4)
    a = matrix.force_symmetry(a)

    row, col = np.tril_indices(4, k=-1)
    b = a.copy()
    b[..., row, col] = 0.0
    np.testing.assert_allclose(a, matrix.fill_other_triangle(b))

    row, col = np.triu_indices(4, k=1)
    c = a.copy()
    c[..., row, col] = 0.0
    np.testing.assert_allclose(a, matrix.fill_other_triangle(c))


def test_mahalanobis():
    rng = np.random.default_rng(858)
    x = rng.random(size=10)
    y = rng.random(size=10)

    np.testing.assert_array_equal(
        vector.mahalanobis(x),
        vector.norm_p(x),
    )
    np.testing.assert_array_equal(
        vector.mahalanobis(x, sqrt=False),
        vector.norm_p(x) ** 2,
    )
    np.testing.assert_array_equal(
        vector.mahalanobis(x, y),
        vector.norm_p(x, y),
    )
    np.testing.assert_array_equal(
        vector.mahalanobis(x, y, np.eye(10)),
        vector.norm_p(x, y),
    )
    np.testing.assert_array_equal(
        vector.mahalanobis(x, y, np.eye(10), solve=True),
        vector.norm_p(x, y),
    )
