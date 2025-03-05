import itertools

import numpy as np
import pytest
from scipy import stats

from corrpops.linalg import matrix, triangle_and_vector as tv
from corrpops.model.covariance_of_correlation import (
    average_covariance_of_correlation,
    covariance_of_correlation,
    covariance_of_fisher_correlation,
    estimated_df,
)
from tests.tests_utils import from_eigh, v_w


@pytest.mark.parametrize(
    "n, p",
    itertools.product(
        [1, 10],
        [4, 6, 8],
    ),
)
def test_covariance_of_correlation_shape(n: int, p: int):
    v, w = v_w(p)
    scale = matrix.cov_to_corr(from_eigh(v, w + 1))

    matrices = stats.wishart.rvs(
        df=5 * p,
        scale=scale,
        size=n,
    ).reshape((n, p, p))
    matrices = matrix.cov_to_corr(matrices)

    result = covariance_of_correlation(matrices)
    assert result.shape[:-2] == matrices.shape[:-2]
    assert result.shape[-2] == result.shape[-1] == tv.vectorized_dim(p)

    scale = matrix.cov_to_corr(from_eigh(v, w))
    with pytest.raises(ValueError):
        covariance_of_correlation(scale, non_positive="raise")

    with pytest.warns():
        covariance_of_correlation(scale, non_positive="warn")

    covariance_of_correlation(scale, non_positive="ignore")


@pytest.mark.parametrize(
    "p, fisher, use_numba",
    [
        (4, False, False),
        (4, True, True),
        (8, False, True),
    ],
)
def test_covariance_of_correlation_empirical(p, fisher, use_numba):
    v, _ = v_w(p)
    w = np.linspace(1, 2, p)
    scale = matrix.cov_to_corr(from_eigh(v, w))

    df = 100_000
    size = 100_000
    empirical = matrix.cov_to_corr(
        stats.wishart.rvs(
            df=df,
            scale=scale,
            size=size,
            random_state=7593,
        ),
    )
    if fisher:
        row, col = np.diag_indices(p)
        empirical[..., row, col] = scale[..., row, col] = 0
        empirical, scale = np.arctanh(empirical), np.arctanh(scale)
        expected_covariance = covariance_of_fisher_correlation(
            scale, use_numba=use_numba
        )
    else:
        expected_covariance = covariance_of_correlation(scale, use_numba=use_numba)

    empirical_covariance = df * np.cov(tv.triangle_to_vector(empirical), rowvar=False)
    np.testing.assert_allclose(
        empirical_covariance,
        expected_covariance,
        atol=np.power(p / size, 1 / 2.5),
        rtol=0,
    )


@pytest.mark.parametrize("p", [4, 6])
def test_estimated_df(p):
    v, w = v_w(p)
    mat1 = from_eigh(v, w + 1)
    np.testing.assert_allclose(estimated_df(mat1, 0.2 * mat1, only_diag=True), 0.2)
    np.testing.assert_allclose(estimated_df(mat1, 0.2 * mat1, only_diag=False), 0.2)


@pytest.mark.parametrize(
    "p, est_n",
    [
        (6, False),
        (6, True),
    ],
)
def test_average_covariance_of_correlation(p, est_n):
    v, w = v_w(p)
    scale = matrix.cov_to_corr(from_eigh(v, w + 1))

    df = 1_000
    size = 100
    empirical = matrix.cov_to_corr(
        stats.wishart.rvs(
            df=df,
            scale=scale,
            size=size,
            random_state=7593,
        ),
    )
    expected_covariance = covariance_of_correlation(scale)
    empirical_covariance, _ = average_covariance_of_correlation(empirical, est_n=est_n)

    if not est_n:
        np.testing.assert_allclose(
            empirical_covariance,
            expected_covariance,
            atol=1 / size,
            rtol=0,
        )
