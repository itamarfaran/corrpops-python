import itertools

import numpy as np
import pytest
from scipy import stats

from corrpops.linalg import matrix, triangle_and_vector as tv
from corrpops.model.covariance_of_correlation import (
    covariance_of_correlation,
    covariance_of_fisher_correlation,
)

DF = 100_000
SIZE = 100_000


@pytest.mark.parametrize(
    "n, p",
    itertools.product(
        [1, 10],
        [4, 6, 8],
    ),
)
def test_covariance_of_correlation_shape(n: int, p: int):
    v = stats.ortho_group.rvs(p, random_state=3548)
    w = np.arange(p)
    scale = matrix.cov_to_corr(np.linalg.multi_dot((v, np.diag(w + 1), v.T)))

    matrices = stats.wishart.rvs(
        df=5 * p,
        scale=scale,
        size=n,
    ).reshape((n, p, p))
    matrices = matrix.cov_to_corr(matrices)

    result = covariance_of_correlation(matrices)
    assert result.shape[:-2] == matrices.shape[:-2]
    assert result.shape[-2] == result.shape[-1] == tv.vectorized_dim(p)

    scale = matrix.cov_to_corr(np.linalg.multi_dot((v, np.diag(w - 1), v.T)))
    with pytest.raises(ValueError):
        covariance_of_correlation(scale, non_positive="raise")

    with pytest.warns():
        covariance_of_correlation(scale, non_positive="warn")

    covariance_of_correlation(scale, non_positive="ignore")


@pytest.mark.parametrize(
    "p, fisher",
    itertools.product(
        [4, 8],
        [False, True],
    ),
)
def test_covariance_of_correlation_empirical(p, fisher):
    v = stats.ortho_group.rvs(p, random_state=3548)
    w = np.linspace(1, 2, p)
    scale = matrix.cov_to_corr(np.linalg.multi_dot((v, np.diag(w), v.T)))

    empirical = matrix.cov_to_corr(
        stats.wishart.rvs(
            df=DF,
            scale=scale,
            size=SIZE,
            random_state=7593,
        ),
    )
    if fisher:
        empirical, scale = np.arctanh(empirical), np.arctanh(scale)
        expected_covariance = covariance_of_fisher_correlation(scale)
    else:
        expected_covariance = covariance_of_correlation(scale)

    empirical_covariance = DF * np.cov(tv.triangle_to_vector(empirical), rowvar=False)
    np.testing.assert_allclose(
        empirical_covariance,
        expected_covariance,
        atol=np.power(p / SIZE, 1 / 2.5),
        rtol=0,
    )
