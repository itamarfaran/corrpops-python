import itertools

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.moment_helpers import cov2corr

from linalg.matrix import cov_to_corr
from model.covariance_of_correlation import covariance_of_correlation

N_P_VALUES = list(
    itertools.product(
        [1, 10],
        [4, 6, 8],
    )
)


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


@pytest.mark.parametrize("n, p", N_P_VALUES)
def test_covariance_of_correlation(n: int, p: int):
    matrices = stats.wishart.rvs(
        df=5 * p,
        scale=np.eye(p),
        size=n,
    ).reshape((n, p, p))
    matrices = cov_to_corr(matrices)

    result = covariance_of_correlation(matrices)
    assert result.shape[-2] == result.shape[-1] == p * (p - 1) / 2
