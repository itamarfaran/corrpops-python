import itertools

import numpy as np
import pytest
from scipy import stats

from corrpops.linalg.matrix import cov_to_corr
from corrpops.model.covariance_of_correlation import covariance_of_correlation


@pytest.mark.parametrize(
    "n, p",
    itertools.product(
        [1, 10],
        [4, 6, 8],
    ),
)
def test_covariance_of_correlation(n: int, p: int):
    matrices = stats.wishart.rvs(
        df=5 * p,
        scale=np.eye(p),
        size=n,
    ).reshape((n, p, p))
    matrices = cov_to_corr(matrices)

    result = covariance_of_correlation(matrices)
    assert result.shape[-2] == result.shape[-1] == p * (p - 1) / 2
