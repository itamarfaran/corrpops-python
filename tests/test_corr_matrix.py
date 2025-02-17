import numpy as np
from model.covariance_of_correlation import covariance_of_correlation
from scipy import stats
from linalg.matrix import cov_to_corr
from statsmodels.stats.moment_helpers import cov2corr


N = 100
P = 4


def test_this():
    putin = stats.wishart(5 * P, np.eye(P)).rvs(N).reshape((N, P, P))
    for i in range(N):
        assert np.allclose(cov_to_corr(putin)[i], cov2corr(putin[i]))
    putin = cov_to_corr(putin)

    result = covariance_of_correlation(putin)
    print(result.max())
