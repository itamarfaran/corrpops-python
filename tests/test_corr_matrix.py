import numpy as np
from corr_matrix_covariance import corr_matrix_covariance
from scipy import stats
from utils import cov_to_corr
from statsmodels.stats.moment_helpers import cov2corr


N = 100
P = 4


def test_this():
    putin = stats.wishart(5 * P, np.eye(P)).rvs(N).reshape((N, P, P))
    for i in range(N):
        assert np.allclose(cov_to_corr(putin)[i], cov2corr(putin[i]))
    putin = cov_to_corr(putin)

    result = corr_matrix_covariance(putin)
    print(result.max())
