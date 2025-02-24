import numpy as np
import pytest
from scipy import stats

from corrpops.linalg import matrix
from corrpops.statistics import efron_rms


@pytest.mark.parametrize("size", [1, (1,), 2, 12, (23, 44)])
def test_efron_rms(size):
    if size in (1, (1,)):
        size_ = ()
    elif not isinstance(size, tuple):
        size_ = (size,)
    else:
        size_ = size

    v = stats.ortho_group.rvs(5, random_state=7549)
    w = np.arange(5)
    scale = v @ np.diag(1 + w) @ v.T

    p = scale.shape[0]
    df = 2 * p

    arr = matrix.cov_to_corr(stats.wishart.rvs(df=df, scale=scale, size=size))

    for bias in (False, True):
        rms = efron_rms.efron_rms(arr, df if bias else None)
        n_eff = efron_rms.efron_effective_sample_size(df, rms)
        assert n_eff.shape == size_

    with pytest.raises(ValueError):
        efron_rms.efron_rms(np.arange(20).reshape((4, 5)))

    with pytest.raises(ValueError):
        efron_rms.efron_rms(np.arange(16).reshape((4, 4)))

    with pytest.raises(ValueError):
        efron_rms.efron_rms(v @ np.diag(w) @ v.T, check_psd=True)

    with pytest.raises(ValueError):
        efron_rms.efron_rms(
            matrix.cov_to_corr(v @ np.diag(w) @ v.T),
            check_psd=True,
        )
