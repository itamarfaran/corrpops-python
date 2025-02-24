import itertools

import numpy as np
import pytest
from scipy import stats

from corrpops.statistics import arma, piece_wise_comparison
from linalg import matrix
from statistics import efron_rms


def test_is_invertible_arma():
    assert arma.is_invertible_arma(0)
    assert arma.is_invertible_arma(0.5)
    assert arma.is_invertible_arma([0.5, 0.1])

    assert not arma.is_invertible_arma(1.5)
    assert not arma.is_invertible_arma([0.5, 0.8])

    with pytest.raises(ValueError):
        arma.is_invertible_arma(np.array([[0.5], [0.8]]))


@pytest.mark.parametrize("size", [1, (1,), 2, 12, (23, 44)])
def test_efron_rms(size):
    rng = np.random.default_rng(17)
    if size in (1, (1,)):
        size_ = ()
    elif not isinstance(size, tuple):
        size_ = (size,)
    else:
        size_ = size

    v = stats.ortho_group.rvs(5, random_state=rng)
    w = np.arange(5)
    scale = v @ np.diag(1 + w) @ v.T

    p = scale.shape[0]
    df = 2 * p

    arr = matrix.cov_to_corr(
        stats.wishart.rvs(df=df, scale=scale, size=size, random_state=rng)
    )

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


@pytest.mark.parametrize(
    "p_adjust_method, alternative",
    itertools.product(
        ["bonferroni", "fdr_bh"],
        ["two-sided", "smaller", "larger"],
    ),
)
def test_piece_wise_comparison(parameters_and_sample, p_adjust_method, alternative):
    _, _, control, diagnosed = parameters_and_sample
    result = piece_wise_comparison.piece_wise_comparison(
        control=control,
        diagnosed=diagnosed,
        p_adjust_method=p_adjust_method,
        alternative=alternative,
    )

    assert isinstance(result, dict)
    for k, v in result.items():
        assert isinstance(k, str)
        assert isinstance(v, np.ndarray)
