import itertools

import numpy as np
import pytest

from corrpops.statistics import arma, piece_wise_comparison


def test_is_invertible_arma():
    assert arma.is_invertible_arma(0.5)
    assert arma.is_invertible_arma([0.5, 0.1])

    assert not arma.is_invertible_arma(1.5)
    assert not arma.is_invertible_arma([0.5, 0.8])


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
