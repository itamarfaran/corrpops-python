import itertools
import warnings
from typing import Literal

import numpy as np
import pytest
from scipy import stats

from corrpops.model.inference import inference, wilks_test
from corrpops.model.link_functions import MultiplicativeIdentity
from tests.tests_utils import from_eigh, v_w

link_function = MultiplicativeIdentity()


@pytest.mark.parametrize(
    "a, p, alternative",
    itertools.product(
        [0.1, 0.05, 0.01],
        [5, 10],
        ["two-sided", "smaller", "larger"],
    ),
)
def test_inference(
    a: float,
    p: int,
    alternative: Literal["two-sided", "smaller", "larger"],
):
    alpha = np.full(p, link_function.null_value + stats.norm.ppf(1 - a))
    cov = np.eye(p)

    results = inference(
        alpha=alpha,
        cov=cov,
        link_function=link_function,
        alternative=alternative,
        known_alpha=np.full_like(alpha, link_function.null_value),
    )
    np.testing.assert_allclose(
        results["z_value"],
        stats.norm.ppf(1 - a),
    )
    if alternative == "two-sided":
        np.testing.assert_allclose(results["p_value"], 2 * a)
    elif alternative == "smaller":
        np.testing.assert_allclose(results["p_value"], 1 - a)
    else:
        np.testing.assert_allclose(results["p_value"], a)

    idx = results["p_value"] < 1 / results["p_value"].size
    np.testing.assert_allclose(
        results["p_value_adjusted"][idx] / results["p_value"][idx],
        results["p_value"].size,
    )
    np.testing.assert_allclose(results["p_value_adjusted"][~idx], 1.0)

    inference(
        alpha=alpha,
        cov=cov,
        link_function=link_function,
        p_adjust_method="fdr_bh",
    )


def test_wilks_wilks_test(parameters_and_sample, p: int = 5):
    theta, alpha, control, diagnosed = parameters_and_sample
    result = wilks_test(
        control_arr=control,
        diagnosed_arr=diagnosed,
        theta=theta,
        alpha=alpha,
        link_function=link_function,
    )
    assert isinstance(result.chi2_val, float)
    assert isinstance(result.df, int)
    assert isinstance(result.p_val, float)

    v, w = v_w(p)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="covariance matrix is not symmetric positive semidefinite, returning nan",
        )

        result = wilks_test(
            control_arr=control,
            diagnosed_arr=diagnosed,
            theta=from_eigh(v, w - 1),
            alpha=alpha,
            link_function=link_function,
            non_positive="ignore",
        )
    assert np.isnan(result.chi2_val)
    assert np.isnan(result.p_val)
