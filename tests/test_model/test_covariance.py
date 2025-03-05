import itertools
import warnings

import numpy as np
import pytest

from corrpops.linalg import triangle_and_vector as tv
from corrpops.model.estimator.covariance import (
    GeeCovarianceEstimator,
    FisherSandwichCovarianceEstimator,
)
from corrpops.model.estimator.optimizer import (
    CorrPopsOptimizerResults,
)
from corrpops.model.link_functions import MultiplicativeIdentity


def test_covariance_get_params():
    cov_est = GeeCovarianceEstimator()

    for param, value in [
        ("est_mu", False),
        ("df_method", "efron"),
    ]:
        assert getattr(cov_est, param) != value
        assert getattr(cov_est, param) == cov_est.get_params()[param]
        cov_est.set_params(**{param: value})
        assert getattr(cov_est, param) == value
        assert getattr(cov_est, param) == cov_est.get_params()[param]

    for param, value in [
        ("this_is", 0.1),
        ("an_example", 0.1),
    ]:
        with pytest.raises(ValueError):
            cov_est.set_params(**{param: value})
        with pytest.raises(KeyError):
            _ = cov_est.get_params()[param]


@pytest.mark.parametrize(
    "est_mu, df_method",
    itertools.product(
        [True, False],
        ["naive", "efron"],
    ),
)
def test_gee_covariance(parameters, est_mu, df_method):
    link_function = MultiplicativeIdentity()
    theta, alpha, _ = parameters
    g11 = link_function(tv.triangle_to_vector(theta), alpha, alpha.shape[1])

    control = tv.triangle_to_vector(np.stack([theta] * 8))
    diagnosed = tv.triangle_to_vector(np.stack([g11] * 8))

    optimizer_results = CorrPopsOptimizerResults(
        theta=tv.triangle_to_vector(theta),
        alpha=alpha.flatten(),
        inv_cov=None,
        link_function=link_function.name,
        p=alpha.shape[0],
        dim_alpha=alpha.shape[1],
        steps=[],
    )

    cov_est = GeeCovarianceEstimator(est_mu=est_mu, df_method=df_method)
    result = cov_est.compute(
        control_arr=control,
        diagnosed_arr=diagnosed,
        link_function=link_function,
        optimizer_results=optimizer_results,
        non_positive="ignore",
    )
    # no variance in static data
    np.testing.assert_allclose(result, 0.0, rtol=1.0, atol=5e-03)


@pytest.mark.parametrize(
    "est_mu, estimated_n",
    itertools.product(
        [True, False],
        [True, False],
    ),
)
def test_fisher_covariance(parameters, est_mu, estimated_n):
    link_function = MultiplicativeIdentity()
    theta, alpha, _ = parameters
    g11 = link_function(tv.triangle_to_vector(theta), alpha, alpha.shape[1])

    control = tv.triangle_to_vector(np.stack([theta] * 8))
    diagnosed = tv.triangle_to_vector(np.stack([g11] * 8))

    optimizer_results = CorrPopsOptimizerResults(
        theta=tv.triangle_to_vector(theta),
        alpha=alpha.flatten(),
        inv_cov=None,
        link_function=link_function.name,
        p=alpha.shape[0],
        dim_alpha=alpha.shape[1],
        steps=[],
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="FisherSandwichCovarianceEstimator is experimental"
        )
        cov_est = FisherSandwichCovarianceEstimator(
            est_mu=est_mu, estimated_n=estimated_n
        )
    assert isinstance(cov_est.get_params(), dict)

    result = cov_est.compute(
        control_arr=control,
        diagnosed_arr=diagnosed,
        link_function=link_function,
        optimizer_results=optimizer_results,
        non_positive="ignore",
    )
    # no variance in static data
    np.testing.assert_allclose(result, 0.0, rtol=1.0, atol=1e-05)


@pytest.mark.parametrize("est_mu", [True, False])
def test_compare_methods(parameters_and_sample, est_mu):
    link_function = MultiplicativeIdentity()
    theta, alpha, control, _ = parameters_and_sample

    control = tv.triangle_to_vector(control)
    diagnosed = tv.triangle_to_vector(
        np.stack([link_function(c, alpha, alpha.shape[1]) for c in control])
    )

    optimizer_results = CorrPopsOptimizerResults(
        theta=tv.triangle_to_vector(theta),
        alpha=alpha.flatten(),
        inv_cov=None,
        link_function=link_function.name,
        p=alpha.shape[0],
        dim_alpha=alpha.shape[1],
        steps=[],
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="FisherSandwichCovarianceEstimator is experimental"
        )
        fisher = FisherSandwichCovarianceEstimator(est_mu=est_mu)
    gee = GeeCovarianceEstimator(est_mu=est_mu)

    fisher_results, gee_results = (
        est.compute(
            control_arr=control,
            diagnosed_arr=diagnosed,
            link_function=link_function,
            optimizer_results=optimizer_results,
            non_positive="ignore",
        )
        for est in (fisher, gee)
    )
    np.testing.assert_allclose(
        fisher_results,
        gee_results,
        atol=1e-02,
    )
