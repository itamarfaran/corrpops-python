import itertools

import numpy as np
import pytest

from corrpops.linalg import matrix
from corrpops.model.link_functions import MultiplicativeIdentity
from corrpops.simulation import sample


@pytest.mark.parametrize(
    "p, dim_alpha, enforce_min_alpha",
    itertools.product(
        [10, 20],
        [1, 2],
        [False, True],
    ),
)
def test_build_parameters(p, dim_alpha, enforce_min_alpha):
    theta, alpha, sigma = sample.build_parameters(
        p=p,
        percent_alpha=0.7,
        alpha_min=0.7,
        alpha_max=0.8,
        dim_alpha=dim_alpha,
        enforce_min_alpha=enforce_min_alpha,
        random_state=847,
    )

    assert theta.shape == (p, p)
    assert matrix.is_positive_definite(theta)
    np.testing.assert_allclose(theta, theta.T)

    assert sigma.shape == (p, p)
    assert matrix.is_positive_definite(sigma)
    np.testing.assert_allclose(sigma, sigma.T)

    np.testing.assert_allclose(theta, matrix.cov_to_corr(sigma))

    assert alpha.shape == (p, dim_alpha)
    alpha_sums = alpha.sum(1)
    alpha_non_null = alpha_sums[alpha_sums < 0.999]

    np.testing.assert_allclose(np.isclose(alpha_sums, 1).mean(), 1 - 0.7)

    if enforce_min_alpha:
        np.testing.assert_allclose(alpha_non_null.min(), 0.7)
    else:
        assert alpha_non_null.min() >= 0.7
    assert alpha_non_null.max() <= 0.8


@pytest.mark.parametrize(
    "control_n, diagnosed_n, size",
    itertools.product(
        [10, 20],
        [10, 20],
        [1, 10],
    ),
)
def test_create_samples(parameters, control_n, diagnosed_n, size):
    theta, alpha, _ = parameters
    p = theta.shape[0]

    control, diagnosed = sample.create_samples(
        control_n=control_n,
        diagnosed_n=diagnosed_n,
        control_covariance=theta,
        diagnosed_covariance=theta,
        size=size,
    )
    assert matrix.is_positive_definite(control).all()
    assert control.shape == (size, control_n, p, p)

    assert matrix.is_positive_definite(diagnosed).all()
    assert diagnosed.shape == (size, diagnosed_n, p, p)

    control, diagnosed = sample.create_samples_from_parameters(
        control_n=control_n,
        diagnosed_n=diagnosed_n,
        theta=theta,
        alpha=alpha,
        link_function=MultiplicativeIdentity(),
        size=size,
    )

    assert matrix.is_positive_definite(control).all()
    assert control.shape == (size, control_n, p, p)

    assert matrix.is_positive_definite(diagnosed).all()
    assert diagnosed.shape == (size, diagnosed_n, p, p)
