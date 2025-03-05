import itertools

import numpy as np
import pytest
from scipy import stats

from corrpops.linalg import matrix
from corrpops.model.link_functions import MultiplicativeIdentity
from corrpops.simulation import wishart, sample
from tests.tests_utils import from_eigh, v_w


def test_generate_covariance_with_random_effect(parameters):
    theta, _, _ = parameters
    theta_new = {
        random_effect: wishart.generate_covariance_with_random_effect(
            theta, random_effect=random_effect
        )
        for random_effect in (0.0, 1e-05, 1e-4, 1e-01)
    }
    np.testing.assert_allclose(theta, theta_new[0.0])
    np.testing.assert_allclose(theta, theta_new[1e-05], atol=0.01)
    np.testing.assert_allclose(theta, theta_new[1e-04], atol=0.1)
    assert np.max(np.abs(theta - theta_new[1e-01])) > 0.1

    with pytest.raises(ValueError):
        wishart.generate_covariance_with_random_effect(theta, random_effect=-1.0)


@pytest.mark.parametrize(
    "df, size",
    itertools.product(
        [10, 20],
        [1, (1,), 12, 40, (23, 44,), (13,)]
    ),
)
def test_multivariate_normal_rvs(parameters, df, size):
    size_ = size if isinstance(size, tuple) else (size,)
    _, _, sigma = parameters

    np.testing.assert_array_equal(
        wishart.multivariate_normal_rvs(
            df=df,
            scale=sigma,
            random_effect=0.0,
            size=size,
            random_state=142,
        ),
        stats.multivariate_normal.rvs(
            cov=sigma,
            size=size_ + (df,),
            random_state=142,
        ),
    )

    if size_ == (1,):
        expected_size = (df, sigma.shape[0])
    else:
        expected_size = size_ + (df, sigma.shape[0])

    result = wishart.multivariate_normal_rvs(
        df=df,
        scale=sigma,
        random_effect=stats.uniform.rvs(scale=100),
        size=size,
        random_state=142,
    )
    assert result.shape == expected_size


@pytest.mark.parametrize(
    "size",
    [1, (1,), 12, 40, (23, 44,), (13,)],
)
def test_generalized_wishart_rvs(parameters, size):
    _, _, sigma = parameters
    p = sigma.shape[-1]

    np.testing.assert_array_equal(
        wishart.generalized_wishart_rvs(
            df=int(p * 2),
            scale=sigma,
            random_effect=0.0,
            size=size,
            random_state=142,
        ),
        stats.wishart.rvs(
            df=int(p * 2),
            scale=sigma,
            size=size,
            random_state=142,
        ),
    )

    positive_definite = wishart.generalized_wishart_rvs(
        df=int(p * 2),
        scale=sigma,
        random_effect=stats.uniform.rvs(scale=10),
        size=size,
        random_state=142,
    )
    assert matrix.is_positive_definite(positive_definite).all()

    not_positive_definite = wishart.generalized_wishart_rvs(
        df=int(p / 2),
        scale=sigma,
        random_effect=0.0,
        size=size,
        random_state=142,
    )
    assert not matrix.is_positive_definite(not_positive_definite).all()

    v, w = v_w(p, random_state=7_689)
    positive_semidefinite = wishart.generalized_wishart_rvs(
        df=int(p * 2),
        scale=from_eigh(v, np.arange(p) > 0),
        random_effect=0.0,
        size=size,
        random_state=142,
    )
    assert not matrix.is_positive_definite(positive_semidefinite).all()

    with pytest.raises(ValueError):  # not positive_semidefinite
        scale = from_eigh(v, w - 1)
        wishart.generalized_wishart_rvs(
            df=2,
            scale=scale,
            random_effect=0.0,
            size=size,
            random_state=142,
        )

    with pytest.raises(ValueError):  # not symmetric
        scale = v @ v
        wishart.generalized_wishart_rvs(
            df=2,
            scale=scale,
            random_effect=0.0,
            size=size,
            random_state=142,
        )


@pytest.mark.parametrize(
    "size",
    [1, (1,), 12, 40, (23, 44,), (13,)],
)
def test_arma_wishart_rvs_same(parameters, size):
    _, _, sigma = parameters
    p = sigma.shape[-1]

    np.testing.assert_array_equal(
        wishart.arma_wishart_rvs(
            df=int(p * 2),
            scale=sigma,
            ar=0.0,
            ma=0.0,
            random_effect=0.0,
            size=size,
            random_state=142,
        ),
        stats.wishart.rvs(
            df=int(p * 2),
            scale=sigma,
            size=size,
            random_state=142,
        ),
    )
    np.testing.assert_array_equal(
        wishart.arma_wishart_rvs(
            df=int(p * 2),
            scale=sigma,
            ar=0.0,
            ma=0.0,
            random_effect=1.0,
            size=size,
            random_state=142,
        ),
        wishart.generalized_wishart_rvs(
            df=int(p * 2),
            scale=sigma,
            random_effect=1.0,
            size=size,
            random_state=142,
        ),
    )
    with pytest.raises(ValueError):
        wishart.arma_wishart_rvs(
            df=int(p * 2),
            scale=sigma,
            ar=(0.1, 0.9),
            ma=0.0,
            random_effect=0.0,
            size=size,
            random_state=142,
        ),
    with pytest.raises(ValueError):
        wishart.arma_wishart_rvs(
            df=int(p * 2),
            scale=sigma,
            ar=0.0,
            ma=(0.1, 0.9),
            random_effect=0.0,
            size=size,
            random_state=142,
        ),


@pytest.mark.parametrize(
    "df, ar, ma, random_effect, size",
    itertools.product(
        [0.5, 2],
        [0.0, 0.5, (0.5, 0.2)],
        [0.0, 0.5, (0.4, 0.1)],
        [0.0, 0.1, 1.0],
        [1, (1,), 12, (23, 44,)],
    ),
)
def test_arma_wishart_rvs(parameters, df, ar, ma, random_effect, size):
    size_ = size if isinstance(size, tuple) else (size,)
    _, _, sigma = parameters
    p = sigma.shape[-1]

    x = wishart.arma_wishart_rvs(
        df=int(p * df),
        scale=sigma,
        ar=ar,
        ma=ma,
        random_effect=random_effect,
        size=size,
        random_state=142,
    )
    if size_ == (1,):
        expected_size = (p, p)
    else:
        expected_size = size_ + (p, p)

    assert x.shape == expected_size


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
