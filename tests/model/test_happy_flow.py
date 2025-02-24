import itertools

import numpy as np
import pytest

from corrpops.linalg.triangle_vector import triangle_to_vector
from corrpops.model import estimator, link_functions


@pytest.mark.parametrize(
    "link_function_class, n_control, n_diagnosed",
    itertools.product(
        [link_functions.MultiplicativeIdentity],
        # todo: add AdditiveQuotient with null alpha = 0.0
        [10, 20],
        [10, 20],
    ),
)
def test_happy_flow(parameters, link_function_class, n_control, n_diagnosed):
    theta, alpha = parameters
    link_function = link_function_class()

    control = np.stack(n_control * [theta])
    diagnosed = np.stack(
        n_diagnosed * [link_function(t=triangle_to_vector(theta), a=alpha, d=1)]
    )

    model = estimator.CorrPopsEstimator(
        link_function=link_function,
        optimizer_kwargs={"verbose": False},
    ).fit(control, diagnosed)

    for results in (model.optimizer_results_, model.naive_optimizer_results_):
        np.testing.assert_allclose(results.alpha, alpha.flatten(), rtol=0.1, atol=0.01)
        np.testing.assert_allclose(results.theta, triangle_to_vector(theta), rtol=0.1, atol=0.01)
    np.testing.assert_allclose(model.cov_, 0.0, rtol=0.1, atol=0.01)  # no variance in sample
