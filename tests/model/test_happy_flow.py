import itertools

import numpy as np
import pytest

from corrpops.linalg.triangle_vector import triangle_to_vector
from corrpops.model import estimator, link_functions


@pytest.mark.parametrize(
    "link_function_class, n_control, n_diagnosed",
    itertools.product(
        link_functions.BaseLinkFunction.__subclasses__(),
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
    ).fit(control, diagnosed, compute_cov=False)

    atol = 1 / n_control + 1 / n_diagnosed
    np.testing.assert_allclose(model.alpha_, alpha.flatten(), atol=atol)
    np.testing.assert_allclose(model.theta_, triangle_to_vector(theta), atol=atol)
