import itertools

import numpy as np
import pytest

from corrpops.corrpops_logger import corrpops_logger
from corrpops.linalg.triangle_and_vector import triangle_to_vector
from corrpops.model import estimator, link_functions
from corrpops.simulation.sample import build_parameters

corrpops_logger().setLevel(30)


@pytest.mark.parametrize(
    "link_function_class, n_control, n_diagnosed",
    itertools.product(
        link_functions.BaseLinkFunction.__subclasses__(),
        [12, 18],
        [12, 18],
    ),
)
def test_happy_flow(parameters, link_function_class, n_control, n_diagnosed):
    link_function: link_functions.BaseLinkFunction = link_function_class()
    theta, alpha, _ = build_parameters(
        p=4,
        percent_alpha=0.2,
        alpha_min=0.4,
        alpha_max=0.6,
        alpha_null=link_function.null_value,
        random_state=42,
    )

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
        np.testing.assert_allclose(
            results.theta, triangle_to_vector(theta), rtol=0.1, atol=0.01
        )

    # no variance in sample
    np.testing.assert_allclose(model.cov_, 0.0, rtol=0.1, atol=0.01)
