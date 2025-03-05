import itertools
import json

import numpy as np
import pytest

from corrpops.corrpops_logger import corrpops_logger
from corrpops.linalg import triangle_and_vector as tv
from corrpops.model import estimator, link_functions
from corrpops.simulation.sample import build_parameters

corrpops_logger().setLevel(30)


def test_init():
    params = {"early_stop": True, "verbose": False}
    optimizer = estimator.CorrPopsOptimizer(**params)

    est_1 = estimator.CorrPopsEstimator(optimizer=optimizer)
    est_2 = estimator.CorrPopsEstimator(optimizer=optimizer, naive_optimizer=optimizer)
    est_3 = estimator.CorrPopsEstimator(
        optimizer_kwargs=params,
        naive_optimizer_kwargs={"early_stop": False},
    )

    assert est_1.optimizer.get_params() == est_3.optimizer.get_params()
    assert est_1.naive_optimizer.get_params() == est_1.optimizer.get_params()
    assert est_3.naive_optimizer.get_params() != est_3.optimizer.get_params()
    assert est_2.naive_optimizer is est_2.optimizer
    assert estimator.CorrPopsEstimator(naive_optimizer="skip").naive_optimizer is None

    with pytest.raises(ValueError):
        estimator.CorrPopsEstimator(
            optimizer=optimizer,
            optimizer_kwargs=params,
        )

    with pytest.raises(ValueError):
        estimator.CorrPopsEstimator(
            naive_optimizer=optimizer,
            naive_optimizer_kwargs=params,
        )

    params = {"est_mu": False, "df_method": "efron"}
    covariance_estimator = estimator.GeeCovarianceEstimator(**params)

    est_1 = estimator.CorrPopsEstimator(covariance_estimator=covariance_estimator)
    est_2 = estimator.CorrPopsEstimator(covariance_estimator_kwargs=params)
    assert est_1.optimizer.get_params() == est_2.optimizer.get_params()

    with pytest.raises(ValueError):
        estimator.CorrPopsEstimator(
            covariance_estimator=covariance_estimator,
            covariance_estimator_kwargs=params,
        )


@pytest.mark.parametrize(
    "save_results_and_naive, save_params",
    [
        (True, False),
        (False, True),
    ],
)
def test_json(parameters_and_sample, save_results_and_naive, save_params):
    orig = estimator.CorrPopsEstimator()

    if save_results_and_naive:
        _, _, control, diagnosed = parameters_and_sample
        orig.fit(control, diagnosed)

    json_ = json.dumps(
        orig.to_json(
            save_results=save_results_and_naive,
            save_params=save_params,
            save_naive=save_results_and_naive,
        ),
    )
    new = estimator.CorrPopsEstimator.from_json(
        json.loads(json_),
        link_functions.MultiplicativeIdentity(),
    )

    for field_name in (
        "link_function",
        "dim_alpha",
        "non_positive",
        "is_fitted",
        "alpha_",
        "theta_",
        "cov_",
    ):
        orig_value = getattr(orig, field_name)
        new_value = getattr(new, field_name)

        if isinstance(orig_value, np.ndarray):
            np.testing.assert_allclose(orig_value, new_value)
        elif orig_value is None:
            assert new_value is None
        elif hasattr(orig_value, "name"):
            assert orig_value.name == new_value.name
        else:
            assert orig_value == new_value

    if orig.is_fitted:
        with pytest.raises(ValueError):
            estimator.CorrPopsEstimator.from_json(
                json.loads(json_),
                link_functions.AdditiveQuotient(),
            )


@pytest.mark.parametrize(
    "link_function_class, n_control, n_diagnosed, naive_and_cov",
    itertools.product(
        link_functions.BaseLinkFunction.__subclasses__(),
        [12, 18],
        [12, 18],
        [True, False],
    ),
)
def test_happy_flow(link_function_class, n_control, n_diagnosed, naive_and_cov):
    link_function: link_functions.BaseLinkFunction = link_function_class()
    theta, alpha, _ = build_parameters(
        p=4,
        percent_alpha=0.2,
        alpha_min=0.4,
        alpha_max=0.6,
        alpha_null=link_function.null_value,
        random_state=42,
    )
    g11 = link_function(tv.triangle_to_vector(theta), alpha, alpha.shape[1])

    control = np.stack(n_control * [theta])
    diagnosed = np.stack(n_diagnosed * [g11])

    model = estimator.CorrPopsEstimator(
        link_function=link_function,
        optimizer_kwargs={"verbose": False},
        naive_optimizer=None if naive_and_cov else "skip",
    ).fit(control, diagnosed, compute_cov=naive_and_cov)

    results = [model.optimizer_results_]
    if naive_and_cov:
        results.append(model.naive_optimizer_results_)

        # no variance in sample
        np.testing.assert_allclose(model.cov_, 0.0, atol=1e-06)
        assert isinstance(model.inference(), dict)
        assert isinstance(model.score(control, diagnosed), tuple)

    for result in results:
        np.testing.assert_allclose(result.alpha, alpha.flatten())
        np.testing.assert_allclose(result.theta, tv.triangle_to_vector(theta))
