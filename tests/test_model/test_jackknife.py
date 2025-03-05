import json
import warnings

import pytest

from corrpops.corrpops_logger import corrpops_logger
from corrpops.model.estimator import CorrPopsEstimator, CorrPopsJackknifeEstimator

corrpops_logger().setLevel(30)


@pytest.mark.parametrize(
    "jack_control, steps_back",
    [(True, 3), (False, 3), (False, 200)],
)
def test_happy_flow(parameters_and_sample, jack_control, steps_back):
    theta, alpha, control, diagnosed = parameters_and_sample
    control = control[:2, ...]
    diagnosed = diagnosed[:2, ...]

    estimator = CorrPopsJackknifeEstimator(
        CorrPopsEstimator(optimizer_kwargs={"verbose": False}),
        jack_control=jack_control,
        steps_back=steps_back,
        non_positive="ignore",
    )
    assert estimator.to_json()["results"] == {}

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="not enough steps in base_estimator optimizer results"
        )
        estimator.fit(
            control_arr=control,
            diagnosed_arr=diagnosed,
            compute_cov=True,
        )

    assert isinstance(estimator.inference(), dict)
    assert isinstance(estimator.score(control, diagnosed), tuple)

    json_ = json.dumps(estimator.to_json(save_stacks=True))
    assert isinstance(
        CorrPopsJackknifeEstimator.results_from_json(json.loads(json_)), dict
    )
