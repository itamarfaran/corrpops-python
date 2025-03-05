import json
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pytest

from corrpops.linalg import triangle_and_vector as tv
from corrpops.model.estimator.optimizer import (
    format_time_delta,
    CorrPopsOptimizer,
    CorrPopsOptimizerResults,
)
from corrpops.model.link_functions import MultiplicativeIdentity
from tests.tests_utils import v_w


def test_format_time_delta():
    formatted = format_time_delta(
        datetime(2024, 1, 1, 12, 0, 0) - datetime(2024, 1, 1, 10, 30, 0)
    )
    assert formatted == "01:30:00"

    formatted = format_time_delta(
        datetime(2024, 1, 1, 12, 0, 0) - datetime(2024, 1, 1, 11, 30, 0)
    )
    assert formatted == "30:00"

    formatted = format_time_delta(
        datetime(2024, 1, 1, 12, 0, 0) - datetime(2024, 1, 1, 11, 30, 30)
    )
    assert formatted == "29:30"


@pytest.mark.parametrize("with_inv_cov", [False, True])
def test_corrpops_optimizer_results(parameters, with_inv_cov):
    theta, alpha, _ = parameters
    if with_inv_cov:
        v, _ = v_w(alpha.size)
        inv_cov = v @ v.T
    else:
        inv_cov = None

    orig = CorrPopsOptimizerResults(
        theta=theta,
        alpha=alpha,
        inv_cov=inv_cov,
        link_function=MultiplicativeIdentity().name,
        p=alpha.shape[0],
        dim_alpha=alpha.shape[1],
        steps=[{"this": "is an example"}],  # type: ignore
    )
    new = CorrPopsOptimizerResults.from_json(json.loads(json.dumps(orig.to_json())))

    for field_name, orig_value in asdict(orig).items():
        new_value = getattr(new, field_name)
        if isinstance(orig_value, np.ndarray):
            np.testing.assert_allclose(orig_value, new_value)
        elif orig_value is None:
            assert new_value is None
        elif field_name == "steps":
            assert not new_value
        else:
            assert orig_value == new_value


def test_corrpops_optimizer_get_params():
    optimizer = CorrPopsOptimizer()

    for param, value in [
        ("rel_tol", 0.1),
        ("abs_tol", 0.1),
        ("early_stop", True),
    ]:
        assert getattr(optimizer, param) != value
        assert getattr(optimizer, param) == optimizer.get_params()[param]
        optimizer.set_params(**{param: value})
        assert getattr(optimizer, param) == value
        assert getattr(optimizer, param) == optimizer.get_params()[param]

    for param, value in [
        ("this_is", 0.1),
        ("an_example", 0.1),
    ]:
        with pytest.raises(ValueError):
            optimizer.set_params(**{param: value})
        with pytest.raises(KeyError):
            _ = optimizer.get_params()[param]

    with pytest.raises(ValueError):
        CorrPopsOptimizer(abs_tol=0.01, rel_tol=0.01)


@pytest.mark.parametrize("rel_tol, abs_tol", [(1e-06, 0), (0, 1e-06)])
def test_optimize(parameters, rel_tol, abs_tol):
    link_function = MultiplicativeIdentity()
    theta, alpha, _ = parameters
    g11 = link_function(tv.triangle_to_vector(theta), alpha, alpha.shape[1])

    control = tv.triangle_to_vector(np.stack([theta] * 10))
    diagnosed = tv.triangle_to_vector(np.stack([g11] * 12))

    optimizer = CorrPopsOptimizer(rel_tol=rel_tol, abs_tol=abs_tol)
    result = optimizer.optimize(control, diagnosed, link_function)

    np.testing.assert_allclose(alpha.flatten(), result.alpha, atol=0.001)
    np.testing.assert_allclose(tv.triangle_to_vector(theta), result.theta, atol=0.001)
