from typing import Tuple

import numpy as np
import pytest

from corrpops.model.link_functions import MultiplicativeIdentity
from corrpops.simulation.sample import build_parameters, create_samples_from_parameters


@pytest.fixture(scope="module")
def parameters() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return build_parameters(
        p=5,
        percent_alpha=0.2,
        alpha_min=0.7,
        random_state=42,
    )


@pytest.fixture(scope="module")
def parameters_and_sample() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta, alpha, _ = build_parameters(
        p=5,
        percent_alpha=0.2,
        alpha_min=0.7,
        random_state=42,
    )
    control, diagnosed = create_samples_from_parameters(
        control_n=9,
        diagnosed_n=11,
        theta=theta,
        alpha=alpha,
        link_function=MultiplicativeIdentity(),
        random_state=12,
    )
    return theta, alpha, control[0], diagnosed[0]
