from typing import Callable, Optional

import numpy as np
import pytest

from linalg.triangle_vector import triangle_to_vector
from model.link_functions import BaseLinkFunction, Transformer


@pytest.mark.parametrize(
    "link_function_class",
    BaseLinkFunction.__subclasses__(),
)
def test_link_function(
    parameters_and_sample,
    link_function_class: Callable[[Optional[Transformer]], BaseLinkFunction],
):
    link_function = link_function_class(None)
    theta, alpha, control, diagnosed = parameters_and_sample

    np.testing.assert_array_equal(
        link_function(
            t=triangle_to_vector(theta),
            a=alpha,
            d=1,
        ),
        link_function.forward(
            t=triangle_to_vector(theta),
            a=alpha,
            d=1,
        ),
    )

    with pytest.raises(ValueError):
        link_function_class(Transformer(lambda x: x, lambda x: x + 0.0001))

    np.testing.assert_allclose(
        triangle_to_vector(control),
        link_function.inverse(
            data=triangle_to_vector(control),
            a=np.full_like(alpha, link_function.null_value),
            d=1,
        ),
    )

    after_inversion = link_function.inverse(
        data=triangle_to_vector(
            link_function.forward(
                t=triangle_to_vector(theta),
                a=alpha,
                d=1,
            )
        ),
        a=alpha,
        d=1,
    )
    np.testing.assert_allclose(
        triangle_to_vector(theta),
        after_inversion,
    )
