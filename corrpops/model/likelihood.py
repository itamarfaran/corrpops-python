from typing import Optional

import numpy as np

from linalg.triangle_and_vector import triangle_to_vector
from linalg.vector import norm_p
from .link_functions import BaseLinkFunction


def theta_of_alpha(
    alpha: np.ndarray,
    control_arr: np.ndarray,
    diagnosed_arr: np.ndarray,
    link_function: BaseLinkFunction,
    dim_alpha: int = 1,
) -> np.ndarray:
    reversed_diagnosed_arr = link_function.inverse(
        data=diagnosed_arr, a=alpha, d=dim_alpha
    )
    arr = np.concatenate((control_arr, reversed_diagnosed_arr))
    return arr.mean(0)


def sum_of_squares(
    alpha: np.ndarray,
    theta: np.ndarray,
    diagnosed_arr: np.ndarray,
    link_function: BaseLinkFunction,
    inv_sigma: Optional[np.ndarray] = None,
    dim_alpha: int = 1,
    reg_lambda: float = 0.0,
    reg_p: float = 2.0,
) -> float:
    g11 = triangle_to_vector(link_function(t=theta, a=alpha, d=dim_alpha))

    if inv_sigma is None:
        sse = np.sum(g11 * (0.5 * g11 - diagnosed_arr.mean(0)))
    else:
        sse = np.squeeze(
            np.linalg.multi_dot(
                ((0.5 * g11 - diagnosed_arr.mean(0)), inv_sigma, g11[:, None])
            )
        )

    sse *= diagnosed_arr.shape[0]
    if reg_lambda > 0.0:
        sse += reg_lambda * norm_p(alpha, link_function.null_value, reg_p)
    return sse
