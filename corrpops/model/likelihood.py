import numpy as np

from linalg.triangle_vector import triangle_to_vector
from linalg.vector import norm_p


def theta_of_alpha(
    alpha,
    control_arr,
    diagnosed_arr,
    link_function,
    d=1,
):
    reversed_diagnosed_arr = link_function.reverse(diagnosed_arr, alpha, d)
    arr = np.concatenate((control_arr, reversed_diagnosed_arr))
    return arr.mean(0)


def sum_of_squares(
    alpha,
    theta,
    diagnosed_arr,
    link_function,
    inv_sigma=None,
    dim_alpha=1,
    reg_lambda=0.0,
    reg_p=2.0,
):
    g11 = triangle_to_vector(link_function.func(theta, alpha, dim_alpha))

    if inv_sigma is None:
        sse = np.sum(g11 * (0.5 * g11 - diagnosed_arr.mean(0)))
    else:
        sse = (0.5 * g11 - diagnosed_arr.mean(0)) @ inv_sigma @ g11[:, None]

    sse *= diagnosed_arr.shape[0]
    if reg_lambda > 0.0:
        sse += reg_lambda * norm_p(alpha, link_function.null_value, reg_p)
    return sse
