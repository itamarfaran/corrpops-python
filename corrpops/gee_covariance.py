from functools import partial
from estimation_utils import theta_of_alpha
from jacobian import simple_jacobian
from triangle_vector import triangle_to_vector


def diagnosed_mu_alpha_jacobian(
        link_function,
        alpha,
        control_arr,
        diagnosed_arr,
        d=1,
):
    def _inner(a):
        theta = theta_of_alpha(a, control_arr, diagnosed_arr, link_function, d)
        return triangle_to_vector(link_function.func(theta, a, d))
    return simple_jacobian(_inner, alpha)


def control_mu_alpha_jacobian(
        link_function,
        alpha,
        control_arr,
        diagnosed_arr,
        d=1,
):
    def _inner(a):
        return theta_of_alpha(a, control_arr, diagnosed_arr, link_function, d)
    return simple_jacobian(_inner, alpha)
