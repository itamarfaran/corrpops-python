import numpy as np
import pytest
from scipy import optimize

from corrpops.linalg.triangle_and_vector import triangle_to_vector
from corrpops.linalg.vector import mahalanobis
from corrpops.model.likelihood import theta_of_alpha, sum_of_squares
from corrpops.model.link_functions import MultiplicativeIdentity


def test_theta_of_alpha(parameters_and_sample):
    theta, alpha, control, diagnosed = parameters_and_sample
    control = triangle_to_vector(control)
    diagnosed = triangle_to_vector(diagnosed)
    link_function = MultiplicativeIdentity()

    np.testing.assert_allclose(
        theta_of_alpha(
            alpha=np.full_like(alpha, link_function.null_value),
            control_arr=control,
            diagnosed_arr=diagnosed,
            link_function=link_function,
            dim_alpha=alpha.shape[-1],
        ),
        np.concatenate((control, diagnosed)).mean(0),
    )

    control = triangle_to_vector(np.stack([theta] * 8))
    diagnosed = triangle_to_vector(
        np.stack(
            [
                link_function(t=triangle_to_vector(theta), a=alpha, d=alpha.shape[-1]),
            ]
            * 12
        )
    )
    np.testing.assert_allclose(
        theta_of_alpha(
            alpha=alpha,
            control_arr=control,
            diagnosed_arr=diagnosed,
            link_function=link_function,
            dim_alpha=alpha.shape[-1],
        ),
        triangle_to_vector(theta),
    )


@pytest.mark.parametrize("with_inv_cov", [True, False])
def test_sum_of_squares(parameters_and_sample, with_inv_cov):
    theta, alpha, _, diagnosed = parameters_and_sample
    theta = triangle_to_vector(theta)
    diagnosed = triangle_to_vector(diagnosed)
    link_function = MultiplicativeIdentity()

    if with_inv_cov:
        dim = diagnosed.shape[-1]
        inv_cov = np.linalg.inv(
            np.full((dim, dim), 0.5) + np.diag(np.full(dim, 0.5))
        ).round(2)
    else:
        inv_cov = None

    x1 = optimize.approx_fprime(
        alpha.flatten(),
        lambda a_: sum_of_squares(
            alpha=a_,
            theta=theta,
            diagnosed_arr=diagnosed,
            link_function=link_function,
            inv_sigma=inv_cov,
            dim_alpha=alpha.shape[-1],
            reg_lambda=0.0,
        )
        * 2,
    )
    x2 = optimize.approx_fprime(
        alpha.flatten(),
        lambda a_: mahalanobis(
            x=diagnosed,
            y=triangle_to_vector(link_function(t=theta, a=a_, d=alpha.shape[-1])),
            m=inv_cov,
            inverse=False,
            sqrt=False,
        ).sum(),
    )
    # derivatives shouldn't care about constants
    np.testing.assert_allclose(x1 / x2, 1, atol=1e-05)

    differences = []
    for a in (
        alpha,
        np.full_like(alpha, link_function.null_value),
    ):
        differences.append(
            sum_of_squares(
                alpha=a,
                theta=theta,
                diagnosed_arr=diagnosed,
                link_function=link_function,
                inv_sigma=inv_cov,
                dim_alpha=alpha.shape[-1],
                reg_lambda=0.0,
            )
            * 2
            - mahalanobis(
                x=diagnosed,
                y=triangle_to_vector(link_function(t=theta, a=a, d=alpha.shape[-1])),
                m=inv_cov,
                inverse=False,
                sqrt=False,
            ).sum()
        )
    for d in differences:
        # difference should be constant
        np.testing.assert_allclose(d, differences[0], atol=1e-05)

    sums = {
        l: sum_of_squares(
            alpha=alpha,
            theta=theta,
            diagnosed_arr=diagnosed,
            link_function=link_function,
            inv_sigma=inv_cov,
            dim_alpha=alpha.shape[-1],
            reg_lambda=l,
        )
        for l in (0.0, 1.0, 2.0)
    }
    assert sums[2] > sums[1] > sums[0]

    # derivatives = {
    #     l: optimize.approx_fprime(
    #         alpha.flatten(),
    #         lambda a_: sum_of_squares(
    #             alpha=a_,
    #             theta=theta,
    #             diagnosed_arr=diagnosed,
    #             link_function=link_function,
    #             inv_sigma=inv_cov,
    #             dim_alpha=alpha.shape[-1],
    #             reg_lambda=l,
    #         ),
    #     )
    #     for l in (0.0, 1.0, 2.0)
    # }
