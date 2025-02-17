from typing import Tuple

import numpy as np
from scipy import stats

from linalg.matrix import force_symmetry, cov_to_corr
from linalg.triangle_vector import triangle_to_vector
from simulation.wishart import arma_wishart_rvs


def build_parameters(
    p: int,
    percent_alpha: float,
    alpha_min: float,
    alpha_max: float,
    dim_alpha: int = 1,
    theta_loc: float = 0.0,
    theta_scale: float = 1.0,
    enforce_min_alpha: bool = False,
    random_state=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)

    theta = stats.norm.rvs(
        loc=theta_loc,
        scale=theta_scale,
        size=(2 * p, p),
        random_state=rng,
    )
    theta = force_symmetry(cov_to_corr(theta.T @ theta))
    stds = np.diag(np.abs(stats.logistic.rvs(size=p, random_state=rng)))
    sigma = stds @ theta @ stds

    n_non_null_alphas = int(percent_alpha * p)
    non_null_alphas = rng.choice(p, n_non_null_alphas, replace=False)

    alpha_row_sums = np.ones(p)
    alpha_row_sums[non_null_alphas] = stats.uniform.rvs(
        loc=alpha_min,
        scale=alpha_max - alpha_min,
        size=n_non_null_alphas,
        random_state=rng,
    )
    if enforce_min_alpha:
        alpha_row_sums[rng.choice(non_null_alphas)] = alpha_min

    alpha = stats.uniform.rvs(size=(dim_alpha, p), random_state=rng)
    alpha /= alpha.sum(axis=0)  # normalize rows to 1
    alpha *= alpha_row_sums  # reduce row sums to alpha_row_sums
    return theta, alpha.T, sigma


def create_samples(
    control_n,
    diagnosed_n,
    control_correlation,
    diagnosed_correlation,
    t_length=100,
    control_ar=None,
    control_ma=None,
    diagnosed_ar=None,
    diagnosed_ma=None,
    size=1,
    random_effect=0.0,
    random_state=None,
):
    rng = np.random.default_rng(random_state)
    p = control_correlation.shape[-1]

    control = arma_wishart_rvs(
        df=t_length,
        scale=control_correlation,
        ar=control_ar,
        ma=control_ma,
        random_effect=random_effect,
        size=control_n * size,
        random_state=rng,
    ).reshape(size, control_n, p, p)

    diagnosed = arma_wishart_rvs(
        df=t_length,
        scale=diagnosed_correlation,
        ar=diagnosed_ar,
        ma=diagnosed_ma,
        random_effect=random_effect,
        size=diagnosed_n * size,
        random_state=rng,
    ).reshape(size, diagnosed_n, p, p)

    control = force_symmetry(cov_to_corr(control))
    diagnosed = force_symmetry(cov_to_corr(diagnosed))

    return control, diagnosed


def create_samples_from_parameters(
    control_n,
    diagnosed_n,
    theta,
    alpha,
    link_function,
    t_length=100,
    control_ar=None,
    control_ma=None,
    diagnosed_ar=None,
    diagnosed_ma=None,
    size=1,
    random_effect=0.0,
    random_state=None,
):
    g11 = link_function.func(
        t=triangle_to_vector(theta),
        a=alpha,
        d=alpha.shape[-1],
    )
    return create_samples(
        control_n=control_n,
        diagnosed_n=diagnosed_n,
        control_correlation=theta,
        diagnosed_correlation=g11,
        t_length=t_length,
        control_ar=control_ar,
        control_ma=control_ma,
        diagnosed_ar=diagnosed_ar,
        diagnosed_ma=diagnosed_ma,
        size=size,
        random_effect=random_effect,
        random_state=random_state,
    )
