import warnings
from collections import namedtuple
from typing import Dict, Literal, Optional

import numpy as np
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests

    _statsmodels_installed: bool = True
except ModuleNotFoundError:  # pragma: no cover
    _statsmodels_installed: bool = False

from linalg.triangle_and_vector import triangle_to_vector
from .covariance_of_correlation import covariance_of_correlation
from .link_functions import BaseLinkFunction

WilksTestResult = namedtuple("WilksTestResult", ["chi2_val", "df", "p_val"])


def inference(
    alpha: np.ndarray,
    cov: np.ndarray,
    link_function: BaseLinkFunction,
    p_adjust_method: str = "bonferroni",
    alternative: Literal["two-sided", "smaller", "larger"] = "two-sided",
    sig_level: float = 0.05,
    std_const: float = 1.0,
    known_alpha: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    alpha_sd = np.sqrt(np.diagonal(cov))
    critical_value = stats.norm.ppf(1 - sig_level / 2)
    # critical_value = stats.multivariate_normal.ppf(1 - sig_level / 2, cov=cov_to_corr(self.cov_))

    result = {
        "estimate": alpha,
        "std": std_const * alpha_sd,
        "z_value": (alpha - link_function.null_value) / alpha_sd,
        "ci_lower": alpha - critical_value * alpha_sd,
        "ci_upper": alpha + critical_value * alpha_sd,
    }

    if alternative == "smaller":
        result["p_value"] = stats.norm.cdf(result["z_value"])
    elif alternative == "larger":
        result["p_value"] = stats.norm.sf(result["z_value"])
    elif alternative == "two-sided":
        result["p_value"] = 2 * stats.norm.sf(np.abs(result["z_value"]))
    else:  # pragma: no cover
        raise ValueError(
            f"alternative should be one of ['two-sided', 'smaller', 'larger'], "
            f"got {alternative} instead"
        )

    if p_adjust_method == "bonferroni":
        result["p_value_adjusted"] = np.minimum(
            result["p_value"] * result["p_value"].size, 1.0
        )
    elif _statsmodels_installed:
        result["p_value_adjusted"] = multipletests(
            pvals=result["p_value"], method=p_adjust_method
        )[1]
    else:  # pragma: no cover
        raise ModuleNotFoundError(
            "missing optional dependency statsmodels. "
            "only p_adjust_method='bonferroni' is supported. "
            "for other methods please install statsmodels."
        )

    if known_alpha is not None:
        result["known_alpha"] = link_function.transformer.inverse_transform(
            known_alpha.flatten()
        )

    return result


def wilks_test(
    control_arr: np.ndarray,
    diagnosed_arr: np.ndarray,
    theta: np.ndarray,
    alpha: np.ndarray,
    link_function: BaseLinkFunction,
    dim_alpha: int = 1,
    non_positive: Literal["raise", "warn", "ignore"] = "raise",
) -> WilksTestResult:
    null_mean = np.concatenate((control_arr, diagnosed_arr)).mean(0)
    null_cov = covariance_of_correlation(null_mean, non_positive)
    g11 = link_function(
        t=triangle_to_vector(theta),
        a=alpha,
        d=dim_alpha,
    )

    control_arr = triangle_to_vector(control_arr)
    diagnosed_arr = triangle_to_vector(diagnosed_arr)

    control_full_cov = covariance_of_correlation(theta, non_positive)
    diagnosed_full_cov = covariance_of_correlation(g11, non_positive)

    try:
        full_log_likelihood = (
            stats.multivariate_normal.logpdf(
                x=control_arr,
                mean=triangle_to_vector(theta),
                cov=control_full_cov,
                allow_singular=True,
            ).sum()
            + stats.multivariate_normal.logpdf(
                x=diagnosed_arr,
                mean=triangle_to_vector(g11),
                cov=diagnosed_full_cov,
                allow_singular=True,
            ).sum()
        )
        null_log_likelihood = (
            stats.multivariate_normal.logpdf(
                x=control_arr,
                mean=triangle_to_vector(null_mean),
                cov=null_cov,
            ).sum()
            + stats.multivariate_normal.logpdf(
                x=diagnosed_arr,
                mean=triangle_to_vector(null_mean),
                cov=null_cov,
            ).sum()
        )
    except ValueError as ex:
        if "positive semidefinite" not in str(ex):
            raise ex  # pragma: no cover
        warnings.warn(
            "covariance matrix is not symmetric positive semidefinite, returning nan"
        )
        full_log_likelihood = null_log_likelihood = np.float32(np.nan)

    chi2_val = 2 * (full_log_likelihood - null_log_likelihood)
    df = alpha.size
    p_val = stats.chi2.sf(chi2_val, df)
    return WilksTestResult(chi2_val.item(), df, p_val.item())
