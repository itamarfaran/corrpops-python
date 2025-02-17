import copy
from typing import Optional, Union, Literal

import numpy as np
from scipy import stats

from utils.covariance_of_correlation import average_covariance_of_correlation
from utils.matrix import cov_to_corr
from utils.triangle_vector import triangle_to_vector
from .gee_covariance import GeeCovarianceEstimator
from .link_functions import MultiplicativeIdentity
from .optimizer import CorrPopsOptimizer


class CorrPopsEstimator:
    def __init__(
        self,
        link_function=MultiplicativeIdentity(),
        *,
        optimizer: CorrPopsOptimizer = CorrPopsOptimizer(),
        gee_estimator: Optional[GeeCovarianceEstimator] = GeeCovarianceEstimator(),
        naive_optimizer: Union[CorrPopsOptimizer, bool] = True,
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
    ):
        self.optimizer = optimizer
        self.optimizer.link_function = link_function

        self.gee_estimator = gee_estimator

        if type(naive_optimizer) is bool:
            if naive_optimizer:
                self.naive_optimizer = copy.deepcopy(self.optimizer)
            else:
                self.naive_optimizer = None
        else:
            self.naive_optimizer = naive_optimizer
            self.naive_optimizer.link_function = link_function

        self.non_positive = non_positive
        self.alpha_ = None
        self.theta_ = None
        self.cov_ = None

    def fit(self, control_arr, diagnosed_arr):
        weight_matrix, _ = average_covariance_of_correlation(
            diagnosed_arr,
            non_positive=self.non_positive,
        )

        control_arr = triangle_to_vector(control_arr)
        diagnosed_arr = triangle_to_vector(diagnosed_arr)

        if self.naive_optimizer is None:
            alpha_ = None
            theta_ = None
        else:
            self.naive_optimizer.optimize(control_arr, diagnosed_arr)
            alpha_ = self.naive_optimizer.alpha_
            theta_ = self.naive_optimizer.theta_

        self.optimizer.optimize(
            control_arr,
            diagnosed_arr,
            alpha_,
            theta_,
            weight_matrix,
        )

        self.alpha_ = self.optimizer.alpha_
        self.theta_ = self.optimizer.theta_

        if self.gee_estimator is not None:
            self.cov_ = self.gee_estimator.compute(
                control_arr,
                diagnosed_arr,
                self.optimizer,
            )

    def inference(
        self,
        p_adjust_method: str = None,
        alternative: Literal["two-sided", "smaller", "larger"] = "two-sided",
        sig_level: float = 0.05,
        std_const: float = 1.0,
        known_alpha: Optional[np.ndarray] = None,
    ):
        alpha_sd = np.sqrt(np.diagonal(self.cov_))
        critical_value = stats.norm.ppf(1 - sig_level / 2)
        # critical_value = stats.multivariate_normal.ppf(
        #     1 - sig_level / 2,
        #     cov=cov_to_corr(self.cov_),
        # )

        result = {
            "estimate": self.alpha_,
            "std": std_const * alpha_sd,
            "z_value": (self.alpha_ - self.optimizer.link_function.null_value)
            / alpha_sd,
            "ci_lower": self.alpha_ - critical_value * alpha_sd,
            "ci_upper": self.alpha_ - critical_value * alpha_sd,
        }

        if alternative == "smaller":
            result["p_vals"] = stats.norm.cdf(result["z_value"])
        elif alternative == "larger":
            result["p_vals"] = stats.norm.sf(result["z_value"])
        elif alternative == "two-sided":
            result["p_vals"] = 2 * stats.norm.sf(np.abs(result["z_value"]))
        else:
            raise ValueError(
                f"alternative should be one of ['two-sided', 'smaller', 'larger'], "
                f"got {alternative} instead"
            )

        if p_adjust_method is not None:
            from statsmodels.stats.multitest import multipletests

            result["p_vals_adjusted"] = multipletests(
                pvals=result["p_vals"], method=p_adjust_method
            )[1]
        else:
            result["p_vals_adjusted"] = np.full_like(result["p_vals"], np.nan)

        if known_alpha is None:
            result["known_alpha"] = np.full_like(self.alpha_, np.nan)
        else:
            result[
                "known_alpha"
            ] = self.optimizer.link_function.transformer.inv_transform(
                known_alpha.flatten()
            )

        return result
