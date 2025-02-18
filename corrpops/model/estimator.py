import copy
from collections import namedtuple
from typing import Any, Dict, Optional, Union, Literal

import numpy as np
from scipy import stats

from linalg.triangle_vector import triangle_to_vector, vector_to_triangle
from .covariance_of_correlation import (
    average_covariance_of_correlation,
    covariance_of_correlation,
)
from .gee_covariance import GeeCovarianceEstimator
from .link_functions import BaseLinkFunction, MultiplicativeIdentity
from .optimizer import (
    CorrPopsOptimizer,
    CorrPopsOptimizerResults,
    results_to_json,
    results_from_json,
)

WilksTestResult = namedtuple("WilksTestResult", ["chi2_val", "df", "p_val"])


class CorrPopsEstimator:
    def __init__(
        self,
        link_function: BaseLinkFunction = MultiplicativeIdentity(),
        *,
        optimizer: CorrPopsOptimizer = CorrPopsOptimizer(),
        gee_estimator: GeeCovarianceEstimator = GeeCovarianceEstimator(),
        naive_optimizer: Union[CorrPopsOptimizer, bool] = True,
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
    ):
        self.optimizer = optimizer.set_params(link_function=link_function)
        self.gee_estimator = gee_estimator.set_params(link_function=link_function)

        if isinstance(naive_optimizer, CorrPopsOptimizer):
            self.naive_optimizer = naive_optimizer.set_params(
                link_function=link_function
            )
        elif naive_optimizer:
            self.naive_optimizer = copy.deepcopy(self.optimizer)
        else:
            self.naive_optimizer = None

        self.non_positive = non_positive
        self.is_fitted = False

        self.alpha_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None
        self.cov_: Optional[np.ndarray] = None
        self.optimizer_results_: Optional[CorrPopsOptimizerResults] = None
        self.naive_optimizer_results_: Optional[CorrPopsOptimizerResults] = None

    @property
    def link_function(self):
        return self.optimizer.link_function

    @classmethod
    def from_json(
        cls,
        link_function: BaseLinkFunction,
        json_: Dict[str, Dict[str, Dict[str, Any]]],
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
    ):
        if "params" in json_:
            link_function.check_name_equal(
                json_["params"]["optimizer"]["link_function"]
            )
            estimator = cls(
                link_function=link_function,
                optimizer=CorrPopsOptimizer(**json_["params"]["optimizer"]),
                naive_optimizer=CorrPopsOptimizer(**json_["params"]["naive_optimizer"]),
                gee_estimator=GeeCovarianceEstimator(
                    **json_["params"]["gee_estimator"]
                ),
                non_positive=non_positive,
            )
        else:
            estimator = cls(link_function=link_function, non_positive=non_positive)

        if "results" in json_:
            if "optimizer" in json_["results"]:
                link_function.check_name_equal(
                    json_["results"]["optimizer"]["link_function"]
                )
                estimator.optimizer_results_ = results_from_json(
                    json_["results"]["optimizer"]
                )
                estimator.alpha_ = estimator.optimizer_results_["alpha"]
                estimator.theta_ = estimator.optimizer_results_["theta"]
            if "naive_optimizer" in json_["results"]:
                estimator.naive_optimizer_results_ = results_from_json(
                    json_["results"]["naive_optimizer"]
                )
            if "gee_estimator" in json_["results"]:
                estimator.cov_ = vector_to_triangle(
                    np.array(json_["results"]["gee_estimator"]["cov"]), diag=True
                )
        return estimator

    def to_json(
        self,
        save_results: bool = True,
        save_params: bool = True,
        save_naive: bool = False,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        json_ = {}

        if save_params:
            json_["params"] = {
                "optimizer": self.optimizer.get_params(),
                "naive_optimizer": self.naive_optimizer.get_params()
                if self.naive_optimizer
                else {},
                "gee_estimator": self.gee_estimator.get_params()
                if self.gee_estimator
                else {},
            }

        if save_results:
            json_["results"] = {}

            if self.optimizer_results_:
                json_["results"]["optimizer"] = results_to_json(self.optimizer_results_)

            if save_naive and self.naive_optimizer_results_:
                json_["results"]["naive_optimizer"] = results_to_json(
                    self.naive_optimizer_results_
                )

            if self.cov_ is not None:
                json_["results"]["gee_estimator"] = {
                    "cov": triangle_to_vector(self.cov_, diag=True).tolist()
                }

        return json_

    def fit(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        compute_cov: bool = True,
    ):
        weight_matrix, _ = average_covariance_of_correlation(
            diagnosed_arr,
            non_positive=self.non_positive,
        )
        control_arr = triangle_to_vector(control_arr)
        diagnosed_arr = triangle_to_vector(diagnosed_arr)

        if self.naive_optimizer is None:
            alpha0 = None
            theta0 = None
        else:
            naive_optimizer_results = self.naive_optimizer.optimize(
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
            )
            alpha0 = naive_optimizer_results["alpha"]
            theta0 = naive_optimizer_results["theta"]

        self.optimizer_results_ = self.optimizer.optimize(
            control_arr=control_arr,
            diagnosed_arr=diagnosed_arr,
            alpha0=alpha0,
            theta0=theta0,
            weights=weight_matrix,
        )
        self.alpha_ = self.optimizer_results_["alpha"]
        self.theta_ = self.optimizer_results_["theta"]
        self.is_fitted = True

        if compute_cov:
            self.compute_covariance(
                control_arr=control_arr, diagnosed_arr=diagnosed_arr
            )
        return self

    def compute_covariance(self, control_arr: np.ndarray, diagnosed_arr: np.ndarray):
        try:
            self.cov_ = self.gee_estimator.compute(
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                optimizer_results=self.optimizer_results_,
                non_positive=self.non_positive,
            )
        except ValueError:
            self.cov_ = self.gee_estimator.compute(
                control_arr=triangle_to_vector(control_arr),
                diagnosed_arr=triangle_to_vector(diagnosed_arr),
                optimizer_results=self.optimizer_results_,
                non_positive=self.non_positive,
            )
        return self

    def inference(
        self,
        p_adjust_method: Optional[str] = None,
        alternative: Literal["two-sided", "smaller", "larger"] = "two-sided",
        sig_level: float = 0.05,
        std_const: float = 1.0,
        known_alpha: Optional[np.ndarray] = None,
    ):
        alpha_sd = np.sqrt(np.diagonal(self.cov_))
        critical_value = stats.norm.ppf(1 - sig_level / 2)
        # critical_value = stats.multivariate_normal.ppf(1 - sig_level / 2, cov=cov_to_corr(self.cov_))

        result = {
            "estimate": self.alpha_,
            "std": std_const * alpha_sd,
            "z_value": (self.alpha_ - self.link_function.null_value) / alpha_sd,
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
            result["known_alpha"] = self.link_function.transformer.inv_transform(
                known_alpha.flatten()
            )

        return result

    def score(self, control_arr, diagnosed_arr):
        null_mean = np.concatenate((control_arr, diagnosed_arr)).mean(0)
        null_cov = covariance_of_correlation(null_mean, self.non_positive)

        g11 = self.link_function.func(
            t=self.theta_,
            a=self.alpha_,
            d=self.optimizer.dim_alpha,
        )

        control_arr = triangle_to_vector(control_arr)
        diagnosed_arr = triangle_to_vector(diagnosed_arr)

        full_log_likelihood = (
            stats.multivariate_normal.logpdf(
                x=control_arr,
                mean=self.theta_,
                cov=covariance_of_correlation(
                    vector_to_triangle(self.theta_, diag_value=1),
                    self.non_positive,
                ),
            ).sum()
            + stats.multivariate_normal.logpdf(
                x=diagnosed_arr,
                mean=triangle_to_vector(g11),
                cov=covariance_of_correlation(g11, self.non_positive),
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

        chi2_val = 2 * (full_log_likelihood - null_log_likelihood)
        df = np.size(self.alpha_)
        p_val = stats.chi2.sf(chi2_val, df)
        return WilksTestResult(chi2_val, df, p_val)
