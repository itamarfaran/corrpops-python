from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy import stats

from linalg.triangle_vector import triangle_to_vector, vector_to_triangle
from .covariance_of_correlation import (
    average_covariance_of_correlation,
    covariance_of_correlation,
)
from .estimator import CorrPopsEstimator, WilksTestResult
from .gee_covariance import GeeCovarianceEstimator
from .optimizer import CorrPopsOptimizer, CorrPopsOptimizerResults


# todo: add ray
def jackknife(
    index: Union[int, Tuple[int, ...]],
    optimizer: CorrPopsOptimizer,
    control_arr: np.ndarray,
    diagnosed_arr: np.ndarray,
    alpha0: np.ndarray,
    theta0: np.ndarray,
    jack_diagnosed: bool = True,
    weights: Optional[np.ndarray] = None,
    gee_estimator: Optional[GeeCovarianceEstimator] = None,
):
    if jack_diagnosed:
        diagnosed_arr = np.delete(diagnosed_arr, index, axis=0)
    else:
        control_arr = np.delete(control_arr, index, axis=0)

    if jack_diagnosed or weights is None:
        weights, _ = average_covariance_of_correlation(
            diagnosed_arr,
            non_positive="ignore",
        )

    control_arr = triangle_to_vector(control_arr)
    diagnosed_arr = triangle_to_vector(diagnosed_arr)

    results = optimizer.optimize(
        control_arr=control_arr,
        diagnosed_arr=diagnosed_arr,
        alpha0=alpha0,
        theta0=theta0,
        weights=weights,
    )
    try:
        cov = gee_estimator.compute(
            control_arr=control_arr,
            diagnosed_arr=diagnosed_arr,
            optimizer_results=results,
            non_positive="ignore",
        )
    except (AttributeError, ValueError):
        cov = None

    return {
        "theta": results["theta"],
        "alpha": results["alpha"],
        "status": results["steps"][-1]["status"],
        "cov": cov,
    }


class CorrPopsJackknifeEstimator:
    def __init__(
        self,
        base_estimator: CorrPopsEstimator,
        jack_control: bool = True,
        steps_back: int = 3,
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
    ):
        self.base_estimator = base_estimator
        self.jack_control = jack_control
        self.steps_back = steps_back
        self.non_positive = non_positive
        self.is_fitted = False

        self.alpha_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None
        self.cov_: Optional[np.ndarray] = None
        self.gee_cov_: Optional[np.ndarray] = None

        self.control_index_: Optional[np.ndarray] = None
        self.diagnosed_index_: Optional[np.ndarray] = None

        self.alpha_stack_: Optional[np.ndarray] = None
        self.theta_stack_: Optional[np.ndarray] = None
        self.gee_cov_stack_: Optional[np.ndarray] = None

    @staticmethod
    def stack_if_not_none(
        results: List[Dict[str, Optional[np.ndarray]]],
        key: str,
        param_shape: Tuple[int, ...],
    ) -> np.ndarray:
        return np.stack(
            [
                np.full(param_shape, np.nan) if element[key] is None else element[key]
                for element in results
            ]
        )

    @classmethod
    def aggregate_stacks(
        cls,
        alpha_stack: np.ndarray,
        control_index: np.ndarray,
        diagnosed_index: np.ndarray,
        theta_stack: Optional[np.ndarray] = None,
        gee_cov_stack: Optional[np.ndarray] = None,
    ):
        diagnosed_n = len(diagnosed_index)
        diagnosed_constant = (diagnosed_n - 1) ** 2 / diagnosed_n
        # diagnosed_mean = alpha_stack[diagnosed_index].mean(0)
        diagnosed_variance = diagnosed_constant * np.cov(
            alpha_stack[diagnosed_index], rowvar=False
        )

        control_n = len(control_index)
        control_constant = (control_n - 1) ** 2 / control_n
        # control_mean = alpha_stack[control_index].mean(0)
        control_variance = control_constant * np.cov(
            alpha_stack[control_index], rowvar=False
        )

        return {
            "alpha": alpha_stack.mean(0),
            # "alpha": (diagnosed_mean + control_mean) / 2,
            "cov": diagnosed_variance + control_variance,
            "theta": None if theta_stack is None else theta_stack.mean(0),
            "gee_cov": None if gee_cov_stack is None else gee_cov_stack.mean(0),
        }

    def to_json(
        self,
        save_results: bool = True,
        save_params: bool = True,
        save_naive: bool = False,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        raise NotImplementedError

    def fit(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        *,
        compute_cov: bool = False,
        optimizer_results: Optional[CorrPopsOptimizerResults] = None,
    ):
        self.control_index_ = []
        self.diagnosed_index_ = []

        if optimizer_results is None:
            self.base_estimator.fit(
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                compute_cov=False,
            )
            steps = self.base_estimator.optimizer_results_["steps"]
        else:
            steps = optimizer_results["steps"]

        last_step = steps[-min(1 + self.steps_back, len(steps))]
        alpha0 = last_step["alpha"]
        theta0 = last_step["theta"]

        weight_matrix, _ = average_covariance_of_correlation(
            diagnosed_arr,
            non_positive=self.non_positive,
        )

        jackknife_kwargs = {
            "optimizer": self.base_estimator.optimizer,
            "control_arr": control_arr,
            "diagnosed_arr": diagnosed_arr,
            "alpha0": alpha0,
            "theta0": theta0,
            "gee_estimator": self.base_estimator.gee_estimator if compute_cov else None,
        }

        results = [
            jackknife(index=index, jack_diagnosed=True, **jackknife_kwargs)
            for index in range(diagnosed_arr.shape[0])
        ]
        self.diagnosed_index_ = np.arange(diagnosed_arr.shape[0])

        if self.jack_control:
            results += [
                jackknife(index=index, jack_diagnosed=False, **jackknife_kwargs)
                for index in range(control_arr.shape[0])
            ]
            self.control_index_ = self.diagnosed_index_.size + np.arange(
                control_arr.shape[0]
            )

        self.alpha_stack_ = self.stack_if_not_none(results, "alpha", alpha0.shape)
        self.theta_stack_ = self.stack_if_not_none(results, "theta", theta0.shape)
        if compute_cov:
            self.gee_cov_stack_ = self.stack_if_not_none(
                results, "cov", (alpha0.size, alpha0.size)
            )

        aggregates = self.aggregate_stacks(
            alpha_stack=self.alpha_stack_,
            control_index=self.control_index_,
            diagnosed_index=self.diagnosed_index_,
            theta_stack=self.theta_stack_,
            gee_cov_stack=self.gee_cov_stack_,
        )
        self.alpha_ = aggregates["alpha"]
        self.theta_ = aggregates["theta"]
        self.cov_ = aggregates["cov"]
        self.gee_cov_ = aggregates["gee_cov"]
        return self

    # todo: remove duplications ...
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
            "z_value": (self.alpha_ - self.base_estimator.link_function.null_value)
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
            ] = self.base_estimator.link_function.transformer.inv_transform(
                known_alpha.flatten()
            )

        return result

    def score(self, control_arr, diagnosed_arr):
        null_mean = np.concatenate((control_arr, diagnosed_arr)).mean(0)
        null_cov = covariance_of_correlation(null_mean, self.non_positive)

        g11 = self.base_estimator.link_function.func(
            t=self.theta_,
            a=self.alpha_,
            d=self.base_estimator.optimizer.dim_alpha,
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
