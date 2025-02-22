import warnings
from typing import Any, Dict, Literal

import numpy as np
from scipy import optimize

from linalg.matrix import force_symmetry
from linalg.triangle_vector import triangle_to_vector, vector_to_triangle
from model.covariance_of_correlation import average_covariance_of_correlation
from model.estimator.optimizer import CorrPopsOptimizerResults
from model.likelihood import theta_of_alpha, sum_of_squares
from model.link_functions import BaseLinkFunction


class FisherSandwichCovarianceEstimator:
    def __init__(
        self,
        *,
        est_mu: bool = True,
        estimated_n: bool = True,
    ):
        warnings.warn(f"{type(self).__name__} is experimental")
        self.est_mu = est_mu
        self.estimated_n = estimated_n

    def get_params(self) -> Dict[str, Any]:
        return {
            "est_mu": self.est_mu,
            "estimated_n": self.estimated_n,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter {k} for estimator {self}.")
            setattr(self, k, v)
        return self

    @staticmethod
    def compute_by_gradient(
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        expected_value: np.ndarray,
        inv_cov: np.ndarray,
        link_function: BaseLinkFunction,
        optimizer_results: CorrPopsOptimizerResults,
    ) -> np.ndarray:
        def jacobian_func(a):
            theta = theta_of_alpha(
                alpha=a,
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                link_function=link_function,
                dim_alpha=optimizer_results.dim_alpha,
            )
            return triangle_to_vector(
                link_function(
                    t=theta,
                    a=a,
                    d=optimizer_results.dim_alpha,
                )
            )

        jacobian = optimize.approx_fprime(optimizer_results.alpha, jacobian_func)
        residuals = control_arr - expected_value
        left = jacobian.T @ inv_cov @ residuals.T
        return left @ left.T

    @staticmethod
    def compute_by_hessian(
        diagnosed_arr: np.ndarray,
        inv_cov: np.ndarray,
        link_function: BaseLinkFunction,
        optimizer_results: CorrPopsOptimizerResults,
    ) -> np.ndarray:
        def f(a):
            return sum_of_squares(
                alpha=a,
                theta=optimizer_results.theta,
                diagnosed_arr=diagnosed_arr,
                link_function=link_function,
                inv_sigma=inv_cov,
                dim_alpha=optimizer_results.dim_alpha,
            )

        def jacobian_func(a):
            return optimize.approx_fprime(a, f)

        return force_symmetry(
            optimize.approx_fprime(optimizer_results.alpha, jacobian_func)
        )

    def compute(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        link_function: BaseLinkFunction,
        optimizer_results: CorrPopsOptimizerResults,
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
    ) -> np.ndarray:
        if self.est_mu:
            expected_value = triangle_to_vector(
                link_function(
                    t=optimizer_results.theta,
                    a=optimizer_results.alpha,
                    d=optimizer_results.dim_alpha,
                )
            )
        else:
            expected_value = np.mean(control_arr, axis=-1)

        cov, _ = average_covariance_of_correlation(
            vector_to_triangle(control_arr, diag_value=1),
            est_n=self.estimated_n,
            non_positive=non_positive,
        )
        inv_cov = np.linalg.inv(cov)

        fisher_by_gradiant = self.compute_by_gradient(
            control_arr=control_arr,
            diagnosed_arr=diagnosed_arr,
            expected_value=expected_value,
            inv_cov=inv_cov,
            link_function=link_function,
            optimizer_results=optimizer_results,
        )
        fisher_by_hessian = self.compute_by_hessian(
            diagnosed_arr=diagnosed_arr,
            inv_cov=inv_cov,
            link_function=link_function,
            optimizer_results=optimizer_results,
        )
        inv_fisher_by_hessian = np.linalg.inv(fisher_by_hessian)
        return inv_fisher_by_hessian @ fisher_by_gradiant @ inv_fisher_by_hessian
