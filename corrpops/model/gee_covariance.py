from typing import Callable, Literal, TypedDict, Optional

import numpy as np

from linalg.jacobian import simple_jacobian, richardson_jacobian
from linalg.triangle_vector import triangle_to_vector, vector_to_triangle
from statistics.efron_rms import efron_effective_sample_size, efron_rms_sample
from .covariance_of_correlation import average_covariance_of_correlation
from .likelihood import theta_of_alpha
from .link_functions import BaseLinkFunction
from .optimizer import CorrPopsOptimizerResults


class GeeProperties(TypedDict):
    data: np.ndarray
    jacobian: np.ndarray
    expected_value: np.ndarray
    inv_cov: np.ndarray
    df: float


class GeeCovarianceEstimator:
    def __init__(
        self,
        link_function: Optional[BaseLinkFunction] = None,
        *,
        est_mu: bool = True,
        jacobian_method: Literal["simple", "richardson"] = "richardson",
        sample_size: Literal["estimated", "naive"] = "estimated",
        df_method: Literal["naive", "efron"] = "naive",
    ):
        self.link_function = link_function
        self.est_mu = est_mu
        self.jacobian_method = jacobian_method
        self.sample_size = sample_size
        self.df_method = df_method

    def get_params(self):
        return {
            "link_function": self.link_function.name,
            "est_mu": self.est_mu,
            "jacobian_method": self.jacobian_method,
            "sample_size": self.sample_size,
            "df_method": self.df_method,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter {k} for estimator {self}.")
            setattr(self, k, v)
        return self

    def create_properties(
        self,
        arr: np.ndarray,
        optimizer_results: CorrPopsOptimizerResults,
        jacobian_func: Callable[[np.ndarray], np.ndarray],
        expected_value: np.ndarray,
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
    ):
        if self.jacobian_method == "simple":
            jacobian = simple_jacobian(jacobian_func, optimizer_results["alpha"])
        else:
            jacobian = richardson_jacobian(jacobian_func, optimizer_results["alpha"])

        cov, _ = average_covariance_of_correlation(
            vector_to_triangle(arr, diag_value=1),
            non_positive=non_positive,
            # don't I need estimated n for something?
        )
        inv_cov = np.linalg.inv(cov)

        if self.df_method == "efron":
            df = efron_effective_sample_size(
                n=np.prod(arr.shape[:-1]),
                rms=efron_rms_sample(arr),
            )
        else:
            df = np.prod(arr.shape[0])

        return GeeProperties(
            data=arr,
            jacobian=jacobian,
            expected_value=expected_value,
            inv_cov=inv_cov,
            df=df,
        )

    def create_control_properties(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        optimizer_results: CorrPopsOptimizerResults,
        non_positive: Literal["raise", "warn", "ignore"],
    ):
        def _inner(a):
            return theta_of_alpha(
                alpha=a,
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                link_function=self.link_function,
                dim_alpha=optimizer_results["dim_alpha"],
            )

        expected_value = (
            optimizer_results["theta"] if self.est_mu else np.mean(control_arr, axis=-1)
        )

        return self.create_properties(
            arr=control_arr,
            optimizer_results=optimizer_results,
            jacobian_func=_inner,
            expected_value=expected_value,
            non_positive=non_positive,
        )

    def create_diagnosed_properties(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        optimizer_results: CorrPopsOptimizerResults,
        non_positive: Literal["raise", "warn", "ignore"],
    ):
        def _inner(a):
            theta = theta_of_alpha(
                alpha=a,
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                link_function=self.link_function,
                dim_alpha=optimizer_results["dim_alpha"],
            )
            return triangle_to_vector(
                self.link_function.func(t=theta, a=a, d=optimizer_results["dim_alpha"])
            )

        expected_value = (
            triangle_to_vector(
                self.link_function.func(
                    t=optimizer_results["theta"],
                    a=optimizer_results["alpha"],
                    d=optimizer_results["dim_alpha"],
                )
            )
            if self.est_mu
            else np.mean(diagnosed_arr, axis=-1)
        )

        return self.create_properties(
            arr=diagnosed_arr,
            optimizer_results=optimizer_results,
            jacobian_func=_inner,
            expected_value=expected_value,
            non_positive=non_positive,
        )

    @staticmethod
    def calc_i0(properties: GeeProperties):
        return properties["data"].shape[0] * (
            properties["jacobian"].T @ properties["inv_cov"] @ properties["jacobian"]
        )

    @staticmethod
    def calc_i1(properties: GeeProperties):
        residuals = properties["data"] - properties["expected_value"]
        covariance = residuals.T @ residuals / properties["df"]
        return properties["data"].shape[0] * (
            properties["jacobian"].T
            @ properties["inv_cov"]
            @ covariance
            @ properties["inv_cov"]
            @ properties["jacobian"]
        )

    def compute(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        optimizer_results: CorrPopsOptimizerResults,
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
    ):
        self.link_function.check_name_equal(optimizer_results["link_function"])

        control_properties = self.create_control_properties(
            control_arr=control_arr,
            diagnosed_arr=diagnosed_arr,
            optimizer_results=optimizer_results,
            non_positive=non_positive,
        )
        diagnosed_properties = self.create_diagnosed_properties(
            control_arr=control_arr,
            diagnosed_arr=diagnosed_arr,
            optimizer_results=optimizer_results,
            non_positive=non_positive,
        )
        i0 = self.calc_i0(control_properties) + self.calc_i0(diagnosed_properties)
        i1 = self.calc_i1(control_properties) + self.calc_i1(diagnosed_properties)
        i1_inv = np.linalg.inv(i1)
        return i0 @ i1_inv @ i0
