import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, Literal, TypedDict

import numpy as np
from scipy import optimize

from linalg.matrix import force_symmetry
from linalg.triangle_and_vector import triangle_to_vector, vector_to_triangle
from model.covariance_of_correlation import average_covariance_of_correlation
from model.estimator.optimizer import CorrPopsOptimizerResults
from model.likelihood import theta_of_alpha, sum_of_squares
from model.link_functions import BaseLinkFunction
from statistics.efron_rms import efron_effective_sample_size, efron_rms_from_vectors


class CovarianceEstimator(ABC):
    def _get_params(self, **params: Any) -> Dict[str, Any]:
        return {"name": type(self).__name__, **params}

    def get_params(self) -> Dict[str, Any]:
        return self._get_params()

    def set_params(self, **params: Any) -> "CovarianceEstimator":
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter {k} for estimator {self}.")
            setattr(self, k, v)
        return self

    @abstractmethod
    def compute(  # pragma: no cover
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        link_function: BaseLinkFunction,
        optimizer_results: CorrPopsOptimizerResults,
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
    ) -> np.ndarray:
        raise NotImplementedError


class _GeeProperties(TypedDict):
    data: np.ndarray
    jacobian: np.ndarray
    expected_value: np.ndarray
    inv_cov: np.ndarray
    df: float


class GeeCovarianceEstimator(CovarianceEstimator):
    def __init__(
        self,
        *,
        est_mu: bool = True,
        df_method: Literal["naive", "efron"] = "naive",
    ):
        self.est_mu = est_mu
        self.df_method = df_method

    def get_params(self) -> Dict[str, Any]:
        return super()._get_params(est_mu=self.est_mu, df_method=self.df_method)

    def create_properties(
        self,
        arr: np.ndarray,
        optimizer_results: CorrPopsOptimizerResults,
        jacobian_func: Callable[[np.ndarray], np.ndarray],
        expected_value: np.ndarray,
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
    ) -> _GeeProperties:
        # todo: migrate to scipy.differentiate.jacobian (1.15.0)
        jacobian = optimize.approx_fprime(optimizer_results.alpha, jacobian_func)

        cov, _ = average_covariance_of_correlation(
            vector_to_triangle(arr),
            non_positive=non_positive,
            # don't I need estimated n for something?
        )
        inv_cov = np.linalg.inv(cov)

        if self.df_method == "efron":
            df = efron_effective_sample_size(
                n=np.prod(arr.shape[:-1]),
                rms=efron_rms_from_vectors(arr).mean(),
            )
        else:
            df = np.prod(arr.shape[:-1])

        return _GeeProperties(
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
        link_function: BaseLinkFunction,
        optimizer_results: CorrPopsOptimizerResults,
        non_positive: Literal["raise", "warn", "ignore"],
    ) -> _GeeProperties:
        def jacobian_func(a: np.ndarray) -> np.ndarray:
            theta = theta_of_alpha(
                alpha=a,
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                link_function=link_function,
                dim_alpha=optimizer_results.dim_alpha,
            )
            return theta

        expected_value = (
            optimizer_results.theta
            if self.est_mu
            else np.mean(control_arr, axis=tuple(range(control_arr.ndim - 1)))
        )

        return self.create_properties(
            arr=control_arr,
            optimizer_results=optimizer_results,
            jacobian_func=jacobian_func,
            expected_value=expected_value,
            non_positive=non_positive,
        )

    def create_diagnosed_properties(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        link_function: BaseLinkFunction,
        optimizer_results: CorrPopsOptimizerResults,
        non_positive: Literal["raise", "warn", "ignore"],
    ) -> _GeeProperties:
        def jacobian_func(a: np.ndarray) -> np.ndarray:
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

        expected_value = (
            triangle_to_vector(
                link_function(
                    t=optimizer_results.theta,
                    a=optimizer_results.alpha,
                    d=optimizer_results.dim_alpha,
                )
            )
            if self.est_mu
            else np.mean(diagnosed_arr, axis=tuple(range(diagnosed_arr.ndim - 1)))
        )

        return self.create_properties(
            arr=diagnosed_arr,
            optimizer_results=optimizer_results,
            jacobian_func=jacobian_func,
            expected_value=expected_value,
            non_positive=non_positive,
        )

    @staticmethod
    def calc_i0(properties: _GeeProperties) -> np.ndarray:
        return properties["data"].shape[0] * np.linalg.multi_dot(
            (properties["jacobian"].T, properties["inv_cov"], properties["jacobian"])
        )

    @staticmethod
    def calc_i1(properties: _GeeProperties) -> np.ndarray:
        residuals = properties["data"] - properties["expected_value"]
        return (
            np.linalg.multi_dot(
                (
                    properties["jacobian"].T,
                    properties["inv_cov"],
                    residuals.T,
                    residuals,
                    properties["inv_cov"],
                    properties["jacobian"],
                )
            )
            * properties["data"].shape[0]
            / properties["df"]
        )

    def compute(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        link_function: BaseLinkFunction,
        optimizer_results: CorrPopsOptimizerResults,
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
    ) -> np.ndarray:
        link_function.check_name_equal(optimizer_results.link_function)

        control_properties = self.create_control_properties(
            control_arr=control_arr,
            diagnosed_arr=diagnosed_arr,
            link_function=link_function,
            optimizer_results=optimizer_results,
            non_positive=non_positive,
        )
        diagnosed_properties = self.create_diagnosed_properties(
            control_arr=control_arr,
            diagnosed_arr=diagnosed_arr,
            link_function=link_function,
            optimizer_results=optimizer_results,
            non_positive=non_positive,
        )
        i0 = self.calc_i0(control_properties) + self.calc_i0(diagnosed_properties)
        i1 = self.calc_i1(control_properties) + self.calc_i1(diagnosed_properties)
        i0_inv = np.linalg.inv(i0)
        return np.linalg.multi_dot((i0_inv, i1, i0_inv))


class FisherSandwichCovarianceEstimator(CovarianceEstimator):
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
        return super()._get_params(est_mu=self.est_mu, estimated_n=self.estimated_n)

    @staticmethod
    def compute_by_gradient(
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        expected_value: np.ndarray,
        inv_cov: np.ndarray,
        link_function: BaseLinkFunction,
        optimizer_results: CorrPopsOptimizerResults,
    ) -> np.ndarray:
        def jacobian_func(a: np.ndarray) -> np.ndarray:
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
        return np.linalg.multi_dot(
            (
                jacobian.T,
                inv_cov,
                residuals.T,
                residuals,
                inv_cov,
                jacobian,
            )
        )

    @staticmethod
    def compute_by_hessian(
        diagnosed_arr: np.ndarray,
        inv_cov: np.ndarray,
        link_function: BaseLinkFunction,
        optimizer_results: CorrPopsOptimizerResults,
    ) -> np.ndarray:
        def f(a: np.ndarray) -> float:
            return a.size * sum_of_squares(
                alpha=a,
                theta=optimizer_results.theta,
                diagnosed_arr=diagnosed_arr,
                link_function=link_function,
                inv_sigma=inv_cov,
                dim_alpha=optimizer_results.dim_alpha,
            )

        def jacobian_func(a: np.ndarray) -> np.ndarray:
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
        expected_value = (
            optimizer_results.theta
            if self.est_mu
            else np.mean(control_arr, axis=tuple(range(control_arr.ndim - 1)))
        )

        cov, _ = average_covariance_of_correlation(
            vector_to_triangle(control_arr),
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
        return np.linalg.multi_dot(
            (inv_fisher_by_hessian, fisher_by_gradiant, inv_fisher_by_hessian)
        )
