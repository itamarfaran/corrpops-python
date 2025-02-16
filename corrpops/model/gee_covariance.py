from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np

from utils.covariance_of_correlation import average_covariance_of_correlation
from utils.efron_rms import efron_effective_sample_size, efron_rms_sample
from utils.jacobian import simple_jacobian, richardson_jacobian
from utils.triangle_vector import triangle_to_vector, vector_to_triangle
from .likelihood import theta_of_alpha
from .optimizer import CorrPopsOptimizer


@dataclass
class GeeProperties:
    data: np.ndarray
    jacobian: np.ndarray
    expected_value: np.ndarray
    inv_cov: np.ndarray
    dof: float


class GeeCovarianceEstimator:
    def __init__(
            self,
            est_mu: bool = True,
            jacobian_method: Literal["simple", "richardson"] = "richardson",
            sample_size: Literal["estimated", "naive"] = "estimated",
            dof_method: Literal["naive", "efron"] = "naive",
    ):
        self.est_mu = est_mu
        self.jacobian_method = jacobian_method
        self.sample_size = sample_size
        self.dof_method = dof_method

    def create_properties(
            self,
            arr: np.ndarray,
            optimizer: CorrPopsOptimizer,
            jacobian_func: Callable[[np.ndarray], np.ndarray],
            expected_value: Optional[np.ndarray],
    ):
        if self.jacobian_method == "simple":
            jacobian = simple_jacobian(jacobian_func, optimizer.alpha_)
        else:
            jacobian = richardson_jacobian(jacobian_func, optimizer.alpha_)

        cov, _ = average_covariance_of_correlation(
            vector_to_triangle(arr, diag_value=1),  # don't I need estimated n for something?
        )
        inv_cov = np.linalg.inv(cov)

        if self.dof_method == "efron":
            dof = efron_effective_sample_size(
                n=np.prod(arr.shape[:-1]),
                rms=efron_rms_sample(arr),
            )
        else:
            dof = np.prod(arr.shape[0])

        return GeeProperties(
            arr,
            jacobian,
            expected_value,
            inv_cov,
            dof,
        )

    def create_control_properties(
            self,
            control_arr,
            diagnosed_arr,
            optimizer,
            d,
    ):
        def _inner(a):
            return theta_of_alpha(
                a,
                control_arr,
                diagnosed_arr,
                optimizer.link_function,
                d,
            )

        expected_value = (
            optimizer.theta_
            if self.est_mu else
            np.mean(control_arr, axis=-1)
        )

        return self.create_properties(
            arr=control_arr,
            optimizer=optimizer,
            jacobian_func=_inner,
            expected_value=expected_value,
        )

    def create_diagnosed_properties(
            self,
            control_arr,
            diagnosed_arr,
            optimizer,
            d,
    ):
        def _inner(a):
            theta = theta_of_alpha(
                a,
                control_arr,
                diagnosed_arr,
                optimizer.link_function,
                d,
            )
            return triangle_to_vector(
                optimizer.link_function.func(theta, a, d)
            )

        expected_value = (
            triangle_to_vector(
                optimizer.link_function.func(
                    t=optimizer.theta_,
                    a=optimizer.alpha_,
                    d=optimizer.alpha_.shape[-1],
                )
            ) if self.est_mu else
            np.mean(diagnosed_arr, axis=-1)
        )

        return self.create_properties(
            arr=diagnosed_arr,
            optimizer=optimizer,
            jacobian_func=_inner,
            expected_value=expected_value,
        )

    @staticmethod
    def calc_i0(properties: GeeProperties):
        return properties.data.shape[0] * (
            properties.jacobian.T
            @ properties.inv_cov
            @ properties.jacobian
        )

    @staticmethod
    def calc_i1(properties: GeeProperties):
        residuals = properties.data - properties.expected_value
        covariance = residuals.T @ residuals / properties.dof
        return properties.data.shape[0] * (
            properties.jacobian.T
            @ properties.inv_cov
            @ covariance
            @ properties.inv_cov
            @ properties.jacobian
        )

    def compute(
            self,
            control_arr,
            diagnosed_arr,
            optimizer,
    ):
        p = 0.5 + np.sqrt(1 + 8 * control_arr.shape[-1]) / 2
        d = int(optimizer.alpha_.size / p)

        control_properties = self.create_control_properties(control_arr, diagnosed_arr, optimizer, d)
        diagnosed_properties = self.create_diagnosed_properties(control_arr, diagnosed_arr, optimizer, d)
        i0 = self.calc_i0(control_properties) + self.calc_i0(diagnosed_properties)
        i1 = self.calc_i1(control_properties) + self.calc_i1(diagnosed_properties)
        i1_inv = np.linalg.inv(i1)
        return i0 @ i1_inv @ i0
