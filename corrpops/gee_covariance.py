from dataclasses import dataclass
from typing import Callable, Literal
import numpy as np

from corr_matrix_covariance import average_covariance_matrix
from efron_rms import efrons_effective_sample_size, efron_rms_sample
from estimation_utils import CorrPopsOptimizer, theta_of_alpha
from jacobian import simple_jacobian, richardson_jacobian
from triangle_vector import triangle_to_vector


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
            jacobian_method: Literal["simple", "richardson"] = "simple",
            dof_method: Literal["naive", "efron"] = "naive",
    ):
        self.est_mu = est_mu
        self.jacobian_method = jacobian_method
        self.dof_method = dof_method

    def create_properties(
            self,
            arr: np.ndarray,
            optimizer: CorrPopsOptimizer,
            jacobian_func: Callable[[np.ndarray], np.ndarray],
            expected_value: np.ndarray | None,
    ):
        data = triangle_to_vector(arr)

        if self.jacobian_method == "simple":
            jacobian = simple_jacobian(jacobian_func, optimizer.alpha_)
        else:
            jacobian = richardson_jacobian(jacobian_func, optimizer.alpha_)

        cov = average_covariance_matrix(arr, est_n=True)
        inv_cov = np.linalg.inv(cov)

        if self.dof_method == "efron":
            dof = efrons_effective_sample_size(
                n=np.prod(data.shape[:-1]),
                rms=efron_rms_sample(data),
            )
        else:
            dof = np.prod(data.shape[0])

        return GeeProperties(
            data,
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
    ):
        def _inner(a):
            return theta_of_alpha(
                a,
                control_arr,
                diagnosed_arr,
                optimizer.link_function,
                optimizer.alpha_.shape[-1],
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
    ):
        def _inner(a):
            theta = theta_of_alpha(
                a,
                control_arr,
                diagnosed_arr,
                optimizer.link_function,
                optimizer.alpha_.shape[-1],
            )
            return triangle_to_vector(
                optimizer.link_function.func(
                    theta,
                    a,
                    optimizer.alpha_.shape[-1],
                )
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
        residuals = properties.data - properties.expected_value[:, None]
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
        control_properties = self.create_control_properties(control_arr, diagnosed_arr, optimizer)
        diagnosed_properties = self.create_control_properties(control_arr, diagnosed_arr, optimizer)
        i0 = self.calc_i0(control_properties) + self.calc_i0(diagnosed_properties)
        i1 = self.calc_i1(control_properties) + self.calc_i1(diagnosed_properties)
        i1_inv = np.linalg.inv(i1)
        return i0 @ i1_inv @ i0
