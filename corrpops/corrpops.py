import copy
from typing import Optional, Union, Literal
from corr_matrix_covariance import average_covariance_matrix
from link_functions import MultiplicativeIdentity
from estimation_utils import CorrPopsOptimizer
from gee_covariance import GeeCovarianceEstimator
from triangle_vector import triangle_to_vector


class CorrPops:
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
        weight_matrix, _ = average_covariance_matrix(
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
