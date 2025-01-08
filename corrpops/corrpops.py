import copy
from typing import Optional, Union, Literal
from corr_matrix_covariance import corr_matrix_covariance
from link_functions import MultiplicativeIdentity
from estimation_utils import CorrPopsOptimizer
from gee_covariance import GeeCovarianceEstimator


class CorrPops:
    def __init__(
            self,
            optimizer: CorrPopsOptimizer = CorrPopsOptimizer(MultiplicativeIdentity()),
            gee_estimator: Optional[GeeCovarianceEstimator] = GeeCovarianceEstimator(),
            naive_optimizer: Union[CorrPopsOptimizer, bool] = True,
            non_positive: Literal["raise", "warn", "ignore"] = "raise",
    ):
        self.optimizer = optimizer
        self.gee_estimator = gee_estimator

        if type(naive_optimizer) is bool:
            self.naive_optimizer = copy.deepcopy(optimizer) if naive_optimizer else None
        else:
            if naive_optimizer.link_function != optimizer.link_function:
                raise ValueError
            self.naive_optimizer = naive_optimizer

        self.non_positive = non_positive
        self.alpha_ = None
        self.theta_ = None
        self.cov_ = None

    def fit(self, control_arr, diagnosed_arr):

        if self.naive_optimizer is None:
            alpha_ = None
            theta_ = None
        else:
            self.naive_optimizer.optimize(control_arr, diagnosed_arr)
            alpha_ = self.naive_optimizer.alpha_
            theta_ = self.naive_optimizer.theta_

        weight_matrix = corr_matrix_covariance(diagnosed_arr, self.non_positive)
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
