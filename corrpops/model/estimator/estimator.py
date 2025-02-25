import copy
import logging
from typing import Any, Dict, Optional, Union, Literal

import numpy as np

from linalg.triangle_and_vector import triangle_to_vector, vector_to_triangle
from model.covariance_of_correlation import average_covariance_of_correlation
from model.estimator.covariance import CovarianceEstimator, GeeCovarianceEstimator
from model.estimator.optimizer import (
    CorrPopsOptimizer,
    CorrPopsOptimizerResults,
)
from model.inference import inference, wilks_test, WilksTestResult
from model.link_functions import BaseLinkFunction, MultiplicativeIdentity

logger = logging.getLogger("corrpops")
logging.basicConfig(level=logging.INFO)


_init_value_error_msg = (
    "cannot pass both an instantiated {0} and non-empty {0}_kwargs. "
    "either pass an instantiated {0} with non-default attributes "
    "or pass non-empty {0}_kwargs to modify a new {1} "
    "instantiated in CorrPopsEstimator.__init__."
)


class CorrPopsEstimator:
    def __init__(
        self,
        link_function: BaseLinkFunction = MultiplicativeIdentity(),
        dim_alpha: int = 1,
        *,
        optimizer: Optional[CorrPopsOptimizer] = None,
        naive_optimizer: Union[CorrPopsOptimizer, Literal["skip"], None] = None,
        covariance_estimator: Optional[CovarianceEstimator] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        naive_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        covariance_estimator_kwargs: Optional[Dict[str, Any]] = None,
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
    ):
        if optimizer is None:
            optimizer_kwargs = optimizer_kwargs or {}
            self.optimizer = CorrPopsOptimizer(**optimizer_kwargs)
        elif optimizer_kwargs:
            raise ValueError(
                _init_value_error_msg.format("optimizer", "CorrPopsOptimizer")
            )
        else:
            self.optimizer = optimizer

        if naive_optimizer is None:
            naive_optimizer_kwargs = naive_optimizer_kwargs or {}
            self.naive_optimizer = copy.deepcopy(self.optimizer).set_params(
                **naive_optimizer_kwargs
            )
        elif naive_optimizer == "skip":
            self.naive_optimizer = None
        elif naive_optimizer_kwargs:
            raise ValueError(
                _init_value_error_msg.format("naive_optimizer", "CorrPopsOptimizer")
            )
        else:
            self.naive_optimizer = naive_optimizer

        if covariance_estimator is None:
            covariance_estimator_kwargs = covariance_estimator_kwargs or {}
            self.covariance_estimator = GeeCovarianceEstimator(
                **covariance_estimator_kwargs
            )
        elif covariance_estimator_kwargs:
            raise ValueError(
                _init_value_error_msg.format(
                    "covariance_estimator", "CovarianceEstimator"
                )
            )
        else:
            self.covariance_estimator = covariance_estimator

        self.link_function = link_function
        self.dim_alpha = dim_alpha
        self.non_positive = non_positive
        self.is_fitted = False

        self.alpha_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None
        self.cov_: Optional[np.ndarray] = None
        self.optimizer_results_: Optional[CorrPopsOptimizerResults] = None
        self.naive_optimizer_results_: Optional[CorrPopsOptimizerResults] = None

    @classmethod
    def from_json(
        cls,
        json_: Dict[str, Dict[str, Dict[str, Any]]],
        link_function: BaseLinkFunction,
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
    ):
        if "params" in json_:
            covariance_estimator_kwargs = json_["params"]["covariance_estimator"].copy()
            if (
                covariance_estimator_kwargs.pop("name")
                != GeeCovarianceEstimator.__name__
            ):
                raise ValueError(
                    "currently only GeeCovarianceEstimator supported in from_json"
                )

            estimator = cls(
                link_function=link_function,
                optimizer=CorrPopsOptimizer(**json_["params"]["optimizer"]),
                naive_optimizer=CorrPopsOptimizer(**json_["params"]["naive_optimizer"]),
                covariance_estimator=GeeCovarianceEstimator(
                    **covariance_estimator_kwargs
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
                estimator.optimizer_results_ = CorrPopsOptimizerResults.from_json(
                    json_["results"]["optimizer"]
                )
                estimator.alpha_ = estimator.optimizer_results_.alpha
                estimator.theta_ = estimator.optimizer_results_.theta
            if "naive_optimizer" in json_["results"]:
                estimator.naive_optimizer_results_ = CorrPopsOptimizerResults.from_json(
                    json_["results"]["naive_optimizer"]
                )
            if "covariance_estimator" in json_["results"]:
                estimator.cov_ = vector_to_triangle(
                    np.array(json_["results"]["covariance_estimator"]["cov"]), diag=True
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
                "covariance_estimator": self.covariance_estimator.get_params()
                if self.covariance_estimator
                else {},
            }

        if save_results:
            json_["results"] = {}

            if self.optimizer_results_:
                json_["results"]["optimizer"] = self.optimizer_results_.to_json()

            if save_naive and self.naive_optimizer_results_:
                json_["results"][
                    "naive_optimizer"
                ] = self.naive_optimizer_results_.to_json()

            if self.cov_ is not None:
                json_["results"]["covariance_estimator"] = {
                    "cov": triangle_to_vector(self.cov_, diag=True).tolist()
                }

        return json_

    def fit(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        *,
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
            logger.info("estimator:fit: run naive_optimizer.optimize")
            self.naive_optimizer_results_ = self.naive_optimizer.optimize(
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                link_function=self.link_function,
                dim_alpha=self.dim_alpha,
            )
            alpha0 = self.naive_optimizer_results_.alpha
            theta0 = self.naive_optimizer_results_.theta

        logger.info("estimator:fit: run optimizer.optimize")
        self.optimizer_results_ = self.optimizer.optimize(
            control_arr=control_arr,
            diagnosed_arr=diagnosed_arr,
            link_function=self.link_function,
            dim_alpha=self.dim_alpha,
            alpha0=alpha0,
            theta0=theta0,
            weights=weight_matrix,
        )
        self.alpha_ = self.optimizer_results_.alpha
        self.theta_ = self.optimizer_results_.theta
        self.is_fitted = True

        if compute_cov and self.covariance_estimator is not None:
            logger.info("estimator:fit: run covariance_estimator.compute")
            self.compute_covariance(
                control_arr=vector_to_triangle(control_arr),
                diagnosed_arr=vector_to_triangle(diagnosed_arr),
            )
        return self

    def compute_covariance(self, control_arr: np.ndarray, diagnosed_arr: np.ndarray):
        self.cov_ = self.covariance_estimator.compute(
            control_arr=triangle_to_vector(control_arr),
            diagnosed_arr=triangle_to_vector(diagnosed_arr),
            link_function=self.link_function,
            optimizer_results=self.optimizer_results_,
            non_positive=self.non_positive,
        )
        return self

    def inference(
        self,
        p_adjust_method: str = "bonferroni",
        alternative: Literal["two-sided", "smaller", "larger"] = "two-sided",
        sig_level: float = 0.05,
        std_const: float = 1.0,
        known_alpha: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        return inference(
            alpha=self.alpha_,
            cov=self.cov_,
            link_function=self.link_function,
            p_adjust_method=p_adjust_method,
            alternative=alternative,
            sig_level=sig_level,
            std_const=std_const,
            known_alpha=known_alpha,
        )

    def score(
        self, control_arr: np.ndarray, diagnosed_arr: np.ndarray
    ) -> WilksTestResult:
        return wilks_test(
            control_arr=control_arr,
            diagnosed_arr=diagnosed_arr,
            theta=self.theta_,
            alpha=self.alpha_,
            link_function=self.link_function,
            dim_alpha=self.dim_alpha,
            non_positive=self.non_positive,
        )
