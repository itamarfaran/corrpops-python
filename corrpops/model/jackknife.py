from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

try:
    import ray

    _ray_installed: bool = True
except ModuleNotFoundError:
    _ray_installed: bool = False


from linalg.triangle_vector import triangle_to_vector
from .covariance_of_correlation import average_covariance_of_correlation
from .estimator import CorrPopsEstimator
from .gee_covariance import GeeCovarianceEstimator
from .inference import inference, wilks_test, WilksTestResult
from .optimizer import CorrPopsOptimizer, CorrPopsOptimizerResults


def _jackknife(
    index: Union[int, Tuple[int, ...]],
    jack_diagnosed: bool,
    optimizer: CorrPopsOptimizer,
    control_arr: np.ndarray,
    diagnosed_arr: np.ndarray,
    alpha0: np.ndarray,
    theta0: np.ndarray,
    weights: Optional[np.ndarray] = None,
    gee_estimator: Optional[GeeCovarianceEstimator] = None,
):
    # todo: perhaps the deletion should happen outside of this function?
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
        use_ray: bool = False,
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
    ):
        self.base_estimator = base_estimator
        self.jack_control = jack_control
        self.steps_back = steps_back
        self.use_ray = use_ray
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

    def set_indices(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
    ):
        self.diagnosed_index_ = np.arange(diagnosed_arr.shape[0])
        if self.jack_control:
            self.control_index_ = self.diagnosed_index_.size + np.arange(
                control_arr.shape[0]
            )
        else:
            self.control_index_ = np.array([])

    def get_jackknife(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        alpha0: np.ndarray,
        theta0: np.ndarray,
        weights: np.ndarray,
        compute_cov: bool,
    ) -> List[Dict[str, np.ndarray]]:
        def _partial_jackknife(
            index: Union[int, Tuple[int, ...]],
            jack_diagnosed: bool,
        ):
            return _jackknife(
                index=index,
                jack_diagnosed=jack_diagnosed,
                optimizer=self.base_estimator.optimizer,
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                alpha0=alpha0,
                theta0=theta0,
                weights=weights,
                gee_estimator=self.base_estimator.gee_estimator if compute_cov else None,
            )
        results = [
            _partial_jackknife(index, True) for index in range(diagnosed_arr.shape[0])
        ]
        if self.jack_control:
            results += [
                _partial_jackknife(index, False) for index in range(control_arr.shape[0])
            ]
        return results

    def get_jackknife_ray(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        alpha0: np.ndarray,
        theta0: np.ndarray,
        weights: np.ndarray,
        compute_cov: bool,
    ) -> List[Dict[str, np.ndarray]]:
        if not _ray_installed:
            raise ModuleNotFoundError(
                "missing optional dependency ray. "
                "use_ray=True is not supported. "
                "for multiprocessing install ray."
            )

        @ray.remote
        def _partial_jackknife(
            index: Union[int, Tuple[int, ...]],
            jack_diagnosed: bool,
        ):
            return _jackknife(
                index=index,
                jack_diagnosed=jack_diagnosed,
                optimizer=self.base_estimator.optimizer,
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                alpha0=alpha0,
                theta0=theta0,
                weights=weights,
                gee_estimator=self.base_estimator.gee_estimator if compute_cov else None,
            )

        futures = [
            _partial_jackknife.remote(index, True) for index in range(diagnosed_arr.shape[0])
        ]
        if self.jack_control:
            futures += [
                _partial_jackknife.remote(index, False) for index in range(control_arr.shape[0])
            ]
        results = ray.get(futures)
        return results

    def fit(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        *,
        compute_cov: bool = False,
        optimizer_results: Optional[CorrPopsOptimizerResults] = None,
    ):
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

        get_jackknife = self.get_jackknife_ray if self.use_ray else self.get_jackknife

        results = get_jackknife(
            control_arr=control_arr,
            diagnosed_arr=diagnosed_arr,
            alpha0=alpha0,
            theta0=theta0,
            weights=weight_matrix,
            compute_cov=compute_cov,
        )
        self.set_indices(control_arr=control_arr, diagnosed_arr=diagnosed_arr)

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

    def inference(
        self,
        p_adjust_method: Optional[str] = None,
        alternative: Literal["two-sided", "smaller", "larger"] = "two-sided",
        sig_level: float = 0.05,
        std_const: float = 1.0,
        known_alpha: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        return inference(
            alpha=self.alpha_,
            cov=self.cov_,
            link_function=self.base_estimator.link_function,
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
            link_function=self.base_estimator.link_function,
            dim_alpha=self.base_estimator.dim_alpha,
            non_positive=self.non_positive,
        )
