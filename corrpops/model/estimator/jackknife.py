import warnings
from collections import namedtuple
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np

try:
    import ray

    _ray_installed: bool = True
except ModuleNotFoundError:  # pragma: no cover
    _ray_installed: bool = False  # type: ignore

from corrpops_logger import corrpops_logger
from linalg.triangle_and_vector import triangle_to_vector, vector_to_triangle
from model.covariance_of_correlation import average_covariance_of_correlation
from model.estimator.estimator import CorrPopsEstimator
from model.estimator.covariance import GeeCovarianceEstimator
from model.inference import inference, wilks_test, WilksTestResult
from model.link_functions import BaseLinkFunction
from model.estimator.optimizer import CorrPopsOptimizer

logger = corrpops_logger()


JackknifeConstantArgs = namedtuple(
    "JackknifeConstantArgs",
    [
        "optimizer",
        "link_function",
        "control_arr",
        "diagnosed_arr",
        "weights",
        "alpha0",
        "theta0",
        "covariance_estimator",
    ],
)


class JackknifeResult(TypedDict):
    theta: np.ndarray
    alpha: np.ndarray
    status: int
    cov: Optional[np.ndarray]


def _jackknife_single(
    index_to_drop: int,
    drop_in_diagnosed: bool,
    optimizer: CorrPopsOptimizer,
    link_function: BaseLinkFunction,
    control_arr: np.ndarray,
    diagnosed_arr: np.ndarray,
    weights: np.ndarray,
    alpha0: np.ndarray,
    theta0: np.ndarray,
    covariance_estimator: Optional[GeeCovarianceEstimator],
) -> JackknifeResult:
    if drop_in_diagnosed:
        diagnosed_arr = np.delete(diagnosed_arr, index_to_drop, axis=0)
        weights, _ = average_covariance_of_correlation(
            vector_to_triangle(diagnosed_arr),
            non_positive="ignore",
            # maybe we can avoid this and use the general weights?
        )
    else:
        control_arr = np.delete(control_arr, index_to_drop, axis=0)

    results = optimizer.optimize(
        control_arr=control_arr,
        diagnosed_arr=diagnosed_arr,
        link_function=link_function,
        alpha0=alpha0,
        theta0=theta0,
        weights=weights,
    )
    try:
        cov = covariance_estimator.compute(  # type: ignore
            control_arr=control_arr,
            diagnosed_arr=diagnosed_arr,
            link_function=link_function,
            optimizer_results=results,
            non_positive="ignore",
        )
    except (AttributeError, ValueError):  # pragma: no cover
        cov = None

    return {
        "theta": results.theta,
        "alpha": results.alpha,
        "status": results.steps[-1]["status"],
        "cov": cov,
    }


class CorrPopsJackknifeEstimator:
    def __init__(
        self,
        base_estimator: CorrPopsEstimator,
        *,
        jack_control: bool = True,
        steps_back: int = 3,
        use_ray: bool = False,
        ray_options: Optional[Dict[str, Any]] = None,
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
    ):
        self.base_estimator = base_estimator
        self.jack_control = jack_control
        self.steps_back = steps_back
        self.use_ray = use_ray
        self.ray_options = ray_options or {}
        self.non_positive = non_positive
        self.is_fitted = False

        self.alpha_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None
        self.cov_: Optional[np.ndarray] = None
        self.cov_est_: Optional[np.ndarray] = None

        self.control_index_: Optional[np.ndarray] = None
        self.diagnosed_index_: Optional[np.ndarray] = None

        self.alpha_stack_: Optional[np.ndarray] = None
        self.theta_stack_: Optional[np.ndarray] = None
        self.cov_est_stack_: Optional[np.ndarray] = None

    @staticmethod
    def stack_if_not_none(
        results: List[JackknifeResult],
        key: Literal["theta", "alpha", "status", "cov"],
        param_shape: Tuple[int, ...],
    ) -> np.ndarray:
        return np.stack(
            [
                np.full(param_shape, np.nan) if element[key] is None else element[key]
                for element in results
            ]
        )

    @staticmethod
    def aggregate_stacks(
        alpha_stack: np.ndarray,
        control_index: np.ndarray,
        diagnosed_index: np.ndarray,
        theta_stack: Optional[np.ndarray] = None,
        cov_est_stack: Optional[np.ndarray] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        diagnosed_n = len(diagnosed_index)
        diagnosed_constant = (diagnosed_n - 1) ** 2 / diagnosed_n
        # diagnosed_mean = alpha_stack[diagnosed_index].mean(0)
        diagnosed_variance = diagnosed_constant * np.cov(
            alpha_stack[diagnosed_index], rowvar=False
        )

        if len(control_index):
            control_n = len(control_index)
            control_constant = (control_n - 1) ** 2 / control_n
            # control_mean = alpha_stack[control_index].mean(0)
            control_variance = control_constant * np.cov(
                alpha_stack[control_index], rowvar=False
            )
        else:
            control_variance = np.zeros_like(diagnosed_variance)

        return {
            "alpha": alpha_stack.mean(0),
            # "alpha": (diagnosed_mean + control_mean) / 2,
            "cov": diagnosed_variance + control_variance,
            "theta": None if theta_stack is None else theta_stack.mean(0),
            "cov_est": None if cov_est_stack is None else cov_est_stack.mean(0),
        }

    @staticmethod
    def results_from_json(json_: Dict[str, Any]) -> Dict[str, Any]:
        results = json_["results"]
        out = {
            "alpha": np.array(results["alpha"]),
            "theta": np.array(results["theta"]),
            "cov": vector_to_triangle(np.array(results["cov"]), diag=True),
            "cov_est_": vector_to_triangle(np.array(results["cov_est_"]), diag=True)
            if results["cov_est_"]
            else None,
        }

        if "stacks" in results:
            stacks = results["stacks"]
            out["stacks"] = {
                "alpha": np.array(stacks["alpha"]),
                "theta": np.array(stacks["theta"]),
                "cov_est_": vector_to_triangle(np.array(stacks["cov_est_"]), diag=True)
                if stacks["cov_est_"]
                else None,
                "control_index": np.array(stacks["control_index"]),
                "diagnosed_index": np.array(stacks["diagnosed_index"]),
            }
        return out

    def to_json(
        self,
        save_results: bool = True,
        save_params: bool = True,
        save_naive: bool = False,
        save_base_results: bool = False,
        save_stacks: bool = False,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        json_ = {}

        if save_params:
            json_["params"] = {
                "base_estimator": self.base_estimator.to_json(
                    save_results=save_base_results,
                    save_params=True,
                    save_naive=save_naive,
                ),
                "jackknife": {
                    "jack_control": self.jack_control,
                    "steps_back": self.steps_back,
                },
            }
        if save_results and self.is_fitted:
            json_["results"] = {  # type: ignore
                "alpha": self.alpha_.tolist(),
                "theta": self.theta_.tolist(),
                "cov": triangle_to_vector(self.cov_, diag=True).tolist(),
                "cov_est_": triangle_to_vector(self.cov_est_, diag=True).tolist()
                if self.cov_est_ is not None
                else [],
            }
            if save_stacks:
                json_["results"]["stacks"] = {
                    "alpha": self.alpha_stack_.tolist(),
                    "theta": self.theta_stack_.tolist(),
                    "cov_est_": triangle_to_vector(
                        self.cov_est_stack_, diag=True
                    ).tolist()
                    if self.cov_est_stack_ is not None
                    else [],
                    "control_index": self.control_index_.tolist(),
                    "diagnosed_index": self.diagnosed_index_.tolist(),
                }
        elif save_results:
            json_["results"] = {}

        return json_

    def _set_indices(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
    ) -> None:
        self.diagnosed_index_ = np.arange(diagnosed_arr.shape[0])
        if self.jack_control:
            self.control_index_ = self.diagnosed_index_.size + np.arange(
                control_arr.shape[0]
            )
        else:
            self.control_index_ = np.array([])

    def _get_jackknife(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        alpha0: np.ndarray,
        theta0: np.ndarray,
        weights: np.ndarray,
        compute_cov: bool,
    ) -> List[JackknifeResult]:
        args = JackknifeConstantArgs(
            optimizer=self.base_estimator.optimizer,
            link_function=self.base_estimator.link_function,
            control_arr=control_arr,
            diagnosed_arr=diagnosed_arr,
            weights=weights,
            alpha0=alpha0,
            theta0=theta0,
            covariance_estimator=self.base_estimator.covariance_estimator
            if compute_cov
            else None,
        )

        results = []
        for index in range(diagnosed_arr.shape[0]):
            current = _jackknife_single(index, True, *args)
            results.append(current)
            print("+", end="")
        if self.jack_control:
            for index in range(control_arr.shape[0]):
                current = _jackknife_single(index, False, *args)
                results.append(current)
                print("+", end="")
        print("")
        return results

    def _get_jackknife_ray(  # pragma: no cover
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        alpha0: np.ndarray,
        theta0: np.ndarray,
        weights: np.ndarray,
        compute_cov: bool,
        ray_options: Optional[Dict[str, Any]],
    ) -> List[JackknifeResult]:
        if not _ray_installed:
            raise ModuleNotFoundError(
                "missing optional dependency ray. "
                "use_ray=True is not supported. "
                "for multiprocessing install ray."
            )
        options = self.ray_options.copy()
        options.update(ray_options or {})
        _jackknife_remote = ray.remote(_jackknife_single).options(**options)

        args = JackknifeConstantArgs(
            optimizer=ray.put(self.base_estimator.optimizer),
            link_function=ray.put(self.base_estimator.link_function),
            control_arr=ray.put(control_arr),
            diagnosed_arr=ray.put(diagnosed_arr),
            weights=ray.put(weights),
            alpha0=ray.put(alpha0),
            theta0=ray.put(theta0),
            covariance_estimator=ray.put(self.base_estimator.covariance_estimator)
            if compute_cov
            else None,
        )

        futures = []
        for index in range(diagnosed_arr.shape[0]):
            run_id = _jackknife_remote.remote(index, True, *args)
            futures.append(run_id)
        if self.jack_control:
            for index in range(control_arr.shape[0]):
                run_id = _jackknife_remote.remote(index, False, *args)
                futures.append(run_id)
        return ray.get(futures)

    def fit(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        *,
        compute_cov: bool = False,
        ray_options: Optional[Dict[str, Any]] = None,
    ) -> "CorrPopsJackknifeEstimator":
        if self.base_estimator.optimizer_results_ is None:
            logger.info("jackknife:fit: run base_estimator.fit")
            self.base_estimator.fit(
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                compute_cov=False,
            )
        steps = self.base_estimator.optimizer_results_.steps  # type: ignore

        if len(steps) > self.steps_back:
            alpha0 = steps[-self.steps_back - 1]["alpha"]
            theta0 = steps[-self.steps_back - 1]["theta"]
        else:
            warnings.warn(
                f"not enough steps in base_estimator optimizer results "
                f"({len(steps)} < {self.steps_back}). taking last step instead"
            )
            alpha0 = self.base_estimator.alpha_
            theta0 = self.base_estimator.theta_

        weight_matrix, _ = average_covariance_of_correlation(
            diagnosed_arr,
            non_positive=self.non_positive,
        )

        logger.info("jackknife:fit: starting jackknife procedure")
        if self.use_ray:  # pragma: no cover
            results = self._get_jackknife_ray(
                control_arr=triangle_to_vector(control_arr),
                diagnosed_arr=triangle_to_vector(diagnosed_arr),
                alpha0=alpha0,
                theta0=theta0,
                weights=weight_matrix,
                compute_cov=compute_cov,
                ray_options=ray_options,
            )
        else:
            results = self._get_jackknife(
                control_arr=triangle_to_vector(control_arr),
                diagnosed_arr=triangle_to_vector(diagnosed_arr),
                alpha0=alpha0,
                theta0=theta0,
                weights=weight_matrix,
                compute_cov=compute_cov,
            )
        self._set_indices(control_arr=control_arr, diagnosed_arr=diagnosed_arr)

        self.alpha_stack_ = self.stack_if_not_none(results, "alpha", alpha0.shape)
        self.theta_stack_ = self.stack_if_not_none(results, "theta", theta0.shape)
        if compute_cov:
            self.cov_est_stack_ = self.stack_if_not_none(
                results, "cov", (alpha0.size, alpha0.size)
            )

        aggregates = self.aggregate_stacks(
            alpha_stack=self.alpha_stack_,
            control_index=self.control_index_,
            diagnosed_index=self.diagnosed_index_,
            theta_stack=self.theta_stack_,
            cov_est_stack=self.cov_est_stack_,
        )
        self.alpha_ = aggregates["alpha"]
        self.theta_ = aggregates["theta"]
        self.cov_ = aggregates["cov"]
        self.cov_est_ = aggregates["cov_est"]
        self.is_fitted = True
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
            theta=vector_to_triangle(self.theta_),
            alpha=self.alpha_,
            link_function=self.base_estimator.link_function,
            dim_alpha=self.base_estimator.dim_alpha,
            non_positive=self.non_positive,
        )
