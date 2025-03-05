import copy
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

import numpy as np
from scipy import linalg, optimize

from corrpops_logger import corrpops_logger
from linalg import matrix, triangle_and_vector as tv, vector
from model.likelihood import theta_of_alpha, sum_of_squares
from model.link_functions import BaseLinkFunction

logger = corrpops_logger()


def format_time_delta(time_delta: timedelta) -> str:
    time_delta_in_seconds = time_delta.seconds
    hours = time_delta_in_seconds // 3_600
    minutes = (time_delta_in_seconds - hours * 3_600) // 60
    seconds = time_delta_in_seconds - hours * 3_600 - minutes * 60

    times = {
        "hours": str(hours),
        "minutes": str(minutes),
        "seconds": str(seconds),
    }
    for k, v in times.items():
        if len(v) == 1:
            times[k] = "0" + v

    out = f"{times['minutes']}:{times['seconds']}"
    if times["hours"] != "00":
        out = f"{times['hours']}:{out}"
    return out


class StepDict(TypedDict):
    theta: np.ndarray
    alpha: np.ndarray
    value: float
    status: int
    optimize_results: Union[dict, optimize.OptimizeResult]


@dataclass
class CorrPopsOptimizerResults:
    theta: np.ndarray
    alpha: np.ndarray
    inv_cov: Union[np.ndarray, None]
    link_function: str
    p: int
    dim_alpha: int
    steps: List[StepDict]

    def to_json(self) -> Dict[str, Any]:
        json_ = asdict(self)
        json_.update(
            theta=json_["theta"].tolist(),
            alpha=json_["alpha"].tolist(),
            inv_cov=tv.triangle_to_vector(json_["inv_cov"], diag=True).tolist()
            if json_["inv_cov"] is not None
            else [],
        )
        json_.pop("steps")
        return json_

    @classmethod
    def from_json(cls, json_):
        return cls(
            theta=np.array(json_["theta"]),
            alpha=np.array(json_["alpha"]),
            inv_cov=(
                tv.vector_to_triangle(np.array(json_["inv_cov"]), diag=True)
                if json_["inv_cov"]
                else None
            ),
            link_function=json_["link_function"],
            p=json_["p"],
            dim_alpha=json_["dim_alpha"],
            steps=[],
        )


class CorrPopsOptimizer:
    def __init__(
        self,
        *,
        rel_tol: float = -1.0,
        abs_tol: float = -1.0,
        tol_p: float = -1.0,
        early_stop: bool = False,
        min_iter: int = 3,
        max_iter: int = 50,
        reg_lambda: float = 0.0,
        reg_p: float = 2.0,
        mat_reg_const: float = 0.0,
        mat_reg_method: Literal["constant", "avg_diag", "increase_diag"] = "constant",
        mat_reg_only_if_singular: bool = False,
        minimize_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = True,
        save_optimize_results: bool = False,
    ):
        self._set_tol(rel_tol, abs_tol, tol_p)

        self.early_stop = early_stop
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
        self.reg_p = reg_p
        self.mat_reg_const = mat_reg_const
        self.mat_reg_method = mat_reg_method
        self.mat_reg_only_if_singular = mat_reg_only_if_singular
        self.minimize_kwargs = minimize_kwargs or {}
        self.verbose = verbose > 0
        self.save_optimize_results = save_optimize_results

        if "options" not in self.minimize_kwargs:
            self.minimize_kwargs["options"] = {}

    def _set_tol(self, rel_tol: float, abs_tol: float, tol_p: float):
        if rel_tol > 0.0:
            if abs_tol > 0.0:
                raise ValueError(
                    "at most one of 'rel_tol', " "'abs_tol' should greater than 0.0"
                )
            else:
                self.rel_tol = rel_tol
                self.abs_tol = 0.0
        else:
            if abs_tol > 0.0:
                self.rel_tol = 0.0
                self.abs_tol = abs_tol
            else:
                # defaults:
                self.rel_tol = np.sqrt(np.finfo(float).eps)
                self.abs_tol = 0.0

        if tol_p > 0.0:
            self.tol_p = tol_p
        elif self.rel_tol:
            self.tol_p = 1.0
        elif self.abs_tol:
            self.tol_p = 2.0

    @staticmethod
    def _check_positive_definite(
        theta: np.ndarray,
        alpha: np.ndarray,
        link_function: BaseLinkFunction,
        dim_alpha: int,
    ):
        is_positive_definite_ = (
            matrix.is_positive_definite(tv.vector_to_triangle(theta)),
            matrix.is_positive_definite(link_function(t=theta, a=alpha, d=dim_alpha)),
        )
        if not all(is_positive_definite_):  # pragma: no cover
            warnings.warn("initial parameters dont yield positive-definite matrices")

    def _log(self, msg_type: Literal["start", "progress", "end"], **kwargs):
        now = datetime.now()

        if msg_type == "start":
            msg = f"optimizer:optimize: start ({now})"
        elif msg_type == "progress":
            msg = (
                f"optimizer:optimize: iteration {len(kwargs['steps'])} "
                f"(elapsed: {(now - kwargs['start_time']).seconds}s, "
                f"current: {(now - kwargs['last_start_time']).seconds}s, "
                f"status: {kwargs['steps'][-1]['status']}, "
                f" distance: {np.round(kwargs['distance'], 5)})"
            )
        elif msg_type == "end":
            msg = (
                f"optimizer:optimize: end ({now}) | "
                f"iterations: {len(kwargs['steps'])}, "
                f"total time: {format_time_delta(now - kwargs['start_time'])}"
            )
        else:  # pragma: no cover
            raise ValueError(f"unrecognized msg_type {msg_type}")

        (logger.info if self.verbose else logger.debug)(msg)
        return now

    def get_params(self) -> Dict[str, Any]:
        return {
            "rel_tol": self.rel_tol,
            "abs_tol": self.abs_tol,
            "tol_p": self.tol_p,
            "early_stop": self.early_stop,
            "min_iter": self.min_iter,
            "max_iter": self.max_iter,
            "reg_lambda": self.reg_lambda,
            "reg_p": self.reg_p,
            "mat_reg_const": self.mat_reg_const,
            "mat_reg_method": self.mat_reg_method,
            "mat_reg_only_if_singular": self.mat_reg_only_if_singular,
            "minimize_kwargs": self.minimize_kwargs,
        }

    def set_params(self, **params):
        if "rel_tol" in params or "abs_tol" in params:
            self._set_tol(
                rel_tol=params.pop("rel_tol", -1.0),
                abs_tol=params.pop("abs_tol", -1.0),
                tol_p=params.pop("tol_p", -1.0),
            )

        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter {k} for estimator {self}.")
            setattr(self, k, v)
        return self

    def update_steps(
        self,
        steps: List[Dict[str, Any]],
        theta: np.ndarray,
        alpha: np.ndarray,
        diagnosed_arr: np.ndarray,
        inv_cov: np.ndarray,
        link_function: BaseLinkFunction,
        dim_alpha: int,
        status: int,
        optimize_results: Union[optimize.OptimizeResult, dict],
    ):
        if not self.save_optimize_results:
            optimize_results = {}

        step: StepDict = {
            "theta": theta,
            "alpha": alpha,
            "value": sum_of_squares(
                theta=theta,
                alpha=alpha,
                diagnosed_arr=diagnosed_arr,
                inv_sigma=inv_cov,
                link_function=link_function,
                dim_alpha=dim_alpha,
                reg_lambda=self.reg_lambda,
                reg_p=self.reg_p,
            ),
            "status": status,
            "optimize_results": optimize_results,
        }
        steps.append(step)

    def optimize(
        self,
        control_arr: np.ndarray,
        diagnosed_arr: np.ndarray,
        link_function: BaseLinkFunction,
        dim_alpha: int = 1,
        alpha0: Optional[np.ndarray] = None,
        theta0: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> CorrPopsOptimizerResults:
        p = tv.triangular_dim(diagnosed_arr.shape[-1])

        if alpha0 is None:
            alpha = np.full((p, dim_alpha), link_function.null_value)
        else:
            alpha = alpha0

        if theta0 is None:
            theta = theta_of_alpha(
                alpha=alpha,
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                link_function=link_function,
                dim_alpha=dim_alpha,
            )
        else:
            theta = theta0

        if weights is not None:
            weights = matrix.regularize_matrix(
                weights,
                const=self.mat_reg_const,
                method=self.mat_reg_method,
                only_if_singular=self.mat_reg_only_if_singular,
            )
            inv_cov = linalg.inv(weights)
        else:
            inv_cov = None

        minimize_kwargs = copy.deepcopy(self.minimize_kwargs)
        adaptive_maxiter = (
            minimize_kwargs.get("method", "") != "TNC"
            and "maxiter" not in minimize_kwargs["options"]
        )

        self._check_positive_definite(theta, alpha, link_function, dim_alpha)

        steps: List[StepDict] = []
        self.update_steps(
            steps=steps,
            theta=theta,
            alpha=alpha,
            diagnosed_arr=diagnosed_arr,
            inv_cov=inv_cov,
            link_function=link_function,
            dim_alpha=dim_alpha,
            status=-1,
            optimize_results={},
        )
        start_time = self._log("start")

        i = 0
        for i in range(self.max_iter):
            last_start_time = datetime.now()

            if adaptive_maxiter:
                minimize_kwargs["options"]["maxiter"] = np.clip(i * 100, 500, 2_000)

            theta = theta_of_alpha(
                alpha=alpha,
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                link_function=link_function,
                dim_alpha=dim_alpha,
            )
            optimize_results = optimize.minimize(
                partial(
                    sum_of_squares,
                    theta=theta,
                    diagnosed_arr=diagnosed_arr,
                    link_function=link_function,
                    inv_sigma=inv_cov,
                    dim_alpha=dim_alpha,
                    reg_lambda=self.reg_lambda,
                    reg_p=self.reg_p,
                ),
                alpha.flatten(),
                **minimize_kwargs,
            )
            alpha = optimize_results["x"]
            self.update_steps(
                steps=steps,
                theta=theta,
                alpha=alpha,
                diagnosed_arr=diagnosed_arr,
                inv_cov=inv_cov,
                link_function=link_function,
                dim_alpha=dim_alpha,
                status=optimize_results.status,
                optimize_results=optimize_results,
            )
            distance = vector.norm_p(
                x=steps[-2]["alpha"],
                y=steps[-1]["alpha"],
                p=self.tol_p,
                agg="mean",
            )
            self._log(
                "progress",
                start_time=start_time,
                last_start_time=last_start_time,
                steps=steps,
                distance=distance,
            )

            if self.abs_tol:
                threshold = self.abs_tol
            else:
                threshold = self.rel_tol * (
                    vector.norm_p(x=steps[-2]["value"], p=self.tol_p, agg="mean")
                    + self.rel_tol
                )

            if (
                distance < threshold
                and i > self.min_iter
                and all(s["status"] == 0 for s in steps[-self.min_iter :])
            ):
                break

            if (  # pragma: no cover
                self.early_stop
                and i > self.min_iter
                and steps[-1]["value"] > steps[-2]["value"]
                and steps[-1]["status"] == 0
                and steps[-2]["status"] == 0
            ):
                steps.pop(-1)
                theta = steps[-1]["theta"]
                alpha = steps[-1]["alpha"]
                warnings.warn(
                    "early stopping used; last iteration didn't minimize target"
                )
                break

        if i + 1 >= self.max_iter:  # pragma: no cover
            warnings.warn("optimization reached maximum iterations")

        self._log("end", start_time=start_time, steps=steps)
        return CorrPopsOptimizerResults(
            theta=theta,
            alpha=alpha,
            inv_cov=inv_cov,
            link_function=link_function.name,
            p=p,
            dim_alpha=dim_alpha,
            steps=steps,
        )
