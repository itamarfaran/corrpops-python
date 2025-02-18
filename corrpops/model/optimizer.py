import warnings
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional, TypedDict, Union

import numpy as np
from scipy import linalg, optimize

from linalg.matrix import is_positive_definite
from linalg.triangle_vector import vector_to_triangle
from linalg.vector import norm_p
from .likelihood import theta_of_alpha, sum_of_squares
from .link_functions import BaseLinkFunction


class CorrPopsOptimizerResults(TypedDict):
    theta: np.ndarray
    alpha: np.ndarray
    inv_cov: Union[np.ndarray, None]
    p: int
    steps: List[Dict[str, Any]]
    dim_alpha: int
    link_function: str


class CorrPopsOptimizer:
    def __init__(
        self,
        *,
        link_function: BaseLinkFunction = None,
        dim_alpha: int = 1,
        rel_tol: float = 1e-06,
        abs_tol: float = 0.0,
        abs_p: float = 2.0,
        early_stop: bool = False,
        min_iter: int = 3,
        max_iter: int = 50,
        reg_lambda: float = 0.0,
        reg_p: float = 2.0,
        minimize_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        save_optimize_results: bool = False,
    ):
        self.link_function = link_function
        self.dim_alpha = dim_alpha
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.abs_p = abs_p
        self.early_stop = early_stop
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
        self.reg_p = reg_p
        self.minimize_kwargs = minimize_kwargs.copy() if minimize_kwargs else {}
        self.verbose = verbose
        self.save_optimize_results = save_optimize_results

        self.stopping_rule = "abs_tol" if self.abs_tol else "rel_tol"
        if "options" not in self.minimize_kwargs:
            self.minimize_kwargs["options"] = {}
        self.adaptive_maxiter = "maxiter" not in self.minimize_kwargs["options"]

    @classmethod
    def to_json(
        cls,
        results: CorrPopsOptimizerResults,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = results.copy()

        out.pop("steps")
        out["theta"] = out["theta"].tolist()
        out["alpha"] = out["alpha"].tolist()
        out["inv_cov"] = [] if out["inv_cov"] is None else out["inv_cov"].tolist()
        if params is not None:
            out["params"] = params
        return out

    def _check_positive_definite(self, theta: np.ndarray, alpha: np.ndarray):
        is_positive_definite_ = (
            is_positive_definite(vector_to_triangle(theta, diag_value=1)),
            is_positive_definite(self.link_function.func(theta, alpha, self.dim_alpha)),
        )
        if not all(is_positive_definite_):
            warnings.warn(
                "initial parameters dont yield with non positive-definite matrices"
            )

    def get_params(self):
        return {
            "link_function": self.link_function.name,
            "dim_alpha": self.dim_alpha,
            "rel_tol": self.rel_tol,
            "abs_tol": self.abs_tol,
            "abs_p": self.abs_p,
            "early_stop": self.early_stop,
            "min_iter": self.min_iter,
            "max_iter": self.max_iter,
            "reg_lambda": self.reg_lambda,
            "reg_p": self.reg_p,
            "minimize_kwargs": self.minimize_kwargs,
        }

    def update_steps(
        self,
        steps: List[Dict[str, Any]],
        theta: np.ndarray,
        alpha: np.ndarray,
        diagnosed_arr: np.ndarray,
        inv_cov: np.ndarray,
        status: int,
        optimize_results: Optional[Dict[str, Any]] = None,
    ):
        if self.save_optimize_results:
            if optimize_results is None:
                raise NameError
        else:
            optimize_results = {}

        step = {
            "theta": theta,
            "alpha": alpha,
            "value": sum_of_squares(
                theta=theta,
                alpha=alpha,
                diagnosed_arr=diagnosed_arr,
                inv_sigma=inv_cov,
                link_function=self.link_function,
                dim_alpha=self.dim_alpha,
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
        alpha0: Optional[np.ndarray] = None,
        theta0: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> CorrPopsOptimizerResults:
        p = (1 + np.sqrt(1 + 8 * diagnosed_arr.shape[1])) / 2

        if p == round(p):
            p = int(p)
        else:
            raise ValueError(
                f"array shape ({diagnosed_arr.shape[-1]}) does "
                f"not fit size of triangular matrix"
            )

        if alpha0 is None:
            alpha = np.full((p, self.dim_alpha), self.link_function.null_value)
        else:
            alpha = alpha0

        if theta0 is None:
            theta = theta_of_alpha(
                alpha=alpha,
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                link_function=self.link_function,
                dim_alpha=self.dim_alpha,
            )
        else:
            theta = theta0

        self._check_positive_definite(theta, alpha)
        if weights is not None:
            inv_cov = linalg.inv(weights)
        else:
            inv_cov = None

        steps = []
        self.update_steps(
            steps=steps,
            theta=theta,
            alpha=alpha,
            diagnosed_arr=diagnosed_arr,
            inv_cov=inv_cov,
            status=-1,
            optimize_results={},
        )
        stop = False
        start_time = datetime.now()

        if self.verbose:
            print(
                f"Time of initialization: {start_time}; "
                f"Progress: 'Loop, (Time, Status, Distance)'"
            )

        for i in range(self.max_iter):
            if (
                self.minimize_kwargs.get("method", "") != "TNC"
                and self.adaptive_maxiter
            ):
                self.minimize_kwargs["options"]["maxiter"] = np.clip(
                    i * 100, 500, 2_000
                )

            theta = theta_of_alpha(
                alpha=alpha,
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                link_function=self.link_function,
                dim_alpha=self.dim_alpha,
            )

            optimize_results = optimize.minimize(
                partial(
                    sum_of_squares,
                    theta=theta,
                    diagnosed_arr=diagnosed_arr,
                    link_function=self.link_function,
                    inv_sigma=inv_cov,
                    dim_alpha=self.dim_alpha,
                    reg_lambda=self.reg_lambda,
                    reg_p=self.reg_p,
                ),
                alpha.flatten(),
                **self.minimize_kwargs,
            )
            alpha = optimize_results["x"]
            self.update_steps(
                steps=steps,
                theta=theta,
                alpha=alpha,
                diagnosed_arr=diagnosed_arr,
                inv_cov=inv_cov,
                status=optimize_results.status,
                optimize_results=optimize_results,
            )

            if self.stopping_rule == "abs_tol":
                distance = norm_p(steps[-2]["alpha"], steps[-1]["alpha"], self.abs_p)
                stop = distance < self.abs_tol
            else:
                distance = np.abs(steps[-2]["value"] - steps[-1]["value"])
                stop = distance < (
                    self.rel_tol * (np.abs(steps[-1]["value"]) + self.rel_tol)
                )
            stop = stop and i > self.min_iter

            if self.verbose:
                print(
                    f"{i} "
                    f"({(datetime.now() - start_time).seconds}s, "
                    f"{steps[-1]['status']}, "
                    f"{np.round(distance, 5)})"
                )

            if stop:
                for j in range(self.min_iter):
                    if steps[-(1 + j)]["status"] != 0:
                        stop = False
                        break

            if self.early_stop:
                maximized = steps[-1]["value"] >= steps[-2]["value"]
                if maximized and steps[-1]["status"] == 0:
                    steps.pop(-1)
                    theta = steps[-1]["theta"]
                    alpha = steps[-1]["alpha"]
                    warnings.warn(
                        "early stopping used; last iteration didn't minimize target"
                    )
                    stop = True

            if stop:
                break

        if not stop:
            warnings.warn("optimization reached maximum iterations")

        if self.verbose:
            total_time = datetime.now() - start_time
            print(
                f"total time: {total_time.seconds // 60} minutes and {total_time.seconds} seconds"
            )

        results: CorrPopsOptimizerResults = {
            "theta": theta,
            "alpha": alpha,
            "inv_cov": inv_cov,
            "p": p,
            "steps": steps,
            "dim_alpha": self.dim_alpha,
            "link_function": self.link_function.name,
        }
        return results
