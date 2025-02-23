import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from scipy import linalg, optimize

from linalg.matrix import is_positive_definite, regularize_matrix
from linalg.triangle_vector import triangle_to_vector, vector_to_triangle
from linalg.vector import norm_p
from model.likelihood import theta_of_alpha, sum_of_squares
from model.link_functions import BaseLinkFunction


@dataclass
class CorrPopsOptimizerResults:
    theta: np.ndarray
    alpha: np.ndarray
    inv_cov: Union[np.ndarray, None]
    link_function: str
    p: int
    dim_alpha: int
    steps: List[Dict[str, Any]]

    def to_dict(self):
        return {
            "theta": self.theta,
            "alpha": self.alpha,
            "inv_cov": self.inv_cov,
            "link_function": self.link_function,
            "p": self.p,
            "dim_alpha": self.dim_alpha,
            "steps": self.steps,
        }

    def to_json(self):
        out = self.to_dict()
        out["theta"] = out["theta"].tolist()
        out["alpha"] = out["alpha"].tolist()
        out["inv_cov"] = (
            triangle_to_vector(out["inv_cov"], diag=True).tolist()
            if out["inv_cov"] is not None
            else []
        )
        out.pop("steps")
        return out

    @classmethod
    def from_dict(cls, dict_):
        return cls(**dict_)

    @classmethod
    def from_json(cls, json_):
        return cls(
            theta=np.array(json_["theta"]),
            alpha=np.array(json_["alpha"]),
            inv_cov=(
                vector_to_triangle(np.array(json_["inv_cov"]), diag=True)
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
        rel_tol: float = 1e-06,
        abs_tol: float = 0.0,
        abs_p: float = 2.0,
        early_stop: bool = False,
        min_iter: int = 3,
        max_iter: int = 50,
        reg_lambda: float = 0.0,
        reg_p: float = 2.0,
        mat_reg_const: float = 0.0,
        mat_reg_method: Literal["constant", "avg_diag", "increase_diag"] = "constant",
        mat_reg_only_if_singular: bool = False,
        minimize_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        save_optimize_results: bool = False,
    ):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.abs_p = abs_p
        self.early_stop = early_stop
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
        self.reg_p = reg_p
        self.mat_reg_const = mat_reg_const
        self.mat_reg_method = mat_reg_method
        self.mat_reg_only_if_singular = mat_reg_only_if_singular
        self.minimize_kwargs = minimize_kwargs.copy() if minimize_kwargs else {}
        self.verbose = verbose
        self.save_optimize_results = save_optimize_results

        self.stopping_rule = "abs_tol" if self.abs_tol else "rel_tol"
        if "options" not in self.minimize_kwargs:
            self.minimize_kwargs["options"] = {}
        self.adaptive_maxiter = "maxiter" not in self.minimize_kwargs["options"]

    @staticmethod
    def _check_positive_definite(
        theta: np.ndarray,
        alpha: np.ndarray,
        link_function: BaseLinkFunction,
        dim_alpha: int,
    ):
        is_positive_definite_ = (
            is_positive_definite(vector_to_triangle(theta, diag_value=1)),
            is_positive_definite(link_function(t=theta, a=alpha, d=dim_alpha)),
        )
        if not all(is_positive_definite_):
            warnings.warn("initial parameters dont yield positive-definite matrices")

    def get_params(self) -> Dict[str, Any]:
        return {
            "rel_tol": self.rel_tol,
            "abs_tol": self.abs_tol,
            "abs_p": self.abs_p,
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
        p = (1 + np.sqrt(1 + 8 * diagnosed_arr.shape[1])) / 2

        if p == round(p):
            p = int(p)
        else:
            raise ValueError(
                f"array shape ({diagnosed_arr.shape[-1]}) does "
                f"not fit size of triangular matrix"
            )

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

        self._check_positive_definite(theta, alpha, link_function, dim_alpha)
        if weights is not None:
            weights = regularize_matrix(
                weights,
                const=self.mat_reg_const,
                method=self.mat_reg_method,
                only_if_singular=self.mat_reg_only_if_singular,
            )
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
            link_function=link_function,
            dim_alpha=dim_alpha,
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
                self.minimize_kwargs["options"]["maxiter"] = int(
                    np.clip(i * 100, 500, 2_000)
                )

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
                **self.minimize_kwargs,
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

        return CorrPopsOptimizerResults(
            theta=theta,
            alpha=alpha,
            inv_cov=inv_cov,
            link_function=link_function.name,
            p=p,
            dim_alpha=dim_alpha,
            steps=steps,
        )
