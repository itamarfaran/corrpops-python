from datetime import datetime
from functools import partial
import warnings
import numpy as np
from scipy import linalg, optimize
from link_functions import BaseLinkFunction
from triangle_vector import triangle_to_vector
from utils import is_positive_definite


def norm_p(x, y, p):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


def theta_of_alpha(
        alpha,
        control_arr,
        diagnosed_arr,
        link_function,
        d=1,
):
    reversed_diagnosed_arr = link_function.reverse(diagnosed_arr, alpha, d)
    arr = np.concatenate((control_arr, reversed_diagnosed_arr))
    return arr.mean(0)


def sum_of_squares(
        alpha,
        theta,
        diagnosed_arr,
        link_function,
        inv_sigma=None,
        dim_alpha=1,
        reg_lambda=0.0,
        reg_p=2.0,
):
    g11 = triangle_to_vector(link_function.func(theta, alpha, dim_alpha))

    if inv_sigma is None:
        sse = np.sum(g11 * (0.5 * g11 - diagnosed_arr.mean(0)))
    else:
        g11 = g11.reshape((theta.size, 1))
        sse = g11.T @ inv_sigma @ (0.5 * g11 - diagnosed_arr.mean(0)[:, None])

    sse *= diagnosed_arr.shape[0]
    if reg_lambda > 0.0:
        sse += reg_lambda * norm_p(alpha, link_function.null_value, reg_p)
    return sse


class CorrPopsOptimizer:
    def __init__(
            self,
            link_function: BaseLinkFunction,
            *,
            dim_alpha: int = 1,
            rel_tol: float = 1e-06,
            abs_tol: float = 0.0,
            abs_p: float = 2.0,
            early_stop: bool = False,
            min_iter: int = 3,
            max_iter: int = 50,
            reg_lambda: float = 0.0,
            reg_p: float = 2.0,
            minimize_kwargs: dict = None,
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

        self.theta_ = None
        self.alpha_ = None
        self.inv_cov_ = None
        self.p_ = None
        self.steps_ = None
        self.start_time_ = None
        self.end_time_ = None
        self.adaptive_maxiter = "maxiter" not in self.minimize_kwargs["options"]

    def update_steps(
            self,
            theta,
            alpha,
            diagnosed_arr,
            inv_cov,
            status,
            optimize_results=None,
    ):
        if self.save_optimize_results:
            if optimize_results is None:
                raise NameError
        else:
            optimize_results = {}

        self.steps_.append({
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
        })

    def optimize(
            self,
            control_arr,
            diagnosed_arr,
            alpha0=None,
            theta0=None,
            weights=None,
    ):
        self.p_ = int(
            (1 + np.sqrt(1 + 8 * diagnosed_arr.shape[1]) / 2)
        )

        if alpha0 is None:
            alpha0 = np.full((self.p_, self.dim_alpha), self.link_function.null_value)
        if theta0 is None:
            theta0 = theta_of_alpha(
                alpha=alpha0,
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                link_function=self.link_function,
                d=self.dim_alpha,
            )

        # todo: check if Initial parameters dont result with positive-definite matrices

        if weights is not None:
            self.inv_cov_ = linalg.inv(weights)

        theta = theta0
        alpha = alpha0

        self.steps_ = []
        self.update_steps(
            theta=theta,
            alpha=alpha,
            diagnosed_arr=diagnosed_arr,
            inv_cov=self.inv_cov_,
            status=-1,
            optimize_results={},
        )
        stop = False
        self.start_time_ = datetime.now()

        if self.verbose:
            print(f"Time of initialization: {self.start_time_}; Progress: 'Loop, (Time, Status, Distance)'")

        for i in range(self.max_iter):
            if self.minimize_kwargs.get("method", "") != "TNC" and self.adaptive_maxiter:
                self.minimize_kwargs["options"]["maxiter"] = np.clip(i * 100, 500, 2_000)

            theta = theta_of_alpha(
                alpha=alpha,
                control_arr=control_arr,
                diagnosed_arr=diagnosed_arr,
                link_function=self.link_function,
                d=self.dim_alpha,
            )

            optimize_results = optimize.minimize(
                partial(
                    sum_of_squares,
                    theta=theta,
                    diagnosed_arr=diagnosed_arr,
                    link_function=self.link_function,
                    inv_sigma=self.inv_cov_,
                    dim_alpha=self.dim_alpha,
                    reg_lambda=self.reg_lambda,
                    reg_p=self.reg_p,
                ),
                alpha.flatten(),
                **self.minimize_kwargs,
            )
            alpha = optimize_results["x"]
            self.update_steps(
                theta=theta,
                alpha=alpha,
                diagnosed_arr=diagnosed_arr,
                inv_cov=self.inv_cov_,
                status=optimize_results.status,
                optimize_results=optimize_results,
            )

            if self.stopping_rule == "abs_tol":
                distance = norm_p(self.steps_[-2]["alpha"], self.steps_[-1]["alpha"], self.abs_p)
                stop = distance < self.abs_tol
            else:
                distance = np.abs(self.steps_[-2]["value"] - self.steps_[-1]["value"])
                stop = distance < (self.rel_tol * (np.abs(self.steps_[-1]["value"]) + self.rel_tol))
            stop = stop and i > self.min_iter

            if self.verbose:
                print(
                    f"{i} "
                    f"({(datetime.now() - self.start_time_).seconds}s, "
                    f"{self.steps_[-1]['status']}, "
                    f"{np.round(distance, 5)})"
                )

            if stop:
                for j in range(self.min_iter):
                    if self.steps_[-(1 + j)]["status"] != 0:
                        stop = False
                        break

            if self.early_stop:
                maximized = self.steps_[-1]["value"] >= self.steps_[-2]["value"]
                if maximized and self.steps_[-1]["status"] == 0:
                    self.steps_.pop(-1)
                    theta = self.steps_[-1]["theta"]
                    alpha = self.steps_[-1]["alpha"]
                    warnings.warn("early stopping used; last iteration didn't minimize target")
                    stop = True

            if stop:
                break

        if not stop:
            warnings.warn("optimization reached maximum iterations")

        self.end_time_ = datetime.now()
        if self.verbose:
            total_time = self.end_time_ - self.start_time_
            print(f"total time: {total_time.seconds // 60} minutes and {total_time.seconds} seconds")

        self.theta_ = theta
        self.alpha_ = alpha
