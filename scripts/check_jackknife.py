import gzip
import json
import os
import warnings

import numpy as np
import ray

from corrpops import corrpops_logger
from corrpops.linalg.triangle_and_vector import triangle_to_vector
from corrpops.model.estimator import CorrPopsEstimator, CorrPopsJackknifeEstimator
from corrpops.model.link_functions import MultiplicativeIdentity
from corrpops.simulation.sample import build_parameters, create_samples_from_parameters
from corrpops.statistics.arma import is_invertible_arma

RNG = np.random.default_rng(237)
RESULTS_DIR = "jackknife_bootstraps"
if not os.path.isdir(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


@ray.remote
def _jackknife(dst_: str, control_: np.ndarray, diagnosed_: np.ndarray):
    corrpops_logger().setLevel(30)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="optimization reached maximum iterations"
        )

        estimator = CorrPopsJackknifeEstimator(
            CorrPopsEstimator(optimizer_kwargs={"verbose": False}),
        ).fit(control_, diagnosed_, compute_cov=True)

    with gzip.open(dst_, "w") as f:
        f.write(json.dumps(estimator.to_json()).encode("utf-8"))


if __name__ == "__main__":
    theta, alpha, _ = build_parameters(
        p=8,
        percent_alpha=0.4,
        alpha_min=0.6,
        alpha_max=0.8,
        dim_alpha=1,
        enforce_min_alpha=True,
        random_state=RNG,
    )

    if not is_invertible_arma([0.4, 0.2]) or not is_invertible_arma([0.5, 0.1]):
        raise ValueError

    control, diagnosed = create_samples_from_parameters(
        control_n=19,
        diagnosed_n=12,
        theta=theta,
        alpha=alpha,
        link_function=MultiplicativeIdentity(),
        t_length=115,
        control_ar=(0.4, 0.2),
        control_ma=(0.4, 0.2),
        diagnosed_ar=(0.5, 0.1),
        diagnosed_ma=(0.5, 0.1),
        size=20,  # 200
        random_state=RNG,
    )

    np.savez_compressed(
        f"{RESULTS_DIR}/parameters.npz",
        theta=triangle_to_vector(theta),
        alpha=alpha.flatten(),
        control=triangle_to_vector(control),
        diagnosed=triangle_to_vector(diagnosed),
    )

    try:
        ray.init()
        running_results = []
        for i in range(control.shape[0]):
            dst = f"{RESULTS_DIR}/{i}.json.gz"
            if os.path.isfile(dst):
                continue
            run_id = _jackknife.remote(dst, control[i], diagnosed[i])
            running_results.append(run_id)
        while running_results:
            done_results, running_results = ray.wait(running_results, num_returns=1)
            print("+", end="")
        print("")

    except Exception as ex:
        raise ex

    finally:
        ray.shutdown()
