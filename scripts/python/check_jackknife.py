import os
import gzip
import json
import numpy as np
import ray

from corrpops.statistics.arma import is_invertible_arma
from corrpops.simulation.sample import build_parameters, create_samples_from_parameters
from corrpops.model.link_functions import MultiplicativeIdentity
from corrpops.model.estimator import CorrPopsEstimator, CorrPopsJackknifeEstimator


RNG = np.random.default_rng(237)
RESULTS_DIR = "jackknife_bootstraps"
if not os.path.isdir(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


@ray.remote
def _jackknife(i_, control_, diagnosed_):
    estimator = CorrPopsJackknifeEstimator(
        CorrPopsEstimator(optimizer_kwargs={"verbose": False}),
    ).fit(control_, diagnosed_, compute_cov=True)

    with gzip.open(f"{RESULTS_DIR}/{i_}.json.gz", "w") as f:
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
        size=20,
        random_state=RNG,
    )

    try:
        ray.init()
        running_results = [
            _jackknife.remote(i, control[i], diagnosed[i])
            for i in range(control.shape[0])
        ]
        while running_results:
            done_results, running_results = ray.wait(running_results, num_returns=1)
            print("+", end="")
        print("")

    except Exception as ex:
        raise ex

    finally:
        ray.shutdown()
