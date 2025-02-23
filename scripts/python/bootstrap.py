import gzip
import json
import os
from collections import namedtuple
from typing import List

import numpy as np
import ray
from scipy import stats

from corrpops.linalg.triangle_vector import triangle_to_vector
from corrpops.model.estimator import CorrPopsEstimator
from corrpops.model.link_functions import MultiplicativeIdentity
from corrpops.simulation.sample import build_parameters, create_samples_from_parameters
from corrpops.statistics.arma import is_invertible_arma

B = 5000
MULTIPLE_RUNS: bool = True
RESULTS_DIR = "bootstraps"
RUNS_PER_B = 20

B_uniq = int(B / RUNS_PER_B)
if not os.path.isdir(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def round_to_tens(a: np.ndarray, decimals: int = 0, tolist: bool = True):
    out = 10 * np.round(a / 10, decimals)
    if not decimals:
        out = out.astype(int)
    if tolist:
        out = out.tolist()
    return out


to_permute = {
    "p": round_to_tens(stats.beta.rvs(a=1.0, b=1.8, loc=20, scale=40, size=B_uniq)),
    "n": round_to_tens(stats.uniform.rvs(loc=80, scale=40, size=B_uniq)),
    "diagnosed_ratio": round_to_tens(
        stats.beta.rvs(a=1.5, b=1.5, loc=0.2, scale=0.6, size=B_uniq), 2
    ),
    "t_length": round_to_tens(
        stats.beta.rvs(a=1.5, b=1.5, loc=80, scale=40, size=B_uniq)
    ),
    "ar": np.random.permutation(
        np.round(
            np.concatenate(
                (
                    stats.uniform.rvs(loc=-0.7, scale=0.4, size=int(B_uniq / 4)),
                    stats.uniform.rvs(loc=0.3, scale=0.4, size=int(B_uniq / 4)),
                    np.zeros(B_uniq - 2 * int(B_uniq / 4)),
                )
            ),
            decimals=1,
        )
    ).tolist(),
}

if MULTIPLE_RUNS:
    for v in to_permute.values():
        v *= RUNS_PER_B


EstimateArguments = namedtuple(
    "EstimateArguments",
    [
        "p",
        "n",
        "diagnosed_ratio",
        "t_length",
        "ar",
        "alpha_min",
        "alpha_max",
        "percent_alpha",
    ],
)
permutations: List[EstimateArguments] = [
    EstimateArguments(
        p=to_permute["p"][i],
        n=to_permute["n"][i],
        diagnosed_ratio=to_permute["diagnosed_ratio"][i],
        t_length=to_permute["t_length"][i],
        ar=to_permute["ar"][i],
        alpha_min=0.8,
        alpha_max=0.9,
        percent_alpha=0.3,
    )
    for i in range(len(to_permute["p"]))
]


@ray.remote
def _estimate(i_: int, args: EstimateArguments):
    rng = np.random.default_rng(i_)
    diagnosed_n = int(args.n * args.diagnosed_ratio)
    control_n = args.n - diagnosed_n

    theta, alpha, _ = build_parameters(
        p=args.p,
        percent_alpha=args.percent_alpha,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        dim_alpha=1,
        enforce_min_alpha=True,
        random_state=rng,
    )
    control, diagnosed = create_samples_from_parameters(
        control_n=control_n,
        diagnosed_n=diagnosed_n,
        theta=theta,
        alpha=alpha,
        link_function=MultiplicativeIdentity(),
        t_length=args.t_length,
        control_ar=args.ar,
        diagnosed_ar=args.ar,
        random_state=rng,
    )

    estimator = CorrPopsEstimator(
        optimizer_kwargs={"verbose": False},
    ).fit(control[0], diagnosed[0])

    json_ = estimator.to_json()
    json_["simulation"] = {
        "theta": triangle_to_vector(theta).tolist(),
        "alpha": alpha.flatten().tolist(),
        **args._asdict(),
    }

    with gzip.open(f"{RESULTS_DIR}/{i_}.json.gz", "w") as f:
        f.write(json.dumps(json_).encode("utf-8"))


if __name__ == "__main__":
    try:
        ray.init()
        running_results = []

        for i, permutation in enumerate(permutations[:1]):
            if not is_invertible_arma(permutation.ar):
                raise ValueError
            run_id = _estimate.remote(i, permutation)
            running_results.append(run_id)
        while running_results:
            done_results, running_results = ray.wait(running_results, num_returns=1)
            print("+", end="")
        print("")

    except Exception as ex:
        raise ex

    finally:
        ray.shutdown()
