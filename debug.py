import gzip
import json

import pandas as pd

from data.loaders import load_data
from data.preproccessing import preprocess
from model.estimator import CorrPopsEstimator
from model.jackknife import CorrPopsJackknifeEstimator
from model.optimizer import CorrPopsOptimizer
from model.link_functions import MultiplicativeIdentity
from simulation.sample import build_parameters, create_samples_from_parameters


REAL_DATA: bool = True
JACKKNIFE: bool = False


if REAL_DATA:
    theta, alpha = None, None
    data = load_data("tga_aal")
    control, diagnosed, dropped_subjects, dropped_columns = preprocess(
        data["control"], data["diagnosed"]
    )
else:
    theta, alpha, _ = build_parameters(
        p=10,
        percent_alpha=0.3,
        dim_alpha=1,
        alpha_min=0.7,
        alpha_max=0.9,
    )
    control, diagnosed = create_samples_from_parameters(
        control_n=19,
        diagnosed_n=12,
        theta=theta,
        alpha=alpha,
        link_function=MultiplicativeIdentity(),
        t_length=100,
        control_ar=[0.5, 0.2],
        control_ma=0.2,
        diagnosed_ar=0.0,
        diagnosed_ma=0.0,
        random_effect=0.05,
        size=1,
        random_state=12,
    )
    control, diagnosed = control[0], diagnosed[0]


model = CorrPopsEstimator(
    # link_function=MultiplicativeIdentity(transformer=Transformer(np.log, np.exp))
    optimizer=CorrPopsOptimizer(verbose=True),
    non_positive="warn",
)
model.fit(control, diagnosed, compute_cov=False)

if REAL_DATA:
    with gzip.open("tga_aal.json.gz", "w") as f:
        f.write(json.dumps(model.to_json()).encode("utf-8"))
    with gzip.open("tga_aal.json.gz", "r") as f:
        new_model_json = json.loads(f.read().decode("utf-8"))
else:
    new_model_json = model.to_json()


new_model = CorrPopsEstimator.from_json(
    model.link_function,
    new_model_json,
    non_positive="warn",
).compute_covariance(control, diagnosed)

print(
    pd.DataFrame(
        new_model.inference(
            p_adjust_method="fdr_bh",
            known_alpha=alpha,
        )
    )
)
print(new_model.score(control, diagnosed))

if JACKKNIFE:
    _ = CorrPopsJackknifeEstimator(
        CorrPopsEstimator(optimizer_kwargs={"verbose": False}), use_ray=True
    ).fit(control, diagnosed, compute_cov=True)

    jackknife = CorrPopsJackknifeEstimator(
        CorrPopsEstimator(optimizer_kwargs={"verbose": False}), use_ray=False
    ).fit(control, diagnosed)

    print(
        pd.DataFrame(
            jackknife.inference(
                p_adjust_method="fdr_bh",
                known_alpha=alpha,
            )
        )
    )
    print(jackknife.score(control, diagnosed))
