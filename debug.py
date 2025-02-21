import gzip
import json

import numpy as np
import pandas as pd

from data.loaders import load_data
from data.preproccessing import preprocess
from model import link_functions
from model.estimator import (
    CorrPopsEstimator,
    CorrPopsJackknifeEstimator,
    CorrPopsOptimizer,
)
from simulation.sample import build_parameters, create_samples_from_parameters

REAL_DATA: bool = False
JACKKNIFE: bool = True


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
        link_function=link_functions.AdditiveQuotient(),
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
    link_function=link_functions.AdditiveQuotient(
        transformer=link_functions.Transformer(np.log, np.exp)
    ),
    optimizer=CorrPopsOptimizer(
        mat_reg_const=0.1,
        early_stop=True,
        verbose=True,
    ),
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

gee_inference = pd.DataFrame(
    new_model.inference(
        p_adjust_method="fdr_bh",
        known_alpha=alpha,
    )
)
gee_wilks_score = new_model.score(control, diagnosed)
print(gee_inference)
print(gee_wilks_score)

if JACKKNIFE:
    _ = CorrPopsJackknifeEstimator(
        new_model,
        use_ray=True,
    ).fit(control, diagnosed, compute_cov=True)

    jackknife_model = CorrPopsJackknifeEstimator(
        CorrPopsEstimator(optimizer_kwargs={"verbose": False}),
        use_ray=False,
    ).fit(control, diagnosed)

    jackknife_inference = pd.DataFrame(
        jackknife_model.inference(
            p_adjust_method="fdr_bh",
            known_alpha=alpha,
        )
    )
    jackknife_wilks_score = jackknife_model.score(control, diagnosed)
    print(jackknife_inference)
    print(jackknife_wilks_score)
