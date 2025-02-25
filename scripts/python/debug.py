import gzip
import json

import numpy as np
import pandas as pd

from corrpops.data.loaders import load_data
from corrpops.data.preproccessing import preprocess
from corrpops.model import estimator, link_functions
from corrpops.simulation.sample import build_parameters, create_samples_from_parameters

REAL_DATA: bool = False
JACKKNIFE: bool = True


if REAL_DATA:
    theta, alpha = None, None
    data = load_data("tga_aal")
    control, diagnosed, dropped_subjects, dropped_columns = preprocess(
        data["control"], data["diagnosed"]
    )
    minimize_kwargs = {"options": {"gtol": 1e-4}}
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
        link_function=link_functions.MultiplicativeIdentity(),
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
    minimize_kwargs = {"options": {}}


model = estimator.CorrPopsEstimator(
    link_function=link_functions.MultiplicativeIdentity(
        transformer=link_functions.Transformer(np.log, np.exp)
    ),
    optimizer=estimator.CorrPopsOptimizer(
        mat_reg_const=0.1,
        early_stop=True,
        minimize_kwargs=minimize_kwargs,
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


new_model = estimator.CorrPopsEstimator.from_json(
    new_model_json,
    model.link_function,
    non_positive="warn",
).compute_covariance(
    control_arr=control,
    diagnosed_arr=diagnosed,
)

new_model.alpha_ = (
    new_model.alpha_ - np.median(new_model.alpha_) + new_model.link_function.null_value
)

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
    _ = estimator.CorrPopsJackknifeEstimator(
        new_model,
        use_ray=True,
    ).fit(control, diagnosed, compute_cov=True)

    jackknife_model = estimator.CorrPopsJackknifeEstimator(
        estimator.CorrPopsEstimator(optimizer_kwargs={"verbose": False}),
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
