import gzip
import json

import numpy as np
import pandas as pd

from corrpops.data.loaders import load_data
from corrpops.data.preprocessing import preprocess
from corrpops.model import estimator, link_functions
from corrpops.simulation.sample import build_parameters, create_samples_from_parameters

REAL_DATA: bool = False
JACKKNIFE: bool = True


if __name__ == "__main__":
    if REAL_DATA:
        theta, alpha = None, None
        data = load_data("tga_aal")
        control, diagnosed, dropped_subjects, dropped_columns = preprocess(
            data["control"], data["diagnosed"], threshold=0.05
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

    model = estimator.CorrPopsEstimator(
        link_function=link_functions.MultiplicativeIdentity(
            # transformer=link_functions.Transformer(np.log, np.exp)
        ),
        optimizer_kwargs={
            "mat_reg_const": 0.1,
            "early_stop": not REAL_DATA,
            "verbose": True,
        },
        non_positive="warn",
    )
    model.fit(control, diagnosed)

    if REAL_DATA:
        with gzip.open("tga_aal.json.gz", "w") as f:
            f.write(json.dumps(model.to_json()).encode("utf-8"))

    model.alpha_ = (
        model.alpha_ - np.median(model.alpha_) + model.link_function.null_value
    )

    gee_inference = pd.DataFrame(
        model.inference(
            p_adjust_method="fdr_bh",
            std_const=1.1 if REAL_DATA else 1.0,
            known_alpha=alpha,
        )
    )
    gee_wilks_score = model.score(control, diagnosed)
    print(
        gee_inference[
            (gee_inference["z_value"] < 0.0)
            & (gee_inference["p_value_adjusted"] < 0.05)
        ]
    )
    print(gee_wilks_score)

    if JACKKNIFE:
        jackknife_model = estimator.CorrPopsJackknifeEstimator(model, use_ray=True)
        jackknife_model.fit(control, diagnosed, compute_cov=True)

        jackknife_inference = pd.DataFrame(
            jackknife_model.inference(
                p_adjust_method="fdr_bh",
                known_alpha=alpha,
            )
        )
        jackknife_wilks_score = jackknife_model.score(control, diagnosed)
        print(jackknife_inference)
        print(jackknife_wilks_score)
