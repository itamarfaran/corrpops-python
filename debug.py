import pandas as pd

from model.estimator import CorrPopsEstimator
from model.optimizer import CorrPopsOptimizer
from model.link_functions import MultiplicativeIdentity
from simulation.sample import build_parameters, create_samples_from_parameters

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
    diagnosed_ar=[0.5, 0.2],
    diagnosed_ma=0.2,
    size=1,
    random_effect=None,
    seed=12,
)


model = CorrPopsEstimator(
    # link_function=MultiplicativeIdentity(transformer=Transformer(np.log, np.exp))
    optimizer=CorrPopsOptimizer(verbose=True),
)
model.fit(control[0], diagnosed[0])
print(
    pd.DataFrame(
        model.inference(
            p_adjust_method="fdr_bh",
            known_alpha=alpha,
        )
    )
)
