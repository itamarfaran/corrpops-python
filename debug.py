import numpy as np
from simulation.wishart import generalized_wishart_rvs
from utils.matrix import cov_to_corr
from model.estimator import CorrPopsEstimator
from model.optimizer import CorrPopsOptimizer


p_ = 0.5
m_ = 4
shape = p_ * np.ones((m_, m_)) + (1 - p_) * np.eye(m_)


diagnosed = generalized_wishart_rvs(
    100,
    shape,
    size=200,
)
control = generalized_wishart_rvs(
    100,
    shape,
    size=200,
)

diagnosed = cov_to_corr(diagnosed)
control = cov_to_corr(control)


model = CorrPopsEstimator(
    # link_function=MultiplicativeIdentity(transformer=Transformer(np.log, np.exp))
    optimizer=CorrPopsOptimizer(verbose=True),
)
model.fit(control, diagnosed)
print(
    model.naive_optimizer.alpha_.round(2),
    model.alpha_.round(2),
)
pass
