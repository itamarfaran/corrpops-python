import copy
from typing import Any, Union, Tuple, Iterable

import numpy as np
from scipy import stats

from utils.matrix import is_positive_definite


# todo: generate_random_effect_sigma


def generalized_wishart_rvs(
    df: int,
    scale: np.ndarray,
    size: Union[int, Tuple[int, ...]] = 1,
    random_state: Any = None,
) -> np.ndarray:
    if isinstance(size, int):
        size = (size,)

    if df > scale.shape[-1] and is_positive_definite(scale):
        return stats.wishart.rvs(df, scale, size, random_state)

    x = stats.multivariate_normal.rvs(None, scale, size + (df,), random_state)
    return np.swapaxes(x, -1, -2) @ x


def is_invertible_arma(coefficients, tol=1e-03):
    x = np.linspace(-1, 1, int(2 / tol))[:, None]
    results = 1 - np.sum(
        np.asarray(coefficients) * x ** (1 + np.arange(len(coefficients))), axis=1
    )
    return (results > 0).all() or (results < 0).all()


def arma_wishart_rvs(
    df: int,
    scale: np.ndarray,
    ar: Union[float, Iterable[float]] = 0.0,
    ma: Union[float, Iterable[float]] = 0.0,
    size: Union[int, Tuple[int, ...]] = 1,
    random_state: Any = None,
):
    if not ar:
        ar_ = []
    elif not isinstance(ar, (tuple, list)):
        ar_ = [ar]
    else:
        ar_ = copy.copy(list(ar))
    if not ma:
        ma_ = []
    elif not isinstance(ma, (tuple, list)):
        ma_ = [ma]
    else:
        ma_ = copy.copy(list(ma))

    if not is_invertible_arma(ar_):
        raise ValueError
    if not is_invertible_arma(ma_):
        raise ValueError
    if isinstance(size, int):
        size = (size,)

    while len(ma_) < len(ar_):
        ma_.append(0)
    while len(ar_) < len(ma_):
        ar_.append(0)

    max_lag = len(ar_)

    if not max_lag:
        return generalized_wishart_rvs(df, scale, size, random_state)

    eps = stats.multivariate_normal.rvs(None, scale, size + (df,), random_state)
    out = np.zeros(size + (df, scale.shape[-1]), float)
    for i in range(df):
        out[..., i, :] = eps[..., i, :]

        for lag in range(max_lag):
            if lag < i:
                out[..., i, :] += ma_[lag] * eps[..., i - lag - 1, :]
                out[..., i, :] += ar_[lag] * out[..., i - lag - 1, :]

    return np.swapaxes(out, -1, -2) @ out
