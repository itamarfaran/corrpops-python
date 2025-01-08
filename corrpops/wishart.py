import copy
from typing import Any, Union, Tuple, Iterable
import numpy as np
from scipy import stats
from utils import is_positive_definite


# todo: generate_random_effect_sigma


def generalized_wishart_rvs(
        df: int,
        scale: np.ndarray,
        size: Union[int, Tuple[int, ...]] = 1,
        random_state: Any = None,
) -> np.ndarray:
    if isinstance(size, int):
        size = (size,)

    if (
        df > scale.shape[-1]
        and is_positive_definite(scale)
    ):
        return stats.wishart.rvs(df, scale, size, random_state)

    x = stats.multivariate_normal.rvs(None, scale, size + (df,), random_state)
    return np.swapaxes(x, -1, -2) @ x


def is_invertible_arma(coefs, tol=1e-03):
    x = np.linspace(-1, 1, int(2 / tol))[:, None]
    results = 1 - np.sum(np.asarray(coefs) * x ** (1 + np.arange(len(coefs))), axis=1)
    return (results > 0).all() or (results < 0).all()


def arma_wishart_rvs(
        df: int,
        scale: np.ndarray,
        ar: Union[float, Iterable[float]] = None,
        ma: Union[float, Iterable[float]] = None,
        size: Union[int, Tuple[int, ...]] = 1,
        random_state: Any = None,
):
    if ar is None:
        ar = []
    elif not isinstance(ar, (tuple, list)):
        ar = [ar]
    else:
        ar = copy.copy(list(ar))
    if ma is None:
        ma = []
    elif not isinstance(ma, (tuple, list)):
        ma = [ma]
    else:
        ma = copy.copy(list(ma))

    if not is_invertible_arma(ar):
        raise ValueError
    if not is_invertible_arma(ma):
        raise ValueError
    if isinstance(size, int):
        size = (size,)

    while len(ma) < len(ar):
        ma.append(0)
    while len(ar) < len(ma):
        ar.append(0)

    max_lag = len(ar)

    eps = stats.multivariate_normal.rvs(None, scale, size + (df,), random_state)
    out = np.zeros(size + (df, scale.shape[-1]), float)
    for i in range(df):
        out[..., i, :] = eps[..., i, :]

        for lag in range(max_lag):
            if lag < i:
                out[..., i, :] += ma[lag] * eps[..., i - lag - 1, :]
                out[..., i, :] += ar[lag] * out[..., i - lag - 1, :]

    return np.swapaxes(out, -1, -2) @ out
