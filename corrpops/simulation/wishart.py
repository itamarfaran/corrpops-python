import copy
from typing import Any, Iterable, List, Tuple, Union

import numpy as np
from scipy import linalg, stats

from linalg.matrix import is_positive_definite
from statistics.arma import is_invertible_arma


def generate_covariance_with_random_effect(
    scale: np.ndarray,
    random_effect: float = 0.0,
    size: Union[int, Tuple[int, ...]] = 1,
    random_state: Any = None,
) -> np.ndarray:
    if random_effect < 0:
        raise ValueError(
            f"expected random_effect to be non-negative, got {random_effect}"
        )

    if random_effect:
        p = scale.shape[-1] * (1 + 1 / random_effect)
        out = stats.wishart.rvs(df=p, scale=scale, size=size, random_state=random_state)
        return out / p

    out = np.repeat(scale[np.newaxis, :, :], size, axis=0)
    return np.squeeze(out)


def multivariate_normal_rvs(
    df: Union[int, float],
    scale: np.ndarray,
    random_effect: float = 0.0,
    size: Union[int, Tuple[int, ...]] = 1,
    random_state: Any = None,
) -> np.ndarray:
    if isinstance(size, int):
        size = (size,)

    if not random_effect:
        return stats.multivariate_normal.rvs(
            cov=scale,
            size=size + (df,),
            random_state=random_state,
        )

    x = stats.multivariate_normal.rvs(
        cov=np.eye(scale.shape[-1]),
        size=size + (df,),
        random_state=random_state,
    )
    scales = generate_covariance_with_random_effect(
        scale=scale,
        random_effect=random_effect,
        size=size,
        random_state=random_state,
    )
    for index in np.ndindex(scales.shape[:-2]):
        scales[index] = linalg.cholesky(scales[index], overwrite_a=True)
    return x @ scales


def generalized_wishart_rvs(
    df: Union[int, float],
    scale: np.ndarray,
    random_effect: float = 0.0,
    size: Union[int, Tuple[int, ...]] = 1,
    random_state: Any = None,
) -> np.ndarray:
    if df > scale.shape[-1] and is_positive_definite(scale) and not random_effect:
        return stats.wishart.rvs(
            df=df,
            scale=scale,
            size=size,
            random_state=random_state,
        )

    x = multivariate_normal_rvs(
        df=df,
        scale=scale,
        random_effect=random_effect,
        size=size,
        random_state=random_state,
    )
    return np.swapaxes(x, -1, -2) @ x


def _prepare_ar_ma(ar_ma: Union[float, Iterable[float]]) -> List[float]:
    if not ar_ma:
        out = []
    elif not isinstance(ar_ma, (tuple, list)):
        out = [ar_ma]
    else:
        out = copy.copy(list(ar_ma))
    if not is_invertible_arma(out):
        raise ValueError
    return out


def _append_until_equal_length(ar: List[float], ma: List[float]) -> int:
    while len(ar) < len(ma):
        ar.append(0)
    while len(ma) < len(ar):
        ma.append(0)
    return len(ar)


def arma_wishart_rvs(
    df: Union[int, float],
    scale: np.ndarray,
    ar: Union[float, Iterable[float]] = 0.0,
    ma: Union[float, Iterable[float]] = 0.0,
    random_effect: float = 0.0,
    size: Union[int, Tuple[int, ...]] = 1,
    random_state: Any = None,
) -> np.ndarray:
    if isinstance(size, int):
        size = (size,)

    ar_ = _prepare_ar_ma(ar)
    ma_ = _prepare_ar_ma(ma)
    max_lag = _append_until_equal_length(ar_, ma_)

    if not max_lag:
        return generalized_wishart_rvs(
            df=df,
            scale=scale,
            random_effect=random_effect,
            size=size,
            random_state=random_state,
        )

    eps = multivariate_normal_rvs(
        df=df,
        scale=scale,
        random_effect=random_effect,
        size=size,
        random_state=random_state,
    )
    out = np.zeros(size + (df, scale.shape[-1]), float)
    for i in range(df):
        out[..., i, :] = eps[..., i, :]

        for lag in range(max_lag):
            if lag < i:
                out[..., i, :] += ma_[lag] * eps[..., i - lag - 1, :]
                out[..., i, :] += ar_[lag] * out[..., i - lag - 1, :]

    return np.swapaxes(out, -1, -2) @ out
