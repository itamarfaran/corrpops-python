from typing import Any, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import linalg, stats

from linalg.matrix import is_positive_definite
from statistics.arma import is_invertible_arma


def generate_scales_with_random_effect(
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

    return np.repeat(scale[np.newaxis, :, :], size, axis=0).squeeze()


def generalized_wishart_rvs(
    df: Union[int, float],
    scale: np.ndarray,
    random_effect: float = 0.0,
    size: Union[int, Tuple[int, ...]] = 1,
    random_state: Any = None,
) -> np.ndarray:
    if not random_effect and df > scale.shape[-1] and is_positive_definite(scale):
        return stats.wishart.rvs(
            df=df,
            scale=scale,
            size=size,
            random_state=random_state,
        )

    if isinstance(size, int):
        size = (size,)

    if random_effect:
        x = stats.norm.rvs(
            size=size + (df, scale.shape[-1]),
            random_state=random_state,
        ).squeeze()
        scales = generate_scales_with_random_effect(
            scale=scale,
            random_effect=random_effect,
            size=size,
            random_state=random_state,
        )
        for index in np.ndindex(x.shape[:-2]):
            x[index] = x[index] @ linalg.cholesky(scales[index], overwrite_a=True)
    else:
        x = stats.multivariate_normal.rvs(
            cov=scale,
            size=size + (df,),
            random_state=random_state,
        )
    return np.swapaxes(x, -1, -2) @ x


def _prepare_arma(
    ar: npt.ArrayLike,
    ma: npt.ArrayLike,
) -> Tuple[np.ndarray, np.ndarray, int]:
    ar = np.atleast_1d(ar)
    ma = np.atleast_1d(ma)

    if not is_invertible_arma(ar):
        raise ValueError("ar not stationary")
    if not is_invertible_arma(ma):
        raise ValueError("ma not invertible")

    shape_diff = ma.size - ar.size
    if shape_diff > 0:
        ar = np.append(ar, np.zeros(shape_diff))
    elif shape_diff < 0:
        ma = np.append(ma, np.zeros(-shape_diff))

    if np.all(ar == 0) and np.all(ma == 0):
        max_lag = 0
    else:
        max_lag = ar.size

    return ar, ma, max_lag


def arma_wishart_rvs(
    df: Union[int, float],
    scale: np.ndarray,
    ar: npt.ArrayLike = 0.0,
    ma: npt.ArrayLike = 0.0,
    random_effect: float = 0.0,
    size: Union[int, Tuple[int, ...]] = 1,
    random_state: Any = None,
) -> np.ndarray:
    if isinstance(size, int):
        size = (size,)

    ar, ma, max_lag = _prepare_arma(ar, ma)

    if not max_lag:
        return generalized_wishart_rvs(
            df=df,
            scale=scale,
            random_effect=random_effect,
            size=size,
            random_state=random_state,
        )

    eps = stats.norm.rvs(
        size=size + (df, scale.shape[-1]),
        random_state=random_state,
    ).squeeze()
    out = np.zeros_like(eps)
    for i in range(df):  # type: ignore
        out[..., i, :] = eps[..., i, :]

        for lag in range(max_lag):
            if lag < i:
                out[..., i, :] += ma[lag] * eps[..., i - lag - 1, :]
                out[..., i, :] += ar[lag] * out[..., i - lag - 1, :]

    if random_effect:
        scales = generate_scales_with_random_effect(
            scale=scale,
            random_effect=random_effect,
            size=size,
            random_state=random_state,
        )
        for index in np.ndindex(out.shape[:-2]):
            out[index] = out[index] @ linalg.cholesky(scales[index], overwrite_a=True)
    else:
        out = out @ linalg.cholesky(scale)

    return np.swapaxes(out, -1, -2) @ out
