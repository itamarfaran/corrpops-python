from typing import Any, Union, Tuple
import numpy as np
from scipy import stats
from utils import is_positive_definite


def generalized_wishart_rvs(
        df: int,
        scale: np.ndarray,
        size: Union[int, Tuple[int]] = 1,
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
    results = 1 - np.sum(coefs * x ** (1 + np.arange(len(coefs))), axis=1)
    return (results > 0).all() or (results < 0).all()


def arma_wishart_rvs(
        df: int,
        scale: np.ndarray,
        ar: Union[float, Tuple[float]] = None,
        ma: Union[float, Tuple[float]] = None,
        size: Union[int, Tuple[int]] = 1,
        random_state: Any = None,
):
    pass
