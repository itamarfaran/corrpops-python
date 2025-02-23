from typing import Literal

import numpy as np
from scipy import linalg


def sqrt_diag(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.diagonal(a, axis1=-2, axis2=-1))


def matrix_power(a: np.ndarray, power: float) -> np.ndarray:
    w, v = linalg.eigh(a)
    w = np.eye(len(w)) * w**power
    return v @ w @ v.T


def fill_other_triangle(a: np.ndarray) -> np.ndarray:
    row, col = np.diag_indices(a.shape[-1])
    diag = np.zeros_like(a)
    diag[..., row, col] = a[..., row, col]
    return a + np.swapaxes(a, -1, -2) - diag


def force_symmetry(a: np.ndarray) -> np.ndarray:
    return (a + np.swapaxes(a, -1, -2)) / 2


def is_positive_definite(a: np.ndarray) -> np.ndarray:
    def _check(a_):
        if np.allclose(a_, a_.T):
            try:
                linalg.cholesky(a_)
                return True
            except linalg.LinAlgError:
                return False
        else:
            return False

    out = np.full(a.shape[:-2], False)
    for index in np.ndindex(a.shape[:-2]):
        out[index] = _check(a[index])
    return out


def regularize_matrix(
    a: np.ndarray,
    const: float = 1.0,
    method: Literal["constant", "avg_diag", "increase_diag"] = "constant",
    only_if_singular: bool = True,
) -> np.ndarray:
    p = a.shape[-1]
    if a.shape[-2] != p:
        raise IndexError
    if only_if_singular:
        if linalg.matrix_rank(a) == p:
            return a

    if method == "constant":
        if const < 0:
            raise ValueError("in method 'constant' const must be greater or equal to 0")
        return a + np.eye(p) * const
    elif method == "avg_diag":
        if not 0 <= const <= 1:
            raise ValueError("in method 'avg_diag' const must be in [0, 1]")
        return (1 - const) * a + const * np.eye(p) * np.diag(a).mean()
    elif method == "increase_diag":
        if not 0 <= const <= 1:
            raise ValueError("in method 'increase_diag' const must be in [0, 1]")
        return (1 - const) * a + const * np.diag(np.diag(a))
    else:
        raise ValueError


def cov_to_corr(a: np.ndarray) -> np.ndarray:
    row, col = np.diag_indices(a.shape[-1])
    diagonals = np.sqrt(a[..., row, col])
    return a / (diagonals[..., None, :] * diagonals[..., :, None])
