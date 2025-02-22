from typing import Literal

import numpy as np
from scipy import linalg


def sqrt_diag(a: np.ndarray):
    return np.sqrt(np.diagonal(a, axis1=-2, axis2=-1))


def matrix_power(arr: np.ndarray, power: float) -> np.ndarray:
    w, v = linalg.eigh(arr)
    w = np.eye(len(w)) * w**power
    return v @ w @ v.T


def fill_other_triangle(arr):
    row, col = np.diag_indices(arr.shape[-1])
    diag = np.zeros_like(arr)
    diag[..., row, col] = arr[..., row, col]
    return arr + np.swapaxes(arr, -1, -2) - diag


def force_symmetry(arr):
    return (arr + np.swapaxes(arr, -1, -2)) / 2


def is_positive_definite(arr):
    def _check(a):
        if np.allclose(a, a.T):
            try:
                linalg.cholesky(a)
                return True
            except linalg.LinAlgError:
                return False
        else:
            return False

    out = np.full(arr.shape[:-2], False)
    for index in np.ndindex(arr.shape[:-2]):
        out[index] = _check(arr[index])
    return out


def regularize_matrix(
    arr: np.ndarray,
    const: float = 1.0,
    method: Literal["constant", "avg_diag", "increase_diag"] = "constant",
    only_if_singular: bool = True,
) -> np.ndarray:
    p = arr.shape[-1]
    if arr.shape[-2] != p:
        raise IndexError
    if only_if_singular:
        if linalg.matrix_rank(arr) == p:
            return arr

    if method == "constant":
        if const < 0:
            raise ValueError("in method 'constant' const must be greater or equal to 0")
        return arr + np.eye(p) * const
    elif method == "avg_diag":
        if not 0 <= const <= 1:
            raise ValueError("in method 'avg_diag' const must be in [0, 1]")
        return (1 - const) * arr + const * np.eye(p) * np.diag(arr).mean()
    elif method == "increase_diag":
        if not 0 <= const <= 1:
            raise ValueError("in method 'increase_diag' const must be in [0, 1]")
        return (1 - const) * arr + const * np.diag(np.diag(arr))
    else:
        raise ValueError


def cov_to_corr(arr):
    row, col = np.diag_indices(arr.shape[-1])
    diagonals = np.sqrt(arr[..., row, col])
    return arr / (diagonals[..., None, :] * diagonals[..., :, None])
