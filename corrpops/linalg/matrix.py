from typing import Literal

import numpy as np


def sqrt_diag(a: np.ndarray) -> np.ndarray:  # pragma: no cover
    return np.sqrt(np.diagonal(a, axis1=-2, axis2=-1))


def matrix_power(a: np.ndarray, power: float) -> np.ndarray:
    w, v = np.linalg.eigh(a)
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
                np.linalg.cholesky(a_)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False

    out = np.full(a.shape[:-2], False)
    for index in np.ndindex(a.shape[:-2]):
        out[index] = _check(a[index])
    return out


def regularize_matrix(
    a: np.ndarray,
    const: float,
    method: Literal["constant", "avg_diag", "increase_diag"] = "constant",
    only_if_singular: bool = False,
) -> np.ndarray:
    if a.ndim != 2:
        raise ValueError(f"expected 2d array, got {a.ndim}d instead")

    p = a.shape[0]
    if a.shape[1] != p:
        raise ValueError(f"array is not square: {a.shape}")

    if const < 0:
        raise ValueError("const must be greater or equal to 0")
    if const > 1:
        if method != "constant":
            raise ValueError(f"in method '{method}' const must be in [0, 1]")

    if only_if_singular:
        if np.linalg.matrix_rank(a) == p:
            return a

    if method == "constant":
        return a + np.diag(np.full(p, const))

    diag = np.diag(a)
    if method == "avg_diag":
        diag = np.full(p, np.mean(diag))
    elif method != "increase_diag":
        raise ValueError(  # pragma: no cover
            f"expected method to be one of 'constant', "
            f"'avg_diag', 'increase_diag, got {method} instead"
        )
    return (1 - const) * a + const * np.diag(diag)


def cov_to_corr(a: np.ndarray) -> np.ndarray:
    row, col = np.diag_indices(a.shape[-1])
    diagonals = np.sqrt(a[..., row, col])
    return a / (diagonals[..., None, :] * diagonals[..., :, None])
