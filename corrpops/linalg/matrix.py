from typing import Literal

import numpy as np


def sqrt_diag(a: np.ndarray) -> np.ndarray:  # pragma: no cover
    return np.sqrt(np.diagonal(a, axis1=-2, axis2=-1))


def matrix_power(a: np.ndarray, power: float) -> np.ndarray:
    w, v = np.linalg.eigh(a)
    w = np.diag(w**power)
    return np.linalg.multi_dot((v, w, v.T))


def fill_other_triangle(a: np.ndarray) -> np.ndarray:
    row, col = np.diag_indices(a.shape[-1])
    diag = np.zeros_like(a)
    diag[..., row, col] = a[..., row, col]
    return a + np.swapaxes(a, -1, -2) - diag


def force_symmetry(a: np.ndarray) -> np.ndarray:
    return (a + np.swapaxes(a, -1, -2)) / 2


def is_positive_definite(a: np.ndarray) -> np.ndarray:
    def _check(a_: np.ndarray) -> bool:
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

    if only_if_singular and np.linalg.matrix_rank(a) == p:
        diag_value = np.zeros(p)
    elif method == "constant":
        diag_value = np.ones(p)
    elif method == "increase_diag":
        diag_value = np.diag(a)
    elif method == "avg_diag":
        diag_value = np.full(p, np.diag(a).mean())
    else:
        raise ValueError(  # pragma: no cover
            f"expected method to be one of 'constant', "
            f"'avg_diag', 'increase_diag, got {method} instead"
        )
    return a + const * np.diag(diag_value)


def cov_to_corr(a: np.ndarray) -> np.ndarray:
    row, col = np.diag_indices(a.shape[-1])
    diagonals = np.sqrt(a[..., row, col])
    return a / (diagonals[..., None, :] * diagonals[..., :, None])
