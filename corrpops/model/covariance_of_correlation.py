import warnings
from typing import Literal, Optional, Tuple

import numpy as np
from numba import njit

from linalg import matrix, triangle_and_vector as tv


@njit
def _correlation_covariance(  # pragma: no cover
    m: np.ndarray,
    n: int,
    order_vector_i: np.ndarray,
    order_vector_j: np.ndarray,
) -> np.ndarray:
    out = np.zeros(m.shape[:-2] + (n, n), float)

    for row in range(0, n):
        for col in range(row, n):
            i = order_vector_i[row]
            j = order_vector_j[row]
            k = order_vector_i[col]
            l = order_vector_j[col]

            m_ij = m[..., i, j]
            m_kl = m[..., k, l]
            m_ik = m[..., i, k]
            m_il = m[..., i, l]
            m_jk = m[..., j, k]
            m_jl = m[..., j, l]

            out[..., row, col] = (
                (m_ij * m_kl / 2) * (m_ik**2 + m_il**2 + m_jk**2 + m_jl**2)
                - m_ij * (m_ik * m_il + m_jk * m_jl)
                - m_kl * (m_ik * m_jk + m_il * m_jl)
                + (m_ik * m_jl + m_il * m_jk)
            )
    return out


def covariance_of_correlation(
    arr: np.ndarray,
    non_positive: Literal["raise", "warn", "ignore"] = "raise",
) -> np.ndarray:
    if non_positive != "ignore":
        which_positive = matrix.is_positive_definite(arr)
        if not which_positive.all():
            msg = "some matrices are not symmetric positive semidefinite"
            if non_positive == "raise":
                raise np.linalg.LinAlgError(msg)
            elif non_positive == "warn":
                warnings.warn(msg)
            else:  # pragma: no cover
                raise ValueError(
                    f'non_positive should be one of "raise", "warn", "ignore", '
                    f"got {non_positive} instead",
                )

    # todo: support no numba?
    p = arr.shape[-1]
    result = _correlation_covariance(
        arr,
        tv.vectorized_dim(p),
        np.concatenate([np.repeat(i, p - i - 1) for i in range(p)]),
        np.concatenate([np.arange(i + 1, p) for i in range(p)]),
    )
    result = matrix.fill_other_triangle(result)
    return result


def covariance_of_fisher_correlation(
    arr: np.ndarray,
    non_positive: Literal["raise", "warn", "ignore"] = "raise",
) -> np.ndarray:
    p = arr.shape[-1]
    row, col = np.diag_indices(p)
    arr = np.tanh(arr)
    arr[..., row, col] = 1.0

    row, col = np.diag_indices(tv.vectorized_dim(p))
    result = covariance_of_correlation(arr, non_positive)
    grad = np.zeros_like(result)
    grad[..., row, col] = 1 / (1 - tv.triangle_to_vector(arr) ** 2)

    return np.linalg.multi_dot((grad, result, grad))


def estimated_df(
    est: np.ndarray,
    theo: np.ndarray,
    only_diag: bool = False,
) -> float:
    if only_diag:
        row, col = np.diag_indices(theo.shape[-1])
        x = theo[..., row, col]
        y = est[..., row, col]
    else:
        x = tv.triangle_to_vector(theo, True)
        y = tv.triangle_to_vector(est, True)

    x, y = x.flatten(), y.flatten()
    return np.linalg.lstsq(x[:, np.newaxis], y)[0][0].item()


def average_covariance_of_correlation(
    arr: np.ndarray,
    fisher: bool = False,
    est_n: bool = False,
    only_diag: bool = True,
    non_positive: Literal["raise", "warn", "ignore"] = "raise",
) -> Tuple[np.ndarray, Optional[float]]:
    if fisher:
        cov = covariance_of_fisher_correlation(arr, non_positive)
    else:
        cov = covariance_of_correlation(arr, non_positive)
    cov = cov.mean(tuple(range(cov.ndim - 2)))

    if est_n:
        mat = tv.triangle_to_vector(arr)
        est = np.swapaxes(mat, -1, -2) @ mat / np.prod(cov.shape[:-2])
        estimated_n = estimated_df(est=est, theo=cov, only_diag=only_diag)
        cov = cov / estimated_n
        return cov, estimated_n
    return cov, None
