import warnings
from typing import Literal

import numpy as np
from numba import njit

from utils.matrix import fill_other_triangle, is_positive_definite
from utils.triangle_vector import triangle_to_vector


@njit
def _correlation_covariance(
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
                (m_ij * m_kl / 2) * (m_ik ** 2 + m_il ** 2 + m_jk ** 2 + m_jl ** 2)
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
        which_positive = is_positive_definite(arr)
        if not which_positive.all():
            if non_positive == "raise":
                raise np.linalg.LinAlgError("some matrices are not positive definite")
            elif non_positive == "warn":
                warnings.warn("some matrices are not positive definite")
            else:
                raise ValueError(
                    f"non_positive should be one of \"raise\", \"warn\", \"ignore\", "
                    f"got {non_positive} instead",
                )

    p = arr.shape[-1]
    result = _correlation_covariance(
        arr,
        int(0.5 * p * (p - 1)),
        np.concatenate([np.repeat(i, p - i - 1) for i in range(p)]),
        np.concatenate([np.arange(i + 1, p) for i in range(p)]),
    )
    result = fill_other_triangle(result)
    return result


def covariance_of_fisher_correlation(
        arr: np.ndarray,
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
) -> np.ndarray:
    arr = np.tanh(arr)
    result = covariance_of_correlation(arr, non_positive)

    grad = np.zeros_like(result)
    row, col = np.diag_indices(grad.shape[-1])
    grad[..., row, col] = 1 / (1 - triangle_to_vector(arr) ** 2)

    return grad @ result @ grad


def estimated_dof(est, theo, only_diag=False):
    if only_diag:
        row, col = np.diag_indices_from(theo)
        x = theo[..., row, col].flatten()
        y = est[..., row, col].flatten()
    else:
        x = triangle_to_vector(theo, True)
        y = triangle_to_vector(est, True)

    return np.linalg.lstsq(x[:, np.newaxis], y)[0]


def average_covariance_of_correlation(
        arr,
        fisher: bool = False,
        est_n: bool = False,
        only_diag: bool = True,
        non_positive: Literal["raise", "warn", "ignore"] = "raise",
):
    if fisher:
        cov = covariance_of_fisher_correlation(arr, non_positive)
    else:
        cov = covariance_of_correlation(arr, non_positive)

    cov = cov.mean(tuple(i for i in range(cov.ndim - 2)))

    if est_n:
        mat = triangle_to_vector(arr)
        est = np.swapaxes(mat, -1, -2) @ mat / np.prod(cov.shape[:-2])
        estimated_n = estimated_dof(est=est, theo=cov, only_diag=only_diag)
        cov = cov / estimated_n
        return cov, estimated_n
    return cov, None
