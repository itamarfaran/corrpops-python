import warnings
from typing import Literal, Tuple

import numpy as np
from numba import njit

from linalg import matrix, triangle_and_vector as tv


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
                (m_ij * m_kl / 2) * (m_ik**2 + m_il**2 + m_jk**2 + m_jl**2)
                - m_ij * (m_ik * m_il + m_jk * m_jl)
                - m_kl * (m_ik * m_jk + m_il * m_jl)
                + (m_ik * m_jl + m_il * m_jk)
            )
    return out


def covariance_of_correlation(
    arr: np.ndarray,
    non_positive: Literal["raise", "warn", "ignore"] = "raise",
    use_numba: bool = True,
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

    p = arr.shape[-1]
    result = (
        _correlation_covariance if use_numba else _correlation_covariance.py_func
    )(
        m=arr,
        n=tv.vectorized_dim(p),
        order_vector_i=np.concatenate([np.repeat(i, p - i - 1) for i in range(p)]),
        order_vector_j=np.concatenate([np.arange(i + 1, p) for i in range(p)]),
    )
    result = matrix.fill_other_triangle(result)
    return result


def covariance_of_fisher_correlation(
    arr: np.ndarray,
    non_positive: Literal["raise", "warn", "ignore"] = "raise",
    use_numba: bool = True,
) -> np.ndarray:
    p = arr.shape[-1]
    row, col = np.diag_indices(p)
    arr = np.tanh(arr)
    arr[..., row, col] = 1.0

    row, col = np.diag_indices(tv.vectorized_dim(p))
    result = covariance_of_correlation(arr, non_positive, use_numba)
    grad = np.zeros_like(result)
    grad[..., row, col] = 1 / (1 - tv.triangle_to_vector(arr) ** 2)

    return np.linalg.multi_dot((grad, result, grad))


def estimated_df(
    est: np.ndarray,
    theo: np.ndarray,
    only_diag: bool = False,
) -> float:
    if only_diag:
        row, col = np.diag_indices(est.shape[-1])
        est = est[..., row, col]
        theo = theo[..., row, col]
    else:
        est = tv.triangle_to_vector(est, True)
        theo = tv.triangle_to_vector(theo, True)

    est = est.flatten()[:, np.newaxis]
    theo = theo.flatten()

    result = np.linalg.lstsq(a=est, b=theo, rcond=-1)
    return result[0][0].item()


def average_covariance_of_correlation(
    arr: np.ndarray,
    fisher: bool = False,
    est_n: bool = False,
    only_diag: bool = True,
    non_positive: Literal["raise", "warn", "ignore"] = "raise",
    use_numba: bool = True,
) -> Tuple[np.ndarray, float]:
    cov = (covariance_of_fisher_correlation if fisher else covariance_of_correlation)(
        arr=arr,
        non_positive=non_positive,
        use_numba=use_numba,
    )
    cov = cov.mean(axis=tuple(range(arr.ndim - 2)))

    if est_n:
        arr = tv.triangle_to_vector(arr)
        est = np.swapaxes(arr, -1, -2) @ arr / np.prod(arr.shape[:-1])
        estimated_n = estimated_df(est=est, theo=cov, only_diag=only_diag)
    else:
        estimated_n = 1.0
    cov /= estimated_n
    return cov, estimated_n
