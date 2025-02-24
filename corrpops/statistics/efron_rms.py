"""
Efron, Bradley. “Correlation Questions.” Large-Scale Inference: Empirical
Bayes Methods for Estimation, Testing and Prediction, Stanford University,
2010, pp. 129, 137.
"""
from typing import Optional, Union

import numpy as np

from linalg.matrix import is_positive_definite
from linalg.triangle_vector import triangle_to_vector

FloatIntOrArray = Union[float, int, np.ndarray]


def efron_bias_correction(
    rms: FloatIntOrArray,
    n: FloatIntOrArray,
) -> Union[float, np.ndarray]:
    return np.sqrt(n / (n - 1) * (rms**2 - 1 / (n - 1)))


def efron_rms_from_vectors(
    arr: np.ndarray,
    n: Optional[FloatIntOrArray] = None,
) -> Union[float, np.ndarray]:
    rms = np.sqrt(np.mean(arr**2, axis=-1))
    return rms if n is None else efron_bias_correction(rms, n=n)


def efron_rms(
    arr: np.ndarray,
    n: Optional[FloatIntOrArray] = None,
    check_psd: bool = False,
) -> Union[float, np.ndarray]:
    if not arr.shape[-1] == arr.shape[-2]:
        raise ValueError("arr is not square")

    if not np.allclose(arr, np.swapaxes(arr, -1, -2)):
        raise ValueError("arr is not symmetric")

    if check_psd:
        row, col = np.diag_indices(arr.shape[-1])
        if not np.allclose(arr[..., row, col], 1):
            raise ValueError("diag of arr is not 1")

        if not is_positive_definite(arr).all():
            raise ValueError("arr is not positive_definite")

    return efron_rms_from_vectors(triangle_to_vector(arr), n)


def efron_effective_sample_size(
    n: FloatIntOrArray,
    rms: FloatIntOrArray,
) -> Union[float, np.ndarray]:
    return n / (1 + (n - 1) * rms**2)
