from typing import Optional, Union

import numpy as np

from linalg.triangle_vector import triangle_to_vector


def efron_bias_correction(rms: Union[float, int], p: Union[float, int]) -> float:
    return np.sqrt(p / (p - 1) * (rms**2 - 1 / (p - 1)))


def efron_rms_sample(
    arr: np.ndarray, p: Optional[Union[float, int]] = None
) -> float:
    rms_mean = np.mean(
        np.sqrt(
            np.mean(
                arr**2,
                axis=tuple(i for i in range(arr.ndim - 1)),
            )
        )
    )

    if p is not None:
        rms_mean = efron_bias_correction(rms_mean, p)
    return rms_mean


def efron_rms(arr: np.ndarray, p: Union[float, int, None] = None) -> float:
    if arr.shape[-2] != arr.shape[-1]:
        raise ValueError("m is not square")

    if np.any(np.diag(arr) != 1):
        raise ValueError("diag of m is not 1")

    arr = triangle_to_vector(arr, False)
    rms = np.sqrt(np.mean(arr**2))

    if p is not None:
        rms = efron_bias_correction(rms, p)
    return rms


def efron_effective_sample_size(n: Union[float, int], rms: Union[float, int]) -> float:
    return n / (1 + (n - 1) * rms**2)
