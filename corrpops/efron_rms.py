import numpy as np
from triangle_vector import triangle_to_vector


def efron_bias_correction(rms, p):
    return np.sqrt(
        p / (p - 1) * (rms ^ 2 - 1 / (p - 1))
    )


def efron_rms_sample(arr, p=None):
    arr = triangle_to_vector(arr)
    rms_mean = np.mean(np.sqrt(np.mean(arr ** 2, arr.shape[:-1])))

    if p is not None:
        rms_mean = efron_bias_correction(rms_mean, p)
    return rms_mean


def efron_rms(arr, p=None):
    if arr.shape[-2] != arr.shape[-1]:
        raise ValueError("m is not square")

    if (np.diag(arr) != 1).any():
        raise ValueError("diag of m is not 1")

    arr = triangle_to_vector(arr, False)
    rms = np.sqrt(np.mean(arr ** 2))

    if p is not  None:
        rms = efron_bias_correction(rms, p)
    return rms


def efron_effective_sample_size(n, rms):
    return n / (1 + (n - 1) * rms ** 2)
