import warnings
from typing import Collection, Optional, Tuple

import numpy as np


def count_na_by_threshold(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    na_counts = np.isnan(arr).mean(0)[0]
    thresholds = np.sort(np.unique(na_counts))
    percent_omitted = np.array([np.mean(na_counts > t) for t in thresholds])
    return thresholds, percent_omitted


def drop_columns_by_na_threshold(
    arr: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if threshold is None:
        potential_thresholds, percent_omitted = count_na_by_threshold(arr)

        potential_thresholds = potential_thresholds[percent_omitted > 0]
        percent_omitted = percent_omitted[percent_omitted > 0]

        threshold = potential_thresholds[percent_omitted.argmin()]
        warnings.warn(f"using {threshold.round(5)} as threshold")

    na_counts = np.isnan(arr).mean(0)[0]
    idx_keep = np.argwhere(na_counts <= threshold).flatten()
    idx_drop = np.argwhere(na_counts > threshold).flatten()
    arr = arr[..., idx_keep, :][..., :, idx_keep]
    return arr, idx_drop


def preprocess(
    control: np.ndarray,
    diagnosed: np.ndarray,
    threshold: Optional[float] = None,
    subset: Optional[Collection[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.concatenate((control, diagnosed))

    if subset is not None:
        arr = arr[..., subset, :][..., :, subset]

    arr, dropped_columns = drop_columns_by_na_threshold(arr, threshold)
    dropped_subjects = np.argwhere(np.isnan(arr).any(axis=(1, 2))).flatten()

    control_indices = np.arange(0, control.shape[0])
    control_indices = control_indices[~np.isin(control_indices, dropped_subjects)]

    diagnosed_indices = control.shape[0] + np.arange(0, diagnosed.shape[0])
    diagnosed_indices = diagnosed_indices[~np.isin(diagnosed_indices, dropped_subjects)]

    return (
        arr[control_indices],
        arr[diagnosed_indices],
        dropped_subjects,
        dropped_columns,
    )
