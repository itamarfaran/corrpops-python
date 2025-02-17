import warnings
from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union

import numpy as np
from scipy.io import loadmat


def count_na_by_threshold(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    na_counts = np.isnan(arr).mean(0)[0]
    thresholds = np.sort(np.unique(na_counts))
    percent_omitted = np.array([(na_counts > t).mean() for t in thresholds])
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


def matlab_to_dict(
    path: Union[Path, str],
    array_key: str,
    control_indices_key: str,
    diagnosed_indices_key: str,
) -> Dict[str, Union[str, np.ndarray]]:
    data = loadmat(path)
    arr = data[array_key].squeeze()
    arr = np.moveaxis(arr, -1, 0)

    control_indices = data[control_indices_key].flatten() - 1
    diagnosed_indices = data[diagnosed_indices_key].flatten() - 1

    return {
        "header": data["__header__"].decode("utf-8"),
        "version": data["__version__"],
        "control": arr[control_indices],
        "diagnosed": arr[diagnosed_indices],
    }


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
