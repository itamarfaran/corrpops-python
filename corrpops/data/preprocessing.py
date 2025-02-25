import logging
from typing import Collection, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("corrpops")
logging.basicConfig(level=logging.INFO)


def count_na_by_threshold(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    na_counts = np.isnan(arr).mean(1).mean(0)
    thresholds = np.sort(np.unique(np.append(na_counts, [0, 1])))
    percent_omitted = np.array([np.mean(na_counts >= t) for t in thresholds])
    return thresholds, percent_omitted


def drop_columns_by_na_threshold(
    arr: np.ndarray,
    threshold: Optional[float] = None,
    max_percent_omitted: Union[float, int, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(max_percent_omitted, int):
        max_percent_omitted = max_percent_omitted / arr.shape[-1]
    if isinstance(max_percent_omitted, float) and not 0 <= max_percent_omitted <= 1:
        raise ValueError("expected max_percent_omitted to be in [0, 1]")

    if threshold is None:
        thresholds, percents_omitted = count_na_by_threshold(arr)

        if max_percent_omitted is not None:
            thresholds = thresholds[percents_omitted <= max_percent_omitted]
            percents_omitted = percents_omitted[percents_omitted <= max_percent_omitted]
            threshold = thresholds[percents_omitted.argmax()]

        else:
            threshold = np.max(thresholds[thresholds < 1])
        logger.warning(f"preprocessing: selected {threshold.round(5)} as threshold")

    elif not 0 <= threshold <= 1:
        raise ValueError("expected threshold to be in [0, 1]")

    na_counts = np.isnan(arr).mean(1).mean(0)
    idx_keep = np.argwhere(na_counts < threshold).flatten()
    idx_drop = np.argwhere(na_counts >= threshold).flatten()
    arr = arr[..., idx_keep, :][..., :, idx_keep]
    return arr, idx_drop


def preprocess(
    control: np.ndarray,
    diagnosed: np.ndarray,
    threshold: Optional[float] = None,
    max_percent_omitted: Union[float, int, None] = None,
    subset: Optional[Collection[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.concatenate((control, diagnosed))

    if subset is not None:
        arr = arr[..., subset, :][..., :, subset]

    arr, dropped_columns = drop_columns_by_na_threshold(arr, threshold, max_percent_omitted)
    dropped_subjects = np.argwhere(np.isnan(arr).any(axis=(1, 2))).flatten()

    control_indices = np.arange(0, control.shape[0])
    dropped_control_indices = dropped_subjects[np.isin(dropped_subjects, control_indices)]
    kept_control_indices = control_indices[~np.isin(control_indices, dropped_subjects)]

    diagnosed_indices = control.shape[0] + np.arange(0, diagnosed.shape[0])
    dropped_diagnosed_indices = dropped_subjects[np.isin(dropped_subjects, diagnosed_indices)] - control.shape[0]
    kept_diagnosed_indices = diagnosed_indices[~np.isin(diagnosed_indices, dropped_subjects)]

    if len(dropped_control_indices):
        logger.warning(
            f"preprocessing: dropped {len(dropped_control_indices)} control subjects: "
            + ", ".join(str(i) for i in dropped_control_indices)
        )
    if len(dropped_diagnosed_indices):
        logger.warning(
            f"preprocessing: dropped {len(dropped_diagnosed_indices)} diagnosed subjects: "
            + ", ".join(str(i) for i in dropped_diagnosed_indices)
        )
    if len(dropped_columns):
        logger.warning(
            f"preprocessing: dropped {len(dropped_columns)} columns "
            + ", ".join(str(i) for i in dropped_columns)
        )

    return (
        arr[kept_control_indices],
        arr[kept_diagnosed_indices],
        dropped_subjects,
        dropped_columns,
    )
