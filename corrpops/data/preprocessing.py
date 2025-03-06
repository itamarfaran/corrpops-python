from collections import namedtuple
from typing import Collection, Dict, List, Optional, Tuple, Union

import numpy as np

from corrpops_logger import corrpops_logger

logger = corrpops_logger()


NaSummary = namedtuple("NaSummary", ["index", "columns", "percent_na"])


def calculate_percent_na(arr: np.ndarray, threshold: float = 0) -> List[NaSummary]:
    results = []
    for index in np.ndindex(arr.shape[:-2]):
        percents_na: List[float] = []
        columns: List[int] = []
        a = arr[index].copy()

        while a.shape[-1]:
            percent_na = np.isnan(a).sum(-1) / (a.shape[-1] - 1)
            max_ = percent_na.max()

            if max_ <= threshold:
                break

            argmax_ = percent_na.argmax()
            a = np.delete(a, argmax_, -2)
            a = np.delete(a, argmax_, -1)
            percents_na.append(max_)
            columns.append(argmax_ + len(columns))

        columns.sort()
        percents_na = [p for _, p in sorted(zip(columns, percents_na))]
        results.append(NaSummary(index, columns, percents_na))
    return results


def na_thresholds(
    arr: np.ndarray,
    threshold: float = 0,
) -> Tuple[List[NaSummary], Dict[int, float]]:
    na_summaries = calculate_percent_na(arr, threshold)
    results_columns = [summary.columns for summary in na_summaries]

    unique_columns = np.unique(np.concatenate(results_columns).astype(int))
    n = np.prod(arr.shape[:-2])
    thresholds = {
        col: sum(col in cols for cols in results_columns) / n for col in unique_columns
    }
    return na_summaries, thresholds


def drop_columns_by_max_omitted(
    arr: np.ndarray,
    max_omitted: Union[float, int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(max_omitted, float):
        if 0 <= max_omitted <= 1:
            max_omitted = int(max_omitted * arr.shape[-1])
        else:
            raise ValueError(
                f"expected max_omitted to be an integer or a "
                f"float in [0, 1], got {max_omitted} instead"
            )
    _, thresholds = na_thresholds(arr)
    sorted_thresholds = sorted(thresholds.items(), key=lambda kv: kv[1], reverse=True)
    idx_drop = [col for col, _ in sorted_thresholds[:max_omitted]]
    arr = np.delete(arr, idx_drop, -2)
    arr = np.delete(arr, idx_drop, -1)
    return arr, np.array(idx_drop)


def drop_columns_by_na_threshold(
    arr: np.ndarray,
    threshold: Union[float, int] = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(threshold, int):
        threshold = min(threshold / arr.shape[-1], 1.0)
    elif not 0 <= threshold <= 1:
        raise ValueError(f"expected threshold to be in [0, 1], got {threshold} instead")

    idx_drop: List[int] = []
    for _ in range(arr.shape[-1]):
        _, thresholds = na_thresholds(arr)
        max_threshold = max(thresholds.values(), default=0.0)
        if max_threshold < threshold:
            break

        current_drop = [
            col for col, percent in thresholds.items() if percent >= max_threshold
        ]
        arr = np.delete(arr, current_drop, -2)
        arr = np.delete(arr, current_drop, -1)
        idx_drop.extend([i + sum(j <= i for j in idx_drop) for i in current_drop])
    return arr, np.array(idx_drop)


def preprocess(
    control: np.ndarray,
    diagnosed: np.ndarray,
    threshold: Optional[float] = None,
    max_omitted: Union[float, int, None] = None,
    subset: Optional[Collection[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.concatenate((control, diagnosed))
    row, col = np.diag_indices(arr.shape[-1])
    arr[..., row, col] = 1.0

    if subset is not None:
        arr = arr[..., subset, :][..., :, subset]

    if max_omitted is None:
        if threshold is None:
            arr, dropped_columns = drop_columns_by_na_threshold(arr)
        else:
            arr, dropped_columns = drop_columns_by_na_threshold(arr, threshold)
    else:
        if threshold is None:
            arr, dropped_columns = drop_columns_by_max_omitted(arr, max_omitted)
        else:
            raise ValueError(
                "at most one of 'threshold', 'max_omitted' can be not none"
            )

    dropped_subjects = np.argwhere(np.isnan(arr).any(axis=(1, 2))).flatten()

    control_indices = np.arange(0, control.shape[0])
    indices = np.isin(dropped_subjects, control_indices)
    dropped_control_indices = dropped_subjects[indices]

    indices = ~np.isin(control_indices, dropped_subjects)
    kept_control_indices = control_indices[indices]

    diagnosed_indices = control.shape[0] + np.arange(0, diagnosed.shape[0])
    indices = np.isin(dropped_subjects, diagnosed_indices)
    dropped_diagnosed_indices = dropped_subjects[indices] - control.shape[0]

    indices = ~np.isin(diagnosed_indices, dropped_subjects)
    kept_diagnosed_indices = diagnosed_indices[indices]

    if len(dropped_control_indices):
        logger.warning(
            f"preprocessing: dropped {len(dropped_control_indices)} control subjects: "
            + ", ".join(str(i) for i in sorted(dropped_control_indices))
        )
    if len(dropped_diagnosed_indices):
        logger.warning(
            f"preprocessing: dropped {len(dropped_diagnosed_indices)} diagnosed subjects: "
            + ", ".join(str(i) for i in sorted(dropped_diagnosed_indices))
        )
    if len(dropped_columns):
        logger.warning(
            f"preprocessing: dropped {len(dropped_columns)} columns "
            + ", ".join(str(i) for i in sorted(dropped_columns))
        )

    return (
        arr[kept_control_indices],
        arr[kept_diagnosed_indices],
        dropped_subjects,
        dropped_columns,
    )
