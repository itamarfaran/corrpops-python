import inspect

import numpy as np
import pytest

from corrpops.data.preprocessing import (
    count_na_by_threshold,
    drop_columns_by_na_threshold,
    preprocess,
)


def test_loaders_imports():
    from corrpops.data import loaders

    inspect.ismodule(loaders)


@pytest.mark.parametrize("seed", np.random.randint(10_000, size=10))
def test_count_na_by_threshold(parameters_and_sample, seed):
    rng = np.random.default_rng(seed)
    _, _, control, diagnosed = parameters_and_sample
    arr = np.concatenate((control, diagnosed))

    i = rng.integers(arr.shape[0])
    j1 = j2 = rng.integers(arr.shape[-1])
    while j2 == j1:
        j2 = rng.integers(arr.shape[-1])

    arr[:, j1, :] = arr[:, :, j1] = np.nan
    arr[i, j2, :] = arr[i, :, j2] = np.nan

    thresholds, percent_omitted = count_na_by_threshold(arr)
    assert thresholds.shape == percent_omitted.shape

    np.testing.assert_allclose(
        thresholds,
        np.array([0, 1 / 20, 1]),
        # rows 1, 2, 5 have 0 nulls across all samples - to omit them the required threshold is <0
        # row 4 has one null out of 20 samples - to omit it the required threshold is <0.05 (or 1 / 20)
        # row 3 has nulls in all samples - to omit it the required threshold is <1
    )
    np.testing.assert_allclose(
        percent_omitted,
        np.array([2 / 5, 1 / 5, 0]),
        # if we select threshold = 0, we will omit 2 / 5 columns (3, 4)
        # if we select threshold = 0.05, we will omit 1 / 5 column (3)
        # if we select threshold = 1, we will not omit columns
    )


def test_drop_columns_by_na_threshold(parameters_and_sample):
    _, _, control, diagnosed = parameters_and_sample
    arr = np.concatenate((control, diagnosed)).copy()
    arr[:, 0, :] = arr[:, :, 0] = np.nan
    arr[0, 1, :] = arr[0, :, 1] = np.nan
    arr[1, 1, :] = arr[1, :, 1] = np.nan
    arr[1, 2, :] = arr[1, :, 2] = np.nan

    # by default, remove only columns with all nulls
    result, idx_drop = drop_columns_by_na_threshold(arr)
    assert result.shape == (arr.shape[0], arr.shape[1] - 1, arr.shape[2] - 1)
    np.testing.assert_allclose(idx_drop, [0])

    # alternatively, set a threshold yourself
    result, idx_drop = drop_columns_by_na_threshold(arr, threshold=1)
    assert result.shape == arr.shape
    np.testing.assert_allclose(idx_drop, [])

    result, idx_drop = drop_columns_by_na_threshold(arr, threshold=0)
    assert result.shape == (arr.shape[0], arr.shape[1] - 3, arr.shape[2] - 3)
    np.testing.assert_allclose(idx_drop, [0, 1, 2])

    # or set what ratio of columns you are willing to drop
    result, idx_drop = drop_columns_by_na_threshold(arr, max_percent_omitted=1 / 5)
    assert result.shape == (arr.shape[0], arr.shape[1] - 1, arr.shape[2] - 1)
    np.testing.assert_allclose(idx_drop, [0])

    # or set how many columns you are willing to drop with an integer
    result, idx_drop = drop_columns_by_na_threshold(arr, max_percent_omitted=1)
    assert result.shape == (arr.shape[0], arr.shape[1] - 1, arr.shape[2] - 1)
    np.testing.assert_allclose(idx_drop, [0])

    # but this will raise a value error if you are mistaken
    with pytest.raises(ValueError):
        drop_columns_by_na_threshold(arr, max_percent_omitted=6)

    # happy flow
    arr = np.concatenate((control, diagnosed))
    result, idx_drop = drop_columns_by_na_threshold(arr)
    assert result.shape == arr.shape
    np.testing.assert_allclose(idx_drop, [])

    with pytest.raises(ValueError):
        drop_columns_by_na_threshold(arr, threshold=1.1)


def test_preprocess(parameters_and_sample):
    _, _, control, diagnosed = parameters_and_sample
    p = diagnosed.shape[-1]

    control_res, diagnosed_res, dropped_subjects, dropped_columns = preprocess(
        control=control,
        diagnosed=diagnosed,
        subset=[1, 2],
    )
    assert diagnosed_res.shape == (diagnosed.shape[0], 2, 2)
    assert control_res.shape == (control.shape[0], 2, 2)

    control[:, 0, :] = control[:, :, 0] = np.nan
    control[0, 1, :] = control[0, :, 1] = np.nan
    diagnosed[:, 0, :] = diagnosed[:, :, 0] = np.nan
    diagnosed[0, 1, :] = diagnosed[0, :, 1] = np.nan
    diagnosed[1, 1, :] = diagnosed[1, :, 1] = np.nan

    control_res, diagnosed_res, dropped_subjects, dropped_columns = preprocess(
        control, diagnosed
    )
    # drop one column, drop one sample
    assert diagnosed_res.shape == (diagnosed.shape[0] - 2, p - 1, p - 1)
    assert control_res.shape == (control.shape[0] - 1, p - 1, p - 1)
