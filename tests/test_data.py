import inspect

import numpy as np
import pytest

from corrpops.corrpops_logger import corrpops_logger
from corrpops.data.preprocessing import (
    calculate_percent_na,
    na_thresholds,
    drop_columns_by_na_threshold,
    drop_columns_by_max_omitted,
    preprocess,
)

corrpops_logger().setLevel(40)


def assert_shape(a, shape):
    if hasattr(shape, "shape"):
        shape = shape.shape
    if isinstance(a, tuple):
        for a_, s_ in zip(a, shape):
            assert a_.shape == s_
    else:
        assert a.shape == shape


def test_loaders_imports():
    from corrpops.data import loaders

    inspect.ismodule(loaders)


def test_calculate_percent_na(parameters_and_sample):
    _, _, control, diagnosed = parameters_and_sample
    arr = np.concatenate((control, diagnosed))
    n, p = np.prod(arr.shape[:-2]), arr.shape[-1]
    row, col = np.diag_indices(p)

    # happy flow
    na_summaries_0 = calculate_percent_na(arr)
    na_summaries, thresholds = na_thresholds(arr)
    assert na_summaries_0 == na_summaries
    assert len(na_summaries) == n
    assert not thresholds

    arr[..., 1, :] = arr[..., :, 1] = np.nan
    arr[..., row, col] = 1
    _, thresholds = na_thresholds(arr)
    assert thresholds == {1: 1.0}

    arr[1, 2, :] = arr[1, :, 2] = np.nan
    arr[..., row, col] = 1
    _, thresholds = na_thresholds(arr)
    assert thresholds == {1: 1.0, 2: 1 / n}


@pytest.mark.parametrize("max_omitted", [0, 1, 2])
def test_drop_columns_by_max_omitted(parameters_and_sample, max_omitted):
    _, _, control, diagnosed = parameters_and_sample
    arr = np.concatenate((control, diagnosed))
    n, p = arr.shape[:-2], arr.shape[-1]
    row, col = np.diag_indices(p)

    result, indices = drop_columns_by_max_omitted(arr, max_omitted)
    np.testing.assert_array_equal(arr, result)
    np.testing.assert_array_equal(indices, [])

    arr[..., 1, :] = arr[..., :, 1] = np.nan
    arr[..., row, col] = 1
    result, indices = drop_columns_by_max_omitted(arr, max_omitted)
    p_ = p - min(1, max_omitted)
    assert_shape(result, n + (p_, p_))

    result, indices = drop_columns_by_max_omitted(arr, max_omitted / p)
    p_ = p - min(1, max_omitted)
    assert_shape(result, n + (p_, p_))

    arr[..., 2, :] = arr[..., :, 2] = np.nan
    arr[..., 1, 3, :] = arr[..., 1, :, 3] = np.nan
    arr[..., row, col] = 1
    result, indices = drop_columns_by_max_omitted(arr, max_omitted)
    p_ = p - min(2, max_omitted)
    assert_shape(result, n + (p_, p_))

    with pytest.raises(ValueError):
        drop_columns_by_max_omitted(arr, 1.1)


def test_drop_columns_by_na_threshold(parameters_and_sample):
    _, _, control, diagnosed = parameters_and_sample
    arr = np.concatenate((control, diagnosed))
    n, p = arr.shape[:-2], arr.shape[-1]
    row, col = np.diag_indices(p)

    result, indices = drop_columns_by_na_threshold(arr)
    np.testing.assert_array_equal(arr, result)
    np.testing.assert_array_equal(indices, [])

    arr[..., 1, :] = arr[..., :, 1] = np.nan
    arr[..., row, col] = 1
    result, indices = drop_columns_by_na_threshold(arr)
    assert_shape(result, n + (p - 1, p - 1))

    arr[..., 1, 2, :] = arr[..., 1, :, 2] = np.nan
    arr[..., row, col] = 1
    result, indices = drop_columns_by_na_threshold(arr)
    assert_shape(result, n + (p - 1, p - 1))

    result, indices = drop_columns_by_na_threshold(arr, 0.0)
    assert_shape(result, n + (p - 2, p - 2))

    result, indices = drop_columns_by_na_threshold(arr, 1)
    assert_shape(result, n + (p - 1, p - 1))

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
    assert_shape(diagnosed_res, (diagnosed.shape[0], 2, 2))
    assert_shape(control_res, (control.shape[0], 2, 2))

    control[:, 0, :] = control[:, :, 0] = np.nan
    control[0, 1, :] = control[0, :, 1] = np.nan
    diagnosed[:, 0, :] = diagnosed[:, :, 0] = np.nan
    diagnosed[0, 1, :] = diagnosed[0, :, 1] = np.nan
    diagnosed[1, 1, :] = diagnosed[1, :, 1] = np.nan

    # drop one column, drop one sample
    expected_shapes = (
        (diagnosed.shape[0] - 2, p - 1, p - 1),
        (control.shape[0] - 1, p - 1, p - 1),
    )

    control_res, diagnosed_res, _, _ = preprocess(control, diagnosed)
    assert_shape((diagnosed_res, control_res), expected_shapes)

    control_res, diagnosed_res, _, _ = preprocess(control, diagnosed, 1.0)
    assert_shape((diagnosed_res, control_res), expected_shapes)

    control_res, diagnosed_res, _, _ = preprocess(control, diagnosed, max_omitted=1)
    assert_shape((diagnosed_res, control_res), expected_shapes)

    # drop all na columns
    expected_shapes = (
        (diagnosed.shape[0], p - 2, p - 2),
        (control.shape[0], p - 2, p - 2),
    )

    control_res, diagnosed_res, _, _ = preprocess(control, diagnosed, 0.0)
    assert_shape((diagnosed_res, control_res), expected_shapes)

    control_res, diagnosed_res, _, _ = preprocess(control, diagnosed, max_omitted=2)
    assert_shape((diagnosed_res, control_res), expected_shapes)

    control_res, diagnosed_res, _, _ = preprocess(control, diagnosed, max_omitted=5)
    assert_shape((diagnosed_res, control_res), expected_shapes)

    # don't drop columns - drop all samples :(
    control_res, diagnosed_res, _, _ = preprocess(control, diagnosed, max_omitted=0)
    assert_shape((diagnosed_res, control_res), 2 * ((0, p, p),))

    with pytest.raises(ValueError):
        preprocess(control, diagnosed, threshold=0.0, max_omitted=0)
