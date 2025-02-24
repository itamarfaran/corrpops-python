import itertools

import numpy as np
import pytest

from corrpops.linalg.matrix import force_symmetry
from corrpops.linalg.triangle_vector import (
    triangle_to_vector,
    vector_to_triangle,
    triangular_dim,
    vectorized_dim,
)


@pytest.mark.parametrize(
    "n, p, diag",
    itertools.product(
        [1, 2, 5],
        [4, 6, 10],
        [False, True],
    ),
)
def test_triangle_vector_duality(n: int, p: int, diag: bool):
    triangle = np.arange(p**2).reshape((p, p))
    triangle = force_symmetry(np.stack(n * [triangle]))

    diag_value = np.random.random()
    if not diag:
        row, col = np.diag_indices(p)
        triangle[..., row, col] = diag_value

    vector = triangle_to_vector(triangle, diag)
    np.testing.assert_array_equal(
        vector_to_triangle(vector, diag, diag_value=diag_value),
        triangle,
    )


def test_triangle_to_vector():
    triangle = 1 + np.arange(4**2).reshape((4, 4))

    vector = triangle_to_vector(triangle, diag=False)
    assert vector.shape == (6,)
    assert vectorized_dim(4, diag=False) == 6

    np.testing.assert_array_equal(vector, np.array([5, 9, 10, 13, 14, 15]))

    vector = triangle_to_vector(triangle, diag=True)
    assert vector.shape == (10,)
    assert vectorized_dim(4, diag=True) == 10

    np.testing.assert_array_equal(
        triangle_to_vector(triangle, diag=True),
        np.array([1, 5, 6, 9, 10, 11, 13, 14, 15, 16]),
    )

    with pytest.raises(ValueError):
        triangle_to_vector(triangle, check=True)
    with pytest.raises(ValueError):
        triangle_to_vector(np.arange(4 * 5).reshape((4, 5)))


@pytest.mark.parametrize("diag", [False, True])
def test_vector_to_triangle(diag):
    vector = np.arange(6)
    m = 3 if diag else 4

    triangle = vector_to_triangle(vector, diag=diag)
    assert triangle.shape == (m, m)
    assert triangular_dim(6, diag=diag) == m
    row, col = np.tril_indices(m, k=0 if diag else -1)
    np.testing.assert_array_equal(
        triangle[row, col],
        vector,
    )
    with pytest.raises(ValueError):
        vector_to_triangle(np.arange(5), diag=diag)


@pytest.mark.parametrize(
    "diag, diag_value",
    itertools.product(
        [False, True],
        [0, 1, np.nan],
    ),
)
def test_multi_index(diag, diag_value):
    rng = np.random.default_rng(358)
    triangles = rng.random((10, 12, 4, 4))
    vectors = triangle_to_vector(triangles, diag=diag)

    for index in np.ndindex(triangles.shape[:-2]):
        np.testing.assert_array_equal(
            vectors[index], triangle_to_vector(triangles[index], diag=diag)
        )

    vectors = rng.random((12, 12, 6))
    triangles = vector_to_triangle(vectors, diag=diag, diag_value=diag_value)

    for index in np.ndindex(triangles.shape[:-2]):
        np.testing.assert_array_equal(
            triangles[index],
            vector_to_triangle(vectors[index], diag=diag, diag_value=diag_value),
        )

    if not diag:
        row, col = np.diag_indices(4)
        np.testing.assert_array_equal(
            triangles[..., row, col],
            diag_value,
        )
