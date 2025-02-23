import itertools

import numpy as np
import pytest

from linalg.matrix import force_symmetry
from linalg.triangle_vector import triangle_to_vector, vector_to_triangle


@pytest.mark.parametrize(
    "n, p, diag",
    itertools.product(
        [1, 2, 5],
        [4, 6, 10],
        [False, True],
    ),
)
def test_triangle_to_vector(n: int, p: int, diag: bool):
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
