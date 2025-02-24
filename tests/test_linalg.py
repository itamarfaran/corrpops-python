import itertools

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.moment_helpers import cov2corr

from corrpops.linalg import matrix, vector
from linalg.matrix import force_symmetry
from linalg.triangle_vector import (
    triangle_to_vector,
    vector_to_triangle,
    vectorized_dim,
    triangular_dim,
)


def test_matrix_power():
    mat = np.random.random(size=(5, 10))
    mat = np.cov(mat)

    mat_sqr = matrix.matrix_power(mat, 2)
    np.testing.assert_allclose(
        mat_sqr,
        mat @ mat,
    )
    mat_sqrt = matrix.matrix_power(mat, 1 / 2)
    np.testing.assert_allclose(
        mat_sqrt @ mat_sqrt,
        mat,
    )


def test_force_symmetry():
    a = np.arange(5 * 4 * 4).reshape(5, 4, 4)
    a_sym = matrix.force_symmetry(a)

    for a_ in a_sym:
        np.testing.assert_allclose(a_, a_.T)

    with pytest.raises(ValueError):
        a = np.arange(5 * 4 * 5).reshape(5, 4, 5)
        matrix.force_symmetry(a)


def test_is_positive_definite():
    v = stats.ortho_group.rvs(5, random_state=7549)
    w = np.arange(5)

    assert matrix.is_positive_definite(v @ np.diag(w + 1) @ v.T)
    assert not matrix.is_positive_definite(v @ np.diag(w - 1) @ v.T)
    assert not matrix.is_positive_definite(v @ np.diag(w + 1) @ v)


@pytest.mark.parametrize(
    "n, p",
    itertools.product(
        [1, 10],
        [4, 6, 8],
    ),
)
def test_cov_to_corr(n: int, p: int):
    matrices = stats.wishart.rvs(
        df=5 * p,
        scale=np.eye(p),
        size=n,
    ).reshape((n, p, p))

    for i in np.random.choice(n, min(n, 5), replace=False):
        np.testing.assert_allclose(
            matrix.cov_to_corr(matrices)[i],
            cov2corr(matrices[i]),
        )


def test_fill_other_triangle():
    a = 1 + np.arange(5 * 4 * 4).reshape(5, 4, 4)
    a = matrix.force_symmetry(a)

    row, col = np.tril_indices(4, k=-1)
    b = a.copy()
    b[..., row, col] = 0.0
    np.testing.assert_allclose(a, matrix.fill_other_triangle(b))

    row, col = np.triu_indices(4, k=1)
    c = a.copy()
    c[..., row, col] = 0.0
    np.testing.assert_allclose(a, matrix.fill_other_triangle(c))


@pytest.mark.parametrize(
    "p, const, method",
    itertools.product(
        [3, 5, 10],
        [0.0, 1.0],
        ["constant", "avg_diag", "increase_diag"],
    ),
)
def test_regularize_matrix(p, const, method):
    v = stats.ortho_group.rvs(p, random_state=7549)
    orig = v @ np.diag(1 + np.arange(p)) @ v.T
    singular = v @ np.diag(np.arange(p)) @ v.T

    np.testing.assert_array_equal(
        matrix.regularize_matrix(orig, const, method, only_if_singular=True),
        orig,
    )

    regularized = matrix.regularize_matrix(
        singular, const, method, only_if_singular=True
    )
    assert np.any(regularized != singular) == bool(const)

    with pytest.raises(ValueError):
        matrix.regularize_matrix(orig, -1, method)

    if method != "constant":
        with pytest.raises(ValueError):
            matrix.regularize_matrix(orig, 2, method)

    with pytest.raises(ValueError):
        matrix.regularize_matrix(np.arange(4 * 5 * 5).reshape(4, 5, 5), const, method)

    with pytest.raises(ValueError):
        matrix.regularize_matrix(np.arange(4 * 5).reshape(4, 5), const, method)

    row, col = np.diag_indices(p)
    regularized = matrix.regularize_matrix(orig, const, method)

    if not const:
        np.testing.assert_allclose(regularized, orig)

    elif const != 1:
        regularized_diag = regularized.copy()
        np.fill_diagonal(regularized_diag, 0)

        if method == "constant":
            diff = regularized - orig
            np.fill_diagonal(diff, 0)
            np.testing.assert_allclose(diff, 0.0)

        elif method == "avg_diag":
            np.testing.assert_allclose(
                regularized[row, col], regularized[row, col].min()
            )
            np.testing.assert_allclose(regularized_diag, 0.0)

        else:
            np.testing.assert_allclose(regularized[row, col], orig[row, col])
            np.testing.assert_allclose(regularized_diag, 0.0)


def test_mahalanobis():
    rng = np.random.default_rng(858)
    x = rng.random(size=10)
    y = rng.random(size=10)

    np.testing.assert_array_equal(
        vector.mahalanobis(x),
        vector.norm_p(x),
    )
    np.testing.assert_array_equal(
        vector.mahalanobis(x, sqrt=False),
        vector.norm_p(x) ** 2,
    )
    np.testing.assert_array_equal(
        vector.mahalanobis(x, y),
        vector.norm_p(x, y),
    )
    np.testing.assert_array_equal(
        vector.mahalanobis(x, y, np.eye(10)),
        vector.norm_p(x, y),
    )
    np.testing.assert_array_equal(
        vector.mahalanobis(x, y, np.eye(10), inverse=False),
        vector.norm_p(x, y),
    )


@pytest.mark.parametrize(
    "d, p",
    itertools.product(
        [3, 5, 10],
        [0.5, 1, 1.5, 2, 3, 4],
    ),
)
def test_norm_p(d, p):
    x = np.ones(d)
    np.testing.assert_allclose(vector.norm_p(x, p=p), d ** (1 / p))
    np.testing.assert_allclose(vector.norm_p(x, p=p, reduce=False), d)
    np.testing.assert_allclose(vector.norm_p(x, p=p, agg="mean"), 1)
    np.testing.assert_allclose(vector.norm_p(x, x, p=p, agg="mean"), 0)


def test_triangle_to_vector():
    triangle_ = 1 + np.arange(4**2).reshape((4, 4))

    vector_ = triangle_to_vector(triangle_, diag=False)
    assert vector_.shape == (6,)
    assert vectorized_dim(4, diag=False) == 6

    np.testing.assert_array_equal(vector_, np.array([5, 9, 10, 13, 14, 15]))

    vector_ = triangle_to_vector(triangle_, diag=True)
    assert vector_.shape == (10,)
    assert vectorized_dim(4, diag=True) == 10

    np.testing.assert_array_equal(
        triangle_to_vector(triangle_, diag=True),
        np.array([1, 5, 6, 9, 10, 11, 13, 14, 15, 16]),
    )

    with pytest.raises(ValueError):
        triangle_to_vector(triangle_, check=True)
    with pytest.raises(ValueError):
        triangle_to_vector(np.arange(4 * 5).reshape((4, 5)))


@pytest.mark.parametrize("diag", [False, True])
def test_vector_to_triangle(diag):
    vector_ = np.arange(6)
    m = 3 if diag else 4

    triangle_ = vector_to_triangle(vector_, diag=diag)
    assert triangle_.shape == (m, m)
    assert triangular_dim(6, diag=diag) == m
    row, col = np.tril_indices(m, k=0 if diag else -1)
    np.testing.assert_array_equal(
        triangle_[row, col],
        vector_,
    )
    with pytest.raises(ValueError):
        vector_to_triangle(np.arange(5), diag=diag)


@pytest.mark.parametrize(
    "n, p, diag",
    itertools.product(
        [1, 2, 5],
        [4, 6, 10],
        [False, True],
    ),
)
def test_triangle_vector_duality(n: int, p: int, diag: bool):
    triangle_ = np.arange(p**2).reshape((p, p))
    triangle_ = force_symmetry(np.stack(n * [triangle_]))

    diag_value = np.random.random()
    if not diag:
        row, col = np.diag_indices(p)
        triangle_[..., row, col] = diag_value

    vector_ = triangle_to_vector(triangle_, diag)
    np.testing.assert_array_equal(
        vector_to_triangle(vector_, diag, diag_value=diag_value),
        triangle_,
    )


@pytest.mark.parametrize(
    "diag, diag_value",
    itertools.product(
        [False, True],
        [0, 1, np.nan],
    ),
)
def test_vector_to_triangle_multi_index(diag, diag_value):
    rng = np.random.default_rng(358)
    triangles_ = rng.random((10, 12, 4, 4))
    vectors_ = triangle_to_vector(triangles_, diag=diag)

    for index in np.ndindex(triangles_.shape[:-2]):
        np.testing.assert_array_equal(
            vectors_[index], triangle_to_vector(triangles_[index], diag=diag)
        )

    vectors_ = rng.random((12, 12, 6))
    triangles_ = vector_to_triangle(vectors_, diag=diag, diag_value=diag_value)

    for index in np.ndindex(triangles_.shape[:-2]):
        np.testing.assert_array_equal(
            triangles_[index],
            vector_to_triangle(vectors_[index], diag=diag, diag_value=diag_value),
        )

    if not diag:
        row, col = np.diag_indices(4)
        np.testing.assert_array_equal(
            triangles_[..., row, col],
            diag_value,
        )
