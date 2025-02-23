import numpy as np

from linalg.matrix import fill_other_triangle


def triangular_dim(vector_length: int, diag: bool = False):
    p = 0.5 * ((-1 if diag else 1) + np.array((-1, 1)) * np.sqrt(1 + 8 * vector_length))
    p = p[np.allclose(p, p.astype(int)) & (p > 0)]
    if len(p) == 1:
        return int(p[0])
    raise ValueError(
        f"array shape ({vector_length}) does not fit size of triangular matrix"
    )


def vectorized_dim(matrix_length: int, diag: bool = False):
    if diag:
        return int(matrix_length * (matrix_length + 1) / 2)
    else:
        return int(matrix_length * (matrix_length - 1) / 2)


def triangle_to_vector(
    a: np.ndarray,
    diag: bool = False,
    check: bool = False,
) -> np.ndarray:
    if a.shape[-2] != a.shape[-1]:
        raise IndexError(f"array is not square ({a.shape[-2]} != {a.shape[-1]}")
    if check and not np.allclose(a, np.swapaxes(a, -1, -2)):
        raise ValueError("a is not symmetric")
    row, col = np.tril_indices(a.shape[-1], 0 if diag else -1)
    return a[..., row, col]


def vector_to_triangle(
    a: np.ndarray,
    diag: bool = False,
    diag_value: float = np.nan,
) -> np.ndarray:
    p = triangular_dim(a.shape[-1], diag=diag)
    out = np.zeros(a.shape[:-1] + (p, p))
    row, col = np.tril_indices(p, 0 if diag else -1)
    out[..., row, col] = a
    out = fill_other_triangle(out)

    if not diag:
        row, col = np.diag_indices(p)
        out[..., row, col] = diag_value

    return out
