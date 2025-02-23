import numpy as np

from linalg.matrix import fill_other_triangle


def triangular_dim(vector_dim: int, diag: bool = False) -> int:
    p = 0.5 * ((-1 if diag else 1) + np.array((-1, 1)) * np.sqrt(1 + 8 * vector_dim))
    p = p[np.allclose(p, p.astype(int)) & (p > 0)]
    if len(p) == 1:
        return int(p[0])
    raise ValueError(
        f"array shape ({vector_dim}) does not fit size of triangular matrix"
    )


def vectorized_dim(triangle_dim: int, diag: bool = False) -> int:
    if diag:
        return int(triangle_dim * (triangle_dim + 1) / 2)
    else:
        return int(triangle_dim * (triangle_dim - 1) / 2)


def triangle_to_vector(
    a: np.ndarray,
    diag: bool = False,
    check: bool = False,
) -> np.ndarray:
    if a.shape[-2] != a.shape[-1]:
        raise ValueError(f"array is not square ({a.shape[-2]} != {a.shape[-1]}")
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
