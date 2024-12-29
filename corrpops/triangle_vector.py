import numpy as np


def triangle_to_vector(
        arr: np.ndarray,
        diag: bool = False,
) -> np.ndarray:
    if arr.shape[-2] != arr.shape[-1]:
        raise IndexError(f"array is not square ({arr.shape[-2]} != {arr.shape[-1]}")

    row, col = np.tril_indices(arr.shape[-1], 0 if diag else -1)
    return arr[..., row, col]


def vector_to_triangle(
        arr: np.ndarray,
        diag: bool = False,
        diag_value: float = np.nan,
) -> np.ndarray:
    p = 0.5 * (
        (-1 if diag else 1)
        + np.array((-1, 1))
        * np.sqrt(1 + 8 * arr.shape[-1])
    )
    p = p[
        (p == np.round(p))
        & (p == np.abs(p))
    ]
    if len(p) == 1:
        p = int(p[0])
    else:
        raise IndexError(f"array shape ({arr.shape[-1]}) does not fit size of triangular matrix")

    out = np.empty(arr.shape[:-1] + (p, p), dtype=float)
    row, col = np.tril_indices(p, 0 if diag else -1)
    out[..., row, col] = arr
    out = out + out.T - np.diagonal(out, axis1=-2, axis2=-1) * np.eye(p)

    if not diag:
        row, col = np.diag_indices(p)
        out[..., row, col] = diag_value

    return out
