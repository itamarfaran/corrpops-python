import numpy as np
from scipy import linalg


def mahalanobis(
    x: np.ndarray,
    y: np.ndarray = None,
    m: np.ndarray = None,
    solve: bool = True,
    sqrt: bool = True,
) -> float:
    if y is not None:
        x = x - y
    x = np.atleast_2d(x)

    if m is None:
        out = np.sum(x**2)
    else:
        if solve:
            m = linalg.inv(m)
        out = x @ m @ np.swapaxes(x, -1, -2)

    if sqrt:
        out = np.sqrt(out)
    return float(out)


def norm_p(x: np.ndarray, y: np.ndarray = None, p: float = 2) -> float:
    if y is not None:
        x = x - y
    return float(np.sum(np.abs(x) ** p) ** (1 / p))
