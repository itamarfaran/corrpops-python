from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import linalg


def mahalanobis(
    x: npt.ArrayLike,
    y: Optional[npt.ArrayLike] = None,
    m: Optional[npt.ArrayLike] = None,
    solve: bool = True,
    sqrt: bool = True,
) -> float:
    x = np.asarray(x)
    if y is not None:
        x = x - np.asarray(y)
    x = np.atleast_2d(x)

    if m is None:
        out = np.sum(x**2)
    else:
        m = np.asarray(m)
        if solve:
            m = linalg.inv(m)
        out = x @ m @ np.swapaxes(x, -1, -2)

    return np.sqrt(out) if sqrt else out


def norm_p(x: npt.ArrayLike, y: Optional[npt.ArrayLike] = None, p: float = 2) -> float:
    x = np.asarray(x)
    if y is not None:
        x = x - np.asarray(y)
    return np.sum(np.abs(x) ** p) ** (1 / p)
