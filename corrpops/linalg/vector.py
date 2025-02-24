from typing import Literal, Optional

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


def norm_p(
    x: npt.ArrayLike,
    y: Optional[npt.ArrayLike] = None,
    p: float = 2,
    agg: Literal["sum", "mean"] = "sum",
    reduce: bool = True,
) -> float:
    x = np.asarray(x)
    if y is not None:
        x = x - np.asarray(y)

    x_raised = np.abs(x) ** p

    if agg == "sum":
        aggregated = np.sum(x_raised)
    elif agg == "mean":
        aggregated = np.mean(x_raised)
    else:
        raise ValueError(  # pragma: no cover
            f"expected 'agg' to be either 'sum' or 'mean', got {agg} instead"
        )

    return aggregated ** (1 / p) if reduce else aggregated
