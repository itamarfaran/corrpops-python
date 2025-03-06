from typing import Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from linalg.matrix import is_positive_definite


def mahalanobis(
    x: npt.ArrayLike,
    y: Optional[npt.ArrayLike] = None,
    m: Optional[npt.ArrayLike] = None,
    inverse: bool = True,
    agg: Literal["sum", "mean"] = "sum",
    reduce: bool = True,
    check_psd: bool = False,
) -> Union[np.ndarray, float]:
    x = np.atleast_2d(x)
    if y is not None:
        x = x - np.atleast_2d(y)
    if m is None:
        out = np.sum(x**2, axis=-1)  # type: ignore
    else:
        m = np.asarray(m)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("m must be a 2d square matrix")
        if check_psd:
            if not np.allclose(m, m.T) or not is_positive_definite(m):
                raise ValueError("m is not positive semi definite")
        if inverse:
            m = np.linalg.inv(m)
        out = np.squeeze(
            np.diagonal(x @ m @ np.swapaxes(x, -1, -2), axis1=-2, axis2=-1)
        )
    if agg == "mean":
        out = out / x.size  # type: ignore
    elif agg != "sum":
        raise ValueError(  # pragma: no cover
            f"expected 'agg' to be either 'sum' or 'mean', got {agg} instead"
        )
    return np.sqrt(out) if reduce else out


def norm_p(
    x: npt.ArrayLike,
    y: Optional[npt.ArrayLike] = None,
    p: float = 2,
    axis: Union[Tuple[int, ...], int, None] = None,
    agg: Literal["sum", "mean"] = "sum",
    reduce: bool = True,
) -> float:
    x = np.asarray(x)
    if y is not None:
        x = x - np.asarray(y)

    x_raised = np.abs(x) ** p

    if agg == "sum":
        aggregated = np.sum(x_raised, axis=axis)
    elif agg == "mean":
        aggregated = np.mean(x_raised, axis=axis)
    else:
        raise ValueError(  # pragma: no cover
            f"expected 'agg' to be either 'sum' or 'mean', got {agg} instead"
        )

    return aggregated ** (1 / p) if reduce else aggregated
