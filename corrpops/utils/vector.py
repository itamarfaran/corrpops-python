import numpy as np
from scipy import linalg


def mahalanobis(
        x,
        y=None,
        m=None,
        solve=True,
        sqrt=True,
):
    if y is not None:
        x = x - y
    x = np.atleast_2d(x)

    if m is None:
        out = np.sum(x ** 2)
    else:
        if solve:
            m = linalg.inv(m)
        out = x @ m @ np.swapaxes(x, -1, -2)

    if sqrt:
        out = np.sqrt(out)
    return out


def norm_p(x, y, p):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)
