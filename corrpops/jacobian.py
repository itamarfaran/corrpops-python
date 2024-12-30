# https://gist.github.com/J-Kahn/af71332cd3499666e70e3159272262e2
import numpy as np
from scipy.special import factorial


def simple_jacobian(f, x0, h=1e-06, how="two_sided", *, args=()):
    x = np.asarray(x0)
    xn = x.shape[0]
    fn = f(x0, *args).shape[0]

    out = np.empty((xn, fn), dtype=float)
    for i in range(xn):
        h_ = np.zeros_like(x0, dtype=float)
        h_[i] += h
        if how == "left":
            out[i] = (f(x0 + h_, *args) - f(x0, *args)) / h
        elif how == "right":
            out[i] = (f(x0, *args) - f(x0 - h_, *args)) / h
        elif how == "two_sided":
            out[i] = (f(x0 + h_, *args) - f(x0 - h_, *args)) / (2 * h)
        else:
            raise ValueError
    return out.T


def _richardson(f, x0, o, h1, v, args):
    """
    Numerical Jacobian using Richardson extrapolation
    Jay Kahn - University of Rochester, November 19, 2012
    f - function to take derivative over
    x - point (vector) at which to evaluate derivatives
    o - order of error term desired relative to f
    h1 - starting factor
    *control - any extra arguments to be passed to f
    """
    x = np.asarray(x0)
    d = x.shape[0]
    i = 1
    r = o / 2

    while i <= d:
        j = 1
        while j <= r:
            if j == 1:
                h = h1
            else:
                h = h / v

            idd = np.eye(d) * h
            xup = x + idd[:, i - 1]
            xdown = x - idd[:, i - 1]
            fat = f(x, *args)
            fup = f(xup, *args)
            fdown = f(xdown, *args)
            ddu = fup - fat
            ddd = fdown - fat
            hp = h

            if j == 1:
                dds = np.array([ddu, ddd])
                hhs = np.array([[hp, -hp]])
            else:
                dds = np.concatenate((dds, np.array([ddu, ddd])), 0)
                hhs = np.concatenate((hhs, np.array([[hp, -hp]])), 1)

            j = j + 1

        mat = hhs

        j = 2
        while j <= o:
            mat = np.concatenate((mat, np.power(hhs, j) / factorial(j)), 0)
            j = j + 1

        der = np.dot(np.transpose(np.linalg.inv(mat)), dds)

        if i == 1:
            g = der
        else:
            g = np.concatenate((g, der), 1)

        i = i + 1
    return g


def richardson_jacobian(f, x0, o=6, h1=0.5, v=2, *, args=()):
    """
     Jacobian running as shell of Richardson. Ends up with matrix
     whose rows are derivatives with respect to different elements
     of x and columns are derivatives of different elements of f(x).
     For scalar valued f(x) simplifies to column gradient.
     Jay Kahn - University of Rochester, November 19, 2012
     f - function to take derivative over
     x - point (vector) at which to evaluate derivatives
     o - order of error term desired relative to f
     h1 - starting factor
     *control - any extra arguments to be passed to f
    """
    x = np.asarray(x0)
    xn = x.shape[0]
    fn = f(x0, *args).shape[0]

    out = np.empty((xn, fn), dtype=float)
    g = _richardson(f, x, o, h1, v, args)
    for i in range(xn):
        for j in range(fn):
            out[i, j] = g[0, j + i * fn]
    return out.T


if __name__ == '__main__':
    res = richardson_jacobian(
        lambda x: np.array([x[0] * x[3], x[2] ** 2]),
        np.array([2, 3, 4, 5]),
    )
    print(res)
    res = simple_jacobian(
        lambda x: np.array([x[0] * x[3], x[2] ** 2]),
        np.array([2, 3, 4, 5]),
    )
    print(res)
