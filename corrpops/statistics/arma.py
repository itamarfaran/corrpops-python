import numpy as np


def is_invertible_arma(coefficients, tol=1e-03):
    coefficients = np.asarray(coefficients)
    degrees = 1 + np.arange(coefficients.size)
    x = np.linspace(-1, 1, int(2 / tol))[:, None]
    results = 1 - np.sum(coefficients * x**degrees, axis=1)
    return (results > 0).all() or (results < 0).all()
