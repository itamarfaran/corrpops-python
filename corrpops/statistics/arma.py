import numpy as np


def is_invertible_arma(coefficients, tol: float = 1e-03) -> bool:
    coefficients = np.asarray(coefficients)
    degrees = 1 + np.arange(coefficients.size)
    x = np.linspace(-1, 1, int(2 / tol))[:, None]
    results = 1 - np.sum(coefficients * x**degrees, axis=1)
    return bool((results > 0).all() or (results < 0).all())
