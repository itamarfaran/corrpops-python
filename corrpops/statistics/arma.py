import numpy as np
import numpy.typing as npt


def is_invertible_arma(coefficients: npt.ArrayLike, tol: float = 1e-03) -> bool:
    coefficients = np.atleast_1d(coefficients)

    if coefficients.ndim > 1:
        raise ValueError(
            f"expected arma coefficients to be 1d, "
            f"got {coefficients.ndim}d instead",
        )

    degrees = 1 + np.arange(coefficients.size)
    x = np.linspace(-1, 1, int(2 / tol))[:, None]
    results = 1 - np.sum(coefficients * x**degrees, axis=1)
    return bool((results > 0).all() or (results < 0).all())
