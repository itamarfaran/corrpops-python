from functools import cache
from typing import Any, Tuple

import numpy as np
from scipy import stats


def from_eigh(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.linalg.multi_dot((v, np.diag(w), v.T))


@cache
def v_w(p: int, random_state: Any = 0) -> Tuple[np.ndarray, np.ndarray]:
    v = stats.ortho_group.rvs(p, random_state=random_state)
    w = np.arange(p)
    return v, w
