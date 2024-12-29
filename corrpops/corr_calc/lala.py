import numpy as np
import pyximport
pyximport.install()

from corrpops.corr_calc_cy import corr_calc

p = 4
matrix = np.full((p, p), 0.25) + 0.75 * np.eye(p, dtype=float)
m = int(0.5 * p * (p - 1))

# Generate order_vecti
order_vecti = np.concatenate([np.repeat(i, p - i - 1) for i in range(p)])

# Generate order_vectj
order_vectj = np.concatenate([np.arange(i + 1, p) for i in range(p)])

result = corr_calc_cy(matrix, m, order_vecti, order_vectj)
result = result + result.T - np.diag(np.diag(result))
print(result)
