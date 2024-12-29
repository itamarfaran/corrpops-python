import ctypes
import numpy as np


clib = np.ctypeslib.load_library("corr_calc.so", "../")
matrix = np.eye(2, dtype=float)
p = matrix.shape[0]

# Generate order_vecti
order_vecti = np.concatenate([np.repeat(i, p - i - 1) for i in range(p)])

# Generate order_vectj
order_vectj = np.concatenate([np.arange(i + 1, p) for i in range(p)])

corr_calc = clib.corr_calc
corr_calc.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
]

# value = 5
results = corr_calc(
    matrix,
    int(p * (p - 1) / 2),
    order_vecti,
    order_vectj,
)
print(results)
