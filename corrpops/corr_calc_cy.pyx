import numpy as np
cimport numpy as cnp


# Declare the function signature with type annotations
def corr_calc(
        cnp.ndarray[cnp.float64_t] m,
        int n,
        cnp.ndarray[cnp.int32_t] order_vector_i,
        cnp.ndarray[cnp.int32_t] order_vector_j
) -> cnp.ndarray[cnp.float64_t]:
    # Allocate the output array
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.empty((n, n), dtype=np.float64)

    # Declare variables used in the loop
    cdef int row, col, i, j, k, l
    cdef double m_ij, m_kl, m_ik, m_il, m_jk, m_jl

    # Loop through the indices
    for row in range(n):
        for col in range(row, n):
            i = order_vector_i[row]
            j = order_vector_j[row]
            k = order_vector_i[col]
            l = order_vector_j[col]

            m_ij = m[i, j]
            m_kl = m[k, l]
            m_ik = m[i, k]
            m_il = m[i, l]
            m_jk = m[j, k]
            m_jl = m[j, l]

            out[row, col] = (
                (m_ij * m_kl / 2) * (m_ik ** 2 + m_il ** 2 + m_jk ** 2 + m_jl ** 2)
                - m_ij * (m_ik * m_il + m_jk * m_jl)
                - m_kl * (m_ik * m_jk + m_il * m_jl)
                + (m_ik * m_jl + m_il * m_jk)
            )

    return out
