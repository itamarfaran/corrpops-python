import numpy as np


def corr_calc(
        m: np.ndarray,
        n: int,
        order_vector_i: np.ndarray,
        order_vector_j: np.ndarray,
) -> np.ndarray:
    out = np.empty((n, n), float)

    for row in range(0, n):
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


if __name__ == '__main__':
    p = 4
    matrix = np.full((p, p), 0.25) + 0.75 * np.eye(p, dtype=float)
    m = int(0.5 * p * (p - 1))

    # Generate order_vecti
    order_vecti = np.concatenate([np.repeat(i, p - i - 1) for i in range(p)])

    # Generate order_vectj
    order_vectj = np.concatenate([np.arange(i + 1, p) for i in range(p)])

    result = corr_calc(matrix, m, order_vecti, order_vectj)
    result = result + result.T - np.diag(np.diag(result))
    print(result)
