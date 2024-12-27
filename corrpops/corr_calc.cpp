#include <vector>
#include <cmath>


using Matrix = std::vector< std::vector<double> >;


extern "C" Matrix corr_calc(
    const Matrix& m,
    int n,
    const std::vector<int>& order_vector_i,
    const std::vector<int>& order_vector_j
) {
    // Initialize the output matrix with the size n x n filled with zeros.
    Matrix out(n, std::vector<double>(n, 0.0));

    for (int row = 0; row < n; row++) {
        for (int col = row; col < n; col++) {
            int i = order_vector_i[row];
            int j = order_vector_j[row];
            int k = order_vector_i[col];
            int l = order_vector_j[col];

            double m_ij = m[i][j];
            double m_kl = m[k][l];
            double m_ik = m[i][k];
            double m_il = m[i][l];
            double m_jk = m[j][k];
            double m_jl = m[j][l];

            out[row][col] = (
                (m_ij * m_kl / 2) * (std::pow(m_ik, 2) + std::pow(m_il, 2) + std::pow(m_jk, 2) + std::pow(m_jl, 2))
                - m_ij * (m_ik * m_il + m_jk * m_jl)
                - m_kl * (m_ik * m_jk + m_il * m_jl)
                + (m_ik * m_jl + m_il * m_jk)
            );
        }
    }
    return out;
}
