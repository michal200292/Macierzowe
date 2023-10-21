import numpy as np
from basic_matrix_operations import multiply


def binet(A, B):
    m, n = A.shape
    k, l = B.shape

    if n != k:
        return None

    if m == 1 or n == 1 or l == 1:
        return multiply(A, B)

    m, n, l = m // 2, n // 2, l // 2

    A11 = A[:m, :n]
    A12 = A[:m, n:]
    A21 = A[m:, :n]
    A22 = A[m:, n:]

    B11 = B[:n, :l]
    B12 = B[:n, l:]
    B21 = B[n:, :l]
    B22 = B[n:, l:]

    C11 = binet(A11, B11) + binet(A12, B21)
    C12 = binet(A11, B12) + binet(A12, B22)
    C21 = binet(A21, B11) + binet(A22, B21)
    C22 = binet(A21, B12) + binet(A22, B22)

    C = np.zeros((2*m, 2*l), dtype=np.double)

    C[:m,:l] = C11
    C[:m, l:] = C12
    C[m:, :l] = C21
    C[m:, l:] = C22

    return C

