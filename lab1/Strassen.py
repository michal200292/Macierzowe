import numpy as np
from basic_matrix_operations import multiply


def strassen(A, B):
    m, n = A.shape
    k, l = B.shape

    if n != k:
        return None

    if m % 2 == 1 or n % 2 == 1 or k % 2 == 1 or l % 2 == 1:
        return multiply(A, B)

    m, n, k, l = m // 2, n // 2, k // 2, l // 2

    A11 = A[:m, :n]
    A12 = A[:m, n:]
    A21 = A[m:, :n]
    A22 = A[m:, n:]

    B11 = B[:k, :l]
    B12 = B[:k, l:]
    B21 = B[k:, :l]
    B22 = B[k:, l:]

    P1 = strassen(A11 + A22, B11 + B22)
    P2 = strassen(A21 + A22, B11)
    P3 = strassen(A11, B12 - B22)
    P4 = strassen(A22, B21 - B11)
    P5 = strassen(A11 + A12, B22)
    P6 = strassen(A21 - A11, B11 + B12)
    P7 = strassen(A12 - A22, B21 + B22)

    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6

    C = np.zeros((2*m, 2*l), dtype=np.double)

    C[:m, :l] = C11
    C[:m, l:] = C12
    C[m:, :l] = C21
    C[m:, l:] = C22

    return C

