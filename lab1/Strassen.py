import numpy as np
from basic_matrix_operations import multiply


def strassen(A, B):
    m, n = A.shape
    k, l = B.shape

    if n != k:
        raise ValueError("Incorrect matrix dimensions")

    if m % 2 == 1 or n % 2 == 1 or l % 2 == 1:
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

    P1 = strassen(A11 + A22, B11 + B22)
    P2 = strassen(A21 + A22, B11)
    P3 = strassen(A11, B12 - B22)
    P4 = strassen(A22, B21 - B11)
    P5 = strassen(A11 + A12, B22)
    P6 = strassen(A21 - A11, B11 + B12)
    P7 = strassen(A12 - A22, B21 + B22)

    counter = P1[1] + P2[1] + P3[1] + P4[1] + P5[1] + P6[1] + P7[1]
    counter.add += 18 * m * l

    C11 = P1[0] + P4[0] - P5[0] + P7[0]
    C12 = P3[0] + P5[0]
    C21 = P2[0] + P4[0]
    C22 = P1[0] - P2[0] + P3[0] + P6[0]

    C = np.zeros((2*m, 2*l), dtype=np.double)

    C[:m, :l] = C11
    C[:m, l:] = C12
    C[m:, :l] = C21
    C[m:, l:] = C22

    return C, counter

