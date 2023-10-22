import numpy as np
from basic_matrix_operations import multiply


def binet(A, B):
    m, n = A.shape
    k, l = B.shape

    if n != k:
        raise ValueError("Incorrect matrix dimensions")

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

    P1 = binet(A11, B11)
    P2 = binet(A12, B21)
    P3 = binet(A11, B12)
    P4 = binet(A12, B22)
    P5 = binet(A21, B11)
    P6 = binet(A22, B21)
    P7 = binet(A21, B12)
    P8 = binet(A22, B22)

    count_mul = P1[2] + P2[2] + P3[2] + P4[2] + P5[2] + P6[2] + P7[2] + P8[2]
    count_add = P1[1] + P2[1] + P3[1] + P4[1] + P5[1] + P6[1] + P7[1] + P8[1]
    count_add += 4 * m * l

    C11 = P1[0] + P2[0]
    C12 = P3[0] + P4[0]
    C21 = P5[0] + P6[0]
    C22 = P7[0] + P8[0]

    C = np.zeros((2*m, 2*l), dtype=np.double)
    C[:m, :l] = C11
    C[:m, l:] = C12
    C[m:, :l] = C21
    C[m:, l:] = C22
    return C, count_add, count_mul

