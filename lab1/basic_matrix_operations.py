import numpy as np


def multiply(A, B):
    m, n = A.shape
    k, l = B.shape

    if n != k:
        return None

    C = np.zeros((m, l), dtype=np.double)

    for row in range(m):
        for col in range(l):
            value = 0.0

            for curr in range(n):
                value += A[row][curr] * B[curr][col]

            C[row][col] = value

    return C

