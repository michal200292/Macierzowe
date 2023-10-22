import numpy as np


def multiply(A, B):
    m, n = A.shape
    k, l = B.shape

    if n != k:
        raise ValueError("Incorrect matrix dimensions")

    C = np.zeros((m, l), dtype=np.double)

    for row in range(m):
        for col in range(l):
            C[row][col] = A[row][0] * B[0][col]
            for curr in range(1, n):
                C[row][col] += A[row][curr] * B[curr][col]

    return C, n*m*(l-1), n*m*l

