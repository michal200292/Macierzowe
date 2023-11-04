import numpy as np
from Arithmetic_counter import Counter


def inverse(A):
    m, n = A.shape

    if m != n:
        raise ValueError("Cannot compute inverse of a non-square matrix")

    if m == 1 and n == 1:
        return np.array([[1 / A[0][0]]]), Counter(0, 0, 1)

    A11 = A[: m // 2, :n // 2]
    A12 = A[: m // 2, n // 2:]
    A21 = A[m // 2:, :n // 2]
    A22 = A[m // 2:, n // 2:]

    A11_inv, count = inverse(A11)

    temp1 = A21 @ A11_inv
    S22 = A22 - (temp1 @ A12)

    S22_inv, count2 = inverse(S22)
    count += count2

    temp2 = A11_inv @ A12 @ S22_inv

    C11 = A11_inv + (temp2 @ temp1)
    C12 = -temp2
    C21 = -S22_inv @ temp1
    C22 = S22_inv

    C = np.zeros((m, n), dtype=np.double)

    C[: m // 2, :n // 2] = C11
    C[: m // 2, n // 2:] = C12
    C[m // 2:, :n // 2] = C21
    C[m // 2:, n // 2:] = C22

    m //= 2
    count.add += 6*m*m*(m - 1) + 2*m*m
    count.mul += 6*m*m*m
    return C, count

