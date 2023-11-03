import numpy as np


def inverse(A):
    m, n = A.shape

    if m == 1 and n == 1:
        return np.array([[1 / A[0][0]]])

    A11 = A[: m // 2, :n // 2]
    A12 = A[: m // 2, n // 2:]
    A21 = A[m // 2:, :n // 2]
    A22 = A[m // 2:, n // 2:]

    A11_inv = inverse(A11)

    S22 = A22 - (A21 @ A11_inv @ A12)

    S22_inv = inverse(S22)

    C11 = A11_inv + (A11_inv @ A12 @ S22_inv @ A21 @ A11_inv)
    C12 = -A11_inv @ A12 @ S22_inv
    C21 = -S22_inv @ A21 @ A11_inv
    C22 = S22_inv

    C = np.zeros((m, n), dtype=np.double)

    C[: m // 2, :n // 2] = C11
    C[: m // 2, n // 2:] = C12
    C[m // 2:, :n // 2] = C21
    C[m // 2:, n // 2:] = C22

    return C