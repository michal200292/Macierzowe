import numpy as np
from Matrix_inverse import inverse


def lu(A):
    m, n = A.shape

    if m == 1 and n == 1:
        return (np.ones((1, 1), dtype=np.double), A.copy())

    A11 = A[:m // 2, :n // 2]
    A12 = A[:m // 2, n // 2:]
    A21 = A[m // 2:, :n // 2]
    A22 = A[m // 2:, n // 2:]

    L11, U11 = lu(A11)

    U11_inv = inverse(U11)

    L21 = A21 @ U11_inv

    L11_inv = inverse(L11)

    U12 = L11_inv @ A12

    L22 = A22 - A21 @ U11_inv @ L11_inv @ A12

    L22, U22 = lu(L22)

    L = np.zeros((m, n), dtype=np.double)
    U = np.zeros((m, n), dtype=np.double)

    L[:m // 2, : n // 2] = L11
    L[m // 2:, : n // 2] = L21
    L[m // 2:, n // 2:] = L22

    U[:m // 2, : n // 2] = U11
    U[:m // 2, n // 2:] = U12
    U[m // 2:, n // 2:] = U22

    return L, U