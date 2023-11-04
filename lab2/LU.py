import numpy as np
from Matrix_inverse import inverse
from Arithmetic_counter import Counter


def lu(A):
    m, n = A.shape

    if m != n:
        raise ValueError("Cannot compute LU-decomposition of a non-square matrix")

    if m == 1 and n == 1:
        return np.ones((1, 1), dtype=np.double), A.copy(), Counter()

    A11 = A[:m // 2, :n // 2]
    A12 = A[:m // 2, n // 2:]
    A21 = A[m // 2:, :n // 2]
    A22 = A[m // 2:, n // 2:]

    L11, U11, count = lu(A11)
    U11_inv, count2 = inverse(U11)

    count += count2

    temp1 = A21 @ U11_inv
    L21 = temp1
    L11_inv, count2 = inverse(L11)

    count += count2

    temp2 = L11_inv @ A12
    U12 = temp2
    L22, U22, count2 = lu(A22 - temp1 @ temp2)

    count += count2

    L = np.zeros((m, n), dtype=np.double)
    U = np.zeros((m, n), dtype=np.double)

    L[:m // 2, : n // 2] = L11
    L[m // 2:, : n // 2] = L21
    L[m // 2:, n // 2:] = L22

    U[:m // 2, : n // 2] = U11
    U[:m // 2, n // 2:] = U12
    U[m // 2:, n // 2:] = U22

    m //= 2
    count.add += 3*m*m*(m - 1) + m*m
    count.mul += 3*m*m*m

    return L, U, count

