import numpy as np
from Matrix_inverse import inverse
from Arithmetic_counter import Counter
from lab1.Binet import binet


def lu(A, mul_func=binet):
    m, n = A.shape

    if m != n:
        raise ValueError("Cannot compute LU-decomposition of a non-square matrix")

    if m == 1 and n == 1:
        return np.ones((1, 1), dtype=np.double), A.copy(), Counter()

    m, n = m // 2, n // 2
    A11 = A[:m, :n]
    A12 = A[:m, n:]
    A21 = A[m:, :n]
    A22 = A[m:, n:]

    L11, U11, count = lu(A11)
    U11_inv, count2 = inverse(U11)
    count += count2

    temp1, count2 = mul_func(A21, U11_inv)
    count += count2

    L21 = temp1
    L11_inv, count2 = inverse(L11)
    count += count2

    temp2, count2 = mul_func(L11_inv, A12)
    count += count2

    U12 = temp2
    mat, count2 = mul_func(temp1, temp2)
    count += count2
    L22, U22, count2 = lu(A22 - mat)
    count += count2

    L = np.zeros((2*m, 2*n), dtype=np.double)
    U = np.zeros((2*m, 2*n), dtype=np.double)

    L[:m, : n] = L11
    L[m:, : n] = L21
    L[m:, n:] = L22

    U[:m, : n] = U11
    U[:m, n:] = U12
    U[m:, n:] = U22

    m //= 2
    count.add += m*m

    return L, U, count

