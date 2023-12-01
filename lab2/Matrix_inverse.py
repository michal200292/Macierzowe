import numpy as np
from Arithmetic_counter import Counter
from lab1.Binet import binet


def inverse(A, mul_func=binet):
    m, n = A.shape

    if m != n:
        raise ValueError("Cannot compute inverse of a non-square matrix")

    if m == 1 and n == 1:
        return np.array([[1 / A[0][0]]]), Counter(0, 0, 1)

    m, n = m // 2, n // 2

    A11 = A[: m, :n]
    A12 = A[: m, n:]
    A21 = A[m:, :n]
    A22 = A[m:, n:]

    A11_inv, count = inverse(A11, mul_func)

    temp1, count2 = mul_func(A21, A11_inv)
    count += count2
    mat, count2 = mul_func(temp1, A12)
    count += count2
    S22 = A22 - mat
    S22_inv, count2 = inverse(S22, mul_func)
    count += count2

    mat, count2 = mul_func(A11_inv, A12)
    count += count2

    temp2, count2 = mul_func(mat, S22_inv)
    count += count2

    mat, count2 = mul_func(temp2, temp1)
    count += count2
    C11 = A11_inv + mat
    C12 = -temp2
    C21, count2 = mul_func(-S22_inv, temp1)
    count += count2
    C22 = S22_inv

    C = np.zeros((2*m, 2*n), dtype=np.double)

    C[: m, :n] = C11
    C[: m, n:] = C12
    C[m:, :n] = C21
    C[m:, n:] = C22

    count.add += 2*m*m
    return C, count

