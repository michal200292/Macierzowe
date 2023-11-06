import numpy as np
from LU import lu
from lab1.Binet import binet


def determinant(A, mul_func=binet):
    L, U, count = lu(A, mul_func)
    det = np.double(1)
    for i in range(A.shape[0]):
        det *= U[i][i]

    count.mul += A.shape[0]
    return det, count
