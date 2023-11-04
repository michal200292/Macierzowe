import numpy as np
from LU import lu


def determinant(A):
    L, U, count = lu(A)
    det = np.double(1)
    for i in range(A.shape[0]):
        det *= U[i][i]

    count.mul += A.shape[0]
    return det, count
