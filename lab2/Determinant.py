import numpy as np
from LU import lu


def determinant(A):
    L, U = lu(A)

    det = np.double(1)

    for i in range(A.shape[0]):
        det *= U[i][i]

    return det