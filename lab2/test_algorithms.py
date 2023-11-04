import numpy as np
from Matrix_inverse import inverse
from LU import lu
from Determinant import determinant


def norm2(matrix):
    return np.square(np.linalg.norm(matrix))


def test_determinant(matrix, eps=1e-9):
    det, _ = determinant(matrix)
    numpy_det = np.linalg.det(matrix)
    if abs(det - numpy_det) > eps:
        raise AssertionError("Something wrong?!")
    else:
        print("Test passed")


def test_inverse(matrix, eps=1e-9):
    inverted, _ = inverse(matrix)
    np_inverted = np.linalg.inv(matrix)
    if norm2(inverted - np_inverted) > eps:
        raise AssertionError("Something wrong?!")
    else:
        print("Test passed")


def test_lu(matrix, eps=1e-9):
    n = matrix.shape[0]
    lower, upper, _ = lu(matrix)
    if norm2(np.diag(lower) - np.ones(n)) > eps:
        raise AssertionError("Elements on a diagonal of lower matrix are not ones")

    if norm2(lower - np.tril(lower)) > eps:
        raise AssertionError("L-matrix is not lower triangular")

    if norm2(upper - np.triu(upper)) > eps:
        raise AssertionError("U-matrix is not upper triangular")

    if norm2(lower @ upper - matrix) > eps:
        raise AssertionError("Something wrong?!")
    else:
        print("Test passed")

