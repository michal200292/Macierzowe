from scipy.sparse.linalg import svds
from scipy.linalg import svd
from Tree import Tree, InternalNode, Leaf
import numpy as np


def truncated_svd(A, k):
    if not 0 < k < A.shape[0] + 1:
        raise ValueError("k should between 0 and A.shape[0] + 1")
    if k == A.shape[0]:
        U, s, V = svd(A)
    else:
        U, s, V = svds(A, k)
        U, s, V = U[::, ::-1], s[::-1], V[::-1]
    return U, s, V


def calculate_singular_values(matrix):
    return svd(matrix)[1]


def power_of_two(x):
    while x % 2:
        x //= 2
    return x != 1


def compress(matrix, min_value, max_rank, length):
    eps = 1e-10
    if length == 1:
        if isinstance(matrix, float):
            matrix = np.array([[matrix]])
        if len(matrix.shape) == 1:
            matrix = matrix.reshape((-1, 1))
        return Leaf(U=matrix, V=np.array([1])) if abs(matrix[0, 0]) > eps else Leaf(zeros=True)
    else:
        if length <= max_rank + 1:
            max_rank = length - 1
        U, s, V = truncated_svd(matrix, k=max_rank + 1)
        if np.abs(s[-1]) < min_value + eps:
            s_values = s[np.abs(s) >= min_value + eps]
            k = s_values.shape[0]
            if k == 0:
                return Leaf(zeros=True)
            return Leaf(U=U[::, :k] @ np.diag(s_values), V=V[:k])

        length //= 2
        node = InternalNode(
            left_up=compress(matrix[:length, :length], min_value, max_rank, length),
            right_up=compress(matrix[:length, length:], min_value, max_rank, length),
            left_low=compress(matrix[length:, :length], min_value, max_rank, length),
            right_low=compress(matrix[length:, length:], min_value, max_rank,  length)
        )
        return node


def compress_matrix(matrix, min_value, max_rank):
    n, m = matrix.shape
    if max_rank > n - 1:
        raise ValueError("Maximum rank should be strictly less than matrix dimension")
    if n != m:
        raise ValueError("Matrix should be square")
    if not power_of_two(n):
        raise ValueError("Matrix dimension should be power of two")

    return Tree(n, max_rank, compress(matrix, abs(min_value), max_rank, n))

