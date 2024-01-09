import random
import numpy as np


def build_matrix(k):
    n = 2 ** k
    n_2 = n ** 2

    matrix = np.zeros((n ** 3, n ** 3))

    for vertex in range(matrix.shape[0]):
        level = vertex // n_2
        rest = vertex % n_2
        row = rest // n
        col = rest % n

        matrix[vertex][vertex] = random.random()
        if level > 0:
            top_level_neighbor = vertex - n_2
            matrix[vertex][top_level_neighbor] = random.random()

        if level < n - 1:
            bottom_level_neighbor = vertex + n_2
            matrix[vertex][bottom_level_neighbor] = random.random()

        if row > 0:
            top_neighbor = vertex - n
            matrix[vertex][top_neighbor] = random.random()

        if row < n - 1:
            bottom_neighbor = vertex + n
            matrix[vertex][bottom_neighbor] = random.random()

        if col > 0:
            left_neighbor = vertex - 1
            matrix[vertex][left_neighbor] = random.random()

        if col < n - 1:
            right_neighbor = vertex + 1
            matrix[vertex][right_neighbor] = random.random()

    return matrix
