from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Tree:
    length: int
    max_rank: int
    root: Optional['Node'] = None

    def eval(self):
        return self.root.eval(self.length)

    def draw(self):
        def tree_depth(root):
            if isinstance(root, Leaf):
                return 0
            else:
                max_value = 1 + max(tree_depth(root.left_up), tree_depth(root.right_up), tree_depth(root.left_low),
                                    tree_depth(root.right_low))
                return max_value

        def get_draw_sizes(root):
            depth = tree_depth(root)

            draw_sizes = []

            current_size = self.length // (2 ** depth)

            for _ in range(depth + 1):
                draw_sizes.append(current_size)

                current_size = 2 * draw_sizes[-1] + 1

            return draw_sizes[::-1]

        draw_sizes = get_draw_sizes(self.root)

        image_matrix = np.ones((draw_sizes[0], draw_sizes[0]))
        self.root.draw(image_matrix, 0, 0, draw_sizes)
        return image_matrix*255

    def multiply_by_vector(self, vector):
        vector = np.squeeze(vector)
        n = vector.shape[0]
        if len(vector.shape) > 2:
            raise ValueError("Vector should be one dimensional")
        if n != self.length:
            raise ValueError(f"Vector length should be {self.length}")
        vector = self.root.multiply_by_vector(vector)
        return np.squeeze(vector)


class Node(ABC):
    @abstractmethod
    def eval(self, length):
        pass

    @abstractmethod
    def draw(self, image_matrix, left, up, draw_sizes, depth=0):
        pass

    @abstractmethod
    def multiply_by_vector(self, vector):
        pass


@dataclass
class InternalNode(Node):
    left_up: Node
    right_up: Node
    left_low: Node
    right_low: Node

    def eval(self, length):
        matrix = np.zeros((length, length))
        length //= 2
        matrix[:length, :length] = self.left_up.eval(length)
        matrix[:length, length:] = self.right_up.eval(length)
        matrix[length:, :length] = self.left_low.eval(length)
        matrix[length:, length:] = self.right_low.eval(length)
        return matrix

    def draw(self, im_mat, l, u, sizes, depth=0):
        k = sizes[depth] // 2

        im_mat[u: u + sizes[depth], l + k] = 0
        im_mat[u + k, l: l + sizes[depth]] = 0

        self.left_up.draw(im_mat, l, u, sizes, depth + 1)
        self.right_up.draw(im_mat, l+k+1, u, sizes, depth + 1)
        self.left_low.draw(im_mat, l, u+k+1, sizes, depth + 1)
        self.right_low.draw(im_mat, l+k+1, u+k+1, sizes, depth + 1)

    def multiply_by_vector(self, vector):
        half1, half2 = np.split(vector, 2)
        first = self.left_up.multiply_by_vector(half1) + self.right_up.multiply_by_vector(half2)
        second = self.left_low.multiply_by_vector(half1) + self.right_low.multiply_by_vector(half2)
        return np.vstack((first, second))


@dataclass
class Leaf(Node):
    U: Optional[np.array] = None
    V: Optional[np.array] = None
    zeros: bool = False

    def __post_init__(self):
        if self.U is not None and len(self.U.shape) == 1:
            self.U = self.U.reshape(-1, 1)
        if self.V is not None and len(self.V.shape) == 1:
            self.V = self.V.reshape(-1, 1)

    def eval(self, length):
        if self.zeros:
            return 0
        return self.U @ self.V

    def draw(self, image_matrix, left, up, sizes, depth=0):
        if not self.zeros:
            length = sizes[depth]

            k = self.V.shape[0]
            image_matrix[up:up+length, left:left+k] = 0
            image_matrix[up:up+k, left:left+length] = 0

    def multiply_by_vector(self, vector):
        if self.zeros:
            return np.zeros(vector.shape[0]).reshape(-1, 1)
        else:
            return self.U @ np.dot(self.V, vector).reshape(-1, 1)




