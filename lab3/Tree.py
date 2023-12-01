from abc import ABC, abstractmethod
import numpy as np


class Tree:
    def __init__(self, length, root=None):
        self.length = length
        self.root = root

    def eval(self):
        return self.root.eval(self.length)

    def draw(self):
        image_matrix = np.ones((self.length, self.length))
        self.root.draw(image_matrix, 0, 0, self.length)
        return image_matrix*255


class Node(ABC):
    @abstractmethod
    def eval(self, length):
        pass

    @abstractmethod
    def draw(self, image_matrix, left, up, length):
        pass


class InternalNode(Node):
    def __init__(self, left_up, right_up, left_low, right_low):
        self.left_up = left_up
        self.right_up = right_up
        self.left_low = left_low
        self.right_low = right_low

    def eval(self, length):
        matrix = np.zeros((length, length))
        length //= 2
        matrix[:length, :length] = self.left_up.eval(length)
        matrix[:length, length:] = self.right_up.eval(length)
        matrix[length:, :length] = self.left_low.eval(length)
        matrix[length:, length:] = self.right_low.eval(length)
        return matrix

    def draw(self, im_mat, l, u, lg):
        lg //= 2
        self.left_up.draw(im_mat, l, u, lg)
        self.right_up.draw(im_mat, l+lg, u, lg)
        self.left_low.draw(im_mat, l, u+lg, lg)
        self.right_low.draw(im_mat, l+lg, u+lg, lg)


class Leaf(Node):
    def __init__(self, U=None, V=None, zeros=False):
        self.U = U
        self.V = V
        self.zeros = zeros

    def eval(self, length):
        if self.zeros:
            return 0
        return self.U @ self.V

    def draw(self, image_matrix, left, up, length):
        if not self.zeros:
            k = self.V.shape[0]
            image_matrix[up:up+length, left:left+k] = 0
            image_matrix[up:up+k, left:left+length] = 0




