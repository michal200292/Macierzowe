from abc import ABC, abstractmethod
import numpy as np


class Tree:
    def __init__(self, length, root=None):
        self.length = length
        self.root = root

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


class Node(ABC):
    @abstractmethod
    def eval(self, length):
        pass

    @abstractmethod
    def draw(self, image_matrix, left, up, draw_sizes, depth):
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

    def draw(self, im_mat, l, u, sizes, depth=0):
        k = sizes[depth] // 2

        im_mat[u: u + sizes[depth], l + k] = 0 #Siatka
        im_mat[u + k, l: l + sizes[depth]] = 0 #

        self.left_up.draw(im_mat, l, u, sizes, depth + 1)
        self.right_up.draw(im_mat, l+k+1, u, sizes, depth + 1)
        self.left_low.draw(im_mat, l, u+k+1, sizes, depth + 1)
        self.right_low.draw(im_mat, l+k+1, u+k+1, sizes, depth + 1)


class Leaf(Node):
    def __init__(self, U=None, V=None, zeros=False):
        self.U = U
        self.V = V
        self.zeros = zeros

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




