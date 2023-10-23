import numpy as np
from numpy.typing import NDArray
from basic_matrix_operations import multiply


def split_matrix(A, horizontal_parts, vertical_parts):
    m, n = A.shape

    a = [[NDArray for _ in range(vertical_parts)] for _ in range(horizontal_parts)]

    height = m // horizontal_parts
    width = n // vertical_parts

    for i in range(horizontal_parts):
        for j in range(vertical_parts):
            a[i][j] = A[height * i: height * (i + 1), width * j: width * (j + 1)]

    return a


def ai(A, B):
    m, n = A.shape
    k, l = B.shape

    if n != k:
        raise ValueError("Incorrect matrix dimensions")

    if not (m % 4 == 0 and n % 5 == 0 and l % 5 == 0):
        return multiply(A, B)

    a = split_matrix(A, 4, 5)
    b = split_matrix(B, 5, 5)

    height = m // 4
    width = l // 5

    h = [NDArray for _ in range(76)]
    h[0] = ai(a[2][1], -b[1][0] - b[1][4] - b[2][0])
    h[1] = ai(a[1][1] + a[1][4] - a[2][4], -b[1][4] - b[4][0])
    h[2] = ai(-a[2][0] - a[3][0] + a[3][1], -b[0][0] + b[1][4])
    h[3] = ai(a[0][1] + a[0][3] + a[2][3], -b[1][4] - b[3][0])
    h[4] = ai(a[0][4] + a[1][1] + a[1][4], -b[1][3] + b[4][0])
    h[5] = ai(-a[1][1] - a[1][4] - a[3][4], b[1][2] + b[4][0])
    h[6] = ai(-a[0][0] + a[3][0] - a[3][1], b[0][0] + b[1][3])
    h[7] = ai(a[2][1] - a[2][2] - a[3][2], -b[1][2] + b[2][0])
    h[8] = ai(-a[0][1] - a[0][3] + a[3][3], b[1][2] + b[3][0])
    h[9] = ai(a[1][1] + a[1][4], b[4][0])
    h[10] = ai(-a[1][0] - a[3][0] + a[3][1], -b[0][0] + b[1][1])
    h[11] = ai(a[3][0] - a[3][1], b[0][0])
    h[12] = ai(a[0][1] + a[0][3] + a[1][3], b[1][1] + b[3][0])
    h[13] = ai(a[0][2] - a[2][1] + a[2][2], b[1][3] + b[2][0])
    h[14] = ai(-a[0][1] - a[0][3], b[3][0])
    h[15] = ai(-a[2][1] + a[2][2], b[2][0])
    h[16] = ai(a[0][1] + a[0][3] - a[1][0] + a[1][1] - a[1][2] + a[1][3] - a[2][1] + a[2][2] - a[3][0] + a[3][1], b[1][1])
    h[17] = ai(a[1][0], b[0][0] + b[0][1] + b[4][1])
    h[18] = ai(-a[1][2], b[2][0] + b[2][1] + b[4][1])
    h[19] = ai(-a[0][4] + a[1][0] + a[1][2] - a[1][4], -b[0][0] - b[0][1] + b[0][3] - b[4][1])
    h[20] = ai(a[1][0] + a[1][2] - a[1][4], b[4][1])
    h[21] = ai(a[0][2] - a[0][3] - a[1][3], b[0][0] + b[0][1] - b[0][3] - b[2][0] - b[2][1] + b[2][3] + b[3][3])
    h[22] = ai(a[0][2], -b[2][0] + b[2][3] + b[3][3])
    h[23] = ai(a[0][4], -b[3][3] - b[4][0] + b[4][3])
    h[24] = ai(-a[0][0], b[0][0] - b[0][3])
    h[25] = ai(-a[0][2] + a[0][3] + a[0][4], b[3][3])
    h[26] = ai(a[0][2] - a[2][0] + a[2][2], b[0][0] - b[0][3] + b[0][4] + b[2][4])
    h[27] = ai(-a[2][3], -b[2][4] - b[3][0] - b[3][4])
    h[28] = ai(a[2][0], b[0][0] + b[0][4] + b[2][4])
    h[29] = ai(a[2][0] - a[2][2] + a[2][3], b[2][4])
    h[30] = ai(-a[0][3] - a[0][4] - a[2][3], -b[3][3] - b[4][0] + b[4][3] - b[4][4])
    h[31] = ai(a[1][0] + a[3][0] + a[3][3], b[0][2] - b[3][0] - b[3][1] - b[3][2])
    h[32] = ai(a[3][2], -b[2][0] - b[2][2])
    h[33] = ai(a[3][3], -b[0][2] + b[3][0] + b[3][2])
    h[34] = ai(-a[3][4], b[0][2] + b[4][0] + b[4][2])
    h[35] = ai(a[1][2] - a[1][4] - a[3][4], b[2][0] + b[2][1] + b[2][2] + b[4][1])
    h[36] = ai(-a[3][0] - a[3][3] + a[3][4], b[0][2])
    h[37] = ai(-a[1][2] - a[2][0] + a[2][2] - a[2][3], b[2][4] + b[3][0] + b[3][1] + b[3][4])
    h[38] = ai(-a[2][0] - a[3][0] - a[3][3] + a[3][4], b[0][2] + b[4][0] + b[4][2] + b[4][4])
    h[39] = ai(-a[0][2] + a[0][3] + a[0][4] - a[3][3], -b[2][0] - b[2][2] + b[2][3] + b[3][3])
    h[40] = ai(-a[0][0] + a[3][0] - a[3][4], b[0][2] + b[2][0] + b[2][2] - b[2][3] + b[4][0] + b[4][2] - b[4][3])
    h[41] = ai(-a[1][0] + a[1][4] - a[2][4], -b[0][0] - b[0][1] - b[0][4] + b[3][0] + b[3][1] + b[3][4] - b[4][1])
    h[42] = ai(a[1][3], b[3][0] + b[3][1])
    h[43] = ai(a[1][2] + a[2][1] - a[2][2], b[1][1] - b[2][0])
    h[44] = ai(-a[2][2] + a[2][3] - a[3][2], b[2][4] + b[3][0] + b[3][2] + b[3][4] + b[4][0] + b[4][2] + b[4][4])
    h[45] = ai(-a[2][4], -b[4][0] - b[4][4])
    h[46] = ai(a[1][0] - a[1][4] - a[2][0] + a[2][4], b[0][0] + b[0][1] + b[0][4] - b[3][0] - b[3][1] - b[3][4])
    h[47] = ai(-a[1][2] + a[2][2], b[1][1] + b[2][1] + b[2][4] + b[3][0] + b[3][1] + b[3][4])
    h[48] = ai(-a[0][0] - a[0][2] + a[0][3] + a[0][4] - a[1][0] - a[1][2] + a[1][3] + a[1][4], -b[0][0] - b[0][1] + b[0][3])
    h[49] = ai(-a[0][3] - a[1][3], b[1][1] - b[2][0] - b[2][1] + b[2][3] - b[3][1] + b[3][3])
    h[50] = ai(a[1][1], b[1][0] + b[1][1] - b[4][0])
    h[51] = ai(a[3][1], b[0][0] + b[1][0] + b[1][2])
    h[52] = ai(-a[0][1], -b[1][0] + b[1][3] + b[3][0])
    h[53] = ai(a[0][1] + a[0][3] - a[1][1] - a[1][4] - a[2][1] + a[2][2] - a[3][1] + a[3][2] - a[3][3] - a[3][4], b[1][2])
    h[54] = ai(a[0][3] - a[3][3], -b[1][2] + b[2][0] + b[2][2] - b[2][3] + b[3][2] - b[3][3])
    h[55] = ai(a[0][0] - a[0][4] - a[3][0] + a[3][4], b[2][0] + b[2][2] - b[2][3] + b[4][0] + b[4][2] - b[4][3])
    h[56] = ai(-a[2][0] - a[3][0], -b[0][2] - b[0][4] - b[1][4] - b[4][0] - b[4][2] - b[4][4])
    h[57] = ai(-a[0][3] - a[0][4] - a[2][3] - a[2][4], -b[4][0] + b[4][3] - b[4][4])
    h[58] = ai(-a[2][2] + a[2][3] - a[3][2] + a[3][3], b[3][0] + b[3][2] + b[3][4] + b[4][0] + b[4][2] + b[4][4])
    h[59] = ai(a[1][4] + a[3][4], b[1][2] - b[2][0] - b[2][1] - b[2][2] - b[4][1] - b[4][2])
    h[60] = ai(a[0][3] + a[2][3], b[0][0] - b[0][3] + b[0][4] - b[1][4] - b[3][3] + b[3][4] - b[4][0] + b[4][3] - b[4][4])
    h[61] = ai(a[1][0] + a[3][0], b[0][1] + b[0][2] + b[1][1] - b[3][0] - b[3][1] - b[3][2])
    h[62] = ai(-a[2][2] - a[3][2], -b[1][2] - b[2][2] - b[2][4] - b[3][0] - b[3][2] - b[3][4])
    h[63] = ai(a[0][0] - a[0][2] - a[0][3] + a[2][0] - a[2][2] - a[2][3], b[0][0] - b[0][3] + b[0][4])
    h[64] = ai(-a[0][0] + a[3][0], -b[0][2] + b[0][3] + b[1][3] - b[4][0] - b[4][2] + b[4][3])
    h[65] = ai(a[0][0] - a[0][1] + a[0][2] - a[0][4] - a[1][1] - a[1][4] - a[2][1] + a[2][2] - a[3][0] + a[3][1], b[1][3])
    h[66] = ai(a[1][4] - a[2][4], b[0][0] + b[0][1] + b[0][4] - b[1][4] - b[3][0] - b[3][1] - b[3][4] + b[4][1] + b[4][4])
    h[67] = ai(a[0][0] + a[0][2] - a[0][3] - a[0][4] - a[3][0] - a[3][2] + a[3][3] + a[3][4], -b[2][0] - b[2][2] + b[2][3])
    h[68] = ai(-a[0][2] + a[0][3] - a[1][2] + a[1][3], -b[1][3] - b[2][0] - b[2][1] + b[2][3] - b[4][1] + b[4][3])
    h[69] = ai(a[1][2] - a[1][4] + a[3][2] - a[3][4], -b[2][0] - b[2][1] - b[2][2])
    h[70] = ai(-a[2][0] + a[2][2] - a[2][3] + a[2][4] - a[3][0] + a[3][2] - a[3][3] + a[3][4], -b[4][0] - b[4][2] - b[4][4])
    h[71] = ai(-a[1][0] - a[1][3] - a[3][0] - a[3][3], b[3][0] + b[3][1] + b[3][2])
    h[72] = ai(a[0][2] - a[0][3] - a[0][4] + a[1][2] - a[1][3] - a[1][4], b[0][0] + b[0][1] - b[0][3] + b[1][3] + b[4][1] - b[4][3])
    h[73] = ai(a[1][0] - a[1][2] + a[1][3] - a[2][0] + a[2][2] - a[2][3], b[3][0] + b[3][1] + b[3][4])
    h[74] = ai(-a[0][1] - a[0][3] + a[1][1] + a[1][4] + a[2][0] - a[2][1] - a[2][3] - a[2][4] + a[3][0] - a[3][1], b[1][4])
    h[75] = ai(a[0][2] + a[2][2], -b[0][0] + b[0][3] - b[0][4] + b[1][3] + b[2][3] - b[2][4])

    c = [[NDArray for _ in range(5)] for _ in range(4)]
    count_add, count_mul = 168*height*width + 192*width*width, 0

    for i in range(76):
        count_add += h[i][1]
        count_mul += h[i][2]
        h[i] = h[i][0]

    c[0][0] = -h[9] + h[11] + h[13] - h[14] - h[15] + h[52] + h[4] - h[65] - h[6]
    c[1][0] = h[9] + h[10] - h[11] + h[12] + h[14] + h[15] - h[16] - h[43] + h[50]
    c[2][0] = h[9] - h[11] + h[14] + h[15] - h[0] + h[1] + h[2] - h[3] + h[74]
    c[3][0] = -h[9] + h[11] - h[14] - h[15] + h[51] + h[53] - h[5] - h[7] + h[8]
    c[0][1] = h[12] + h[14] + h[19] + h[20] - h[21] + h[22] + h[24] - h[42] + h[48] + h[49]
    c[1][1] = -h[10] + h[11] - h[12] - h[14] - h[15] + h[16] + h[17] - h[18] - h[20] + h[42] + h[43]
    c[2][1] = -h[15] - h[18] - h[20] - h[27] - h[28] - h[37] + h[41] + h[43] - h[46] + h[47]
    c[3][1] = h[10] - h[11] - h[17] + h[20] - h[31] + h[32] - h[33] - h[35] + h[61] - h[69]
    c[0][2] = h[14] + h[22] + h[23] + h[33] - h[36] + h[39] - h[40] + h[54] - h[55] - h[8]
    c[1][2] = -h[9] + h[18] + h[31] + h[34] + h[35] + h[36] - h[42] - h[59] - h[5] - h[71]
    c[2][2] = -h[15] - h[27] + h[32] + h[36] - h[38] + h[44] - h[45] + h[62] - h[70] - h[7]
    c[3][2] = h[9] + h[14] + h[15] - h[32] + h[33] - h[34] - h[36] - h[53] + h[5] + h[7] - h[8]
    c[0][3] = -h[9] + h[11] + h[13] - h[15] + h[22] + h[23] + h[24] + h[25] + h[4] - h[65] - h[6]
    c[1][3] = h[9] + h[17] - h[18] + h[19] - h[21] - h[23] - h[25] - h[4] - h[68] + h[72]
    c[2][3] = -h[13] + h[15] - h[22] - h[25] + h[26] + h[28] + h[30] + h[45] - h[57] + h[75]
    c[3][3] = h[11] + h[24] + h[25] - h[32] - h[34] - h[39] + h[40] + h[64] - h[67] - h[6]
    c[0][4] = h[14] + h[23] + h[24] + h[26] - h[27] + h[29] + h[30] - h[3] + h[60] + h[63]
    c[1][4] = -h[9] - h[17] - h[1] - h[29] - h[37] + h[41] - h[42] + h[45] + h[66] + h[73]
    c[2][4] = -h[9] + h[11] - h[14] + h[27] + h[28] - h[1] - h[29] - h[2] + h[45] + h[3] - h[74]
    c[3][4] = -h[11] - h[28] + h[29] - h[33] + h[34] + h[38] + h[2] - h[44] + h[56] + h[58]

    count_add += 180*height*width
    C = np.zeros((m, l), dtype=np.double)

    for i in range(4):
        for j in range(5):
            C[i * height: (i + 1) * height, j * width: (j + 1) * width] = c[i][j]

    return C, count_add, count_mul