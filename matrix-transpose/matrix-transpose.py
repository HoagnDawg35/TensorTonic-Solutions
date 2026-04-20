import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A, dtype=float)

    row, col = A.shape

    temp = np.zeros((col, row)) # zeros array with shape of (col, row) as temp
    for i in range(row):
        for j in range(col):
            temp[j, i] = A[i, j]

    return temp