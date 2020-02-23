# importing the libraries
import cv2 as cv
import numpy as np


# Defining the homography function


def homography(pt_src, pt_dst):
	# img = cv2.imread('reference_images/ref_marker.png')
    A = np.matrix([[-x1, -y1, -1, 0, 0, 0, x1 * xp1, y1 * xp1, xp1],
                   [0, 0, 0, -x1, -y1, -1, x1 * yp1, y1 * yp1, yp1],
                   [-x2, -y2, -1, 0, 0, 0, x2 * xp2, y2 * xp2, xp2],
                   [0, 0, 0, -x2, -y2, -1, x2 * yp2, y2 * yp2, yp2],
                   [-x3, -y3, -1, 0, 0, 0, x3 * xp3, y3 * xp3, xp3],
                   [0, 0, 0, -x3, -y3, -1, x3 * yp3, y3 * yp3, yp3],
                   [-x4, -y4, -1, 0, 0, 0, x4 * xp4, y4 * xp4, xp4],
                   [0, 0, 0, -x4, -y4, -1, x4 * yp4, y4 * yp4, yp4]])

    # Transpose of the matrix A
    At = np.transpose(A)

    # Finding U in the Singular value decomposition form A = U*s*V(transpose)
    matrix_product_1 = np.dot(A, At)
    w, U = np.linalg.eig(matrix_product_1)

    print("Matrix U:", U)

    # Finding the s in the singular value decomposition form.
    s = np.diag(np.sqrt(w))
    print("Matrix s", s)

    # Finding V(transpose) in the Singular value Decomposition form A = U*s*V(transpose)
    matrix_product_2 = np.dot(At, A)
    p, v = np.linalg.eig(matrix_product_2)

    # Taking the transpose of matrix v to get V_transpose
    V_transpose = np.transpose(v)
    print("Matrix Transpose of V:", V_transpose)

    # The solution x of the equation Ax = 0
    V = np.linalg.pinv(V_transpose)
    x = (V[:, 8])
    print("Solution of the linear equation Ax = 0 is x = ", x)
