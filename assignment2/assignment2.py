#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------------------
@Authors: Nicola Lea Libera (117073), Jenny Döring (119611), Josephin Kröger (124068)
Description: Source code to Assignment 1 of the course Image Analysis and Object Recognition
             at Bauhaus-Universität Weimar.
--------------------------------------------------------------------------------------------
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt



SIGMA = 0.5


def load_image(path):
    print("loading image...")
    # reads image as colour image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # convert image from BGR to RGB representation and return said picture
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def create_kernel():
    g_y = np.zeros((5, 5), dtype=np.float32)
    for y in range(-2, 3):
        for x in range(-2, 3):
            tmp1 = - (x / (2 * np.pi * SIGMA**4))
            tmp2 = - ((x**2 + y**2) / 2 * SIGMA**2)
            g_y[x + 2, y + 2] = tmp1 * np.exp(tmp2)
    g_y = -(g_y)
    g_x = np.transpose(g_y)

    return g_x, g_y


def autocorrelation_matrix(gradient_x_squared, gradient_x_y, gradient_y_squared, current_x, current_y, window_size):

    weight = 1

    value = window_size // 2

    M = 0

    for i in range(current_x - value, current_x + value):
        for j in range(current_y - value, current_y + value):
            M = M + weight * np.matrix([[gradient_x_squared[i, j], gradient_x_y[i, j]], [gradient_x_y[i, j], gradient_y_squared[i, j]]])

    return M 


def cornerness(M):

    if (det(M) == 0 or trace(M) == 0):
        return 0

    return det(M) / trace(M)


def roundness(M):

    if (det(M) == 0 or trace(M) == 0):
        return 0

    return 4 * det(M) / (trace(M) * trace(M))


def det(M):
    
    m_1 = M[0, 0]
    m_2 = M[0, 1]
    m_3 = M[1, 0]
    m_4 = M[1, 1]

    return m_1 * m_4 - m_2 * m_3


def trace(M):

    m_1 = M[0, 0]
    m_4 = M[1, 1]

    return m_1 + m_4


def binary_mask(image_size):

    threshold_w = 0.004
    threshold_ = 0.5


def foerstner(gradient_x, gradient_y, image_size):

    print("foerstner...")
    print("don't worry, it lasts some time...")

    gradient_x_squared = gradient_x * gradient_x
    gradient_x_y = gradient_x * gradient_y
    gradient_y_squared = gradient_y * gradient_y

    window_size = 5

    W = np.zeros((image_size[0], image_size[1]))
    Q = np.zeros((image_size[0], image_size[1]))

    # for each pixel in the image
    for current_x in range(0, image_size[0]-1):
        for current_y in range(0, image_size[1]-1):

            M = autocorrelation_matrix(gradient_x_squared, gradient_x_y, gradient_y_squared, current_x, current_y, window_size)

            w = cornerness(M)
            W[current_x, current_y] = w

            q = roundness(M)
            Q[current_x, current_y] = q

    # plot images for cornerness and roundness with jet color mapping
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(W, cmap='jet')
    plt.figure()
    plt.subplot(1, 2, 2)
    plt.imshow(Q, cmap='jet')
    plt.show()


def main():

    print("\nGIVEN IMAGE\n")
    img_path_1 = 'ampelmaennchen.png'
    img_1 = load_image(img_path_1)
    #height, width = img_1.shape
    norm = np.zeros(img_1.shape, dtype=np.float32)
    norm = np.float32(cv2.normalize(img_1, norm, 0.0, 1.0, cv2.NORM_MINMAX))
    g_x, g_y = create_kernel()

    gradient_x = cv2.filter2D(norm, -1, g_x)
    gradient_y = cv2.filter2D(norm, -1, g_y)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(gradient_x, cmap='gray')
    plt.figure()
    plt.subplot(1, 2, 2)
    plt.imshow(gradient_y, cmap='gray')
    plt.show()

    # ***********************************************************************

    image_size = [int(norm.shape[0]), int(norm.shape[1])]

    foerstner(gradient_x, gradient_y, image_size)



if __name__ == '__main__':
    main()
