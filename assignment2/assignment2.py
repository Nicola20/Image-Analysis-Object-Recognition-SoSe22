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

if __name__ == '__main__':
    main()
