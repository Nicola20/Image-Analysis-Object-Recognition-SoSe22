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


def load_image(path):
    print("loading image...")
    # reads image as colour image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # convert image from BGR to RGB representation and return said picture
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def main():

    print("\nGIVEN IMAGE\n")
    img_path_1 = r'ampelmaennchen.png'
    img_1 = load_image(img_path_1)

    #norm the picture
    norm = np.zeros((800,800))
    norm_image = cv2.normalize(img_1,norm,0,255,cv2.NORM_MINMAX)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img_1, cmap='gray')
    plt.figure()
    plt.subplot(1, 2, 2)
    plt.imshow(norm_image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
