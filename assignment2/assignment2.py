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

#code from https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
# does y and x acxis at the same time :/
def convolve2D(image, kernel, padding=0, strides=1):
    kernel = np.flipud(np.fliplr(kernel))
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    xOutPut = int (((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutPut = int (((yImgShape - yKernShape + 2 * padding) / strides) + 1)

    output_Matrix = np.zeros((xOutPut, yOutPut))

    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding+2, image.shape[1] + padding+2))
        imagePadded[int(padding):int(-1*padding), int (padding):int(-1*padding)] = image
        print(imagePadded)
    else:
        imagePadded = image
    
    for y in range(image.shape[1]):
        if y > image.shape[1] - yKernShape:
            break

        if y % strides == 0:
            for x in range(image.shape[0]):
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    if x % strides ==0:
                        output_Matrix[x,y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break
    return output_Matrix


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


    """
    #norm the picture
    norm = np.zeros((800,800))
    norm_image = cv2.normalize(img_1,norm,0,255,cv2.NORM_MINMAX)
    
    #edge small
    kernel1 = np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])  
    img_2 = convolve2D(img_1, kernel1)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img_1, cmap='gray')
    plt.figure()
    plt.subplot(1, 2, 2)
    plt.imshow(img_2, cmap='gray')
    plt.show()"""


if __name__ == '__main__':
    main()
