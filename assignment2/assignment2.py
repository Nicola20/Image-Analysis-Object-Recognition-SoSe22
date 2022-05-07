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


def main():

    print("\nGIVEN IMAGE\n")
    img_path_1 = r'ampelmaennchen.png'
    img_1 = load_image(img_path_1)

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
    plt.show()

    

if __name__ == '__main__':
    main()
