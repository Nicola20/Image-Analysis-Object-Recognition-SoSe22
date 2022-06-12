#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------------------
@Authors: Nicola Lea Libera (117073), Jenny Döring (119611)
Description: Source code to Assignment 4 of the course Image Analysis and Object Recognition
             at Bauhaus-Universität Weimar.
--------------------------------------------------------------------------------------------
"""

import numpy as np
import cv2
import math
from matplotlib import pyplot as plt


# loading of the image, converting to gray and normalizing it
def load_image(path):
    # reads image as colour image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # convert image from BGR to RGB representation and return said picture
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def plot_image(img, title, img_name):
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(img_name + ".jpg")


def main():
    # ------------------------------------- TASK 1 --------------------------------------------------
    print('task 1\n')

    input_ex5_1_img = load_image('input-images/inputEx5_1.jpg')


if __name__ == '__main__':
    main()
