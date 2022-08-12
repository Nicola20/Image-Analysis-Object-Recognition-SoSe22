#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------------------
@Authors: Nicola Lea Libera (117073), Jenny Döring (119611), Philipp Tornow(118332)
Description: Source code to the final project of the course Image Analysis and Object Recognition
             at Bauhaus-Universität Weimar.
--------------------------------------------------------------------------------------------
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


# loading of the image, converting to gray and normalizing it
def load_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    norm = np.float32(img / np.max(img))

    return norm


def plot_gray_image(img, title, img_name):
    print("I was called")
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.savefig(img_name)


def template_matching(img, template, stride=1):
    padding_y = int(np.floor(template.shape[0] / 2))
    padding_x = int(np.floor(template.shape[1] / 2))
    padded_img = np.pad(img, ((padding_y, padding_y), (padding_x, padding_x)), constant_values=0)


def main():
    # ------------------------------------- TASK 1 --------------------------------------------------
    print('task 1\n')
    # templates
    temp1 = load_gray_image('words/1.jpg')
    temp2 = load_gray_image('words/2.jpg')
    temp3 = load_gray_image('words/3.jpg')
    temp4 = load_gray_image('words/4.jpg')
    temp5 = load_gray_image('words/5.jpg')
    temp6 = load_gray_image('words/6.jpg')

    # images
    img1 = load_gray_image('images/1.jpg')
    img2 = load_gray_image('images/2.jpg')
    img3 = load_gray_image('images/3.jpg')
    img4 = load_gray_image('images/4.jpg')
    img5 = load_gray_image('images/5.jpg')
    img6 = load_gray_image('images/6.jpg')

    templates = [temp1, temp2, temp3, temp4, temp5, temp6]
    images = [img1, img2, img3, img4, img5, img6]

    template_matching(img1, temp1)


    # ------------------------------------- TASK 2 --------------------------------------------------


if __name__ == '__main__':
    main()
