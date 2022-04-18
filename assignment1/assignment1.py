#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------------------
@Authors: Nicola Lea Libera (117073), Jenny Döring (), Josephin Kröger ()
Description: Source code to Assignment 1 of the course Image Analysis and Object Recognition
             at Bauhaus-Universität Weimar.
--------------------------------------------------------------------------------------------
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse


def read_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', help="Path to image that is going to be edited", type=str)
    args = parser.parse_args()
    img = args.image

    return img


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # convert image to RGB representation
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def compute_gray_value(img):
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    # convert the color channels to double so that they won't overflow and result in a false value
    img = np.uint8(((np.uint16(red) + np.uint16(green) + np.uint16(blue)) / 3))
    return img


def image_enhancement(img):
    gray = compute_gray_value(img)
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.show()

#def  binarization():
    # implement task 2

#def morpholigical_operators():
    # implement task 3 here


def main():
    # img_path = read_input()
    img_path = 'input_sat_image.jpg'
    img = load_image(img_path)
    image_enhancement(img)


if __name__ == '__main__':
    main()
