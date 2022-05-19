#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------------------
@Authors: Nicola Lea Libera (117073), Jenny Döring (119611), Josephin Kröger (124068)
Description: Source code to Assignment 3 of the course Image Analysis and Object Recognition
             at Bauhaus-Universität Weimar.
--------------------------------------------------------------------------------------------
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


# loading of the image, converting to gray and normalizing it
def load_image(path):
    # reads image as colour image
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    # convert image from BGR to RGB representation and return said picture
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # normalize the image
    norm = np.zeros(img.shape, dtype=np.float32)
    # normalize the input image
    norm = np.float32(cv2.normalize(img, norm, 0.0, 1.0, cv2.NORM_MINMAX))

    return norm


def main():
    img_path_task1 = 'taskA.png'
    task_1_img = load_image(img_path_task1)


if __name__ == '__main__':
    main()
