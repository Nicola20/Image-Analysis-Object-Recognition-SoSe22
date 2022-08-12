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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    norm = np.float32(img / np.max(img))

    return norm


def plot_image(img, title, img_name):
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig("output-images/" + img_name)


def main():
    # ------------------------------------- TASK 1 --------------------------------------------------

    print('task 1\n')

    # ------------------------------------- TASK 2 --------------------------------------------------


if __name__ == '__main__':
    main()
