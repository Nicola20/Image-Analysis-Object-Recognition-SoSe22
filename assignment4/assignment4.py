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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors


# loading of the image, converting to gray and normalizing it
def load_image(path):
    # reads image as colour image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # convert image from BGR to RGB representation and return said picture
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def rgb_feature_space(img):
    # code from: https://realpython.com/python-opencv-color-spaces/
    r, g, b = cv2.split(img)
    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize()
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    plt.title("3D RGB Feature Space")
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.savefig("output-images/rgb-feature-space.jpg")


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
    rgb_feature_space(input_ex5_1_img)


if __name__ == '__main__':
    main()
