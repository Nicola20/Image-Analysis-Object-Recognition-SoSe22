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
    print("loading image...")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # convert image to RGB representation
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def compute_gray_value(img):
    print("creating grayscale...")
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    # convert the color channels to double so that they won't overflow and result in a false value
    img = np.uint8(((np.float32(red) + green + blue) / 3))
    return img


def contrast_stretching(img, min_val, max_val):
    print("stretching contrast...")
    height, width = img.shape
    contrast_img = np.zeros((height, width), img.dtype)
    # create a threshold for the outlier pixel
    outliers = 0.15 * (max_val - min_val)
    high = max_val - outliers
    low = min_val + outliers

    # go through every pixel and recompute the value
    # according to the threshold stretching formula
    for y in range(0, height):
        for x in range(0, width):
            if img[y, x] <= low:
                stretched_val = 0
            elif img[y, x] >= high:
                stretched_val = 1
            elif low < img[y, x] < high:
                stretched_val = ((img[y, x] - low) / (high - low))
            contrast_img[y, x] = np.uint8(stretched_val * 255)

    return contrast_img


def image_enhancement(gray):
    print("enhancing image...")
    # prepare the gray values for the histogram and contrast stretching
    hist_data_gray = gray.flatten()

    # get the min, max value for the contrast stretching
    min_val = min(hist_data_gray)
    max_val = max(hist_data_gray)
    contrast_img = contrast_stretching(gray, min_val, max_val)
    contrast_hist_data = contrast_img.flatten()

    # Show the resulting images and corresponding histograms
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Gray")
    plt.subplot(1, 2, 2)
    plt.hist(hist_data_gray, bins=(max_val-min_val))
    plt.xlabel('Intensity Value')
    plt.ylabel('Number of Pixels')
    plt.savefig("grayscale.jpg")

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(contrast_img, cmap='gray')
    plt.title("Enhanced")
    plt.subplot(1, 2, 2)
    plt.hist(contrast_hist_data, bins=(max(contrast_hist_data) - min(contrast_hist_data)))
    plt.locator_params(axis="x", nbins=6)
    plt.xlabel('Intensity Value')
    plt.ylabel('Number of Pixels')
    plt.savefig("enhanced.jpg")

    return contrast_img


def  binarization(img):
    print("creating binary mask...")
    # 70 is quite a good value (2nd parameter is the threshold value
    thresh, binary = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY_INV)
    thresh = int(thresh)
    # get the min, max value for the contrast stretching
    plt.figure(3)
    plt.imshow(binary, cmap='gray')
    plt.title("Binary Mask with Threshold=" + str(thresh))
    plt.savefig("binary" + str(thresh) + ".jpg")

    return binary


#def morpholigical_operators():
    # implement task 3 here


def main():
    # img_path = read_input()
    img_path = 'input_sat_image.jpg'
    img = load_image(img_path)
    gray = compute_gray_value(img)
    enhanced = image_enhancement(gray)
    binary = binarization(enhanced)


if __name__ == '__main__':
    main()
