#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------------------
@Authors: Nicola Lea Libera (117073), Jenny Döring (119611), Josephin Kröger ()
Description: Source code to Assignment 1 of the course Image Analysis and Object Recognition
             at Bauhaus-Universität Weimar.
--------------------------------------------------------------------------------------------
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def load_image(path):
    print("loading image...")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # convert image from BGR to RGB representation
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def compute_gray_value(img):
    print("creating grayscale...")
    # extract the 3 different color channels
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    # convert the color channels to double so that they won't overflow and result in a false value
    img = np.uint8(((np.float32(red) + green + blue) / 3))
    return img


def contrast_stretching(img, min_val, max_val):
    print("stretching contrast...")
    height, width = img.shape
    # create an image with the same size as the original
    # and fill it with zeros
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


def image_enhancement(gray, folder):
    print("enhancing image...")
    # prepare the gray values for the histogram and contrast stretching
    hist_data_gray = gray.flatten()

    # get the min, max value for the contrast stretching
    min_val = min(hist_data_gray)
    max_val = max(hist_data_gray)

    contrast_img = contrast_stretching(gray, min_val, max_val)
    contrast_hist_data = contrast_img.flatten()

    # Show the resulting images and corresponding histograms
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Gray")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.hist(hist_data_gray, bins=(max_val-min_val))
    plt.xlabel('Intensity Value')
    plt.ylabel('Number of Pixels')
    plt.tight_layout()
    plt.savefig(str(folder) + "grayscale_hist.jpg")

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(contrast_img, cmap='gray')
    plt.title("Enhanced")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.hist(contrast_hist_data, bins=(max(contrast_hist_data) - min(contrast_hist_data)))
    plt.locator_params(axis="x", nbins=6)
    plt.xlabel('Intensity Value')
    plt.ylabel('Number of Pixels')
    plt.tight_layout()
    plt.savefig(str(folder) + "enhanced_hist.jpg")

    plt.figure()
    plt.imshow(contrast_img, cmap='gray')
    plt.title("Enhanced")
    plt.axis('off')
    plt.savefig(str(folder) + "enhanced.jpg")

    return contrast_img


def binarization(img, folder):
    print("creating binary mask...")
    # 70 is quite a good value (2nd parameter is the threshold value)
    thresh, binary = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY_INV)
    thresh = int(thresh)
    # get the min, max value for the contrast stretching
    plt.figure()
    plt.imshow(binary, cmap='gray')
    plt.title("Binary Mask with Threshold=" + str(thresh))
    plt.axis('off')
    plt.savefig(str(folder) + "binary" + str(thresh) + ".jpg")

    return binary


def morphological_opening(img):
    # inspired by https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    print("doing morphological opening...")
    structuring_element = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, structuring_element)

    return opening


def morphological_closing(img):
    print("doing morphological closing...")
    structuring_element = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, structuring_element)

    return closing


def morphological_operators(binary, folder):
    print("creating filtered image...")
    opening = morphological_opening(binary)
    closing = morphological_closing(opening)

    plt.figure()
    plt.imshow(closing, cmap='gray')
    # plt.title("Morphologically Filtered Mask")
    plt.axis('off')
    plt.savefig(str(folder) + "filtered.jpg")


def overlay(folder):
    print("overlay enhanced image and filtered image...")

    enhanced = cv2.imread(cv2.samples.findFile(str(folder) + 'enhanced.jpg'))
    filtered = cv2.imread(cv2.samples.findFile(str(folder) + 'filtered.jpg'))
    overlay = cv2.addWeighted(enhanced, 0.5, filtered, 1.0, 0.0)

    plt.figure()
    plt.imshow(overlay, cmap='gray')
    plt.title("Overlay")
    plt.axis('off')
    plt.savefig(str(folder) + "overlay.jpg")


def main():

    print("\nGIVEN IMAGE\n")
    folder_1 = "image_1/"
    img_path_1 = str(folder_1) + 'input_sat_image.jpg'
    img_1 = load_image(img_path_1)
    gray_1 = compute_gray_value(img_1)
    enhanced_1 = image_enhancement(gray_1, folder_1)
    binary_1 = binarization(enhanced_1, folder_1)
    morphological_operators(binary_1, folder_1)
    overlay(folder_1)

    # own picture 1
    # source: https://en.wikipedia.org/wiki/Pseudanthium
    print("\nOWN IMAGE\n")
    folder_2 = "image_2/"
    img_path_2 = str(folder_2) + 'flower.jpg'
    img_2 = load_image(img_path_2)
    gray_2 = compute_gray_value(img_2)
    enhanced_2 = image_enhancement(gray_2, folder_2)
    binary_2 = binarization(enhanced_2, folder_2)
    morphological_operators(binary_2, folder_2)
    overlay(folder_2)

    # own picture 2
    print("\nOWN IMAGE\n")
    folder_3 = "image_3/"
    img_path_3 = str(folder_3) + 'berlin.jpg'
    img_3 = load_image(img_path_3)
    gray_3 = compute_gray_value(img_3)
    enhanced_3 = image_enhancement(gray_3, folder_3)
    binary_3 = binarization(enhanced_3, folder_3)
    morphological_operators(binary_3, folder_3)
    overlay(folder_3)


if __name__ == '__main__':
    main()
