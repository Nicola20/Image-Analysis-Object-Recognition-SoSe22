#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------------------
@Authors: Nicola Lea Libera (117073), Jenny Döring (119611), Philipp Tornow(118332)
Description: Source code to the final project of the course Image Analysis and Object Recognition
             at Bauhaus-Universität Weimar.
--------------------------------------------------------------------------------------------
"""

from textwrap import fill
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


# loading of the image, converting to gray and normalizing it
def load_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        norm = np.float32(img / np.max(img))
        return norm
    print("image from "+str(path)+" not found, or could not read.")
    return None




def plot_gray_image(img, title, img_name, boundingBoxOrigin = None, boundingBoxDims = None):

    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    #plt.gca().invert_yaxis()
    plt.gca().add_patch(Rectangle(boundingBoxOrigin, boundingBoxDims[0], boundingBoxDims[1]))
    plt.show()
    plt.savefig(img_name)


def template_matching(img, template, stride=1):

    # if template is even then add padding so that we get an uneven kernel size
    if template.shape[0] % 2 == 0:
        template = np.pad(template, ((1, 0), (0, 0)), constant_values=1)
    if template.shape[1] % 2 == 0:
        template = np.pad(template, ((0, 0), (1, 0)), constant_values=1)

    padding_y = int(np.floor(template.shape[0] / 2))
    padding_x = int(np.floor(template.shape[1] / 2))
    padded_img = np.pad(img, ((padding_y, padding_y), (padding_x, padding_x)), constant_values=1)
    filtered_image = normalized_cross_correlation(padded_img, template, (padding_x,padding_y), stride)
    return filtered_image, padding_x, padding_y


def normalized_cross_correlation(paddedImage, template, padding, stride=1):
    mean_template_patch = np.mean(template)
    filtered_image = np.zeros((paddedImage.shape[0]-2*padding[0], paddedImage.shape[1]-2*padding[1]))

    for m in range(filtered_image.shape[0]):
        for n in range(filtered_image.shape[1]):
            filtered_image[m,n] = ((template - mean_template_patch) * paddedImage[m:m+template.shape[0],n:n+template.shape[1]]).sum()

    return filtered_image

def normalized_cross_correlation(paddedImage, template, padding, stride=1):
    mean_template_patch = np.mean(template)
    filtered_image = np.zeros((paddedImage.shape[0]-2*padding[0], paddedImage.shape[1]-2*padding[1]))

    for m in range(filtered_image.shape[0]):
        for n in range(filtered_image.shape[1]):
            filtered_image[m,n] = ((template - mean_template_patch) * paddedImage[m:m+template.shape[0],n:n+template.shape[1]]).sum()

    return filtered_image

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

    for t in templates:
        if t is None:
            print("Abort")
            return -1

    for i in images:
        if i is None:
            print("Abort")
            return -1

    filtered_image, padding_x, padding_y = template_matching(img1, temp1)
    ncc_max_index = np.unravel_index(np.argmax(filtered_image, axis=None), filtered_image.shape)
    print("max index: " + str(ncc_max_index))
    print("max value: " + str(filtered_image[ncc_max_index]))

    thresholding = np.vectorize(lambda v: 1 if v>0.5 else 0)
    thresholded_image = thresholding(filtered_image)
    #thresholded_image = cv2.threshold(filtered_image, )

    plot_gray_image(filtered_image, "TITLE", "out.jpg", (ncc_max_index[0]-padding_x, ncc_max_index[1]+padding_y), temp1.shape)
    #plot_gray_image(thresholded_image, "TITLE", "out.jpg")


    # ------------------------------------- TASK 2 --------------------------------------------------


if __name__ == '__main__':
    main()
