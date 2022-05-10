#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------------------
@Authors: Nicola Lea Libera (117073), Jenny Döring (119611), Josephin Kröger (124068)
Description: Source code to Assignment 2 of the course Image Analysis and Object Recognition
             at Bauhaus-Universität Weimar.
--------------------------------------------------------------------------------------------
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


SIGMA = 0.5

#loading of the image and converting to gray
def load_image(path):
    # reads image as colour image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # convert image from BGR to RGB representation and return said picture
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

#computation of the kernel
def create_gaussian_kernel():
    #5x5 matrix 
    g_x = np.zeros((5, 5), dtype=np.float32)
    # compute the radius of the kernel
    radius = int(np.ceil(3 * SIGMA))
    #math formula applied and values stored in the matrix
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            tmp1 = - (x / (2 * np.pi * SIGMA ** 4))
            tmp2 = - ((x ** 2 + y ** 2) / 2 * SIGMA ** 2)
            g_x[y, x] = tmp1 * np.exp(tmp2)

    g_y = np.transpose(g_x)

    return g_x, g_y

#computation of magnitude according to formula
def compute_magnitude(gradient_x, gradient_y):
    mag = np.sqrt(np.add(gradient_x**2, gradient_y**2))
    return mag

#autocorrelation matrix 
def autocorrelation_matrix(gradient_x_squared, gradient_x_y, gradient_y_squared, current_x, current_y, window_size):
    # weight is 1 because it is in the window
    weight = 1
    value = window_size // 2
    M = 0
    #for each pixel the matrix is computed and stored
    for i in range(current_x - value, current_x + value):
        for j in range(current_y - value, current_y + value):
            M = M + weight * np.matrix([[gradient_x_squared[i, j], gradient_x_y[i, j]], [gradient_x_y[i, j],
                                                                                         gradient_y_squared[i, j]]])

    return M 


def cornerness(M):

    if (det(M) == 0 or trace(M) == 0):
        return 0

    return det(M) / trace(M)


def roundness(M):

    if det(M) == 0 or trace(M) == 0:
        return 0

    return 4 * det(M) / (trace(M) * trace(M))


def det(M):
    
    m_1 = M[0, 0]
    m_2 = M[0, 1]
    m_3 = M[1, 0]
    m_4 = M[1, 1]

    return m_1 * m_4 - m_2 * m_3


def trace(M):

    m_1 = M[0, 0]
    m_4 = M[1, 1]

    return m_1 + m_4

#binary mask with threshold
def derive_binary_mask(img_name, image_size, W, Q, threshold_w, threshold_q):

    binary_mask = np.zeros((image_size[0], image_size[1]))

    for i in range(0, image_size[0] - 1):
        for j in range(0, image_size[1] - 1):
            if (W[i][j] > threshold_w):
                if (Q[i][j] > threshold_q):
                    binary_mask[i][j] = 1

    plt.figure()
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    # plt.show()
    plt.savefig("Binary_Mask-" + img_name + ".jpg")

    return binary_mask


def foerstner(img_name, gradient_x, gradient_y, image_size):

    print("foerstner...")
    print("don't worry, it lasts some time...")

    gradient_x_squared = gradient_x * gradient_x
    gradient_x_y = gradient_x * gradient_y
    gradient_y_squared = gradient_y * gradient_y

    #thresholds according to task
    threshold_w = 0.004
    threshold_q = 0.5

    window_size = 5

    W = np.zeros((image_size[0], image_size[1]))
    Q = np.zeros((image_size[0], image_size[1]))

    # for each pixel in the image
    for current_x in range(0, image_size[0]-1):
        for current_y in range(0, image_size[1]-1):

            M = autocorrelation_matrix(gradient_x_squared, gradient_x_y, gradient_y_squared, current_x, current_y, window_size)

            w = cornerness(M)
            W[current_x, current_y] = w

            q = roundness(M)
            Q[current_x, current_y] = q

    binary_mask = derive_binary_mask(img_name, image_size, W, Q, threshold_w, threshold_q)

    # plot images for cornerness and roundness with jet color mapping
    plot_cornerness(img_name, W)
    plot_roundness(img_name, Q)

    return binary_mask


def plot_cornerness(img_name, W):

    plt.figure()
    plt.imshow(W, cmap='jet')
    plt.axis('off')
    # plt.show()
    plt.savefig("Cornerness-" + img_name + ".jpg")


def plot_roundness(img_name, Q):

    plt.figure()
    plt.imshow(Q, cmap='jet')
    plt.axis('off')
    # plt.show()
    plt.savefig("Roundness-" + img_name + ".jpg")


def overlay(img_name):
    
    image = cv2.imread("Origin-" + img_name + '.jpg')
    binary_mask = cv2.imread('Binary_Mask-' + img_name + '.jpg')

    overlay = cv2.addWeighted(image, 0.5, binary_mask, 1.0, 0.0)

    plt.figure()
    plt.imshow(overlay, cmap='gray')
    plt.axis('off')
    # plt.show()
    plt.savefig("Overlay-" + img_name + ".jpg")


def task_1(norm, img_name):

    print("---Task 1--- " + img_name + "\n")

    # use gaussian for the creation of the kernel
    print("calculating gaussian kernels...")
    g_x, g_y = create_gaussian_kernel()

    # apply the gradient filter to the normalized input image
    print("applying gradient filter to the normaized input image...")
    gradient_x = cv2.filter2D(norm, -1, g_x)
    gradient_y = cv2.filter2D(norm, -1, g_y)

    # compute the magnitude image out of the two gradient images
    print("computing magnitude...\n")
    mag = compute_magnitude(gradient_x, gradient_y)

    return gradient_x, gradient_y, mag


def plot_task_1(orig_img, gradient_x, gradient_y, mag, img_name):

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(orig_img)
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.title("Ix")
    plt.imshow(gradient_x, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.title("Iy")
    plt.imshow(gradient_y, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.title("Magnitude")
    plt.imshow(mag, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    #plt.show()
    plt.savefig("GoG-Filtering-" + img_name + ".jpg")


def task_2(orig_img, img_name, norm, gradient_x, gradient_y):

    print("\n---Task 2--- " + img_name + "\n")
    image_size = [int(norm.shape[0]), int(norm.shape[1])]
    binary_mask = foerstner(img_name, gradient_x, gradient_y, image_size)

    overlay(img_name)


def get_norm_input(img_path):

    print("loading image...\n")
    img = load_image(img_path)
    img_name = img_path.split('.')[0]
    orig_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    plt.figure()
    plt.imshow(orig_img)
    plt.axis('off')
    # plt.show()
    plt.savefig("Origin-" + img_name + ".jpg")

    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    norm = np.zeros(img.shape, dtype=np.float32)
    # normalize the input image
    norm = np.float32(cv2.normalize(img, norm, 0.0, 1.0, cv2.NORM_MINMAX))

    return orig_img, norm, img_name


def main():

    img_path_1 = 'ampelmaennchen.png'
    img_path_2 = 'flower.jpg'
    img_path_3 = 'berlin.jpg'

    # -------------------- input --------------------------

    orig_img_1, norm_1, img_name_1 = get_norm_input(img_path_1)
    orig_img_2, norm_2, img_name_2 = get_norm_input(img_path_2)
    orig_img_3, norm_3, img_name_3 = get_norm_input(img_path_3)

    # -------------------- Task 1 -------------------------

    gradient_x_1, gradient_y_1, mag_1 = task_1(norm_1, img_name_1)
    plot_task_1(orig_img_1, gradient_x_1, gradient_y_1, mag_1, img_name_1)

    gradient_x_2, gradient_y_2, mag_2 = task_1(norm_2, img_name_2)
    plot_task_1(orig_img_2, gradient_x_2, gradient_y_1, mag_2, img_name_2)
    
    gradient_x_3, gradient_y_3, mag_3 = task_1(norm_3, img_name_3)
    plot_task_1(orig_img_3, gradient_x_3, gradient_y_3, mag_3,  img_name_3)

    # -------------------- Task 2 -------------------------

    task_2(orig_img_1, img_name_1, norm_1, gradient_x_1, gradient_y_1)
    task_2(orig_img_2, img_name_2, norm_2, gradient_x_2, gradient_y_2)
    task_2(orig_img_3, img_name_3, norm_3, gradient_x_3, gradient_y_3)


if __name__ == '__main__':
    main()
