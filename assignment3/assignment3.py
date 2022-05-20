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
import math
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


def add_gaussian_noise(img):
    mean = 0
    variance = 0.01
    sigma = math.sqrt(variance)
    gaussian = np.random.normal(mean, sigma, img.shape)

    return img + gaussian


# computation of the kernel
def create_gaussian_kernel(sigma):
    # compute the radius and size of the kernel - maybe change this according to the result
    radius = int(np.ceil(3 * sigma))
    kernel_size = 2 * radius + 1

    # initialize gaussian kernel
    g = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    # math formula applied and values stored in the matrix
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            tmp1 = - (1 / (2 * np.pi * sigma ** 4))
            tmp2 = - ((x ** 2 + y ** 2) / 2 * sigma ** 2)
            g[y+radius, x+radius] = tmp1 * np.exp(tmp2)

    return g


def plot_image(img, title, img_name):
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.savefig(img_name + ".jpg")


def plot_log_centered_image(img, title, img_name):
    plt.figure()
    plt.title(title)
    plt.imshow(np.log(abs(img)), cmap='viridis')
    plt.savefig(img_name + ".jpg")


def frequency_domain_filtering(img, kernel):

    # fit the size of the kernel to the size of the image by padding it with zeros
    kernel = np.pad(kernel, [(0, img.shape[0] - kernel.shape[0]),
                             (0, img.shape[1] - kernel.shape[1])], mode='constant')

    # apply fourier transformation to kernel and image
    fourier_img = np.fft.fft2(img)
    shifted_fourier_img = np.fft.fftshift(fourier_img)
    plot_log_centered_image(shifted_fourier_img, "Noisy Image", "noisy image log")

    fourier_kernel = np.fft.fft2(kernel)
    shifted_kernel = np.fft.fftshift(fourier_kernel)
    plot_log_centered_image(shifted_kernel, "Gaussian Filter", "gaussian filter log")

    # multiply them elementwise
    res_img = fourier_img * fourier_kernel
    res_img_shifted = np.fft.fftshift(res_img)
    plot_log_centered_image(res_img_shifted, "Filtered Image", "filtered noisy log")

    # apply inverse fourier to the resulting frequency domain
    res_img = np.fft.ifft2(res_img)
    return np.abs(res_img)


def main():
    img_path_task1 = 'taskA.png'
    task_1_img = load_image(img_path_task1)
    # plot_image(task_1_img, "some title", "normal")
    task_1_img_noise = add_gaussian_noise(task_1_img)
    # plot_image(task_1_img_noise, "some title", "gaussian noise")
    sigma = 0.5
    gaussian_kernel = create_gaussian_kernel(sigma)

    filtered_img = frequency_domain_filtering(task_1_img_noise, gaussian_kernel)
    plot_image(filtered_img, "Filtered Image Sigma=" + str(sigma), "filtered sigma " + str(sigma))


if __name__ == '__main__':
    main()
