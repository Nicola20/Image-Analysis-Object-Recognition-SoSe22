#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------------------
@Authors: Nicola Lea Libera (117073), Jenny Döring (119611), Josephin Kröger (124068)
Description: Source code to Assignment 3 of the course Image Analysis and Object Recognition
             at Bauhaus-Universität Weimar.
--------------------------------------------------------------------------------------------
"""

import numpy as np
import sys
import cv2
import math
from matplotlib import pyplot as plt
from skimage.transform import hough_line


# # OLD VERSION
# # loading of the image, converting to gray and normalizing it
def load_image_try(path):
    # reads image as colour image
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    # convert image from BGR to RGB representation and return said picture
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # normalize the image
    norm = np.zeros(img.shape, dtype=np.float32)
    # normalize the input image
    norm = np.float32(cv2.normalize(img, norm, 0.0, 1.0, cv2.NORM_MINMAX))

    return norm


# CORRECTED (?) VERSION
# loading of the image, converting to gray and normalizing it
def load_image(path):
  
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    norm = np.float32(img / np.max(img))
    #print(type(norm[0][0]))
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
            tmp1 = 1 / (2 * np.pi * (sigma**2))
            tmp2 = - ((x**2 + y**2) / (2 * (sigma**2)))
            g[y + radius, x + radius] = tmp1 * np.exp(tmp2)

    return g, radius


def plot_image(img, title, img_name):
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(img_name + ".jpg")


def plot_log_centered_image(img, title, img_name):
    plt.figure()
    plt.title(title)
    plt.imshow(np.log(abs(img)), cmap='viridis')
    plt.axis('off')
    plt.savefig(img_name + ".jpg")


def frequency_domain_filtering(img, kernel, radius):
    #plt.figure()
    #plt.imshow(kernel, cmap='gray')
    #plt.show()
    # fit the size of the kernel to the size of the image by padding it with zeros
    kernel = np.pad(kernel, [(0, img.shape[0] - kernel.shape[0]),
                             (0, img.shape[1] - kernel.shape[1])], mode='constant')
    kernel = np.roll(kernel, [-radius, -radius], axis=(0, 1))

    #plt.figure()
    #plt.imshow(kernel, cmap='gray')
    #plt.show()

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

# ----------------------- from assignment 2 -------------------------------------------


# computation of the kernel
def ass2_create_gaussian_kernel():
    sigma = 0.5
    # 5x5 matrix
    g_x = np.zeros((5, 5), dtype=np.float32)
    # compute the radius of the kernel
    radius = int(np.ceil(3 * sigma))
    # math formula applied and values stored in the matrix
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            tmp1 = - (x / (2 * np.pi * sigma ** 4))
            tmp2 = - ((x ** 2 + y ** 2) / 2 * sigma ** 2)
            g_x[y + radius, x + radius] = tmp1 * np.exp(tmp2)

    g_y = np.transpose(g_x)

    return g_x, g_y


# computation of magnitude according to formula
def ass2_compute_magnitude(gradient_x, gradient_y):
    mag = np.sqrt(np.add(gradient_x**2, gradient_y**2))
    return mag


def ass2_gaussian_filtering(norm):

    # use gaussian for the creation of the kernel
    g_x, g_y = ass2_create_gaussian_kernel()

    # apply the gradient filter to the normalized input image
    gradient_x = cv2.filter2D(norm, -1, g_x)
    gradient_y = cv2.filter2D(norm, -1, g_y)

    # compute the magnitude image out of the two gradient images
    mag = ass2_compute_magnitude(gradient_x, gradient_y)

    return gradient_x, gradient_y, mag


def ass2_plot_gaussian_filtering(orig_img, gradient_x, gradient_y, mag, img_name):

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(orig_img, cmap='gray')
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
    plt.savefig("task_2_GoG-Filtering-" + img_name + ".jpg")


# ------------------------------------------------------------------------------------


def binary_edge_mask(mag, threshold):

    image_size = [int(mag.shape[0]), int(mag.shape[1])]
    mask = np.zeros((image_size[0], image_size[1]))

    for i in range(0, image_size[0] - 1):
        for j in range(0, image_size[1] - 1):
            if mag[i][j] > threshold:
                mask[i][j] = 1

    return mask


def hough_line_detection(binary_edge_mask, gradient_x, gradient_y):

    image_size = [int(binary_edge_mask.shape[0]), int(binary_edge_mask.shape[1])]
    d = int(np.sqrt(image_size[0]**2 + image_size[1]**2))
    hough_voting_array = np.zeros((181, int(2 * d + 1)))

    thetas = np.degrees(np.arctan2(gradient_y, gradient_x))

    for i in range(0, image_size[0] - 1):
        for j in range(0, image_size[1] - 1):
            if binary_edge_mask[i][j] == 1:
                
                # theta = int(np.arctan(gradient_y[i][j] / gradient_x[i][j]))
                theta = int(thetas[i][j])
                rho = int(i * np.cos(theta) + j * np.sin(theta)) + d

                hough_voting_array[theta][rho] += 1

    rho_array = np.zeros((int(d * 2)))
    theta_array = np.zeros((179))

    return hough_voting_array, rho_array, theta_array


# ----------------------------------------- TASK 3 --------------------------------------------------
def fourier_descriptor(img):
    # find a threshold for the image with otsu method and apply it to th eimage to get a binary mask
    thresh, img = cv2.threshold(img.astype("uint8"), 0.0, 1.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #img = np.float32(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #img_contours = np.zeros(img.shape)
    #img_contours = cv2.drawContours(img_contours, contours, -1, (128, 255, 0), 3)
    #cv2.imwrite('contours.png', img_contours)


def main():
    # ------------------------------------- TASK 1 --------------------------------------------------
    print('task 1\n')
    img_path_task1 = 'taskA.png'
    task_1_img = load_image(img_path_task1)
    # plot_image(task_1_img, "some title", "normal")
    task_1_img_noise = add_gaussian_noise(task_1_img)
    plot_image(task_1_img_noise, "some title", "gaussian noise")
    sigma = 2.0
    gaussian_kernel, radius = create_gaussian_kernel(sigma)

    filtered_img = frequency_domain_filtering(task_1_img_noise, gaussian_kernel, radius)
    plot_image(filtered_img, "Filtered Image Sigma=" + str(sigma), "filtered sigma " + str(sigma))

    # ------------------------------------- TASK 2 --------------------------------------------------
    """
    np.set_printoptions(threshold=sys.maxsize)
    print('task 2')
    task_2_img_path = 'input_ex3.jpg'
    
    # task a
    print('task a')
    task_2_img = load_image(task_2_img_path)
    plot_image(task_2_img, 'Grayscale', 'task_2_grayscale')
    
    # task b
    print('task b')
    task_2_img_name = task_2_img_path.split('.')[0]
    task_2_gradient_x, task_2_gradient_y, task_2_mag = ass2_gaussian_filtering(task_2_img)
    #print(task_2_mag)"""

    """
    # Look at the histogram of the magnitude image to look for a fitting threshold
    testi = task_2_mag.flatten()

    plt.figure()
    plt.hist(testi)
    plt.title("Histogram of Magnitude")
    plt.savefig("task_2_magnitude_histogram" +  ".jpg")
    plt.show()
    """
    """
    ass2_plot_gaussian_filtering(task_2_img, task_2_gradient_x, task_2_gradient_y, task_2_mag, task_2_img_name)
    
    # task c
    print('task c')
    task_2_threshold = 0.5
    task_2_binary_edge_mask = binary_edge_mask(task_2_mag, task_2_threshold)
    plot_image(task_2_binary_edge_mask, 
               "Binary Edge Mask (Magnitude) - threshold = " + str(task_2_threshold), 
               "task_2_binary_edge_mask_" + str(task_2_threshold))
    
    # task d
    print('task d')
    task_2_hough_voting_array, task_2_rho_array, task_2_theta_array = hough_line_detection(task_2_binary_edge_mask, task_2_gradient_x, task_2_gradient_y)
    # print(task_2_hough_voting_array)


    # task e
    print('task e')
    plot_image(task_2_hough_voting_array, 'Hough Voting Array', 'task_2_hough_voting_array')

    # comparison with built-in function:
    # https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.hough_line
    built_in_hough_voting_array, angles, d = hough_line(task_2_binary_edge_mask)
    plot_image(built_in_hough_voting_array, 'Hough Voting Array built-in', 'task_2_hough_built_in')

    # ------------------------------------- TASK 3 --------------------------------------------------
    """
    print('\ntask 3')
    task_3_img_path = 'trainB.png'
    
    # task a 
    print('task a')
    task_3_img = load_image_try(task_3_img_path)
    plot_image(task_3_img, 'Grayscale', 'task_3_grayscale')
    fourier_descriptor(task_3_img)
    #plot_image(task_3_img, 'Binary Mask', 'task_3_binary')


if __name__ == '__main__':
    main()
