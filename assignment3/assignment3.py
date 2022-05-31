#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------------------
@Authors: Nicola Lea Libera (117073), Jenny Döring (119611)
Description: Source code to Assignment 3 of the course Image Analysis and Object Recognition
             at Bauhaus-Universität Weimar.
--------------------------------------------------------------------------------------------
"""

import numpy as np
import sys
import cv2
import math
from matplotlib import pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
import peakutils
from peakutils.plot import plot as pplot

# number of points for the fourier descriptor in task 3
N = 24


# loading of the image, converting to gray and normalizing it
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    norm = np.float32(img / np.max(img))

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


def frequency_domain_filtering(img, kernel, radius, sigma):

    # fit the size of the kernel to the size of the image by padding it with zeros
    kernel = np.pad(kernel, [(0, img.shape[0] - kernel.shape[0]),
                             (0, img.shape[1] - kernel.shape[1])], mode='constant')

    # center the center of the kernel at the upper left corner
    kernel = np.roll(kernel, [-radius, -radius], axis=(0, 1))

    # apply fourier transformation to kernel and image
    fourier_img = np.fft.fft2(img)
    shifted_fourier_img = np.fft.fftshift(fourier_img)
    plot_log_centered_image(shifted_fourier_img, "Noisy Image", "noisy image log")

    fourier_kernel = np.fft.fft2(kernel)
    shifted_kernel = np.fft.fftshift(fourier_kernel)
    plot_log_centered_image(shifted_kernel, "Gaussian Filter", "gaussian filter log sigma " + str(sigma))

    # multiply them elementwise
    res_img = fourier_img * fourier_kernel
    res_img_shifted = np.fft.fftshift(res_img)
    plot_log_centered_image(res_img_shifted, "Filtered Image", "filtered noisy log sigma " + str(sigma))

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

    image_size = [binary_edge_mask.shape[0], binary_edge_mask.shape[1]]
    # d = sqrt(w^2 + h^2)
    d = np.ceil(np.sqrt(image_size[0]**2 + image_size[1]**2))

    # set ranges of theta and rho arrays
    theta_array = np.arange(-90, 90)
    rho_array = np.arange(-d, d + 1)

    # H(0...179, -d...d)
    hough_voting_array = np.zeros((len(rho_array), len(theta_array)))

    # np.ceil rounds up 
    thetas = np.ceil(np.arctan2(gradient_y, gradient_x))

    for i in range(0, image_size[0]):
        for j in range(0, image_size[1]):
            if binary_edge_mask[i][j] == 1:
            
                # np.cos and np.sin needs radiants as input
                rho = np.ceil(i * np.cos(np.deg2rad(thetas[i, j])) + j * np.sin(np.deg2rad(thetas[i, j])))

                # getting the index, where theta can be found in the theta_array with range (-90, 89)
                index_theta = np.where(theta_array == thetas[i, j])
                # getting the index, where rho can be found in the rho_array with range (-d, d)
                index_rho = np.where(rho_array == rho)

                # update hough voting array
                hough_voting_array[index_rho, index_theta] += 1

    return hough_voting_array, rho_array, theta_array


# ----------------------------------------- TASK 3 --------------------------------------------------
def fourier_descriptor(img, threshold):
    # create a binary mask with a fitting threshold
    thresh, binary = cv2.threshold(img, threshold, 1.0, cv2.THRESH_BINARY)

    # extract the contours of different objects
    contours, hierarchy = cv2.findContours(binary.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # create the fourier descriptor for each image
    df = create_fourier_descriptor(contours)
    return df, contours


def check_for_similarity(df1, df2, boundaries, img, img_name):
    counter = 0
    for entry in df2:
        # check how similar the two descriptor are.
        # If they are very similar then they are likely to be the same object
        if np.linalg.norm(df1 - entry) < 0.06:
            # draw the contours of the object that is similar to our model (df1)
            cv2.drawContours(img, boundaries, counter, (255, 255, 255), 2)
            cv2.imwrite(img_name, img)
        counter += 1


def create_fourier_descriptor(contours):
    # if more than one object (contour) was detected
    if len(contours) > 1:
        # create a descriptor whre the number of rows is equal to the number of detected boundaries (objects)
        descriptor = np.zeros((len(contours), N))

        counter = 0
        for entry in contours:
            if len(entry) > N:
                mod_entry = np.array(entry)[:, 0]
                # convert the point pairs to complex numbers
                df = mod_entry[:, 1] + 1j * mod_entry[:, 0]
                # apply the 1 dim fourier transformation
                df = np.fft.fft(df)

                # make the descriptor invariant to translation, scaling and orientation
                df = abs(df[1:N + 1] / abs(df[1]))
                descriptor[counter, :] = df
            else:
                # if the number of boundary points is smaller than our N, save them as all zeros
                descriptor[counter, :] = np.zeros((1, N))
            counter += 1
    else:
        # if just one contour was found
        contours = np.array(contours[0])[:, 0]
        df = contours[:, 1] + 1j * contours[:, 0]
        df = np.fft.fft(df)
        df = abs(df[1:N + 1] / abs(df[1]))
        descriptor = df

    return descriptor


def main():
    # ------------------------------------- TASK 1 --------------------------------------------------
    print('task 1\n')
    img_path_task1 = 'taskA.png'
    task_1_img = load_image(img_path_task1)
    task_1_img_noise = add_gaussian_noise(task_1_img)
    plot_image(task_1_img_noise, "some title", "gaussian noise")
    sigma = 1.0
    gaussian_kernel, radius = create_gaussian_kernel(sigma)

    filtered_img = frequency_domain_filtering(task_1_img_noise, gaussian_kernel, radius, sigma)
    plot_image(filtered_img, "Filtered Image Sigma=" + str(sigma), "filtered sigma " + str(sigma))

    # ------------------------------------- TASK 2 --------------------------------------------------
    
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

    # Look at the histogram of the magnitude image to look for a fitting threshold
    task_2_mag_hist = task_2_mag.flatten()

    plt.figure()
    plt.hist(task_2_mag_hist)
    plt.title("Histogram of Magnitude")
    plt.savefig("task_2_magnitude_histogram" +  ".jpg")
    
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
    task_2_hough_voting_array, task_2_rho_array, task_2_theta_array = hough_line_detection(task_2_binary_edge_mask,
                                                                                           task_2_gradient_x,
                                                                                           task_2_gradient_y)

    # task e
    print('task e')
    plot_image(task_2_hough_voting_array, 'Hough Voting Array', 'task_2_hough_voting_array')

    # comparison with built-in function:
    # https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.hough_line
    built_in_hough_voting_array, angles, d = hough_line(task_2_binary_edge_mask)
    plot_image(built_in_hough_voting_array, 'Hough Voting Array built-in', 'task_2_hough_built_in')

    # We used for the tasks 2f-i the results for the hough voting array from the built in function, 
    # because of probably existing issues with the gaussian filtering from assignment 2 and caused 
    # from that a wrong hough voting array in 2d).

    # task f
    print('task f')
    built_in_hough_peaks, angles, dists = hough_line_peaks(built_in_hough_voting_array, angles, d)

    # task g
    print('task g')
    temp = cv2.imread('input_ex3.jpg')

    # source: https://peakutils.readthedocs.io/en/latest/tutorial_a.html
    indexes = peakutils.indexes(dists, 0, 0)
    plt.figure()
    pplot(built_in_hough_peaks, dists, indexes)
    plt.savefig('task_2_hough_peaks_diagram' + ".jpg")

    for i in range(0, len(indexes)):
        cv2.circle(temp, (built_in_hough_peaks[indexes[i]], int(dists[indexes[i]])), 0, (255,0,0), 3)

    cv2.imwrite('task_2_hough_peaks.jpg', temp)

    # task h
    print('task h')
    lines = cv2.HoughLines(task_2_binary_edge_mask.astype("uint8"), 1, math.pi/180, 260)

    # task i
    print('task i')

    # inspired by https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
    for i in range(0, len(lines)): 

        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x_0 = a * rho
        y_0 = b * rho
        x_1 = int(x_0 + 1000 * (-b))
        y_1 = int(y_0 + 1000 * (a))
        x_2 = int(x_0 - 1000 * (-b))
        y_2 = int(y_0 - 1000 * (a))
        point_1 = (x_1, y_1)
        point_2 = (x_2, y_2)

        cv2.line(temp, point_1, point_2, (255, 0, 0), 1)

    cv2.imwrite('task_2_hough_lines.jpg', temp)

    # ------------------------------------- TASK 3 --------------------------------------------------
    
    print('\ntask 3')
    task_3_train_img = load_image('trainB.png')
    test_1b_img = load_image('test1B.jpg')
    test_2b_img = load_image('test2B.jpg')
    test_3b_img = load_image('test3B.jpg')
    df_train, contours_train = fourier_descriptor(task_3_train_img, 0.5)
    df_1b, contours_1b = fourier_descriptor(test_1b_img, 0.25)
    df_2b, contours_2b = fourier_descriptor(test_2b_img, 0.35)
    df_3b, contours_3b = fourier_descriptor(test_3b_img, 0.2)
    check_for_similarity(df_train, df_1b, contours_1b,
                         cv2.imread('test1B.jpg', cv2.IMREAD_COLOR), "test1B_result.jpg")
    check_for_similarity(df_train, df_2b, contours_2b,
                         cv2.imread('test2B.jpg', cv2.IMREAD_COLOR), "test2B_result.jpg")
    check_for_similarity(df_train, df_3b, contours_3b,
                         cv2.imread('test3B.jpg', cv2.IMREAD_COLOR), "test3B_result.jpg")


if __name__ == '__main__':
    main()
