#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------------------
@Authors: Nicola Lea Libera (117073), Jenny Döring (119611)
Description: Source code to Assignment 4 of the course Image Analysis and Object Recognition
             at Bauhaus-Universität Weimar.
--------------------------------------------------------------------------------------------
"""

import numpy as np
from numpy import random
import cv2
import copy
import math
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from scipy import ndimage


MAX_ITER = 300


def load_image(path):
    # reads image as colour image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # convert image from BGR to RGB representation and return said picture
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    norm = np.float32(img / np.max(img))
    return norm


# loading of the image, converting to gray and normalizing it
def load_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    norm = np.float32(img / np.max(img))

    return norm



def rgb_feature_space(img, color_centroids):
    # code from: https://realpython.com/python-opencv-color-spaces/
    r, g, b = cv2.split(img)
    print(color_centroids)
    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(color_centroids)[1], 3))
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
    plt.savefig("output-images/rgb-feature-space-clustering_result.jpg")


def k_means_clustering(img, k):
    iter_count = 0
    loop = True
    feature_space = img.reshape((img.shape[0] * img.shape[1], 3))
    print("shape img: " + str(img.shape))
    print("featurespace shape: " + str(feature_space.shape))
    #print(feature_space)
    previous_centroids = np.random.rand(k, 3) # check here for uniformly distributed rand
    current_centroids = copy.deepcopy(previous_centroids)
    print("starting previous")
    print(previous_centroids)
    print("starting current")
    print(current_centroids)

    while loop and iter_count <= MAX_ITER:
        #print("iteration round: " + str(iter_count))
        # these distances have the structure: in each row we have the distance for the point with
        # [centroid1, centroid2, centroid3, ..., centroidk]
        distances = cdist(feature_space, current_centroids, 'euclidean')
        for c in range(len(current_centroids)):
            # get the indices of the rows with  the smallest distance for c
            mins = np.where(distances[:, c] == distances.min(axis=1))
            mins = mins[0]
            # compute the new centroid out of the points with minimal distance to the old centroid
            new_centroid = centroid(mins, feature_space)
            if new_centroid.all() != 0:
                current_centroids[c] = new_centroid

        if np.array_equal(previous_centroids, current_centroids):
            print("same")
            # assign all points to the nearest centroid
            for c in range(len(current_centroids)):
                # get the indices of the rows with  the smallest distance for c
                mins = np.where(distances[:, c] == distances.min(axis=1))[0]
                # maybe not the best idea to show the categorical data
                feature_space[mins] = copy.deepcopy(current_centroids[c])

            # stop the iteration
            loop = False
        else:
            previous_centroids = copy.deepcopy(current_centroids)

        iter_count += 1
    image_space = np.reshape(feature_space, (img.shape[0], img.shape[1], 3))

    # plot the resulting image from the clustering
    plot_image(image_space, "Clustered image with k =" + str(k), "cluster-img-k-" + str(k))
    rgb_feature_space(img, current_centroids)


def centroid(points, data):

    num_of_points = points.size
    if num_of_points != 0:
        # get all points from the feature space that have a minimal distance to one of the current centroids
        image_pixel = np.take(data, points, axis=0)
        # print(image_pixel[0])
        column_sum = image_pixel.sum(axis=0)

        centroid_x = column_sum[0] / num_of_points
        centroid_y = column_sum[1] / num_of_points
        centroid_z = column_sum[2] / num_of_points

        cent = np.array([centroid_x, centroid_y, centroid_z])
        return cent
    else:
        return np.zeros(3)


def plot_image(img, title, img_name):
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap='nipy_spectral')
    plt.axis('off')
    plt.savefig(img_name + ".jpg")


def gradient_magnitude(img):
    res = ndimage.gaussian_gradient_magnitude(img, sigma=0.5)
    return res



def main():
    # ------------------------------------- TASK 1 --------------------------------------------------
    # print('task 1\n')

    # input_ex5_1_img = load_image('input-images/inputEx5_1.jpg')
    # #rgb_feature_space(input_ex5_1_img)
    # # compute the k means clustering algorithm parameters = img, k
    # k_means_clustering(input_ex5_1_img, 5)

    # ------------------------------------- TASK 2 --------------------------------------------------
    print('task 2')
    print('task a\n')

    task_2_img = load_gray_image("input-images/inputEx5_2.jpg")
    plot_image(task_2_img, 'Grayscale', 'task_2_grayscale')

    task_2_mag = gradient_magnitude(task_2_img)
    plot_image(task_2_mag, 'Gradient Magnitude', 'task_2_magnitude')

    


if __name__ == '__main__':
    main()
