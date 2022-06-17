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


MAX_ITER = 300


# loading of the image, converting to gray and normalizing it
def load_image(path):
    # reads image as colour image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # convert image from BGR to RGB representation and return said picture
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    norm = np.float32(img / np.max(img))
    return norm


def rgb_feature_space(img):
    # code from: https://realpython.com/python-opencv-color-spaces/
    #r, g, b = cv2.split(img)
    #pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    #norm = colors.Normalize()
    #norm.autoscale(pixel_colors)
    #pixel_colors = norm(pixel_colors).tolist()

    #fig = plt.figure()
    #axis = fig.add_subplot(1, 1, 1, projection="3d")
    #plt.title("3D RGB Feature Space")
    #axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    #axis.set_xlabel("Red")
    #axis.set_ylabel("Green")
    #axis.set_zlabel("Blue")
    #plt.savefig("output-images/rgb-feature-space.jpg")
    return 0


def k_means_clustering(img, k):
    iter_count = 0
    loop = True
    feature_space = img.reshape((img.shape[0] * img.shape[1], 3))
    #print(feature_space)
    previous_centroids = np.random.rand(k, 3)
    current_centroids = copy.deepcopy(previous_centroids)
    print("starting previous")
    print(previous_centroids)
    print("starting current")
    print(current_centroids)

    while loop and iter_count <= MAX_ITER:
        print("iteration round: " + str(iter_count))
        # these distances have the structure: in each row we have the distance for the point with
        # [centroid1, centroid2, centroid3, ..., centroidk]
        distances = cdist(feature_space, current_centroids, 'euclidean')
        #print(distances)
        #test_iter = 0
        for c in range(len(current_centroids)):
            #print("Iter test: " + str(test_iter))
            # get the indices of the rows with  the smallest distance for c
            mins = np.where(distances[:, c] == distances.min(axis=1))
            mins = mins[0]
            #print("number of indices with smallest c")
            #print(mins)
            #num_of_points = mins.size
            #print("num of points in min")
            #print(num_of_points)
            # compute the new centroid out of the points with minimal distance to the old centroid
            new_centroid = centroid(mins, feature_space)
            current_centroids[c] = new_centroid
            #print("changed current")
            #print(current_centroids)
            #test_iter += 1
            print("computed new centroids")
            print(iter_count)

        if np.array_equal(previous_centroids, current_centroids):
            print("same")
            # assign all points to the nearest centroid
            for c in range(len(current_centroids)):
                # print("Iter test: " + str(test_iter))
                # get the indices of the rows with  the smallest distance for c
                mins = np.where(distances[:, c] == distances.min(axis=1))[0]
                #print(mins[0])
                #print("feature space:")
                #print(feature_space[mins[0]])
                #print("current centroid")
                #print(current_centroids[c])
                feature_space[mins] = copy.deepcopy(current_centroids[c])
                #print("feature space:")
                #print(feature_space[mins[0]])
            # stop the iteration
            loop = False
            #break
        else:
            previous_centroids = copy.deepcopy(current_centroids)
        #    print("previous: ")
        #    print(previous_centroids)
        #    print("current: ")
        #    print(current_centroids)
        #loop = False
        iter_count += 1
    print(previous_centroids)
    print(current_centroids)
    return feature_space


def centroid(points, data):
    #print(data[points[0]])
    num_of_points = points.size
    #print(num_of_points)
    # get all points from the feature space that have a minimal distance to one of the current centroids
    image_pixel = np.take(data, points, axis=0)
    # print(image_pixel[0])
    column_sum = image_pixel.sum(axis=0)
    #print(column_sum)
    centroid_x = column_sum[0] / num_of_points
    centroid_y = column_sum[1] / num_of_points
    centroid_z = column_sum[2] / num_of_points

    cent = np.array([centroid_x, centroid_y, centroid_z])
    #print(cent)
    return cent


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
    #rgb_feature_space(input_ex5_1_img)
    feature_spac = k_means_clustering(input_ex5_1_img, 3)


if __name__ == '__main__':
    main()
