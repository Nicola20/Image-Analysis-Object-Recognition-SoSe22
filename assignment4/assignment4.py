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
# from skimage.segmentation import watershed
# from skimage.feature import peak_local_max


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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    return ndimage.gaussian_gradient_magnitude(img, sigma=0.5)
     

"""
def neighborhood(img, x, y):

    neighborhood_region_numbers = {}
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i == 0 and j == 0):
                continue
            if (x + i < 0 or y + j < 0):
                continue
            if (x + i >= img.shape[0] or y + j >= img.shape[1]):
                continue
            if (neighborhood_region_numbers.get(img[x + i][y + j]) == None):
                neighborhood_region_numbers[img[x + i][y + j]] = 1
            else:
                neighborhood_region_numbers[img[x + i][y + j]] += 1

    if (neighborhood_region_numbers.get(0) != None):
        del neighborhood_region_numbers[0]

    keys = list(neighborhood_region_numbers)

    keys.sort()

    if (keys[0] == -1):
        if (len(keys) == 1):
            return -1
        elif (len(keys) == 2):
            return keys[1]
        else:
            return 0
    else:
        if(len(keys) == 1):
            return keys[0]
        else:
            return 0


def watershed_algorithm(img):

    intensity_list = []

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            intensity_list.append((img[x][y], (x, y)))

    # intensity_list.sort()

    segmented_img = np.full(img.shape, -1, dtype=int)

    region_number = 0

    for i in range(len(intensity_list)):
   
        x = intensity_list[i][1][0]
        y = intensity_list[i][1][1]

        region_status = neighborhood(segmented_img, x, y)

        if (region_status == -1):
            region_number += 1
            segmented_img[x][y] = region_number
        elif (region_status == 0):
            segmented_img[x][y] = 0
        else:
            segmented_img[x][y] = region_status

    return segmented_img
"""

def watershed_built_in(path):

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=5)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(),255,0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown==255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255,0,0]

    return markers


# def watershed_built_in_2(img):

#     distance = ndimage.distance_transform_edt(img)
#     coords = peak_local_max(distance, footprint=np.ones((3,3)), labels=img)
#     mask = np.zeros(distance.shape, dtype=bool)
#     mask[tuple(coords.T)] = True
#     markers, _ = ndimage.label(maks)
#     labels = watershed(-distance, markers, mask=img)

#     fig, axes = plt.subplots(ncols=3, figsize=(9,3), sharex=True, sharey=True)
#     ax = axes.ravel()

#     ax[0].imshow(img, cmap=plt.cm.gray)
#     ax[0].set_title('Overlapping Area')
#     ax[1].imshow(-distance, cmap=plt.cm.gray)
#     ax[1].set_title('Distances')
#     ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
#     ax[2].set_title('Separated Objects')

#     for a in ax:
#         a.set_axis_off()

#     fig.tight_layout()
#     plt.show()

def plot_seed_points(img, seed_points):
    plt.figure()
    plt.title("Seed Points")
    plt.axis('off')

    for point in seed_points:
        plt.plot(point[0], point[1], marker='o', color="blue")

    plt.imshow(img)
    plt.savefig("task_2_seed_points" + ".jpg")


def watershed(gradient_img):

    seed_points = [[40, 110], [140, 90], [5, 60], [130, 20], [70, 30], [100, 130]]

    measure_of_similarity = 0.1

    # test_neighbor = neighborhood(gradient_img, seed_points[0])
    # test_neighbor = neighborhood(gradient_img, (0, 0))
    # print(test_neighbor)
    # test_colors = get_neighbor_color_values(gradient_img, test_neighbor)
    # print(test_colors)

    region_array = [[] for x in range(len(seed_points))]
    
    for i in range(len(seed_points)):
        
        current_point = seed_points[i]
        current_point_color = get_point_color_value(gradient_img, current_point)
        neighbors = neighborhood(current_point)
        neighbor_colors = get_neighbor_color_values(gradient_img, neighbors)
        
        for j in range(len(neighbors)):
            if (np.abs(neighbor_colors[j] - current_point_color) <= measure_of_similarity):
                region_array[i].append(neighbors[j])
    
    for l in range(len(region_array)):
        print(len(region_array[l]))

    return region_array


def plot_regions(region_array):
    plt.figure()
    plt.title("Regions")
    plt.axis('off')

    colors = ["blue", "red", "green", "yellow", "orange", "black"]

    for i in range(len(region_array)):
        for point in region_array[i]:
            plt.plot(point[0], point[1], marker='o', markersize=3, color=colors[i])

    plt.savefig("task_2_regions" + ".jpg")

    
def neighborhood(point):
    
    neighbors = []

    x = point[0]
    y = point[1]

    neighbors.append((x + 1, y))
    neighbors.append((x - 1, y))
    neighbors.append((x, y + 1))
    neighbors.append((x, y - 1))

    return neighbors


def get_point_color_value(array, point):
    
    x = point[0]
    y = point[1]
    point_color = array[x][y][0]

    return point_color


def get_neighbor_color_values(array, neighbors):

    neighbor_colors = []

    for n in range(len(neighbors)):
        x = neighbors[n][0]
        y = neighbors[n][1]
        neighbor_colors.append(array[x][y][0])

    return neighbor_colors


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

    # task_2_segmented_img = watershed_built_in("input-images/inputEx5_2.jpg")
    # plot_image(task_2_segmented_img, 'Watershed Segmentation Built-In', 'task_2_watershed_built_in')

    print('task b')

    # seed_points = [[40, 110], [140, 90], [5, 60], [130, 20], [70, 30], [100, 130]]
    # plot_seed_points(task_2_mag, seed_points)

    print('task c')
    task_2_region_array = watershed(task_2_mag)
    plot_regions(task_2_region_array)


if __name__ == '__main__':
    main()
