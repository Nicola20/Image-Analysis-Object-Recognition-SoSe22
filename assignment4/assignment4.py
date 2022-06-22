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


# max number of iterations for the k means clustering algorithm in case of an infity loop
MAX_ITER = 300
# number of cluster centroids for the k means clustering algorithm
# testing with k = 3, 5, 10, 15, 20
K = 20


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


def rgb_feature_space(img, color_space, output_name):
    # code from: https://realpython.com/python-opencv-color-spaces/
    r, g, b = cv2.split(img)
    colors = color_space.reshape((np.shape(color_space)[0] * np.shape(color_space)[1], 3))

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    plt.title("3D RGB Feature Space with k = " + str(K))
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=colors,  marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.savefig("output-images/" + output_name)


def k_means_clustering(img, k):
    iter_count = 0
    loop = True
    feature_space = img.reshape((img.shape[0] * img.shape[1], 3))
    color_space = copy.deepcopy(img.reshape((img.shape[0] * img.shape[1], 3)))
    # create list of random cluster centroids
    previous_centroids = np.random.rand(k, 3)
    current_centroids = copy.deepcopy(previous_centroids)

    while loop and iter_count <= MAX_ITER:
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

        # end the algorithm if the cluster centroids do not change anymore or the max number of iterations is reached
        if np.array_equal(previous_centroids, current_centroids) or iter_count == MAX_ITER:
            print("Found cluster centroids")
            # assign all points to the nearest centroid
            color_map_viridis = cm.get_cmap('jet', k)
            for c in range(len(current_centroids)):
                # get the indices of the rows with  the smallest distance for c
                mins = np.where(distances[:, c] == distances.min(axis=1))[0]

                # assign the colors of the pixels to the nearest cluster centroid
                feature_space[mins] = copy.deepcopy(current_centroids[c])

                # assign a color out of the color map to the cluster centroid
                color_space[mins] = color_map_viridis(c)[:3]
            # stop the iteration
            loop = False
        else:
            previous_centroids = copy.deepcopy(current_centroids)

        iter_count += 1
    image_space = np.reshape(feature_space, (img.shape[0], img.shape[1], 3))
    color_space = np.reshape(color_space, (img.shape[0], img.shape[1], 3))
    return image_space, color_space


def centroid(points, data):

    num_of_points = points.size
    if num_of_points != 0:
        # get all points from the feature space that have a minimal distance to one of the current centroids
        image_pixel = np.take(data, points, axis=0)
        # get the sum of the r, g, b columns
        column_sum = image_pixel.sum(axis=0)

        # compute the new centroids
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
    plt.imshow(img)
    plt.axis('off')
    plt.savefig("output-images/" + img_name)


def gradient_magnitude(img):
    # get gradient magnitude of given image
    return ndimage.gaussian_gradient_magnitude(img, sigma=0.5)
     

def plot_seed_points(img, seed_points):
    # plot the seed points onto the gradient magnitude image

    plt.figure()
    plt.title("Seed Points")
    plt.axis('off')

    for point in seed_points:
        plt.plot(point[0], point[1], marker='o', color="blue")

    plt.imshow(img)
    plt.savefig("output-images/task_2_seed_points" + ".jpg")


def non_functional_watershed_segmentation(gradient_img):

    # This function is another approach of implementing the watershed segmentation algorithm.
    # Unfortunately this function cannot be executed, because it ends in an endless loop
    # because of the condition for the while loop.
    # It should break if the seed_points list is empty, but it does not, because of an error,
    # which unfortunately could not be found.
    # This function uses many implemented functions, which work fine. Because of the described 
    # error, only the neighboorhood() function can be used in the watershed_segmentation()
    # function.

    # counter for retracing the iteration steps
    counter = 0
    # given seed points
    seed_points = [[40, 110], [140, 90], [5, 60], [130, 20], [70, 30], [100, 130]]
    # chosen measure_of_similarity
    measure_of_similarity = 0.3

    # array of regions (amount of regions = amount of seed points) 
    region_array = [[] for x in range(len(seed_points))]
    for i in range(len(seed_points)):
        region_array[i].append(seed_points[i])

    # create array for retracing which points in the image were already visited
    visited_points = np.zeros(shape=(gradient_img.shape), dtype=np.uint8)

    for i in range(len(region_array)):

        print("region " + str(i))
        counter = 0
        starting_seed_points = [seed_points[i]]

        # while there are still seed points left
        while (len(seed_points) > 0):
            counter, starting_seed_points, visited_points, region_array[i] = do_watershed_segmentation(counter, gradient_img, starting_seed_points, visited_points, region_array[i], measure_of_similarity)

    # for testing
    for l in range(len(region_array)):
        print(len(region_array[l]))

    return region_array


def do_watershed_segmentation(counter, gradient_img, seed_points, visited_points, region_array, measure_of_similarity):

    counter += 1
    print(counter)
    # new array for updated seed points
    new_seed_points = []

    for i in range(len(seed_points)):

        current_point = seed_points[i]
        # get color of the current point
        current_point_color = get_point_color_value(gradient_img, current_point)
        # get the neighbors of the current point 
        neighbors = neighborhood(gradient_img, current_point)
        try:
            # get the colors of all neighbors
            neighbor_colors = get_neighbor_color_values(gradient_img, neighbors)
        except:
            break
        
        for j in range(len(neighbors)):
            # if the current point was not already visited
            if (not visited_points[neighbors[j][0]][neighbors[j][1]].all()):
                # set point to visited
                visited_points[neighbors[j][0]][neighbors[j][1]] = 1

                # if the neighbor point of the current point matches the measure of similarity
                if (np.abs(neighbor_colors[j] - current_point_color) <= measure_of_similarity):
                    # append neighbor point to the new seed points array and to current region array
                    new_seed_points.append(neighbors[j])
                    region_array.append(neighbors[j])

    return counter, new_seed_points, visited_points, region_array


def watershed_segmentation(gradient_img, seed_points):

    # inspired by https://pythonmana.com/2021/12/20211208035633797D.html 

    # chosen measure_of_similarity
    measure_of_similarity = 0.01
    # create empty array for result image (segmented_array)
    segmented_array = np.zeros(shape=(gradient_img.shape), dtype=np.uint8)
    # create array for retracing which points in the image were already visited
    visited_points = np.zeros(shape=(gradient_img.shape), dtype=np.uint8)

    # get maximum values of gradient image
    max_x = gradient_img.shape[0]
    max_y = gradient_img.shape[1]

    # for every seed point set the suitable point in the segmented array to white (255)
    for point in seed_points:
        x = point[0]
        y = point[1]

        segmented_array[y][x] = 255

    # while there are still seed points left
    while (len(seed_points) > 0):
        # take one seed point
        point = seed_points.pop(0)
        x = point[0]
        y = point[1]

        # mark its corresponding position in the visited_points array as marked (1)
        visited_points[y][x] = 1
        # get the neighbors of the current point
        neighbors = neighborhood(gradient_img, point)

        # for every neighbor
        for n in neighbors:
            current_x = n[0]
            current_y = n[1]

            # if the point is inside the image
            if (current_x < 0 or current_y < 0 or current_x >= max_x or current_y >= max_y):
                continue

            # if the current point was not already visited and
            # if the neighbor point of the current point matches the measure of similarity
            if (not visited_points[current_y][current_x].all()) and (np.abs(gradient_img[current_y][current_x][0] - gradient_img[y][x][0]) <= measure_of_similarity) :
                # mark the point in the result image (segmented_array) white (255)
                segmented_array[current_y][current_x] = 255
                # mark its corresponding position in the visited_points array as marked (1)
                visited_points[current_y][current_x] = 1
                # append point to the seed points array
                seed_points.append((current_x, current_y))

    return segmented_array
        

def put_segmentation_onto_original_image(original_img, segmented_array):
    # puts the resulting image in function watershed_segmentation() 
    # onto the original image

    max_x = original_img.shape[0]
    max_y = original_img.shape[1]
    
    for x in range(max_x):
        for y in range(max_y):
            if segmented_array[x][y][0] != 255:
                original_img[x][y][0] = 0
                original_img[x][y][1] = 0
                original_img[x][y][2] = 0
    
    return original_img


def plot_regions(region_array):

    # plots the different regions (amount dependent on the amount of seed points)

    plt.figure()
    plt.title("Regions")
    plt.axis('off')

    colors = ["blue", "red", "green", "yellow", "orange", "black"]

    for i in range(len(region_array)):
        for point in region_array[i]:
            plt.plot(point[0], point[1], marker='o', markersize=3, color=colors[i])

    plt.savefig("output-images/task_2_regions" + ".jpg")

    
def neighborhood(gradient_img, point):

    # get all the neighbors of the given point
    # --> 8-neighborhood
    
    neighbors = []
    max_x = gradient_img.shape[0]
    max_y = gradient_img.shape[1]
    x = point[0]
    y = point[1]

    if (x <= 0 or y <= 0 or x >= max_x or y >= max_y):
        return neighbors

    neighbors.append((x + 1, y))
    neighbors.append((x - 1, y))
    neighbors.append((x, y + 1))
    neighbors.append((x, y - 1))
    neighbors.append((x + 1, y + 1))
    neighbors.append((x + 1, y - 1))
    neighbors.append((x - 1, y + 1))
    neighbors.append((x - 1, y - 1))

    return neighbors


def get_point_color_value(array, point):

    # get the color of one single point
    
    x = point[0]
    y = point[1]
    point_color = array[x][y][0]

    return point_color


def get_neighbor_color_values(array, neighbors):

    # get the color of an array with neighbor points

    neighbor_colors = []

    if (len(neighbors) > 0):
        for n in range(len(neighbors)):
            x = neighbors[n][0]
            y = neighbors[n][1]
            neighbor_colors.append(array[x][y][0])

    return neighbor_colors


def main():

    # ------------------------------------- TASK 1 --------------------------------------------------

    print('task 1\n')
    input_ex5_1_img = load_image('input-images/inputEx5_1.jpg')
    # compute the k means clustering algorithm parameters = img, k
    original_image_ex5_1 = copy.deepcopy(input_ex5_1_img)
    img_space, color_space = k_means_clustering(input_ex5_1_img, K)

    # plot the resulting image and feature space from the clustering
    rgb_feature_space(original_image_ex5_1, img_space, "feature space ex5_1 k " + str(K) + ".jpg")
    plot_image(img_space, "Clustered image with k =" + str(K), "cluster ex5_1 k-" + str(K))
    # plot the resulting image and feature space from the clustering
    rgb_feature_space(original_image_ex5_1, color_space, "feature space ex5_1 k " + str(K) + " color mapped.jpg")
    plot_image(color_space, "Clustered image with k =" + str(K), "cluster ex5_1 k-" + str(K) + " color mapped.jpg")

    input_ex5_2_img = load_image('input-images/inputEx5_2.jpg')
    original_image_ex5_2 = copy.deepcopy(input_ex5_2_img)
    img_space, color_space = k_means_clustering(input_ex5_2_img, K)

    # plot the resulting image and feature space from the clustering
    rgb_feature_space(original_image_ex5_2, img_space, "feature space ex5_2 k" + str(K) + ".jpg")
    plot_image(img_space, "Clustered image with k =" + str(K), "cluster ex5_2 k-" + str(K))
    # plot the resulting image and feature space from the clustering
    rgb_feature_space(original_image_ex5_2, color_space, "feature space ex5_2 k" + str(K) + " color mapped.jpg")
    plot_image(color_space, "Clustered image with k =" + str(K), "cluster ex5_2 k-" + str(K) + " color mapped.jpg")

    # ------------------------------------- TASK 2 --------------------------------------------------

    print('task 2')
    print('task a')

    task_2_img = load_gray_image("input-images/inputEx5_2.jpg")
    plot_image(task_2_img, 'Grayscale', 'task_2_grayscale')

    task_2_mag = gradient_magnitude(task_2_img)
    plot_image(task_2_mag, 'Gradient Magnitude', 'task_2_magnitude')

    print('task b')

    seed_points = [[40, 110], [140, 90], [5, 60], [130, 20], [70, 30], [100, 130]]
    plot_seed_points(task_2_mag, seed_points)

    print('task c + d')

    task_2_original_img = load_image('input-images/inputEx5_2.jpg')

    task_2_watershed_array = watershed_segmentation(task_2_mag, seed_points)
    plot_image(task_2_watershed_array, 'Watershed', 'task_2_watershed_segementation')

    task_2_overlap_array = put_segmentation_onto_original_image(task_2_original_img, task_2_watershed_array)
    plot_image(task_2_overlap_array, 'Watershed Overlap', 'task_2_watershed_segmentation_overlap')


if __name__ == '__main__':
    main()
