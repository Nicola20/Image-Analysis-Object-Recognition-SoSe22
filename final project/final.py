#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------------------
@Authors: Nicola Lea Libera (117073), Jenny Döring (119611), Philipp Tornow(118332)
Description: Source code to the final project of the course Image Analysis and Object Recognition
             at Bauhaus-Universität Weimar.
--------------------------------------------------------------------------------------------
"""

from cProfile import label
from cmath import sqrt
import math
from turtle import distance
from xml.sax.handler import feature_validation
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
    plt.gca().add_patch(Rectangle(boundingBoxOrigin, boundingBoxDims[0], boundingBoxDims[1], fill=False, edgecolor='r'))
    plt.show()
    plt.savefig(img_name)


def template_matching(img, template, scale=1, stride=1):

    pyramid_scale = (int(img.shape[1]*scale), int(img.shape[0]*scale))
    img = cv2.resize(img, pyramid_scale, interpolation = cv2.INTER_LINEAR)

    # if template is even then add padding so that we get an uneven kernel size
    if template.shape[0] % 2 == 0:
        template = np.pad(template, ((1, 0), (0, 0)), constant_values=1)
    if template.shape[1] % 2 == 0:
        template = np.pad(template, ((0, 0), (1, 0)), constant_values=1)

    padding_y = int(np.floor(template.shape[0] / 2))
    padding_x = int(np.floor(template.shape[1] / 2))
    padded_img = np.pad(img, ((padding_y, padding_y), (padding_x, padding_x)), constant_values=1)
    filtered_image = normalized_cross_correlation(padded_img, template, (padding_y,padding_x), stride)
    return filtered_image, padding_x, padding_y


def cross_correlation(paddedImage, template, padding, stride=1):
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
            sigma_ab = (1 / (template.shape[0]*template.shape[1])) * (template * paddedImage[m:m+template.shape[0],n:n+template.shape[1]]).sum() - mean_template_patch * np.mean(paddedImage[m:m+template.shape[0],n:n+template.shape[1]])
            sigma_a = (1 / (template.shape[0]*template.shape[1])) * (template * template).sum() - (mean_template_patch*mean_template_patch)
            sigma_b = (1 / (template.shape[0]*template.shape[1])) * (paddedImage[m:m+template.shape[0],n:n+template.shape[1]] * paddedImage[m:m+template.shape[0],n:n+template.shape[1]]).sum() - (np.mean(paddedImage[m:m+template.shape[0],n:n+template.shape[1]])**2)
            divider = sqrt(sigma_a*sigma_b)
            filtered_image[m,n] = -1 if divider == 0 else sigma_ab/divider

    return filtered_image


def nearest_neighbour(vectors, query, k=3):
    distances = []
    for v in vectors:
        distance = np.linalg.norm(v[1]-query)
        distances.append((v[0],distance))

    sorting = lambda d : d[1]
    distances = sorted(distances, key=sorting)
    return distances[:3]


def plot_feature_vectors(vectors):
    plt.figure()
    plt.title("Feature Vectors")
    plt.xlabel("Templates")
    plt.ylabel("NCC")
    markers=['.','^','1','s','P','*']
    colors = ['r','g','b','c','m','y']

    for vidx, key in enumerate(vectors):

        x = np.arange(1,len(vectors[key])+1)
        y = vectors[key]
        plt.scatter(x, y, s=200, c=colors[vidx], marker=markers[vidx], alpha=0.5, label=key+".jpg")

    plt.legend(loc='lower left')
    plt.savefig("results/RESULT.jpg")



def main():
    # ------------------------------------- TASK 1 --------------------------------------------------
    print('task 1\n')
    # templates
    templates = dict()
    templates['1'] = load_gray_image('words/1.jpg')
    templates['2'] = load_gray_image('words/2.jpg')
    templates['3'] = load_gray_image('words/3.jpg')
    templates['4'] = load_gray_image('words/4.jpg')
    templates['5'] = load_gray_image('words/5.jpg')
    templates['6'] = load_gray_image('words/6.jpg')

    # images
    images = dict()
    images['1'] = load_gray_image('images/1.jpg')
    images['2'] = load_gray_image('images/2.jpg')
    images['3'] = load_gray_image('images/3.jpg')
    images['4'] = load_gray_image('images/4.jpg')
    images['5'] = load_gray_image('images/5.jpg')
    images['6'] = load_gray_image('images/6.jpg')

    scales = [1.0, 0.1,0.25, 0.5,0.75, 0.9]

    for t in templates.values():
        if t is None:
            print("Abort")
            return -1

    for i in images.values():
        if i is None:
            print("Abort")
            return -1

    feature_vectors = dict()

    for iidx,image in enumerate(images.values()):
        feature_vectors[str(iidx+1)] = np.zeros(len(templates.values()))
        for tidx,template in enumerate(templates.values()):
            max_ncc_over_scales = -1.0
            for sidx,scale in enumerate(scales):
                filtered_image, padding_x, padding_y = template_matching(image, template, scale)
                ncc_max_index = np.unravel_index(np.argmax(filtered_image, axis=None), filtered_image.shape)
                print(filtered_image[ncc_max_index])
                max_ncc_over_scales = max(max_ncc_over_scales, filtered_image[ncc_max_index])
                #print("max index: " + str(ncc_max_index))
                #print("max value: " + str(filtered_image[ncc_max_index]))

                thresholding = np.vectorize(lambda v: 1 if v>0.5 else 0)
                thresholded_image = thresholding(filtered_image)

            feature_vectors[str(iidx+1)][tidx] = max_ncc_over_scales

                #plot_gray_image(image,
                #    "Image "+str(iidx+1)+" Temp "+str(tidx+1)+" Scale "+str(scale),
                #    "results/scale1/Image"+str(iidx+1)+"_temp"+str(tidx+1)+"_scale"+str(scale)+".jpg",
                #    (ncc_max_index[1]-padding_x, ncc_max_index[0]-padding_y),
                #    template.shape
                #)
                #plot_gray_image(thresholded_image, "TITLE", "out.jpg")

    """feature_vectors['1'] = np.array([0.86933977, 0.91854587, 0.89261441, 0.66698355, 0.6340318 , 0.4428864 ])
    feature_vectors['2'] = np.array([0.76703889, 0.80521685, 0.73824314, 0.74996759, 0.61837406, 0.52716945])
    feature_vectors['3'] = np.array([0.67898011, 0.64412394, 0.74683243, 0.87176417, 0.79158227, 0.850059  ])
    feature_vectors['4'] = np.array([0.7230937 , 0.62198174, 0.74330082, 0.92336739, 0.93324744, 0.85880204])
    feature_vectors['5'] = np.array([0.77387798, 0.86428489, 0.68208233, 0.6822719 , 0.6480239 , 0.0]) #9.27746143
    feature_vectors['6'] = np.array([0.65916653, 0.63556474, 0.74821683, 0.79581352, 0.74234451, 0.64304265])
    """
    plot_feature_vectors(feature_vectors)

    training = []
    training.append(('1',feature_vectors.get('1'),'violin'))
    training.append(('2',feature_vectors.get('2'),'violin'))
    training.append(('3',feature_vectors.get('3'),'guitar'))
    training.append(('4',feature_vectors.get('4'),'guitar'))

    test = []
    test.append(('5',feature_vectors.get('5'),'violin'))
    test.append(('6',feature_vectors.get('6'),'guitar'))

    for query in test:
        nn = nearest_neighbour(training, query[1])
        nn = [i[0] for i in nn]
        print(nn)
        frequency = dict()
        for e in training:
            if e[0] in nn:
                if e[2] not in frequency:
                    frequency[e[2]] = 1
                else:
                    frequency[e[2]] += 1

        plt.figure()
        title = "Votes: "
        for k in frequency.keys():
            title = title+k+":"+str(frequency[k])+", "
        plt.title(title)
        plt.imshow(images[query[0]], cmap='gray')
        plt.axis('off')
        plt.savefig(query[0]+'_labeled.jpg')



    # ------------------------------------- TASK 2 --------------------------------------------------



if __name__ == '__main__':
    main()
