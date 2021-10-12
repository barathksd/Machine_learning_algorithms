# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:25:15 2019

@author: Lenovo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import corner_harris,corner_peaks
from skimage.color import rgb2gray
import skimage.io as io
from skimage.exposure import equalize_hist

def show_corners(corners, image):
    fig = plt.figure()
    plt.gray()
    plt.imshow(image)
    y_corner, x_corner = zip(*corners)
    plt.plot(x_corner, y_corner, 'or')
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
    plt.show()
    
img = io.imread('C:/Users/Lenovo/Desktop/IMG_0486.jpg')
img = equalize_hist(rgb2gray(img))
corners = corner_peaks(corner_harris(img), min_distance=2)
show_corners(corners, img)