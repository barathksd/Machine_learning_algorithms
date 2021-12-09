# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

file = 'C:/Users/Lenovo/Desktop/IMG_0060.jpg'
original_image = cv2.imread(file)

renorm_image = np.reshape(original_image,(original_image.shape[0]*original_image.shape[1],3))

renorm_image = renorm_image.astype('float32')
renorm_image -= np.mean(renorm_image, axis=0)
renorm_image /= np.std(renorm_image, axis=0)

cov = np.cov(renorm_image, rowvar=False)

lambdas, p = np.linalg.eig(cov)
alphas = np.random.normal(0, 0.1, 3)

delta = np.dot(p, alphas*lambdas)


mean = np.mean(renorm_image, axis=0)
std = np.std(renorm_image, axis=0)
pca_augmentation_version_renorm_image = renorm_image + delta
pca_color_image = pca_augmentation_version_renorm_image * std + mean
pca_color_image = np.maximum(np.minimum(pca_color_image*255, 255), 0).astype('uint8')
pca_color_image = pca_color_image.reshape(original_image.shape)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.figure()
plt.imshow(cv2.cvtColor(pca_color_image, cv2.COLOR_BGR2RGB))

#cv2.imwrite('C:/Users/Lenovo/Desktop/data/pca_clraug_alps.jpg',pca_color_image)





