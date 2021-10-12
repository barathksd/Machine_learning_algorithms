# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 21:19:59 2019

@author: Lenovo
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
y = data.target
x = data.data
x1 = x - [np.mean(x[:,i]) for i in range(x.shape[1])]
cov = np.cov(x.T)
w,v = np.linalg.eig(cov)

p1 = np.mat(x1)*np.mat(v[:,0]).T
p2 = np.mat(x1)*np.mat(v[:,1]).T

pca = PCA(n_components=2)
reduced_X = pca.fit_transform(x)

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_X)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()

fig = plt.figure()
ax2 = fig.add_subplot(111)
for i in range(len(reduced_X)):
    if y[i] == 0:
        ax2.scatter(np.array(p1)[:,0][i],np.array(p2)[:,0][i],c='r', marker='x')
    elif y[i] == 1:
        ax2.scatter(np.array(p1)[:,0][i],np.array(p2)[:,0][i],c='b', marker='D')
    else:
        ax2.scatter(np.array(p1)[:,0][i],np.array(p2)[:,0][i],c='g', marker='.')
    



