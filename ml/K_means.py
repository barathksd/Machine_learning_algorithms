# -*- coding: utf-8 -*-

"""
Created on Wed Jan 30 20:42:41 2019

@author: Lenovo
"""

import numpy as np
from sklearn import cluster,mixture,datasets
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

KMeans = cluster.KMeans
n_samples = 200

colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00'])
                                      )))


print('noisy circles')
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
plt.scatter(noisy_circles[0][:, 0], noisy_circles[0][:, 1],color=colors[noisy_circles[1]])
plt.show()

print('noisy moons')
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
plt.scatter(noisy_moons[0][:,0],noisy_moons[0][:,1])
plt.show()

print('blobs')
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
plt.scatter(blobs[0][:, 0], blobs[0][:, 1])
plt.show()

print('random')
no_structure = np.random.rand(n_samples, 2)
plt.scatter(no_structure[:, 0], no_structure[:, 1])
plt.show()

# Anisotropicly distributed data
print('elongated')
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples,random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)
plt.scatter(aniso[0][:, 0], aniso[0][:, 1])
plt.show()

print('varied blobs')
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5])
plt.scatter(varied[0][:, 0], varied[0][:, 1])
plt.show()
# ============
# Set up cluster parameters
# ============
#plt.figure(figsize=(9 * 2 + 3, 12.5))










X = noisy_moons[0]

# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)
plt.scatter(X[:, 0], X[:, 1])
plt.show()


# connectivity matrix for structured Ward
connectivity = kneighbors_graph(
    X, n_neighbors=10, include_self=False)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)


ward = cluster.AgglomerativeClustering(
n_clusters=3, linkage='ward')

spectral = cluster.SpectralClustering(
        n_clusters=3, eigen_solver='arpack',
        affinity="nearest_neighbors")

dbscan = cluster.DBSCAN(eps=0.25)

average_linkage = cluster.AgglomerativeClustering(
linkage="average", affinity="cityblock",
n_clusters=3)


algorithm = dbscan
algorithm.fit(X)

if hasattr(algorithm, 'labels_'):
    y_pred = algorithm.labels_.astype(np.int)
else:
    y_pred = algorithm.predict(X)


colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))

colors = np.append(colors, ["#000000"])
plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
                            

                            
                            
'''


c1 = np.random.uniform(0.5,3,10)
c11 = np.random.uniform(0,3,10)

c2 = np.random.uniform(6,9.5,10)
c22 = np.random.uniform(6,9,10)
print('c1',c1)
print('c2',c2)
X = np.hstack((c11,c22)).T
print('x',X)
Y = np.hstack((c1, c2)).T
print('y',Y)
X = np.vstack((X,Y)).T
print('X_data',X)
meandistortions=[]
sc = []
K=range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km.fit(X)
    meandistortions.append(sum(np.min(cdist(X, km.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    if k>2:
        sc.append(metrics.silhouette_score(X,km.labels_,metric='euclidean'))
    
fig = plt.figure()
ax = fig.add_subplot(111)
km = KMeans(n_clusters=2)
km.fit(X)
a = km.cluster_centers_
ax.scatter(a[:,0],a[:,1])
c = ['r','g','k']
m = km.labels_
for i in range(0,19):
    ax.scatter(X[i][0],X[i][1],c=c[m[i]])
    

sc=np.array(sc)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(K,meandistortions,'-x')



'''
