
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster,datasets,mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from itertools import cycle, islice
from scipy.spatial.distance import cdist


''' Dataset Preparation '''

n_samples = 200
data = {}

data['noisy_circles'] = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)

data['noisy_moons'] = datasets.make_moons(n_samples=n_samples, noise=.05)


data['blobs'] = datasets.make_blobs(n_samples=n_samples, random_state=8)


data['random'] = np.random.rand(n_samples, 2),0

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
data['aniso'] = (X_aniso, y)


# blobs with varied variances
data['varied'] = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],random_state=100)

# plot the datasets
fig = plt.figure(figsize=(25,5))
colors = np.array(['b', 'g', 'r', 'c', 'm', 'y'])
print('dataset')                                       
for i0,(data_name,data_val) in enumerate(data.items()):
    ax = fig.add_subplot(1,6,i0+1)
    ax.scatter(data_val[0][:, 0], data_val[0][:, 1],c=colors[data_val[1]])
    ax.set_title(data_name)
plt.show()



''' Clustering Algorithms '''
n_clusters = 3

algorithms = {}
algorithms['kmeans'] = cluster.KMeans(n_clusters=n_clusters)   # K-Means

algorithms['gmm'] = mixture.GaussianMixture(
        n_components=n_clusters, covariance_type='full')     # Gaussian Mixture Model

algorithms['ward_linkage'] = cluster.AgglomerativeClustering(
        n_clusters=n_clusters, linkage='ward')               # Hierarchial CLustering -> Agglomerative Clustering

algorithms['average_linkage'] = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=n_clusters)

algorithms['single_linkage'] = cluster.AgglomerativeClustering(
        linkage="single", affinity="cityblock",
        n_clusters=n_clusters)

algorithms['spectral'] = cluster.SpectralClustering(
        n_clusters=n_clusters, eigen_solver='arpack',
        affinity="nearest_neighbors")                      # Spectral Clustering

eps = 0.5
algorithms['dbscan'] = cluster.DBSCAN(eps=eps) # Doesn't need pre-defined cluster size


fig = plt.figure(figsize=(25,5*5))
for i0,(data_name,data_val) in enumerate(data.items()):
    
    for index,(key,val) in enumerate(algorithms.items()):
        X,y = data_val
        algorithm = val
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
        ax = fig.add_subplot(6, 7, i0*7+index+1)
        ax.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
        if 'dbscan' in key:
            ax.set_title(key+'_'+str(eps))
        else:
            ax.set_title(key)

plt.show()