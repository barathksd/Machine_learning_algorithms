
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mahotas as mh
from sklearn.utils import shuffle
from scipy.spatial.distance import cdist
from sklearn import metrics

original_img = np.array(mh.imread('C:/Users/Lenovo/Desktop/data/alps.jpg'), dtype=np.float64) /255
original_dimensions = tuple(original_img.shape)
width, height, depth = tuple(original_img.shape)
image_flattened = np.reshape(original_img, (width * height, depth))

arr_sample = shuffle(image_flattened,random_state=0)[:1000]

meandistortions=[]
sc=[]

fig = plt.figure()
fig.add_subplot(111).imshow(original_img)
    
K=range(1,8)
for k in K:
    km = KMeans(n_clusters=2**k,random_state=0)
    km.fit(arr_sample)
    meandistortions.append(sum(np.min(cdist(arr_sample, km.cluster_centers_, 'euclidean'), axis=1)) / arr_sample.shape[0])
    if k>=2:
        sc.append(metrics.silhouette_score(arr_sample,km.labels_,metric='euclidean'))
        
    cluster_assignments = km.predict(image_flattened)
    compressed_palette = km.cluster_centers_
    compressed_img = np.zeros((width, height, 3))
    itr=0
    for i in range(width):
        for j in range(height):
            compressed_img[i][j] = compressed_palette[cluster_assignments[itr]]
            itr += 1
    
    fig = plt.figure()
    ax=fig.add_subplot(111)
    plt.title('%i'%2**k)
    ax.imshow(compressed_img)
    plt.show()
    compressed_img*=255
    mh.imsave('cmp_img_%i.jpg'%2**k,compressed_img.astype(np.uint8))



