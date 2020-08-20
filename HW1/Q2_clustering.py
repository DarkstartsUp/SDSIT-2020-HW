import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import preprocessing
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

data = np.load('Data/dji_5yr_20d_log_close_data.npy')
data = preprocessing.minmax_scale(data, axis=1)
print(data.shape)
km = KMeans(
    n_clusters=4, init='random',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(data)
print(str(y_km))
print(len([i for i in y_km if i == 1]))

plt.plot(np.mean(data[y_km == 0], axis=0), color='g', label='Cluster 1 with {} sample'.format(len(data[y_km == 0])))
plt.plot(np.mean(data[y_km == 1], axis=0), color='b', label='Cluster 2 with {} sample'.format(len(data[y_km == 1])))
plt.plot(np.mean(data[y_km == 2], axis=0), color='r', label='Cluster 3 with {} sample'.format(len(data[y_km == 2])))
plt.plot(np.mean(data[y_km == 3], axis=0), color='orange', label='Cluster 4 with {} sample'.format(len(data[y_km == 3])))
plt.legend()
plt.xlabel('Date')
plt.show()

# labels = np.random.randint(1230, size=50)
# linked = linkage(data[labels,:], 'single')

# plt.figure(figsize=(10, 7))
# dendrogram(linked,
#            orientation='top',
#            labels=labels,
#            distance_sort='descending',
#            show_leaf_counts=True,
#            leaf_font_size=14)
# plt.show()
