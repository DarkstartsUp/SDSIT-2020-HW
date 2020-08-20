import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import preprocessing
from sklearn.cluster import KMeans


data = pd.read_csv('Data/30_128_X.csv').values
label = pd.read_csv('Data/30_y.csv').values.astype(np.int).squeeze()

print(data.shape)
print(label.shape)

# plt.plot(np.mean(data[label == 0], axis=0), color='g', label='cylider')
# plt.plot(np.mean(data[label == 1], axis=0), color='b', label='bell')
# plt.plot(np.mean(data[label == 2], axis=0), color='r', label='funnel')
# plt.legend()
# plt.xlabel('Date')
# plt.show()

# linked = linkage(data, 'single')
# plt.figure(figsize=(10, 7))
# dendrogram(linked,
#            orientation='top',
#            labels=np.array([i for i in range(len(data))]),
#            distance_sort='descending',
#            show_leaf_counts=True,
#            leaf_font_size=14)
# plt.show()

# km = KMeans(
#     n_clusters=3, init='random',
#     n_init=10, max_iter=300,
#     tol=1e-04, random_state=0
# )
# y_km = km.fit_predict(data)
#
# plt.plot(np.mean(data[y_km == 0], axis=0), color='g', label='Cluster 1 with {} sample'.format(len(data[y_km == 0])))
# plt.plot(np.mean(data[y_km == 1], axis=0), color='b', label='Cluster 2 with {} sample'.format(len(data[y_km == 1])))
# plt.plot(np.mean(data[y_km == 2], axis=0), color='r', label='Cluster 3 with {} sample'.format(len(data[y_km == 2])))
# plt.legend()
# plt.xlabel('Date')
# plt.show()

data_cat = data.ravel()
print(data_cat.shape)

data_conv = []
pace = 128
for i in range(len(data_cat) - pace):
    data_conv.append(data_cat[i:i+pace])

data_conv = np.array(data_conv)
print(data_conv.shape)
data_conv = preprocessing.minmax_scale(data_conv, axis=1)

# labels = np.random.randint(3584, size=50)
# linked = linkage(data_conv[labels, :], 'single')
#
# plt.figure(figsize=(10, 7))
# dendrogram(linked,
#            orientation='top',
#            labels=labels,
#            distance_sort='descending',
#            show_leaf_counts=True,
#            leaf_font_size=14)
# plt.show()

km = KMeans(
    n_clusters=4, init='random',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(data_conv)

plt.plot(np.mean(data_conv[y_km == 0], axis=0), color='g', label='Cluster 1 with {} sample'.format(len(data_conv[y_km == 0])))
plt.plot(np.mean(data_conv[y_km == 1], axis=0), color='b', label='Cluster 2 with {} sample'.format(len(data_conv[y_km == 1])))
plt.plot(np.mean(data_conv[y_km == 2], axis=0), color='r', label='Cluster 3 with {} sample'.format(len(data_conv[y_km == 2])))
plt.plot(np.mean(data_conv[y_km == 3], axis=0), color='orange', label='Cluster 4 with {} sample'.format(len(data_conv[y_km == 3])))
plt.legend()
plt.show()