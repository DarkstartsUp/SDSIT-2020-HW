import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

data = pd.read_csv(
    'Data/INTLFXD_csv/data/1999_2018_complete.csv')  # 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
# 250d per year
length = 250
countries = data.columns[2:]
print(countries)
data = data.values[:, 2:].transpose(1, 0)
print(data.shape)

data = preprocessing.scale(data, axis=1)

start = 0
while start < data.shape[1]:
    data_year = data[:, start: start + length]

    # 求协方差矩阵
    data_cov = np.cov(data_year)

    # 求特征向量和特征值
    tzz, tzxl = np.linalg.eig(data_cov)
    add = np.abs(tzxl[0]) * tzz[0] / sum(tzz) + np.abs(tzxl[1]) * tzz[1] / sum(tzz)
    print(add)
    if start == 0:
        drivers = add[:, np.newaxis]
    else:
        drivers = np.concatenate((drivers, add[:, np.newaxis]), axis=1)
    start += length

km = KMeans(
    n_clusters=4, init='random',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(preprocessing.scale(drivers, axis=1))
print(y_km)

params = {
    'figure.figsize': '20, 4'
}
plt.rcParams.update(params)
for i in range(len(drivers)):
    if y_km[i] == 0:  # drivers[i][1] - drivers[i][0] > 0.1     np.mean(drivers[i]) > 0.17
        plt.plot([str(1999 + i) for i in range(20)], drivers[i], label=countries[i])  # countries[np.argsort(-drivers)]
plt.legend()

plt.tight_layout()

plt.show()
