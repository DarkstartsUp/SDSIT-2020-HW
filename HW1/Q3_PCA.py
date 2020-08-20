import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

data = pd.read_csv(
    'Data/INTLFXD_csv/data/1999_2018_complete.csv')  # 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
countries = data.columns[2:]
print(countries)
data = data.values[:, 2:].transpose(1, 0)
print(data.shape)

data = preprocessing.scale(data, axis=1)

# pca = PCA(n_components=23)
# pca.fit(data)
#
# print(pca.explained_variance_ratio_)
# print(np.argsort(-pca.explained_variance_ratio_))

# 求协方差矩阵
data_cov = np.cov(data)

# 求特征向量和特征值
tzz, tzxl = np.linalg.eig(data_cov)

print('===================================================')
print(tzz)
for z in tzz:
    print(z / sum(tzz))
print('===================================================')
print(np.array(tzxl).shape)
print(tzxl[0])

drivers = np.abs(tzxl[0]) * tzz[0] / sum(tzz) + np.abs(tzxl[1]) * tzz[1] / sum(tzz)

params = {
    'figure.figsize': '20, 4'
}
plt.rcParams.update(params)
# plt.bar([str(i) for i in range(1, 24)], sorted(tzz / sum(tzz), reverse=True), alpha=0.8,
#         width=0.6)  # countries[np.argsort(-drivers)]
# plt.tight_layout()
# plt.show()

bar_data = [2809, 9249, 7778, 8472, 3536, 2104, 6807, 4898, 6291, 3804, 7768, 8661, 345, 8886, 5670, 8918, 7054, 8976, 9587, 8753, 9129, 2481, 479]
plt.bar([str(i) for i in range(1, 24)], [t / 1000 for t in bar_data], alpha=0.8,
        width=0.6)  # countries[np.argsort(-drivers)]
plt.tight_layout()
plt.show()
