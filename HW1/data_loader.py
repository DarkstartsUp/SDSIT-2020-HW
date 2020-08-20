import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv('Data/dji_5yr_data.csv')  # 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
    data5y = data.values[:, 1:]  # 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
    # Standardization
    # normed_data5y = preprocessing.scale(data5y, axis=0)
    log_close = np.log(data5y[:, 3].astype(np.float64))
    plt.plot(log_close, color='b', label='Log Close')
    plt.legend()
    plt.xlabel('Date')
    plt.show()
    normed_close_data20 = []
    for i in range(len(log_close) - 20):
        normed_close_data20.append(log_close[i:i+20])

    np.save('Data/dji_5yr_20d_log_close_data.npy', np.array(normed_close_data20))
    print(np.array(normed_close_data20).shape)

