import h2o
import numpy as np
import pandas as pd
from h2o.estimators import H2OGeneralizedLowRankEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import preprocessing

# PROBS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
LENGTH = [20, 25, 30, 35, 40, 45, 50, 55, 60]
TIMES = 20

h2o.init()

# Import the dataset into H2O:
data = pd.read_csv(
    'Data/INTLFXD_csv/data/1999_2018_complete.csv')  # 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
data = data.values[-92:, 2:]
data = preprocessing.scale(data, axis=1)
num = data.shape[0] * data.shape[1]
PROBS = [l / num for l in LENGTH]
print(PROBS)

maes = [[] for i in range(len(PROBS))]

for count, prob in enumerate(PROBS):
    for time in range(TIMES):
        miss = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.random.rand() < prob:
                    miss[i, j] = np.nan
                else:
                    miss[i, j] = data[i, j]

        frame_train = h2o.H2OFrame(miss)
        frame_val = h2o.H2OFrame(data)

        glrm_model = H2OGeneralizedLowRankEstimator(k=5, max_iterations=100)
        glrm_model.train(training_frame=frame_train)
        preds = glrm_model.predict(frame_train)
        preds = np.array(preds.as_data_frame(use_pandas=False)[1:])

        # calc mse metric
        gt, res = [], []
        for i in range(miss.shape[0]):
            for j in range(miss.shape[1]):
                if np.isnan(miss[i, j]):
                    gt.append(data[i, j])
                    res.append(float(preds[i, j]) if not np.isnan(float(preds[i, j])) else 0)
        mae = mean_absolute_error(gt, res)
        maes[count].append(mae)

# mean = np.mean(np.array(maes), axis=1)
# error = np.std(np.array(maes), axis=1)
# print(mean)
# print(error)
# # Build the plot
# fig, ax = plt.subplots()
# ax.bar(np.arange(len(PROBS)), mean, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
# ax.set_ylabel('Mean Absolute Error (MAE)')
# ax.set_xticks(np.arange(len(PROBS)))
# ax.set_xticklabels([str(t) for t in PROBS])
# # ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
# ax.yaxis.grid(True)
#
# # Save the figure and show
# plt.tight_layout()
# plt.savefig('bar_plot_with_error_bars.png')
# plt.show()

# maes = [[] for i in range(len(LENGTH))]
#
# for count, length in enumerate(LENGTH):
#     for time in range(TIMES):
#         miss = data.copy()
#
#         miss[2, :length] = np.nan
#
#         frame_train = h2o.H2OFrame(miss)
#         frame_val = h2o.H2OFrame(data)
#
#         glrm_model = H2OGeneralizedLowRankEstimator(k=5, max_iterations=100)
#         glrm_model.train(training_frame=frame_train)
#         preds = glrm_model.predict(frame_train)
#         preds = np.array(preds.as_data_frame(use_pandas=False)[1:])
#
#         # calc mse metric
#         gt, res = [], []
#         for i in range(miss.shape[0]):
#             for j in range(miss.shape[1]):
#                 if np.isnan(miss[i, j]):
#                     gt.append(data[i, j])
#                     res.append(float(preds[i, j]))
#         mae = mean_absolute_error(gt, res)
#         maes[count].append(mae)

mean = np.mean(np.array(maes), axis=1)
error = np.std(np.array(maes), axis=1)
# Build the plot
fig, ax = plt.subplots()
ax.bar(np.arange(len(LENGTH)), mean, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Mean Absolute Error (MAE)')
ax.set_xticks(np.arange(len(LENGTH)))
ax.set_xticklabels([str(t) for t in LENGTH])
# ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.show()
