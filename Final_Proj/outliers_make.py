import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Final_Proj.ARIMA as am
import Final_Proj.outlier_det as od
import statsmodels.api as sm
from sklearn.metrics import f1_score, mean_absolute_error

def purchase_seq_display(timeseries):
    graph = plt.figure(figsize=(10, 4))
    ax = graph.add_subplot(111)
    ax.set(title='Total_Purchase_Amt',
           ylabel='Unit (yuan)', xlabel='Date')
    plt.plot(timeseries)
    plt.show()


def add_outliers(timeseries, add_ids, min_thres=100000000):
    noise = np.random.randn(len(timeseries)) * 200000000
    noise = noise - np.mean(noise)  # 均值为0

    pred_with_noise = np.zeros_like(np.array(timeseries))
    gt = []
    for i in range(len(timeseries)):
        if i in add_ids:
            if np.abs(noise[i]) > min_thres:
                gt.append(True)
                pred_with_noise[i] = timeseries[i] + noise[i]
            else:
                gt.append(False)
                pred_with_noise[i] = timeseries[i]
        else:
            gt.append(False)
            pred_with_noise[i] = timeseries[i]

    return pred_with_noise, gt


def conv_smoothing(timeseries, win_size=15):
    kernel = np.hanning(win_size)  # 随机生成一个卷积核（对称的）
    kernel /= kernel.sum()
    timeseries_convolved = np.convolve(timeseries, kernel, 'same')
    return timeseries_convolved


if __name__ == '__main__':
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    purchase_seq_train = pd.read_csv('./data/Purchase Redemption Data/purchase_seq_train.csv',
                                     parse_dates=['report_date'],
                                     index_col='report_date', date_parser=dateparse)

    conved = conv_smoothing(purchase_seq_train['value'])
    # print(purchase_seq_train)
    purchase_seq_train['value'] = conved
    fit_seq = am.ARIMA_fit(purchase_seq_train)

    # plt.plot(pred_with_noise, color='b', label='noised')
    # plt.scatter([i for i, ele in enumerate(gt) if ele], pred_with_noise[[i for i, ele in enumerate(gt) if ele]], marker='x', c='red', label='outliers')
    # plt.legend()
    # plt.show()

    # plt.plot(fit_seq, color='red', label='fit_seq')
    # plt.plot(purchase_seq_train, color='blue', label='purchase_seq_train')
    # plt.legend(loc='best')
    # plt.show()
    exp_times = 200
    f1_scores = []
    maes = []
    for exp in range(exp_times):
        add_ids = [i for i in range(len(purchase_seq_train['value'])) if i % 10 == 0]
        pred_with_noise, gt = add_outliers(conved, add_ids)
        yhat = od.one_class_SVM(pred_with_noise.reshape(-1, 1))

        # select all rows that are not outliers
        true_mask = yhat != -1
        false_mask = yhat == -1
        true_id = [i for i, ele in enumerate(true_mask) if ele]
        false_id = [i for i, ele in enumerate(true_mask) if not ele]

        pred_recover = []
        for i in range(len(pred_with_noise)):
            if false_mask[i]:
                pred_recover.append(fit_seq[i])
            else:
                pred_recover.append(pred_with_noise[i])

        f1_scores.append(f1_score(gt, false_mask))
        maes.append(mean_absolute_error(conved, pred_recover) / 10000000)

    print(round(np.mean(f1_scores), 3))
    print(round(np.mean(maes), 3))

    # plt.plot(list(range(len(pred_with_noise))), conved, c='b')
    # plt.scatter(false_id, pred_with_noise[false_mask], marker='x', c='red')
    # plt.scatter(false_id, np.array(pred_recover)[false_mask], marker='x', c='green')
    # plt.show()
