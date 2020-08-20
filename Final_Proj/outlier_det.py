# evaluate model performance with outliers removed using isolation forest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt


def isolation_forest(X_train):
    # identify outliers in the training dataset
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X_train)
    return yhat


def minimum_covariance_determinant(X_train):
    # identify outliers in the training dataset
    ee = EllipticEnvelope(contamination=0.01)
    yhat = ee.fit_predict(X_train)
    return yhat


def one_class_SVM(X_train):
    ee = OneClassSVM(nu=0.01)
    yhat = ee.fit_predict(X_train)
    return yhat


def local_outlier_factor(X_train):
    # identify outliers in the training dataset
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X_train)
    return yhat


if __name__ == '__main__':
    # load the dataset
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    purchase_seq_train = pd.read_csv('./data/Purchase Redemption Data/purchase_seq_train.csv', parse_dates=['report_date'],
                                     index_col='report_date', date_parser=dateparse)
    purchase_seq_test = pd.read_csv('./data/Purchase Redemption Data/purchase_seq_test.csv', parse_dates=['report_date'],
                                    index_col='report_date', date_parser=dateparse)

    # retrieve the array
    X_train = purchase_seq_train.values
    print(X_train)
    yhat = one_class_SVM(X_train)

    # select all rows that are not outliers
    true_mask = yhat != -1
    false_mask = yhat == -1
    true_id = [i for i, ele in enumerate(true_mask) if ele]
    false_id = [i for i, ele in enumerate(true_mask) if not ele]

    plt.plot(list(range(len(X_train))), X_train, c='b')
    plt.scatter(false_id, X_train[false_mask, :], marker='x', c='red')
    plt.show()
