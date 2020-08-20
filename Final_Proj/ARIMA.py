import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller as ADF
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA


def generate_purchase_seq():
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
    user_balance = pd.read_csv('./data/Purchase Redemption Data/user_balance_table.csv', parse_dates=['report_date'],
                               index_col='report_date', date_parser=dateparse)

    df = user_balance.groupby(['report_date'])['total_purchase_amt'].sum()

    purchase_seq = pd.Series(df, name='value')
    purchase_seq_201402_201407 = purchase_seq['2014-02-01':'2014-07-31']
    purchase_seq_201402_201407.to_csv(path='./data/Purchase Redemption Data/purchase_seq_201402_201407.csv', header=True)


def generate_purchase_seq_2():
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
    user_balance = pd.read_csv('./data/Purchase Redemption Data/user_balance_table.csv', parse_dates=['report_date'],
                               index_col='report_date', date_parser=dateparse)

    df = user_balance.groupby(['report_date'])['total_purchase_amt'].sum()
    purchase_seq = pd.Series(df, name='value')

    purchase_seq_train = purchase_seq['2014-04-01':'2014-07-31']
    purchase_seq_test = purchase_seq['2014-08-01':'2014-08-10']

    purchase_seq_train.to_csv(path='./data/Purchase Redemption Data/purchase_seq_train.csv', header=True)
    purchase_seq_test.to_csv(path='./data/Purchase Redemption Data/purchase_seq_test.csv', header=True)


def purchase_seq_display(timeseries):
    graph = plt.figure(figsize=(10, 4))
    ax = graph.add_subplot(111)
    ax.set(title='Total_Purchase_Amt',
           ylabel='Unit (yuan)', xlabel='Date')
    plt.plot(timeseries)
    plt.show()


def decomposing(timeseries):
    decomposition = seasonal_decompose(timeseries)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.figure(figsize=(16, 12))
    plt.subplot(411)
    plt.plot(timeseries, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonarity')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residual')
    plt.legend(loc='best')
    plt.show()


def diff(timeseries):
    timeseries_diff1 = timeseries.diff(1)
    timeseries_diff2 = timeseries_diff1.diff(1)

    timeseries_diff1 = timeseries_diff1.fillna(0)
    timeseries_diff2 = timeseries_diff2.fillna(0)
    if 'value' in timeseries.keys():
        timeseries_adf = ADF(timeseries['value'].tolist())
    else:
        timeseries_adf = ADF(timeseries)
    if 'value' in timeseries_diff1.keys():
        timeseries_diff1_adf = ADF(timeseries_diff1['value'].tolist())
    else:
        timeseries_diff1_adf = ADF(timeseries_diff1)
    if 'value' in timeseries_diff2.keys():
        timeseries_diff2_adf = ADF(timeseries_diff2['value'].tolist())
    else:
        timeseries_diff2_adf = ADF(timeseries_diff2)

    print('timeseries_adf : ', timeseries_adf)
    print('timeseries_diff1_adf : ', timeseries_diff1_adf)
    print('timeseries_diff2_adf : ', timeseries_diff2_adf)

    plt.figure(figsize=(12, 8))
    plt.plot(timeseries, label='Original', color='blue')
    plt.plot(timeseries_diff1, label='Diff1', color='red')
    plt.plot(timeseries_diff2, label='Diff2', color='purple')
    plt.legend(loc='best')
    plt.show()


def autocorrelation(timeseries, lags):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(timeseries, lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(timeseries, lags=lags, ax=ax2)
    plt.show()


def ARIMA_Model(timeseries, order):
    model = ARIMA(timeseries, order=order)
    return model.fit(disp=0)


def ARIMA_fit(purchase_seq_train):
    decomposition = seasonal_decompose(purchase_seq_train)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    trend = trend.fillna(0)
    seasonal = seasonal.fillna(0)
    residual = residual.fillna(0)

    # 趋势序列模型训练
    trend_model = ARIMA_Model(trend, (1, 0, 0))
    trend_fit_seq = trend_model.fittedvalues
    trend_predict_seq = trend_model.predict(start='2014-08-01', end='2014-08-10', dynamic=True)

    # 残差序列模型训练
    residual_model = ARIMA_Model(residual, (3, 0, 1))
    residual_fit_seq = residual_model.fittedvalues
    residual_predict_seq = residual_model.predict(start='2014-08-01', end='2014-08-10', dynamic=True)

    # 拟合训练集
    fit_seq = pd.Series(seasonal, index=seasonal.index)
    fit_seq = fit_seq.add(trend_fit_seq, fill_value=0)
    fit_seq = fit_seq.add(residual_fit_seq, fill_value=0)

    return fit_seq


if __name__ == '__main__':
    # dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    # purchase_seq_201402_201407 = pd.read_csv('./data/Purchase Redemption Data/purchase_seq_201402_201407.csv',
    #                                          parse_dates=['report_date'], index_col='report_date',
    #                                          date_parser=dateparse)
    # decomposing(purchase_seq_201402_201407)
    # diff(purchase_seq_201402_201407)
    # purchase_seq_201402_201407_diff1 = purchase_seq_201402_201407.diff(1)
    # purchase_seq_201402_201407_diff1 = purchase_seq_201402_201407_diff1.fillna(0)
    # autocorrelation(purchase_seq_201402_201407_diff1, 20)
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    purchase_seq_train = pd.read_csv('./data/Purchase Redemption Data/purchase_seq_train.csv', parse_dates=['report_date'],
                                     index_col='report_date', date_parser=dateparse)
    purchase_seq_test = pd.read_csv('./data/Purchase Redemption Data/purchase_seq_test.csv', parse_dates=['report_date'],
                                    index_col='report_date', date_parser=dateparse)

    decomposition = seasonal_decompose(purchase_seq_train)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    trend = trend.fillna(0)
    seasonal = seasonal.fillna(0)
    residual = residual.fillna(0)

    # diff(trend)
    # diff(residual)

    # autocorrelation(trend, 20)
    # autocorrelation(residual, 20)

    # 趋势序列模型训练
    trend_model = ARIMA_Model(trend, (1, 0, 0))
    trend_fit_seq = trend_model.fittedvalues
    trend_predict_seq = trend_model.predict(start='2014-08-01', end='2014-08-10', dynamic=True)

    # 残差序列模型训练
    residual_model = ARIMA_Model(residual, (2, 0, 1))
    residual_fit_seq = residual_model.fittedvalues
    residual_predict_seq = residual_model.predict(start='2014-08-01', end='2014-08-10', dynamic=True)

    # 拟合训练集
    fit_seq = pd.Series(seasonal, index=seasonal.index)
    fit_seq = fit_seq.add(trend_fit_seq, fill_value=0)
    fit_seq = fit_seq.add(residual_fit_seq, fill_value=0)

    plt.plot(fit_seq, color='red', label='fit_seq')
    plt.plot(purchase_seq_train, color='blue', label='purchase_seq_train')
    plt.legend(loc='best')
    plt.show()

    # # 预测测试集
    # # 这里测试数据的周期性是根据seasonal对象打印的结果，看到里面的数据每7天一个周期，2014-08-01~2014-08-10的数据正好和2014-04-04~2014-04-13的数据一致
    # seasonal_predict_seq = seasonal['2014-04-04':'2014-04-13']
    #
    # predict_dates = pd.Series(
    #     ['2014-08-01', '2014-08-02', '2014-08-03', '2014-08-04', '2014-08-05', '2014-08-06', '2014-08-07', '2014-08-08',
    #      '2014-08-09', '2014-08-10']).apply(lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d'))
    #
    # seasonal_predict_seq.index = predict_dates
    #
    # predict_seq = pd.Series(seasonal_predict_seq, index=seasonal_predict_seq.index)
    # predict_seq = predict_seq.add(trend_predict_seq, fill_value=0)
    # predict_seq = predict_seq.add(residual_predict_seq, fill_value=0)
    #
    # plt.plot(predict_seq, color='red', label='predict_seq')
    # plt.plot(purchase_seq_test, color='blue', label='purchase_seq_test')
    # plt.legend(loc='best')
    # plt.show()
