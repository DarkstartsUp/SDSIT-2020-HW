import numpy as np
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, precision_score, f1_score
from math import sqrt
# load dataset
series = read_csv('./daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]
# train autoregression
model = AutoReg(train, lags=3)
model_fit = model.fit()
# print('Coefficients: %s' % model_fit.params)


# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+30-1, dynamic=False)
# for i in range(len(predictions)):
# 	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
# rmse = sqrt(mean_squared_error(test, predictions))
# print('Test RMSE: %.3f' % rmse)
# plot results
# pyplot.plot(test)

locs = [4, 9, 14, 19, 24]
gap = 0.01
run_times = 50
test_times = 40
mean_f1s = []
threses = []
for tst in range(test_times):
	thres = 0.001 + tst * gap
	f1s = []
	for run in range(run_times):
		noise = np.random.randn(30) / 20	#产生N(0,0.05)噪声数据
		noise = noise-np.mean(noise)    #均值为0

		larger_noise = np.random.randn(30) / 4
		larger_noise = larger_noise-np.mean(larger_noise)

		pred_with_noise = np.zeros_like(predictions)
		gt = []
		for i in range(len(predictions)):
			if i in locs:
				gt.append(True)
				pred_with_noise[i] = predictions[i] + larger_noise[i]
			else:
				gt.append(False)
				pred_with_noise[i] = predictions[i] + noise[i]


		# pyplot.plot(predictions, color='blue', label='prediction')
		# pyplot.plot(pred_with_noise, color='purple', label='noised')
		# pyplot.scatter(locs, pred_with_noise[locs], marker='x', c='red', label='outliers')
		# pyplot.legend()

		# signal_power = np.linalg.norm( signal )**2 / signal.size	#此处是信号的std**2
		# noise_variance = signal_power/np.power(10,(SNR/10))         #此处是噪声的std**2
		# noise = (np.sqrt(noise_variance) / np.std(noise) )*noise    ##此处是噪声的std**2
		# signal_noise = noise + signal

		# pyplot.show()


		outliers = np.abs(pred_with_noise - predictions) > thres
		f1s.append(f1_score(gt, outliers))
	threses.append(thres)
	mean_f1s.append(np.mean(f1s))

pyplot.scatter(threses, mean_f1s, marker='x', c='red', label='F1 scores')

pyplot.xlabel('Maximum deviation value')
pyplot.ylabel('F1 score')

#用3次多项式拟合
f1 = np.polyfit(threses, mean_f1s, 3)
p1 = np.poly1d(f1)
yvals1 = p1(threses)  #拟合y值
pyplot.plot(threses, yvals1, 'b', label='polyfit values')
pyplot.legend()
pyplot.show()