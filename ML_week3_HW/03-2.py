# 2019115177 배주한
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
path = 'C:/Users/spbab/Desktop/4-1/ML_HW/ML_week3_HW/'
origin = pd.read_csv(path + '03-2_dataset.csv')
origin = origin.iloc[:,1].to_numpy()
data = origin[:, np.newaxis]

#%%
# 1) 좌 원본 데이터 우 히스토그램 plot(100개 구간 bias)
plt.figure(1, figsize=(16, 8))
plt.subplot(121)
plt.plot(data)
plt.subplot(122)
plt.hist(data, bins = 100)
plt.show()

#%%
# 2) 정규화를 하기 위한 함수 작성, 결과 plot. 원본 데이터 최대값, 최소값 출력
#    정규화 된 데이터의 최대, 최소 출력. 또한 이를 scikit-learn 정규화 함수 결과와 비교.

def normalization(x, xmin, xmax):
    x_copy = x.copy()
    for i in range(len(x)):
        x_copy[i] = (x[i]-xmin)/(xmax-xmin)
    return x_copy
#원본 min, max
xmin = np.squeeze(min(data))
xmax = np.squeeze(max(data))

#정규화 및 정규화 min, max
normalized_x = normalization(data, xmin, xmax)
n_xmin = np.squeeze(min(normalized_x))
n_xmax = np.squeeze(max(normalized_x))

#scikit learn min max
scaler = MinMaxScaler()
sci_data = scaler.fit_transform(data)
sci_xmin = np.squeeze(min(sci_data))
sci_xmax = np.squeeze(max(sci_data))

#원본 함수와 normalization 비교
plt.figure(2, figsize=(16, 8))
plt.subplot(121)
plt.plot(data)
plt.subplot(122)
plt.hist(normalized_x, bins=100)
plt.show()

print('원본 데이터의 최대값 : {:.5f} 최소값 : {:.5f}'.format(xmax, xmin))
print('정규화 된 데이터의 최대값 : {:.5f} 최소값 : {:.5f}'.format(n_xmax, n_xmin))
print('scikit-learn 정규화 데이터의 최대값 : {:} 최소값 : {:}'.format(sci_xmax, sci_xmin))
print()
#%%
# 3) 표준화를 하기 위한 함수를 작성하고 히스토그램 plot. 원 데이터의 평균과 표준편차 출력.
#    표준화된 데이터의 평균, 표준편차를 출려기. 또한 이를 scikit-learn의 표준화 함수 결과값과 비교.
def standardization(x, mean, std):
    x_copy = x.copy()
    for i in range(len(x)):
        x_copy[i] = (x[i]-mean)/(std)
    return x_copy

#원본 mean, var
xmean = np.squeeze(np.mean(data))
xstd = np.squeeze(np.std(data))

#표준화 및 표준화 mean, var
standardization_x = standardization(data, xmean, xstd)
s_xmean = np.squeeze(np.mean(standardization_x))
s_xstd = np.squeeze(np.std(standardization_x))

#scikit learn mean var
scaler_stand = StandardScaler()
sci_data_stand = scaler_stand.fit_transform(data)
sci_xmean = np.squeeze(np.mean(sci_data_stand))
sci_xstd = np.squeeze(np.std(sci_data_stand))

#원본 함수와 standardization 비교
plt.figure(3, figsize=(16, 8))
plt.subplot(121)
plt.plot(data)
plt.subplot(122)
plt.hist(standardization_x, bins=100)

print('원본 데이터의 평균값 : {:.2f} 표준편차 : {:.2f}'.format(xmean, xstd))
print('표준화 된 데이터의 평균 : {:.2f} 표준편차 : {:.2f}'.format(s_xmean, s_xstd))
print('scikit-learn 표준화 데이터의 평균 : {:.2f} 표준편차 : {:.2f}'.format(sci_xmean, sci_xstd))

#%%
# 4) y = -2X + 1 + 1.2 * N(0, 1)를 만족하는 y 데이터를 생성하고 train 80%, test 20%로 나누어서 선형회귀
#    모델을 구현.
#    학습의 결과로 얻은 선형 회귀 모델을 이용하여 train데이터와 test 데이터를 입력으로 예측치를 계산.
from sklearn import linear_model
regr1 = linear_model.LinearRegression()
regr2 = linear_model.LinearRegression()
X = origin
X = X[:, np.newaxis]

#1 정규화 minmaxscaler
mm_scaled_data = scaler.fit_transform(X)
y1 = -2*mm_scaled_data + 1 + 1.2*np.random.randn(len(X))[:,np.newaxis]

X1_train, X1_test, y1_train, y1_test = train_test_split(mm_scaled_data, y1, test_size = 0.2)
regr1.fit(X1_train, y1_train) #regr1 학습

y1_hat_train = regr1.predict(X1_train) #정규화 train 예측
y1_hat_test = regr1.predict(X1_test) #정규화 test 예측
y1_concat = np.vstack([y1_hat_train, y1_hat_test])
xy1_range = [min(y1_concat), max(y1_concat)]
mse_mm = mean_squared_error(y1_test, y1_hat_test)

#2 표준화 standardization
std_scaled_data = scaler_stand.fit_transform(X)
y2 = -2*std_scaled_data + 1 + 1.2*np.random.randn(len(X))[:,np.newaxis]

X2_train, X2_test, y2_train, y2_test = train_test_split(mm_scaled_data, y2, test_size = 0.2)
regr2.fit(X2_train, y2_train) #regr2 학습

y2_hat_train = regr2.predict(X2_train) #표준화 train 예측
y2_hat_test = regr2.predict(X2_test) #표준화 test 예측
y2_concat = np.vstack([y2_hat_train, y2_hat_test])
xy2_range = [min(y2_concat), max(y2_concat)]
mse_std = mean_squared_error(y2_test, y2_hat_test)

#3 graph 시각화
plt.figure(4, figsize = (16, 8))
plt.subplot(121)
plt.scatter(y1_train, y1_hat_train, c='r', label = 'train')
plt.scatter(y1_test, y1_hat_test, c='b', label = 'test')
plt.plot(xy1_range, xy1_range, c = 'skyblue', linewidth = '3')

plt.text(-4,0.75,'MSE: {:.2f}'.format(mse_mm), fontsize=15)
plt.title('Normalization')
plt.legend(loc = 'upper left')


plt.subplot(122)
plt.scatter(y2_train, y2_hat_train, c='r', label = 'train')
plt.scatter(y2_test, y2_hat_test, c='b', label = 'test')
plt.plot(xy2_range, xy2_range, c = 'skyblue', linewidth = '3')

plt.text(-6,6,'MSE: {:.2f}'.format(mse_std), fontsize=15)
plt.title('Standardization')
plt.legend(loc = 'upper left')

plt.show()

