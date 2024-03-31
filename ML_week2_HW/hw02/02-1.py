# 2019115177 배주한

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

#%%
# 1) 주어진 데이터 x를 이용하여 아래 함수들을 정의하고 MSE를 구하세요.
path = 'C:/Users/spbab/Desktop/4-1/ML_HW/ML_week2_HW/02-1_dataset.csv'
data = pd.read_csv(path, index_col = 0)
param = [0.5, 0.5] # w, b

def h(x, param):
    return param[0]*x + param[1]
def mse(y, x, param):
    return ((y - h(x, param))**2).sum()/len(x)

x = data['X'].to_numpy()
y = 5*x + 50 * np.random.randn(100)
mse_1 = mse(y, x, param)

plt.figure(1)
plt.scatter(x, y)

plt.plot(x, h(x, param), color = 'red')
plt.text(20, 400, 'MSE Loss: {:.2f}'.format(mse_1))

#sklearn mean_squared_error test code
#sci_mse = mean_squared_error(y, h(x, param))
#print(sci_mse)
#%%
# 2) 앞의 식을 Gradient Descent 방법을 통해 주어진 데이터를 학습하고 plot
learning_rate = 0.0001
learning_iteration = 100

for i in range(learning_iteration):
    error = (h(x, param)-y)
    param[0] -= learning_rate * (error * x).sum()/len(x)
    param[1] -= learning_rate * error.sum()/len(x)
mse_2 = mse(y, x, param)
plt.figure(2)
plt.scatter(x, y)

plt.plot(x, h(x, param), color = 'red')
plt.text(20, 400, 'MSE Loss: {:.2f}'.format(mse_2))
#%%
# 3) 앞 문제에서 사용한 동일한 데이터를 scikit-learn을 사용하여 출력.

x3 = x[:, np.newaxis]
regr = linear_model.LinearRegression()
regr.fit(x3, y)

y_pred = regr.predict(x3)
sci_mse = mean_squared_error(y, y_pred)

plt.figure(3)
plt.scatter(x, y)

plt.plot(x, y_pred, color = 'red')
plt.text(20, 400, 'MSE Loss: {:.2f}'.format(sci_mse))