# 2019115177 배주한

import numpy as np
import pandas as pd
from sklearn import linear_model
#%%
# 1) scikit-learn을 이용하여 선형회귀모델을 구현하고 모델의 절편과 계수, 예측점수 출력
hp = pd.Series([130, 250, 190, 300, 210, 220, 170])
fe = pd.Series([16.3, 10.2, 11.1, 7.1, 12.1, 13.2, 14.2])
df = pd.DataFrame({'HP': hp, 'FE': fe})

x = df['HP'].to_numpy().reshape(-1, 1)
y = df['FE'].to_numpy()
regr = linear_model.LinearRegression()

regr.fit(x, y)
y_pred = regr.predict(x)

print('계수 :', regr.coef_)
print('절편 :', regr.intercept_)
print('예측 점수 :', regr.score(x, y))
print()
#%%
# 2) 앞선 LR 모델을 이용하여 270마력을 가지는 새로운 자동차 예상 연비 출력.
new_pred = regr.predict([[270]])
print('270 마력 자동차의 예상 연비 : {:.2f} km/l'.format(float(new_pred[0])))
print()
#%%
# 3) 총중량 데이터를 추가하고 선형 회귀 모델을 구현.
tw = pd.Series([1900, 2600, 2200, 2900, 2400, 2300, 2100])
df3 = pd.DataFrame({'HP':hp, 'TW':tw, 'FE':fe})

x3 = df3[['HP', 'TW']].to_numpy()
y3 = df3['FE'].to_numpy()
regr3 = linear_model.LinearRegression()

regr3.fit(x3, y3)
y3_pred = regr3.predict(x3)

print('계수 :', regr3.coef_)
print('절편 :', regr3.intercept_)
print('예측 점수 :', regr3.score(x3, y3))
print()
#%%
# 4) 앞의 선형 회귀 모델을 바탕으로 270마력의 
new_pred3 = regr3.predict([[270, 2500]])
print('270 마력 자동차의 예상 연비 : {:.2f} km/l'.format(float(new_pred3[0])))