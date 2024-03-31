# 2019115177 배주한
import numpy as np
import matplotlib.pyplot as plt
#%%
def h(x, theta):
    return x*theta[1] + theta[0]

x = np.random.randn(500)
X = np.c_[np.ones((500, 1)), x]
y = (-2*x + 1 + 1.2*np.random.randn(500))
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

plt.figure(1, figsize=(8, 8))
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('y')

plt.figure(2, figsize=(8, 8))
plt.scatter(x, y)

#범위 지정 없이 x, y 그래프 그리기
#y_pred = theta[1]*x + theta[0]
#plt.plot(x, y_pred)

plt.plot([min(x), max(x)], [h(min(x), theta), h(max(x), theta)], color='orange')
plt.xlabel('X-input')
plt.ylabel('y - target / true')
plt.text(1, 6, 'W = [{:.7f}]'.format(theta[1]))
plt.text(1, 5, 'b = [{:.7f}]'.format(theta[0]))
plt.show()