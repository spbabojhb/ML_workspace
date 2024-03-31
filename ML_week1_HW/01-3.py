# 2019115177 배주한

#%%
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 11, 1)
f_x = x**2
#plt.plot(x, f_x)
#plt.show()
x_new = 10
derivative = []
y = []
learning_rate = 0.1
for i in range(10):
    old_value = x_new
    derivative.append(old_value-learning_rate*2*old_value)
    x_new = old_value - learning_rate*2*old_value
    y.append(x_new**2)
plt.plot(x, f_x)
fig, ax=plt.scatter(derivative, y)
plt.show()
#%%
x2 = np.arange(-3, 3, 0.01)
f2_x = x2*np.sin(x2**2)
#plt.plot(x2, f2_x)
#plt.show()
x2_new = 1.6
derivative2 = []
y2 = []
learning_rate2 = 0.01

#sin function
def f2(x):
    return x*np.sin(x**2)
def f2_der(x):
    return np.sin(x**2)+2*(x**2)*np.cos(x**2)

for i in range(10):
    old_value = x2_new
    derivative2.append(old_value-learning_rate2*(f2_der(old_value)))
    x2_new = old_value -learning_rate2*(f2_der(old_value))
    y2.append(x2_new*np.sin(x2_new**2))

plt.plot(x2, f2_x)
fig, ax=plt.scatter(derivative2, y2)
plt.show()