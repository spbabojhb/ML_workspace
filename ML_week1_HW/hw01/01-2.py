# 2019115177 배주한

#%%
import numpy as np

#hadamard product
def np_hadamard(a, b):
    hadamard = np.zeros((a.shape[0], a.shape[0]))
    for i in range(len(a)):
        for j in range(len(a)):
            hadamard[i][j] = a[i][j]*b[i][j]
    return hadamard

k = int(input('행렬의 k를 입력하세요 >> '))
arr_A = np.random.randint(1, 10, (k, k))
arr_B = np.random.randint(1, 10, (k, k))
hp_numpy = np.multiply(arr_A, arr_B)

print('행렬 A\n', arr_A)
print('행렬 B\n', arr_B)

print('내가 만든 hadamard product\n{}'.format(np_hadamard(arr_A, arr_B)))
print('numpy의 hadamard product\n{}'.format(hp_numpy))

#%%
#dot product
def np_dot(a, b):
    np_dot = np.zeros((a.shape[0], b.shape[-1]))
    for i in range(len(a)):
        for j in range(a.shape[0]):
            for k in range(a.shape[1]):
                np_dot[i][j] += a[i][k]*b[k][j]
    return np_dot

n, m = input('행렬의 n, m을 입력하세요 >> ').split()
n = int(n)
m = int(m)

arr_A2 = np.random.randint(1, 10, (n, m))
arr_B2 = np.random.randint(1, 10, (m, n))
dp_numpy = arr_A2 @ arr_B2
print('행렬 A\n', arr_A2)
print('행렬 B\n', arr_B2)

print('내가 만든 dot product\n{}'.format(np_dot(arr_A2, arr_B2)))
print('numpy의 dot product\n{}'.format(dp_numpy))

