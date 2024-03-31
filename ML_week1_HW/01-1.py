import numpy as np

N = int(input('N을 입력하세요 >> '))
arr = np.random.choice(range(1, 101), N, replace=False)

def l1_norm(x):
    x_norm = np.abs(x)
    x_norm = np.sum(x_norm)
    return x_norm
l1_np = np.linalg.norm(arr, 1) # numpy norm

def l2_norm(x):
    x_norm = x*x
    x_norm = np.sum(x_norm)
    x_norm = np.sqrt(x_norm)
    return x_norm
l2_np = np.linalg.norm(arr, 2) # numpy norm

print('arr :', arr)
print('\n내가 만든 L1-Norm : {}'.format(float(l1_norm(arr))))
print('numpy의 L1-Norm : {}'.format(l1_np))
print('\n내가 만든 L2-Norm : {}'.format(l2_norm(arr)))
print('numpy의 L2-Norm : {}'.format(l2_np))
