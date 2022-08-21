'''
均值移除(标准化)
调整数据的分布状态，缩小列与列之间的差异
每列的均值为0，标准差为1
'''
import numpy as np

raw_sample = np.array([[3.0,-100.0,2.0],
                       [0.0,400.0,3.0],
                       [1.0,-400.0,2.0]])
std_sample = raw_sample.copy()

for col in std_sample.T:
    col_mean = col.mean()
    col_std = col.std()
    col -= col_mean
    col /= col_std

print(std_sample)
print(std_sample.mean(axis=0))
print(std_sample.std(axis=0))

print('*'* 30)
#基于sklearn提供的API实现均值移除
import sklearn.preprocessing as sp #数据预处理

res = sp.scale(raw_sample)
print(res)
print(res.mean(axis=0))
print(res.std(axis=0))

