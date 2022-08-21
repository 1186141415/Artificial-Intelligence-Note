'''
二值化： 设定一个阈值，用所有元素与阈值进行比较
        大于阈值-->1  小于等于阈值-->0
'''
import numpy as np

raw_sample = np.array([[34.5,67.8,99.9],
                       [12.9,100.0,121.2],
                       [78.4,66.6,45.4]])
bin_sample = raw_sample.copy()
# 设定阈值：60  大于60-->1  小于等于60-->0
# np.where(条件,True,False)
# res = np.where(bin_sample>60,1.0,0.0)
# print(res)
bin_sample[bin_sample <= 60] = 0
bin_sample[bin_sample > 60] = 1
print(bin_sample)

#基于sklearn提供的API实现二值化
import sklearn.preprocessing as sp #数据预处理

biner = sp.Binarizer(threshold=60)
res = biner.transform(raw_sample)
print(res)


