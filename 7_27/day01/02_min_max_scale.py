'''
范围缩放
调整数据分布，将最小值和最大值设为相同的区间（以列为单位）
缩小列与列之间的差异
'''
import numpy as np

raw_sample = np.array([[1.0,2.0,3.0],
                       [4.0,5.0,9.0],
                       [7.0,8.0,11.0]])
mms_sample = raw_sample.copy()

for col in mms_sample.T:
    col_min = col.min()
    col_max = col.max()
    col -= col_min
    col /= (col_max - col_min)

print(mms_sample)

#基于sklearn提供的API实现范围缩放
import sklearn.preprocessing as sp #数据预处理

mms = sp.MinMaxScaler(feature_range=(0,1))
# mms.fit(raw_sample)
# res = mms.transform(raw_sample)
res = mms.fit_transform(raw_sample)
print(res)


