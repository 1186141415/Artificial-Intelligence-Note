'''
独热编码：根据特征中不重复值的个数，构建1个1和若干个0组成的序列
'''
import numpy as np
import sklearn.preprocessing as sp #数据预处理

raw_sample = np.array([[1,3,2],
                       [7,5,4],
                       [1,8,6],
                       [7,3,9]])
#构建独热编码器
encoder = sp.OneHotEncoder(sparse=False,
                           dtype='int32',
                           categories='auto')
res = encoder.fit_transform(raw_sample)
print(res)

#解码
inv_res = encoder.inverse_transform(res)
print(inv_res)



