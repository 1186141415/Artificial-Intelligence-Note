'''
标签编码:将字符串转成数值类型（根据特征值的位置）
'''
import numpy as np
import sklearn.preprocessing as sp #数据预处理

data = np.array(['bmw','audi','bmw','benzi','bmw',
                 'audi','yadi'])
# [2,0,2,1,2,0,3]
encoder = sp.LabelEncoder()
res = encoder.fit_transform(data)
print(res)

inv_res = encoder.inverse_transform(res)
print(inv_res)


