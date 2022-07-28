'''
对car.txt的字符串类型进行标签编码
'''
import pandas as pd
import sklearn.preprocessing as sp #数据预处理

data = pd.read_csv('../data_test/car.txt',
                   header=None) #不将第一行设为列名
# print(data.head())
# print(data.dtypes)
new_data = pd.DataFrame()
for i in data:
    encoder = sp.LabelEncoder()
    res = encoder.fit_transform(data[i])
    new_data[i] = res
print(new_data)



