'''
多项式回归
'''

import pandas as pd
import sklearn.preprocessing as sp #数据预处理
import sklearn.linear_model as lm #线性模型
import sklearn.pipeline as pl #数据管线
import matplotlib.pyplot as plt

data = pd.read_csv('../data_test/Salary_Data.csv')

#整理输入和输出
x = data.iloc[:,:-1]
y = data.iloc[:,-1]

model = pl.make_pipeline(sp.PolynomialFeatures(3),
                         lm.LinearRegression())
model.fit(x,y)

pred_y = model.predict(x)

plt.plot(data['YearsExperience'],pred_y,color='orangered')
plt.scatter(data['YearsExperience'],y)
plt.show()










