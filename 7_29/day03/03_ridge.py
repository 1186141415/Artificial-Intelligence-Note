'''
岭回归与Lasso回归
'''
import numpy as np
import pandas as pd
import sklearn.linear_model as lm #线性模型
import matplotlib.pyplot as plt
import sklearn.metrics as sm #评估模块

data = pd.read_csv('../data_test/Salary_Data2.csv')
# print(data.head())

#整理输入数据和输出数据
x = data.iloc[:,:-1]
y = data.iloc[:,-1]

# 构建模型
model = lm.Ridge(alpha=100)
model_lr = lm.LinearRegression()
#训练模型
model.fit(x,y)
model_lr.fit(x,y)
#执行预测
pred_y = model.predict(x)
pred_y_lr = model_lr.predict(x)

plt.plot(data['YearsExperience'],
         pred_y,
         color='orangered',
         label='ridge')
plt.plot(data['YearsExperience'],
         pred_y_lr,
         color='blue',
         label='lr')

plt.scatter(data['YearsExperience'],data['Salary'])
plt.legend()
# plt.show()

# 寻找模型的最优参数
test_x = x.iloc[:30:4]
test_y = y[:30:4] #真实值

params = np.arange(50,151,10)
scores = []
for p in params:
    model = lm.Ridge(alpha=p)
    model.fit(x,y)
    pred_test_y = model.predict(test_x) #预测值
    score = sm.r2_score(test_y,pred_test_y)
    scores.append(score)

scores = pd.Series(scores,index=params)
print(scores.idxmax())




