'''
使用sklearn提供的API实现线性回归
'''
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm # 线性模型
import sklearn.metrics as sm #评估模块

data = pd.read_csv('../data_test/Salary_Data.csv')


#整理输入数据(二维)和输出数据(一维)
x = data.iloc[:,:-1]
y = data.iloc[:,-1]

#构建模型
model = lm.LinearRegression() #y = w1x1+w2x2...+w0
#训练模型
model.fit(x,y)
#执行预测
pred_y = model.predict(x)

print('w1:',model.coef_[0])
print('w0:',model.intercept_)

#绘制回归线
# plt.plot(x,pred_y)
# plt.scatter(data['YearsExperience'],data['Salary'])
# plt.show()

#在全部样本中挑选一些数据，用于测试集（假设测试集没参加过训练）
test_x = x.iloc[::4] #测试集的输入
test_y = y[::4] #测试集的输出

pred_test_y = model.predict(test_x) #测试集的预测值

#平均绝对误差
print(sm.mean_absolute_error(test_y,pred_test_y))
#平均平方误差(均方误差) mse
print(sm.mean_squared_error(test_y,pred_test_y))
#中位数绝对偏差
print(sm.median_absolute_error(test_y,pred_test_y))
#r2_score
print(sm.r2_score(test_y,pred_test_y))

