'''
鸢尾花的分类预测
'''
import sklearn.datasets as sd #数据集合
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms #模型选择
import sklearn.linear_model as lm #线性模型
import sklearn.metrics as sm #评估模块

iris = sd.load_iris()
# print(iris.keys())
# print(iris.feature_names)
# print(iris.DESCR)
# print(iris.target_names)
# print(iris.data.shape)
# print(iris.target)
# print(iris.data)

#整理数据
data = pd.DataFrame(iris.data,
                    columns=iris.feature_names)
data['target'] = iris.target

# print(data)
data.plot.scatter(x='sepal length (cm)',
                  y='sepal width (cm)',
                  c='target',
                  s=50,
                  cmap='brg')

data.plot.scatter(x='petal length (cm)',
                  y='petal width (cm)',
                  c='target',
                  s=50,
                  cmap='brg')
# plt.show()
# print(data)

# 挑选出来1,2类别数据，做二分类
# sub_data = data.iloc[50:]
# sub_data = data.tail(100)
# sub_data = data[(data['target'] == 1) | (data['target'] == 2)]
# sub_data = data[~(data['target'] == 0)]

#整理输入和输出
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
#划分训练集和测试集
train_x,\
test_x,\
train_y,\
test_y = ms.train_test_split(x,y,
                             test_size=0.2,
                             random_state=7,
                             stratify=y)
#建立模型
model = lm.LogisticRegression(solver='liblinear')
#做5次交叉验证
score = ms.cross_val_score(model,
                           x,y,
                           cv=5,
                           scoring='f1_weighted')
print(score)
print('交叉验证均值:',score.mean())

#训练
model.fit(train_x,train_y)
#预测
pred_test_y = model.predict(test_x)
#评估(精度(准确率))   对的个数 / 总个数
print('真实值:',test_y.values)
print('预测值:',pred_test_y)
# print('准确率:',(test_y==pred_test_y).sum() / test_y.size)
print('准确率:',sm.accuracy_score(test_y,pred_test_y))
print('查准率:',sm.precision_score(test_y,
                                pred_test_y,
                                average='macro'))
print('召回率:',sm.recall_score(test_y,
                             pred_test_y,
                             average='macro'))
print('f1得分:',sm.f1_score(test_y,
                          pred_test_y,
                          average='macro'))

print('混淆矩阵:\n',sm.confusion_matrix(test_y,pred_test_y))

print('分类报告:\n',sm.classification_report(test_y,pred_test_y))







