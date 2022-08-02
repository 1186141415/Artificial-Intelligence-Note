'''
支持向量机 SVM
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms #模型选择
import sklearn.svm as svm #支持向量机
import sklearn.metrics as sm #评估模块

data = pd.read_csv('../data_test/multiple2.txt',
                   header=None,
                   names=['x1','x2','y'])
# print(data.head())
# print(data['y'].value_counts())



x = data.iloc[:,:-1]
y = data.iloc[:,-1]

train_x,\
test_x,\
train_y,\
test_y = ms.train_test_split(x,y,test_size=0.1,
                             random_state=7,
                             stratify=y)
# model = svm.SVC(kernel='linear')
# model = svm.SVC(kernel='poly',degree=3)
model = svm.SVC(kernel='rbf',gamma=0.1,C=1)
model.fit(train_x,train_y)
pred_test_y = model.predict(test_x)
print(sm.classification_report(test_y,pred_test_y))

#绘制分类边界线
# 1.将x1的最小值 到x1的最大值 拆成200个点
x1s = np.linspace(data['x1'].min(),data['x1'].max(),200)
# 2.将x2的最小值 到x2的最大值 拆成200个点
x2s = np.linspace(data['x2'].min(),data['x2'].max(),200)
# 3.组合x1和x2的所有点，4W个 （二维）
points = []
for x1 in x1s:
    for x2 in x2s:
        points.append([x1,x2])
points = pd.DataFrame(points,columns=['x1','x2'])
# 4.将4W个点带入模型中得到预测类别
points_label = model.predict(points)
# 5.将4W个点进行散点图的绘制，根据预测类别划分颜色 cmap='gray'
plt.scatter(points['x1'],points['x2'],c=points_label,cmap='gray')
# 6.讲样本数据绘制在图上
plt.scatter(data['x1'],data['x2'],c=data['y'],cmap='brg')
plt.colorbar()
plt.show()



