'''朴素贝叶斯'''

import pandas as pd
import sklearn.model_selection as ms #模型选择
import sklearn.naive_bayes as nb #朴素贝叶斯
import sklearn.metrics as sm #评估模块
import matplotlib.pyplot as plt

data = pd.read_csv('../data_test/multiple1.txt',
                   header=None,
                   names=['x1','x2','y'])
# print(data.head())
x = data.iloc[:,:-1]
y = data.iloc[:,-1]

train_x,\
test_x,\
train_y,\
test_y = ms.train_test_split(x,y,test_size=0.1,
                             random_state=7,
                             stratify=y)
model = nb.GaussianNB()
model.fit(train_x,train_y)
pred_test_y = model.predict(test_x)
print(sm.classification_report(test_y,pred_test_y))


plt.scatter(data['x1'],data['x2'],c=data['y'],cmap='brg')
plt.show()