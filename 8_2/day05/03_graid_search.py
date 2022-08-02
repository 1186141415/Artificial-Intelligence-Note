'''
网格搜索：寻找最优的超参数组合
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

params = [{'kernel':['linear'],'C':[1,10,100]},
          {'kernel':['poly'],'degree':[2,3,4],'C':[1,10,100]},
          {'kernel':['rbf'],'gamma':[1,0.1,0.01],'C':[1,10,100]}]

model = ms.GridSearchCV(svm.SVC(),params,cv=5)

model.fit(x,y)

print('最优秀的模型参数:',model.best_params_)
print('最优秀的模型打分:',model.best_score_)


