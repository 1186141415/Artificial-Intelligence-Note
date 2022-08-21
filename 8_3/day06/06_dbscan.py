'''
DBSCAN算法
'''
import pandas as pd
import sklearn.cluster as sc #聚类模块
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import numpy as np

data = pd.read_csv('../data_test/multiple3.txt',
                   header=None,
                   names=['x1','x2'])
# # print(data.head())
# model = sc.DBSCAN(eps=0.65,min_samples=5)
# model.fit(data)
# #拿到预测类别
# labels = model.labels_
#
# #将几何中心画在散点图上
# plt.scatter(data['x1'],data['x2'],s=50,c=labels,cmap='brg')
# plt.colorbar()
# plt.show()

eps_params = np.arange(0.5,1.0,0.1)
samples_params = np.arange(5,14)
scores = []
index = []
for i in eps_params:
    for j in samples_params:
        model = sc.DBSCAN(eps=i,min_samples=j)
        model.fit(data)
        labels = model.labels_
        score = sm.silhouette_score(data,labels,
                                    sample_size=len(data))
        scores.append(score)
        index.append('eps:{},sample:{}'.format(i,j))

scores = pd.Series(scores,index=index)
print(scores.idxmax())

