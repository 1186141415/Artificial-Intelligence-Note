'''
K-means算法
'''
import pandas as pd
import sklearn.cluster as sc #聚类模块
import matplotlib.pyplot as plt

data = pd.read_csv('../data_test/multiple3.txt',
                   header=None,
                   names=['x1','x2'])
# print(data.head())
model = sc.KMeans(n_clusters=4)
model.fit(data)

#拿到预测类别
labels = model.labels_
#拿到几何中心
center = model.cluster_centers_
# print(center)

#将几何中心画在散点图上
plt.scatter(data['x1'],data['x2'],s=50,c=labels,cmap='brg')
plt.colorbar()
plt.scatter(center[:,0],center[:,-1],c='black',s=200,marker='+')
plt.show()





