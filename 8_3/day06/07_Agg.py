'''
凝聚层次算法
'''
import pandas as pd
import sklearn.cluster as sc  # 聚类模块
import matplotlib.pyplot as plt
import sklearn.metrics as sm  # 评估模块

data = pd.read_csv('../data_test/multiple3.txt',
                   header=None,
                   names=['x1', 'x2'])
# print(data.head())
model = sc.AgglomerativeClustering(n_clusters=4)
model.fit(data)

# 拿到预测类别
labels = model.labels_

# 将几何中心画在散点图上
plt.scatter(data['x1'], data['x2'], s=50, c=labels, cmap='brg')
plt.colorbar()
plt.show()

score = sm.silhouette_score(data,
                            labels,
                            metric='euclidean',
                            sample_size=len(data))
print(score)
