'''
小汽车等级预测
'''
import pandas as pd
import sklearn.preprocessing as sp #数据预处理
import sklearn.ensemble as se #集成学习

data = pd.read_csv('../data_test/car.txt',
                   header=None)

#数据预处理（标签编码）
train_data = pd.DataFrame() #用于保存处理结果
encoders = {}
for i in data:
    encoder = sp.LabelEncoder()
    res = encoder.fit_transform(data[i])
    train_data[i] = res
    encoders[i] = encoder
#整理输入和输出（使用全部样本进行训练）
x = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

#构建模型（随机森林）
model = se.RandomForestClassifier(max_depth=6,
                                  n_estimators=400,
                                  random_state=7)
#执行训练
model.fit(x,y)


test_data = [['high','med','5more','4','big','low','unacc'],
             ['high','high','4','4','med','med','acc'],
             ['low','low','2','4','small','high','good'],
             ['low','med','3','4','med','high','vgood']]

#测试数据和训练数据要保持一致
test_data = pd.DataFrame(test_data)
for i in test_data:
    encoder = encoders[i]#拿到训练集对应的列的编码器
    res = encoder.transform(test_data[i])
    test_data[i] = res
# print(test_data)
test_x = test_data.iloc[:,:-1]
test_y = test_data.iloc[:,-1]
#预测评估（准确率）
pred_test_y = model.predict(test_x)

print('真实值:',encoders[6].inverse_transform(test_y.values))
print('预测值:',encoders[6].inverse_transform(pred_test_y))


