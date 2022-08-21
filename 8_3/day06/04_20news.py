'''
邮件分类
'''
import sklearn.datasets as sd #数据集合
import sklearn.feature_extraction.text as ft #文本特征提取
import sklearn.model_selection as ms #模型选择
import sklearn.linear_model as lm #线性模型
import sklearn.metrics as sm #评估模块
import sklearn.naive_bayes as nb #朴素贝叶斯

data = sd.load_files('../data_test/20news',
                     random_state=7,
                     encoding='latin1')

# print(data.keys())
# print(data.data[0])
# print(data.target[0])
# print(data.target_names)

#整理输入和输出
x = data.data
y = data.target
#对x进行预处理（TFIDF）
# 1.构建词袋模型
cv = ft.CountVectorizer()
bow = cv.fit_transform(x)
# 2.构建tf-idf
tt = ft.TfidfTransformer()
tfidf = tt.fit_transform(bow)
# print(tfidf.shape)

#划分训练集和测试集
train_x,\
test_x,\
train_y,\
test_y = ms.train_test_split(tfidf,y,
                             test_size=0.1,
                             random_state=7,
                             stratify=y)

# model = lm.LogisticRegression(solver='liblinear')
model = nb.MultinomialNB()
model.fit(train_x,train_y)
pred_test_y = model.predict(test_x)
print(sm.classification_report(test_y,pred_test_y))


test_data = ['At the last game, a spectator was hit by a baseball and hospitalized',
             'Recently Lao Wang is working on asymmetric encryption algorithms',
             'These two wheels are pretty fast on the freeway',
             'China will explore Mars next year']

bow = cv.transform(test_data)
test_x = tt.transform(bow)
pred_test_y = model.predict(test_x)
print(data.target_names)
print('预测类别:',pred_test_y)