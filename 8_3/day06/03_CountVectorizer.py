
import nltk.tokenize as tk #分词模块
import sklearn.feature_extraction.text as ft #文本特征提取
import sklearn.preprocessing as sp #数据预处理

doc = 'This hotel is very bad. The toilet in this hotel smells bad. The environment of this hotel is very good.'

sents = tk.sent_tokenize(doc)

# 1.构建词袋模型对象
cv = ft.CountVectorizer()
res = cv.fit_transform(sents).toarray()
# print(res)
#
# print(cv.get_feature_names())

#词频(对词袋模型进行归一化)
# tf = sp.normalize(res,norm='l1')
# print(tf)

# TF-IDF
# 1.构建词袋模型
cv = ft.CountVectorizer()
bow = cv.fit_transform(sents)
# 2.构建词频逆文档频率
tt = ft.TfidfTransformer()
res = tt.fit_transform(bow)
print(res.toarray())