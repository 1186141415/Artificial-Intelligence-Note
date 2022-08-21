## 舆情分析

文本情感分析又称意见挖掘、倾向性分析等。简单而言,是对带有情感色彩的主观性文本进行分析、处理、归纳和推理的过程。互联网产生了大量的诸如人物、事件、产品等有价值的评论信息。这些评论信息表达了人们的各种情感色彩和情感倾向性,如喜、怒、哀、乐和批评、赞扬等。基于此,潜在的用户就可以通过浏览这些主观色彩的评论来了解大众舆论对于某一事件或产品的看法。

酒店评论

```
1. 房间真棒，离大马路很近，非常方便。不错。				好评
2. 房间有点脏，厕所还漏水，空调不制冷，下次再也不来了。   差评  
3. 地板不太干净，电视没信号，但是空调还可以，总之还行。   好评 

张平平安全到家了
```

原理：

先针对训练文本进行分词处理，统计词频，通过词频-逆文档频率算法获得该词对样本语义的贡献，根据每个词的贡献力度，构建有监督分类学习模型。把测试样本交给模型处理，得到测试样本的情感类别。

```
pip3 install nltk -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip3 install jieba -i https://pypi.tuna.tsinghua.edu.cn/simple/


what are you doing now ?
我真帅呀！
```

### 文本分词

分词处理相关API： nltk将会寻找punkt资源：

~/nltk_data/tokenizers/punkt/

```python
import nltk.tokenize as tk

# 把样本按句子进行拆分  sent_list:句子列表
sent_list = tk.sent_tokenize(text)

# 把样本按单词进行拆分  word_list:单词列表
word_list = tk.word_tokenize(text)

#  把样本按单词进行拆分 punctTokenizer：分词器对象
punctTokenizer = tk.WordPunctTokenizer()
word_list = punctTokenizer.tokenize(text)
```

案例：

```python
import nltk.tokenize as tk

doc = "Are you curious about tokenization? "
"Let's see how it works! "
"We need to analyze a couple of sentences "
"with punctuations to see it in action."
print(doc)
tokens = tk.sent_tokenize(doc)
for i, token in enumerate(tokens):
    print("%2d" % (i + 1), token)
print('-' * 15)
tokens = tk.word_tokenize(doc)
for i, token in enumerate(tokens):
    print("%2d" % (i + 1), token)
print('-' * 15)
tokenizer = tk.WordPunctTokenizer()
tokens = tokenizer.tokenize(doc)
for i, token in enumerate(tokens):
    print("%2d" % (i + 1), token)
```

### 文本向量化处理

在训练语义分析模型时，需要以每一个段落为一个样本

```
一句话为一个样本
```

这一段落的语义为类别标签构建训练样本集

```
吧语义（好评，差评）    标签：输出Y
```

需要一种算法可以把一句话（一个样本）转成一个特征向量，需要该向量通过数字表达语义

### 词袋模型

一句话的语义很大程度取决于某个单词出现的次数，所以可以把句子中所有可能出现的单词作为特征名，每一个句子为一个样本，单词在句子中出现的次数为特征值构建数学模型，称为词袋模型。

This hotel is very bad. The toilet in this hotel smells bad. The environment of this hotel is very good.

1 This hotel is very bad.   
2 The toilet in this hotel smells bad.   
3 The environment of this hotel is very good.

| This | hotel | is   | very | bad  | The  | toilet | in   | smells | environment | of   | good |
| ---- | ----- | ---- | ---- | ---- | ---- | ------ | ---- | ------ | ----------- | ---- | ---- |
| 1    | 1     | 1    | 1    | 1    | 0    | 0      | 0    | 0      | 0           | 0    | 0    |
| 1    | 1     | 0    | 0    | 1    | 1    | 1      | 1    | 1      | 0           | 0    | 0    |
| 1    | 1     | 1    | 1    | 0    | 1    | 0      | 0    | 0      | 1           | 1    | 1    |

​

词袋模型化相关API：

```python
import sklearn.feature_extraction.text as ft  # 特征提取模块

# 构建词袋模型对象
cv = ft.CountVectorizer()
# 训练模型，把句子中所有可能出现的单词作为特征名，每一个句子为一个样本，单词在句子中出现的次数为特征值。
bow = cv.fit_transform(sentences).toarray()
print(bow)
# 获取所有特征名
words = cv.get_feature_names()
```

案例：

```python
import nltk.tokenize as tk
import sklearn.feature_extraction.text as ft

doc = 'This hotel is very bad. The toilet in this hotel smells bad. The environment of this hotel is very good.'
print(doc)
sentences = tk.sent_tokenize(doc)
print(sentences)
cv = ft.CountVectorizer()
bow = cv.fit_transform(sentences).toarray()
print(bow)
words = cv.get_feature_names()
print(words)

```

### 词频（TF）

单词在句子中出现的次数除以句子的总词数称为词频。即一个单词在一个句子中出现的频率。词频相比单词的出现次数可以更加客观的评估单词对一句话的语义的贡献度。词频越高，对语义的贡献度越大。对词袋矩阵归一化即可得到词频。



案例：对词袋矩阵进行归一化

```
这家酒店棒，装修棒，早餐棒，环境棒  			1好评      2/7
这家酒店烂，烂烂烂，真的烂，烂				 0差评	  1/2

这家酒店装修棒，服务烂     

```

```python
import nltk.tokenize as tk
import sklearn.feature_extraction.text as ft
import sklearn.preprocessing as sp

doc = 'This hotel is very bad. The toilet in this hotel smells bad. The environment of this hotel is very good.'
print(doc)
sentences = tk.sent_tokenize(doc)
print(sentences)
cv = ft.CountVectorizer()
bow = cv.fit_transform(sentences).toarray()
print(bow)
words = cv.get_feature_names()
print(words)
tf = sp.normalize(bow, norm='l1')
print(tf)
```

有些词可能大部分样本中都会出现 比如代码：我。。。 语气助词

这种词在大多数样本中出现，对判断样本的类别没有太大的作用

需要设计一个算法对这些单词的语义贡献度降低

### 文档频率（DF）

$$DF = \frac{含有某个单词的文档样本数}{总文档样本数}$$

文档频率越高，说明这个单词在文档中出现的概率就越大，越大越没啥用

与样本语义贡献度 反相关

````
我今天非常高兴来到这个酒店，这个酒店很好
我今天不开心，因为，这个酒店很差
我今天，，，，，，
我今天。。。。。
今天非常
我。。。
我，，，
我，，，

我： 文档频率 --》 7/8

文档频率 与 样本语义贡献度 反相关

````

### 逆文档频率（IDF）

$$IDF =log( \frac{总样本数}{1+含有某个单词的样本数})$$

逆文档频率与文档频率 反相关

逆文档频率和样本语义贡献度正相关

当前单词的逆文档频率越大，那该单词对于样本语义贡献度越大

### 词频-逆文档频率(TF-IDF)

词频矩阵中的每一个元素乘以相应单词的逆文档频率，其值越大说明该词对样本语义的贡献越大，根据每个词的贡献力度，构建学习模型。

获取词频逆文档频率（TF-IDF）矩阵相关API：

```python
# 获取词袋模型
cv = ft.CountVectorizer()
bow = cv.fit_transform(sentences).toarray()
# 获取TF-IDF模型训练器
tt = ft.TfidfTransformer()
tfidf = tt.fit_transform(bow).toarray()
```

案例：获取TF_IDF矩阵：

```python
import nltk.tokenize as tk
import sklearn.feature_extraction.text as ft

doc = 'This hotel is very bad. The toilet in this hotel smells bad. The environment of this hotel is very good.'

print(doc)
sentences = tk.sent_tokenize(doc)
print(sentences)
cv = ft.CountVectorizer()
bow = cv.fit_transform(sentences).toarray()
print(bow)
words = cv.get_feature_names()
print(words)
tt = ft.TfidfTransformer()
tfidf = tt.fit_transform(bow).toarray()
print(tfidf)
```

### 文本分类(主题识别)

使用给定的文本数据集进行主题识别训练，自定义测试集测试模型准确性。

识别邮件类别

​

案例：

```python
import sklearn.datasets as sd
import sklearn.feature_extraction.text as ft
import sklearn.naive_bayes as nb

train = sd.load_files('../data/20news', encoding='latin1',
                      shuffle=True, random_state=7)
# 20news 下的文件夹名即是相应子文件的主题类别名
# train.data 返回每个文件的字符串内容
# train.target 返回每个文件的父目录名（主题类别名）
train_data = train.data
train_y = train.target
categories = train.target_names
cv = ft.CountVectorizer()
train_bow = cv.fit_transform(train_data)
tt = ft.TfidfTransformer()
train_x = tt.fit_transform(train_bow)
model = nb.MultinomialNB()
model.fit(train_x, train_y)
test_data = [
    'The curveballs of right handed pitchers tend to curve to the left',
    'Caesar cipher is an ancient form of encryption',
    'This two-wheeler is really good on slippery roads']
test_bow = cv.transform(test_data)
test_x = tt.transform(test_bow)
pred_test_y = model.predict(test_x)
for sentence, index in zip(test_data, pred_test_y):
    print(sentence, '->', categories[index])

```

