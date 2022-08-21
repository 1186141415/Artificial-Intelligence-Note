'''
英文分词
'''
import nltk.tokenize as tk #英文分词

doc = "Are you curious about tokenization? " \
      "Let's see how it works! " \
      "We need to analyze a couple of sentences " \
      "with punctuations to see it in action."

#分句子
# res = tk.sent_tokenize(doc)
# for i in range(len(res)):
#     print(i,':',res[i])

# 分单词
# words = tk.word_tokenize(doc)
# for i in range(len(words)):
#     print(i,':',words[i])

#分词器对象分词
tokenizer = tk.WordPunctTokenizer()
words = tokenizer.tokenize(doc)
for i in range(len(words)):
    print(i,':',words[i])