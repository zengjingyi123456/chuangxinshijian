import pandas as pd
train_df = pd.read_csv('weibo_senti_100k.csv')
print(train_df.head())

# for name, group in train_df.groupby(0):
#     print(name, '\t', len(group))
#
# for name, group in train_df.groupby(0):
#     print(name)
#     print(group)

import jieba, time
print(train_df.shape)

stopword_list = [k.strip() for k in open('stopwords.txt', encoding='utf-8') if k.strip() != '']
cutWords_list = []

i = 0
startTime = time.time()
for article in train_df['review']:
    cutWords = [k for k in jieba.cut(article) if k not in stopword_list]
    i += 1
    if i % 1000 == 0:
        print('前%d条评论分词共花费%.2f秒' % (i, time.time() - startTime))
    cutWords_list.append(cutWords)

with open('cutWords_list.txt', 'w', encoding='utf-8') as file:
    for cutWords in cutWords_list:
        file.write(' '.join(cutWords) + '\n')

with open('cutWords_list.txt',encoding='utf-8') as file:
    cutWords_list = [ k.split() for k in file ]

import warnings
warnings.filterwarnings('ignore')
from gensim.models import Word2Vec
word2vec_model = Word2Vec(cutWords_list, size=100, iter=10, min_count=20)

import time
import pandas as pd
import numpy as np


def getVector_v2(cutWords, word2vec_model):
    vector_list = [word2vec_model[k] for k in cutWords if k in word2vec_model]
    vector_df = pd.DataFrame(vector_list)
    cutWord_vector = vector_df.mean(axis=0).values
    return cutWord_vector


startTime = time.time()
vector_list = []
i = 0
for cutWords in cutWords_list[:5000]:
    i += 1
    if i % 1000 == 0:
        print('前%d篇文章形成词向量花费%.2f秒' % (i, time.time() - startTime))
    vector_list.append(getVector_v2(cutWords, word2vec_model))
X = np.array(vector_list)
print('Total Time You Need To Get X:%.2f秒' % (time.time() - startTime))

import pandas as pd

from sklearn.preprocessing import LabelEncoder
train_df = pd.read_csv('weibo_senti_100k.txt', sep='\t', header=None)
print(train_df.shape)
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(train_df.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

logistic_model = LogisticRegression()
logistic_model.fit(train_X, train_y)
logistic_model.score(test_X, test_y)

from sklearn.externals import joblib

joblib.dump(logistic_model, 'logistic.model')

# 加载模型
logistic_model = joblib.load('logistic.model')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv_split = ShuffleSplit(n_splits=5, train_size=0.7, test_size=0.2)
logistic_model = LogisticRegression()
score_ndarray = cross_val_score(logistic_model, X, y, cv=cv_split)
print(score_ndarray)
print(score_ndarray.mean())