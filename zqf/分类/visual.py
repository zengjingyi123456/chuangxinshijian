import pandas as pd
train_df = pd.read_csv('weibo_senti_100k.txt', sep='\t', header=None)
train_df.head()

for name, group in train_df.groupby(0):
    print(name, '\t', len(group))
train_df.columns = ['Subject', 'Content']
train_df['Subject'].value_counts().sort_index()

test_df = pd.read_csv('weibo_senti_100k.txt', sep='\t', header=None)
for name, group in test_df.groupby(0):
    print(name, '\t', len(group))

for name, group in df_train.groupby(0):
    print(name)
    print(group)

import jieba, time

train_df.columns = ['分类', '文章']
# stopword_list = [k.strip() for k in open('stopwords.txt', encoding='utf-8').readlines() if k.strip() != '']
# 上面的语句不建议这么写，因为readlines()是一下子将所有内容读入内存，如果文件过大，会很耗内存，建议这么写
stopword_list = [k.strip() for k in open('stopwords.txt', encoding='utf-8') if k.strip() != '']

cutWords_list = []

i = 0
startTime = time.time()
for article in train_df['文章']:
    cutWords = [k for k in jieba.cut(article) if k not in stopword_list]
    i += 1
    if i % 1000 == 0:
        print('前%d篇文章分词共花费%.2f秒' % (i, time.time() - startTime))
    cutWords_list.append(cutWords)

with open('cutWords_list.txt', 'w') as file:
    for cutWords in cutWords_list:
        file.write(' '.join(cutWords) + '\n')

with open('cutWords_list.txt') as file:
    cutWords_list = [ k.split() for k in file ]

import warnings

warnings.filterwarnings('ignore')
from gensim.models import Word2Vec
word2vec_model = Word2Vec(cutWords_list, size=100, iter=10, min_count=20)


def getVector_v1(cutWords, word2vec_model):
    count = 0
    article_vector = np.zeros(word2vec_model.layer1_size)
    for cutWord in cutWords:
        if cutWord in word2vec_model:
            article_vector += word2vec_model[cutWord]
            count += 1

    return article_vector / count


startTime = time.time()
vector_list = []
i = 0
for cutWords in cutWords_list[:5000]:
    i += 1
    if i % 1000 == 0:
        print('前%d篇文章形成词向量花费%.2f秒' % (i, time.time() - startTime))
    vector_list.append(getVector_v1(cutWords, word2vec_model))
X = np.array(vector_list)
print('Total Time You Need To Get X:%.2f秒' % (time.time() - startTime))

import pandas as pd

from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('sohu_train.txt', sep='\t', header=None)
train_df.columns = ['分类', '文章']
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(train_df['分类'])

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

import pandas as pd
import numpy as np
from sklearn.externals import joblib
import jieba


def getVectorMatrix(article_series):
    return np.array([getVector_v4(jieba.cut(k), word2vec_model) for k in article_series])


logistic_model = joblib.load('logistic.model')

test_df = pd.read_csv('sohu_test.txt', sep='\t', header=None)
test_df.columns = ['分类', '文章']
for name, group in test_df.groupby('分类'):
    featureMatrix = getVectorMatrix(group['文章'])
    target = labelEncoder.transform(group['分类'])
    print(name, logistic_model.score(featureMatrix, target))

from sklearn.metrics import classification_report
test_df = pd.read_csv('sohu_test.txt', sep='\t', header=None)
test_df.columns = ['分类', '文章']
test_label = labelEncoder.transform(test_df['分类'])
y_pred = logistic_model.predict( getVectorMatrix(test_df['文章']) )
print(labelEncoder.inverse_transform([[x] for x in range(12)]))
print(classification_report(test_label, y_pred))