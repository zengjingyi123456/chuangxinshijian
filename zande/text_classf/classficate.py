import jieba
import jieba.posseg as pseg
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
from gensim import corpora, models
from jieba import analyse
import functools
import pandas as pd
import numpy as np

pd_all = pd.read_csv('weibo_senti_100k.csv')


outstr = []

psd=pd_all[pd_all.label==1][0:40]
nd=pd_all[pd_all.label==0][0:40]

ptd=pd_all[pd_all.label==1][0:2000]
ntd=pd_all[pd_all.label==0][0:2000]

pl=[]
nl=[]

ptl=[]
ntl=[]

pl2=[]
nl2=[]

ptl2=[]
ntl2=[]

pl3=[]
nl3=[]
ptl3=[]
ntl3=[]


r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

np=nd.drop(['label'],axis=1)
psd=psd.drop(['label'],axis=1)

pl=psd.review.tolist()
nl=nd.review.tolist()

ntd=ntd.drop(['label'],axis=1)
ptd=ptd.drop(['label'],axis=1)

ptl=psd.review.tolist()
ntl=nd.review.tolist()


for i in pl:
    pl2.append(re.sub(r1,'',i))

for i in nl:
    nl2.append(re.sub(r1,'',i))

for i in ptl:
    ptl2.append(re.sub(r1,'',i))

for i in ntl:
    ntl2.append(re.sub(r1,'',i))



def StopWords():
    filepath = 'stopwords.txt'
    wlst = [w.strip() for w in open(filepath, 'r', encoding='utf8').readlines()]
    return wlst


def seg_sentence(sentence, stop_words):
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    sentence_seged = pseg.cut(re.sub('\n', '', sentence))
    for word, flag in sentence_seged:
        if word not in stop_words and flag not in stop_flag:
            outstr.append(word)
    return ' '.join(outstr)


# 朴素贝叶斯
def model(tfidf,y):
#     clf = MultinomialNB(alpha=0.1).fit(tfidf,y)
    clf = BernoulliNB(alpha=0.1).fit(tfidf,y)
#     clf = GaussianNB(alpha=0.1).fit(tfidf,y)
    return clf

stop_words = StopWords()

for i in pl2:
    pl3.append(seg_sentence(i, stop_words))
for i in nl2:
    nl3.append(seg_sentence(i, stop_words))
for i in ptl2:
    ptl3.append(seg_sentence(i, stop_words))
for i in ntl2:
    ntl3.append(seg_sentence(i, stop_words))



corpus = pl3+nl3

testdata=ptl3+ntl3

y = [0] * len(pl3) + [1] * len(nl3)

x_train, x_test, y_train, y_test = train_test_split(corpus, y, test_size=0.5)

#vector = CountVectorizer()
vector = TfidfVectorizer()
xtrain = vector.fit_transform(x_train)

xtest = vector.transform(x_test)

#模型
clf = model(xtrain, y_train)
pre = clf.predict(xtrain)

#评估
#print('======================================================')
pre = clf.predict(xtest)
#print(pre)
#print('======================================================')
print('testdata准确率：',metrics.f1_score(pre, y_test, average='micro'))



'''


def get_stopword_list():
    stop_word_path = 'stopwords.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path,encoding='utf-8').readlines()]
    return stopword_list


def seg_to_list(sentence, pos=False):
    if not pos:
        seg_list = jieba.cut(sentence)
    else:
        seg_list = psg.cut(sentence)
    return seg_list


def word_filter(seg_list, pos=False):
    stopword_list = get_stopword_list()
    filter_list = []
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        if not word in stopword_list and len(word) > 1:
            filter_list.append(word)

    return filter_list



'''
