
import numpy as np
import jieba
import re
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score

#coding:utf-8
# _*_ coding: utf-8 _*_
yuhao=pd.read_csv('weibo_senti_100k.csv',encoding='utf-8')

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

def cutstop(text):
   w=jieba.cut(text)
   fenci=' '.join(w)
   stopwords = {}.fromkeys([ line.rstrip() for line in open('stopwords.txt') ])
   fwl=''
   fwl2=''
   word=''   #去停词
   for seg in fenci:
       if seg not in stopwords:
           fwl+=seg
   for seg1 in fwl:
       word+=seg1
       if seg1==' ' and len(word)<3:
           word=''
       if seg1==' ' and  len(word)>2:
               fwl2+=word
               word=''
   return(fwl2)
l1=[]
l2=[]
data=[]
target=[]

for i in range(0,5000):
    a=cutstop(yuhao.review[i])
    a=remove_pattern(a,'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+')
    b=cutstop(yuhao.review[i+65500])
    b=remove_pattern(b,'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+')
    l1.append(a)
    l2.append(b)
x=[]
c=[]

for i in l1:
    x.append(i+"\n")
file_handle2=open('l1.txt',mode='w')
file_handle2.writelines(x)
file_handle2.close()

for i in l2:
    c.append(i+"\n")
file_handle2=open('l2.txt',mode='w')
file_handle2.writelines(c)
file_handle2.close()
k=[]
l=[]
conut1=0
conut2=0
with open('l1.txt','r') as fr,open('l12.txt','w') as fd:
        for text in fr.readlines():
                if text.split():
                        fd.write(text)
        print('输出成功....')
with open('l2.txt','r') as fr,open('l22.txt','w') as fd:
        for text in fr.readlines():
                if text.split():
                        fd.write(text)
        print('输出成功....')

file_handle2=open('l12.txt',mode='r')
for i in file_handle2:
    k.append(i)
    conut1+=1

print(conut1)
file_handle2.close()

file_handle2=open('l22.txt',mode='r')
for i in file_handle2:
    l.append(i)
    conut2+=1

print(conut2)
file_handle2.close()
data2=k+l


list1=[]
for i in range(0,conut1):
    list1.append('1')
list2=[]
for i in range(0,conut2):
    list2.append('0')
target=list1+list2
np.array(target)

import warnings
warnings.filterwarnings('ignore')
from gensim.models import Word2Vec
word2vec_model = Word2Vec(data2, size=20, iter=10, min_count=20)
word2vec_model.save( 'word2vec_model.w2v' )

def getVector_v4(cutWords, word2vec_model):
        i = 0
        index2word_set = set(word2vec_model.wv.index2word)
        article_vector = np.zeros((word2vec_model.layer1_size))
        for cutWord in cutWords:
                if cutWord in index2word_set:
                        article_vector = np.add(article_vector, word2vec_model.wv[cutWord])
                        i += 1
        cutWord_vector = np.divide(article_vector, i)
        return cutWord_vector


vector_list = []
for cutWords in data2:
    vector_list.append( getVector_v4(cutWords, word2vec_model) )

X = np.array(vector_list)
X.dump('articles_vector.txt')


# print(np.isnan(X).any())
# for i in X:
#     i.dropna(inplace=True)


# t=[]
# for i in X:
#     for n in i:
#         if not str(n).isdigit():
#             print(i)
#             if n==nan:

        # n=float(n)
        # t.append(n)
#
#
# print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
# print(max(t))
# print(min(t))
# print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


'''
file_handle=open('x.txt',mode='w')
file_handle.writelines(t)
file_handle.close()
'''

def getVectorMatrix(article_series):
        return np.array([getVector_v4(jieba.cut(k), word2vec_model) for k in article_series])

x_train,x_test,y_train,_y_test=train_test_split(X,target,test_size=0.2, random_state=0)
logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train)
logistic_model.score(x_test, _y_test)

cv_split = ShuffleSplit(n_splits=5, train_size=0.7, test_size=0.2)
score_ndarray = cross_val_score(logistic_model, X, target, cv=cv_split)
print(score_ndarray)
print(score_ndarray.mean())


