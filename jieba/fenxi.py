from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import re
import jieba
import numpy as np
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
    b=cutstop(yuhao.review[i+65000])
    b=remove_pattern(b,'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+')
    l1.append(a)
    l2.append(b)
data=l1+l2

list1=[]
for i in range(0,5000):
    list1.append('1')
list2=[]
for i in range(0,5000):
    list2.append('0')
target=list1+list2

np.array(target)

x_train,x_test,y_train,_y_test=train_test_split(data,target,test_size=0.1,random_state=21)

transfer= TfidfVectorizer()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

estimator=MultinomialNB()
estimator.fit(x_train,y_train)

y_predict=estimator.predict(x_test)
# print('y_predict:',y_predict)
# print('预测值是否正确:',_y_test==y_predict)

score=estimator.score(x_test,_y_test)
print("准确率",score)





