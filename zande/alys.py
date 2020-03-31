# -*- coding: utf-8 -*-

import math
import jieba
import jieba.posseg as psg
from jieba import analyse
import functools


#预处理
'''
fpub=open('pubtime.txt','r+')
fup=open('uptime.txt','r+')
ftx=open('text.txt','r+')
ftag=open('tags.txt','r+')

lpub=fpub.readlines()
lup=fup.readlines()
ltx=ftx.readlines()
ltg=ftag.readlines()

emp=[]
lsp=[]
lsu=[]
lst=[]
ltag=[]
cmp=[]
check=[]

for line in lpub:
    lsp.append(line[0:-1])
for line in lup:
    lsu.append(line[0:-1])
for line in ltx:
    lst.append(line[0:-1])
for line in ltg:
    ltag.append(line[0:-1])

for i in range(9):
    emp=[]
    emp.append(lsp[i])
    cmp.append(emp)
    check.append(emp)


for nu in range(len(lsu)):
    for i in range(9):
        if lsu[nu] >= ((check[i])[0]):
            #check[i].append(lsu[nu])
            cmp[i].append(lst[nu])
            break

douhao='，'

for i in cmp:
    del i[0]

for i in cmp:
    douhao.join(i)
'''

#模型


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


def load_data(pos=False, corpus_path='text.txt'):
    doc_list = []
    for line in open(corpus_path, 'r',encoding='unicode_escape'):
        content = line.strip()
        seg_list = seg_to_list(content, pos)
        filter_list = word_filter(seg_list, pos)
        doc_list.append(filter_list)

    return doc_list


def train_idf(doc_list):
    idf_dic = {}
    tt_count = len(doc_list)

    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0

    for k, v in idf_dic.items():
        idf_dic[k] = math.log(tt_count / (1.0 + v))

    default_idf = math.log(tt_count / (1.0))
    return idf_dic, default_idf


def cmp(e1, e2):
    import numpy as np
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1

class TfIdf(object):

    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        self.word_list = word_list
        self.idf_dic, self.default_idf = idf_dic, default_idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num

    def get_tf_dic(self):
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0.0) + 1.0

        tt_count = len(self.word_list)
        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / tt_count

        return tf_dic

    def get_tfidf(self):
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)

            tfidf = tf * idf
            tfidf_dic[word] = tfidf

        fopen=open("tikey.txt","a+")
        tflis=[]
        tfidf_dic.items()
        for k, v in sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + "/ ", end='')
            tflis.append(k+'/')
        for l in tflis:
            fopen.write(l)
        fopen.write('\n')
        fopen.close
        print()



def tfidf_extract(word_list, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    idf_dic, default_idf = train_idf(doc_list)
    tfidf_model = TfIdf(idf_dic, default_idf, word_list, keyword_num)
    tfidf_model.get_tfidf()


def textrank_extract(text, pos=False, keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num)
    fopen=open("trkey.txt","a+")
    trlis=[]
    for keyword in keywords:
        print(keyword + "/ ", end='')
        trlis.append(keyword+'/')
    for l in trlis:
        fopen.write(l)  
    fopen.write('\n')
    fopen.close
    print()


#run



if __name__ == '__main__':
    pos = True
    
    
    ft=open('jiebats.txt',"r+",encoding='utf-8')#encoding='utf-8'
    text=''
    f=ft.readlines()
    for line in f:
        text=text+line
    seg_list = seg_to_list(text, pos)
    filter_list = word_filter(seg_list, pos)

    print('TF-IDF模型结果：')
    tfidf_extract(filter_list)
    print('TextRank模型结果：')
    textrank_extract(text)

    ft.close()



'''
pos = True
    ft=open('jiebats.txt',"r+",encoding='utf-8')
    text=''
    f=ft.readlines()
    for line in f:
        text=text+line
    seg_list = seg_to_list(text, pos)
    filter_list = word_filter(seg_list, pos)

    print('TF-IDF模型结果：')
    tfidf_extract(filter_list)
    print('TextRank模型结果：')
    textrank_extract(text)

    ft.close()
'''
