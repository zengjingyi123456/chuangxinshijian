import multiprocessing

import jieba
import re
import jieba.posseg as psg

#coding:utf-8
# _*_ coding: utf-8 _*_
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def gethandle(txtname):
   fileobj = open(txtname, 'r',encoding='utf-8')    #读取文本放到strings中
   try:
      strings = fileobj.read()
   finally:
      fileobj.close()

   sentences = re.split('(。|！|\!|\.|？|\?)',strings)         #使用re分句
   new_sents = []
   for i in range(int(len(sentences)/2)):
       sent = sentences[2*i] + sentences[2*i+1]
       new_sents.append(sent)


   #分词
   w=jieba.cut(strings)
   fenci=' '.join(w)
   # file_handle=open('分词结果.txt',mode='w',encoding='utf-8')   #分词结果写入文本
   # file_handle.writelines(fenci)
   # file_handle.close()

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
   file_handle2=open('weibo_cut.txt',mode='w',encoding='utf-8')
   file_handle2.writelines(fwl2)
   file_handle2.close()

gethandle('weibo_senti_100k.txt')


inp = 'weibo_cut.txt'
out_model = 'corpusSegDone_1.model'
out_vector = 'corpusSegDone2_1.vector'

# 训练skip-gram模型
model = Word2Vec(LineSentence(inp), size=1, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())
# 保存模型
model.save(out_model)
# 保存词向量
model.wv.save_word2vec_format(out_vector, binary=False)


