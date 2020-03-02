import jieba
import re
import jieba.posseg as psg
#coding:utf-8
# _*_ coding: utf-8 _*_

fileobj = open('xs.txt', 'r')    #读取文本放到strings中
try:
   strings = fileobj.read()
finally:
   fileobj.close()


sentences = re.split('(。|！|\!|\.|？|\?)',strings)         #使用re分句
new_sents = []
for i in range(int(len(sentences)/2)):
    sent = sentences[2*i] + sentences[2*i+1]
    new_sents.append(sent)
print('分句结果:')
print(new_sents)


#分词
w=jieba.cut(strings)
print('分词结果:')
print(','.join(w))


#词性
print('词性标注结果:')
print([(x.word,x.flag) for x in psg.cut(strings)])


#某些词性的数量统计
print('某些类型词的数量统计：')
v=0
n=0
a=0
ad=0
c=0
for x in psg.cut(strings):
    if (x.flag=='v'):
       v=v+1;
    if (x.flag=='n'):
       n=n+1;
    if (x.flag=='a'):
       a=a+1;
    if (x.flag=='ad'):
       ad=ad+1;
    if (x.flag=='c'):
       c=c+1;
print('v动词:')
print(v)
print('n名词:')
print(n)
print('a形容词:')
print(a)
print('ad副形词:')
print(ad)
print('c连词:')
print(c)

#某些词出现次数
print('某些词出现次数:')
n=0
for x in psg.cut(strings):
       if (x.word=='在'):
         n=n+1
print('在')
print(n)
for x in psg.cut(strings):
       if (x.word=='的'):
         n=n+1
print('的')
print(n)
for x in psg.cut(strings):
       if (x.word=='说'):
         n=n+1
print('说')
print(n)

# a=[]
# print('词性标注结果2:')
# for x in psg.cut(strings):
#     if not(x.flag=='x' or x.flag=='n'):
#        a.append([(x.word,x.flag) for x in psg.cut(strings)])
# print(a)

