import jieba
import re
import jieba.posseg as psg
#coding:gbk
#coding:utf-8
# _*_ coding: utf-8 _*_

fileobj = open('文本.txt', 'r')    #读取文本放到strings中
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
print('分词结果写入文本')
fenci='/'.join(w)
file_handle=open('分词结果.txt',mode='w')   #分词结果写入文本
file_handle.writelines(fenci)
file_handle.close()

#去停词结果写入新文本
print('去停词结果写入文本')
stopwords = {}.fromkeys([ line.rstrip() for line in open('stopwords.txt') ])
final = ''
fwl=''
fwl2=''

for seg in fenci:     #去停词
    if seg not in stopwords:
            final += seg
file_handle1=open('去停词结果.txt',mode='w')
file_handle1.writelines(final)
file_handle1.close()

word=''   #去停词以及长度小于2的词
for seg in fenci:
    if seg not in stopwords:
        fwl+=seg

for seg1 in fwl:
    word+=seg1
    if seg1=='/' and len(word)<3:
        word=''
    if seg1=='/' and  len(word)>2:

            fwl2+=word
            word=''
file_handle2=open('去停词以及长度小于2的词结果.txt',mode='w')
file_handle2.writelines(fwl2)
file_handle2.close()

