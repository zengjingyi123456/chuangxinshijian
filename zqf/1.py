import jieba
import re
import jieba.posseg as psg


fileobj = open('xyj.txt', 'r',encoding='utf-8')    #读取文本放到strings中
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
file_handle=open('分词.txt',mode='w',encoding='utf-8')   #分词结果写入文本
file_handle.writelines(fenci)
file_handle.close()

#去停词结果写入新文本
print('去停词结果写入文本')
stopwords = {}.fromkeys([ line.rstrip() for line in open('停词.txt') ])
final = ''
fwl=''
fwl2=''

for seg in fenci:     #去停词
    if seg not in stopwords:
            final += seg
file_handle1=open('去停词.txt',mode='w')
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
file_handle2=open('去停词以及长度小于2的词.txt',mode='w')
file_handle2.writelines(fwl2)
file_handle2.close()



#词性
print('词性标注结果:')
print([(x.word,x.flag) for x in psg.cut(strings)])


#词性的数量统计
print('部分类型词的数量统计：')
k=0
cixing=['v','n','a','ad','c','an','ag','al','b','bl','cc','d','e','f','h','k','m','mq','nr','nr1','nr2','nrj','nrf','ns','nsf','nt','nz','nl','ng','o','p','pba','pbei','q','qv','qt']
for cx in cixing:
  for x in psg.cut(strings):
    if (x.flag==cx):
       k=k+1;
  print(cx)
  print(k)
  k=0


#某些词出现次数
CI=['在','的','说','呢','走','可以']
print('某些词出现次数:')
n=0
for c in CI:
  for x in psg.cut(strings):
       if (x.word==c):
         n=n+1
  print(c)
  print(n)
  n=0