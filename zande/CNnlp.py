import jieba as j
import jieba.posseg as pg        
import re
import numpy as np
import time


words=[]
flags=[]
wdc=[]
fac=[]
wordst=[]
whole=[]

stop=open('CNstopwords.txt','r+',encoding='utf-8')
stopword=stop.read().split("\n")

print(stopword)

fn=open('jiebats.txt',"r",encoding='utf-8')


for line in fn.readlines():
        voc=pg.cut(line)
        for v in voc:
            whole.append(v)
            words.append(v.word)
            flags.append(v.flag)



for i in words:
    if i not in stopword:
        wordst.append(i)
            


wdset=set(wordst)
faset=set(flags)

for i in wdset:
    num=wordst.count(i)
    wdc.append(num)

for i in faset:
    num=flags.count(i)
    fac.append(num)

wddict=dict(zip(wdset,wdc))
fadict=dict(zip(faset,fac))

print(wddict)
print(fadict)


