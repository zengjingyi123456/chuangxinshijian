fopen=open("keystopw.txt","r+")
rls=fopen.readlines()
keynewstop=[]

for l in rls:
    keynewstop.append(l[:-1])
print(keynewstop)
    
keynewstop=set(keynewstop)
print(keynewstop)
keynewstop=list(keynewstop)
print(keynewstop)

fopen.close()


fw=open("stopwords.txt","a+",encoding='utf-8')

for i in keynewstop:
    fw.write(i)
    fw.write('\n')

fw.close() 
