fopen=open("trkey.txt","r+")
print('trkey')
rls=fopen.readlines()
ti=[]
tiset=[]
for line in rls:
    i=line.split('/');
    ti.append(i[:-1])
    #print(i[:-1])
#print(len(ti))

fstopkey=[]

for i in ti:
    si=set(i)
    tiset.append(si)
print(ti)

for i in range (8):
    f=tiset[i].intersection(tiset[i+1])
    print(f)
    for fi in f:
        fstopkey.append(fi)
    print("sign==================++++++++++++=")

fstopkey=set(fstopkey)
print(fstopkey)
fstopkey=list(fstopkey)

print(fstopkey)

fopen.close()

fo=open("keystopw.txt","a+")
for i in fstopkey:
    fo.write(i)
    fo.write('\n')
    
fo.close()


#f=tiset[0].intersection(tiset[1],tiset[2],tiset[3])    
#print(f)

'''
for i in range (6):
    f=tiset[i].intersection(tiset[i+1],tiset[i+2])
    print(f)
    for fi in f:
        fstopkey.append(fi)
    print("sign==================++++++++++++=")
'''
