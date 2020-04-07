fopen=open("pre-keywords/trkey.txt","r+")
print('ti')
rls=fopen.readlines()
ti=[]
tiset=[]
fstopkey=[]
fenli=[]
fenlic=[]
nsw=[]

for line in rls:
    i=line.split('/');
    ti.append(i[:-1])

#print(i[:-1])
#print(len(ti))

for li in ti:
    for i in li:
        fenli.append(i)

fenli=set(fenli)
fenli=list(fenli)


for i in ti:
    si=set(i)
    si=list(si)
    tiset.append(si)
print(ti)
print(fenli)



for ci in fenli:
    fenlico=0
    for i in tiset:
        co=i.count(ci)
        fenlico=fenlico+co
    fenlic.append(fenlico)
        
print(fenlic)

print(len(fenlic))
print(len(fenli))

leng=len(ti)/3
print(leng)

for i in range(len(fenlic)):
    if fenlic[i]>leng:
        nsw.append(fenli[i])


print(nsw)

fopen.close()
fo=open("keystopw.txt","a+")
for i in nsw:
    fo.write(i)
    fo.write('\n')
    
fo.close()




'''
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


'''
for i in range (6):
    f=tiset[i].intersection(tiset[i+1],tiset[i+2])
    print(f)
    for fi in f:
        fstopkey.append(fi)
    print("sign==================++++++++++++=")
'''
