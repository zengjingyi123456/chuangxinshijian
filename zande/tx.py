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

#print(check)
#print(cmp)
douhao='，'

for i in cmp:
    del i[0]

oput=[]
for i in cmp:
    oput.append(douhao.join(i))
    print(douhao.join(i))
    print('\n')

print(len(oput))   
#print(oput)
for ni in range(10):
    fo=open(ltag[ni]+'issue.txt','w+')
    fo.write(oput[ni])
    fo.close

'''
for i in cmp:
    print(i)
    print('\n')
    del i[0]
    print(i)
    print('\n')

for i in cmp:
    douhao.join(i)
    print(i)
    print('\n')
'''

