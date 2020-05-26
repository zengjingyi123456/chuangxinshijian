import tokenize
from io import  BytesIO
import token
import pandas as pd

sourceCode='''
from math import ceil
n,m,a=map(int,input().split())
print(ceil(n/a)*ceil(m/a))'''

source = pd.read_pickle('Python_3.pkl')
print('total data set size:',source.shape)

for sourceIndex, row in source.iterrows():
    sourceIndex=row['code']
    count=0
    for toknum, tokval, start, end, _ in tokenize.tokenize(BytesIO(sourceCode.encode('utf-8')).readline):
        print(sourceIndex,str(count)+'***************************')
        print('TokenType:',token.tok_name[toknum],'\tToken:',tokval,'\tPosition:',start,end)
