import tokenize
from io import BytesIO
import token

sourceCode='''
    from math import ceil
    n,m,a=map(int,input().split())
    print(ceil(n/a)*ceil(m/a))
'''
count = 0
for toknum,tokval,start,end,_ in tokenize.tokenize(BytesIO(sourceCode.encode('utf-8')).readline):
    print(str(count)+"*******************************************************")
    print("TokenType:",token.tok_name[toknum],'\tToken',tokval,'\tPosition',start,end)
    count =count + 1;
