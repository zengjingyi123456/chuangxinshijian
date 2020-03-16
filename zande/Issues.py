#-*- coding: utf-8 -*-
import ssl
import requests
import os
import json
import time

repos = ['fxsjy/jieba']

def getissues():
    isu = []
    isutimes=[]
    titles = []
    nu=1
    for repo in repos:
        for i in range(17):
            title=[]
            isutime=[]
            repo_url = 'https://api.github.com/repos/fxsjy/jieba/issues?page=%d' %nu# 确定url
            print(repo_url+'\n')
            repoisu = readURL('data/als/%s' % (repo), repo_url)  # 访问url得到数据
            repoisu = repoisu and json.loads(repoisu)  # 将数据类型转换
            for issue in repoisu:
                title.append(issue['title'])
                isutime.append((issue['updated_at'])[:10])
            f=open("text.txt","a+")
            ftime=open("uptime.txt","a+")
            for t in title:
                f.write(t+'\n')
            for it in isutime:
                ftime.write(it+'\n')
            f.close()
            ftime.close()
            print(title)
            print(isutime)
            print(nu)
            nu=nu+1
            titles=titles+title
            isutimes=isutimes+isutime
            time.sleep(10)
    isu=[titles,isutimes]
    return isu

#读取url的信息，并建立缓存

def readURL(cache,url):
	#看看该url是否访问过
	content = requests.get(url).content.decode()

	#吧文件内容保存下来，以免多次重复访问url，类似于缓存
	'''
	folder = cache.rpartition('/')[0]
	not os.path.isdir(folder) and os.makedirs(folder)
	with open(cache, 'w') as f:
		f.write(content)
	'''
	return content
	
    
'''
	cache = 'data/cache/%s' % cache
	if os.path.isfile(cache):
		with open(cache, 'r') as f:
			content = f.read()
    		return content
        '''
getissues()
