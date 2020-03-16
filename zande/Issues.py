#-*- coding: utf-8 -*-
import ssl
import requests
import os
import json

repos = ['fxsjy/jieba']

def getissues():
    isu = {}
    for repo in repos:
        repo_url = 'https://api.github.com/repos/%s/issues' % repo  # 确定url
        repoisu = readURL('data/als/%s' % (repo), repo_url)  # 访问url得到数据
        repoisu = repoisu and json.loads(repoisu)  # 将数据类型转换
        # 提取想要的信息保存在info中
        titles = []
        for issue in repoisu:
            titles.append(issue['title'])
        isu =titles
        print(isu)
        f=open("text.txt","w+")
        for l in isu:
            f.write(l+'\n')
        f.close()
        return isu

#读取url的信息，并建立缓存
def readURL(cache,url):
	#看看该url是否访问过
	cache = 'data/cache/%s' % cache
	if os.path.isfile(cache):
		with open(cache, 'r') as f:
			content = f.read()
		return content

	content = requests.get(url).content.decode()

	#吧文件内容保存下来，以免多次重复访问url，类似于缓存
	folder = cache.rpartition('/')[0]
	not os.path.isdir(folder) and os.makedirs(folder)
	with open(cache, 'w') as f:
		f.write(content)
	return content

getissues()
