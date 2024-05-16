#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
import csv
from bs4 import BeautifulSoup
import re
headers={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0' 
}

def url_get(url):
    response=requests.get(url,headers=headers)
    soup=BeautifulSoup(response.content.decode('utf-8'),'lxml')
    lis=soup.select('ul[class="sellListContent"]>li')
    urls=[]
    i=1
    for li in lis:
        if i==6:
            i+=1
            continue
        liurl=li.select('a')[0]['href']
        urls.append(liurl)
        i+=1
    return urls

def  content_get(url):
    response=requests.get(url,headers=headers)
    soup=BeautifulSoup(response.content.decode('utf-8'),'lxml')
    content={}
    title=soup.find('h1')['title']
    content['标题']=title
    lis1=soup.select('div[class="base"]>div[class="content"]>ul>li')
    s=re.findall('\d',list(lis1[0].stripped_strings)[1])
    content['室']=s[0]
    content['厅']=s[1]
    content['厨']=s[2]
    content['卫']=s[3]
    content['所在楼层']=re.match('...',list(lis1[1].stripped_strings)[1]).group()
    content['总楼层']=re.search('\d+',list(lis1[1].stripped_strings)[1]).group()
    content['建筑面积']=re.match('\d+.\d*',list(lis1[2].stripped_strings)[1]).group()
    content['户型结构']=list(lis1[3].stripped_strings)[1]
    content['建筑类型']=list(lis1[5].stripped_strings)[1]
    content['房屋朝向']=list(lis1[6].stripped_strings)[1]
    content['建筑结构']=list(lis1[7].stripped_strings)[1]
    content['装修情况']=list(lis1[8].stripped_strings)[1]
    content['梯户比例']=list(lis1[9].stripped_strings)[1]
    content['供暖方式']=list(lis1[10].stripped_strings)[1]
    content['配备电梯']=list(lis1[11].stripped_strings)[1]
    
    lis2=soup.select('div[class="transaction"]>div[class="content"]>ul>li')
    content['挂牌时间']=list(lis2[0].stripped_strings)[1]
    content['交易权属']=list(lis2[1].stripped_strings)[1]
    content['房屋年限']=list(lis2[4].stripped_strings)[1]
    content['产权所属']=list(lis2[5].stripped_strings)[1]
    content['抵押信息']=list(lis2[6].stripped_strings)[1]
    total=soup.find('span',class_='total').string
    uni=list(soup.find('span',class_='unitPriceValue').stripped_strings)[0]
    content['单价']=uni
    content['总价(万元)']=total
    content['区']='临潼'
    content['小区']=list(soup.find('div',class_='communityName').stripped_strings)[1]
    return content

f=open('临潼 .csv',mode='w',newline='',encoding='utf-8-sig')
header=['标题','室','厅','厨','卫','所在楼层','总楼层','建筑面积','户型结构','建筑类型','房屋朝向','建筑结构','装修情况','梯户比例','供暖方式','配备电梯','挂牌时间','交易权属','房屋年限','产权所属','抵押信息','单价','总价(万元)','区','小区']
writer=csv.DictWriter(f,header)
writer.writeheader()


for i in range(50):
    try:
        print('正在爬取第{}页'.format(i+1))
        url='https://xa.lianjia.com/ershoufang/lintong/pg{}/'.format(i+1)
        urls=url_get(url)
        for u in urls:
            content=content_get(u)
            writer.writerow(content)
    except IndexError:
        print("IndexError in page{}".format(i+1))
        continue
f.close()


# In[ ]:




