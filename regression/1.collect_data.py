
# coding: utf-8

# ## 获取数据
# 这个项目的数据是从New York State Independent Service Authority (NYISO)获得的
# 

# 数据也可以从[这个页面](http://mis.nyiso.com/public/P-58Blist.htm)直接获得: `http://mis.nyiso.com/public/P-58Blist.htm`

# 整个纽约州的电能由12个“区域”生产和提供，这12个区有自己独立的能源市场。下面这张图可以给你一个直观的印象，大概的一个分布状况。这里采集到的数据，也会按照这12个区域做分割，其中区域的名称会写在"name"字段里。



import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import urllib
import os
import pandas as pd
import numpy as np


# ## 直接下载数据
# 直接找到数据源下载纽约州2001-2015年的相关数据。
# 
# 直接拉下来的数据会以zip形式存储在`../data/nyiso`文件夹下。 
# 
# 然后解压缩打包文件到 `../data/nyiso/all/raw_data` 文件夹。

# In[11]:

print "download and unzipping..."
dates = pd.date_range(pd.to_datetime('2001-01-01'),                        pd.to_datetime('2015-12-31'), freq='M')

for date in dates:
    url = 'http://mis.nyiso.com/public/csv/pal/{0}{1}01pal_csv.zip'.format(date.year, str(date.month).zfill(2))
    urllib.urlretrieve(url, "../data/nyiso/{0}".format(url.split('/')[-1]))

def unzip(source_filename, dest_dir):
    with zipfile.ZipFile(source_filename) as zf:
        try:
            zf.extractall(dest_dir)
        except:
            print source_filename
            return


# In[12]:

zips = []
for file in os.listdir("../data/nyiso"):
    if file.endswith(".zip"):
        zips.append(file)
for z in zips:
    try:
        unzip('../data/nyiso/' + z, '../data/nyiso/all/raw_data')
        print '../data/nyiso/' + z + "extract done!"
    except:
        print '../data/nyiso/' + z
        continue


# 文件多了处理起来麻烦，整合到一个合并的文件里吧: combined_nyiso.csv

# In[2]:

csvs = []
for file in os.listdir("../data/nyiso/all/raw_data"):
    if file.endswith("pal.csv"):
        csvs.append(file)


# In[3]:

fout=open("../data/nyiso/all/combined_iso.csv","a")

# 第一个文件保存头部column name信息:
for line in open("../data/nyiso/all/raw_data/"+csvs[0]):
    fout.write(line)
# 后面的部分可以跳过 headers:    
for file in csvs[1:]:
    f = open("../data/nyiso/all/raw_data/"+file)
    f.next() # 跳过 header
    for line in f:
         fout.write(line)
    f.close() # 关闭文件
fout.close()


# ## 清洗和了解数据
# 
# 这里总共有14年的数据，把它们放到一个完整的pandas数据帧(dataframe)里。

# In[4]:

df = pd.read_csv("../data/nyiso/all/combined_iso.csv")


# 这里所谓的数据清洗实际上是一个数据的选择过程，留下来感兴趣的4列: timestamp, region name, id, and load (说明一下，load是电力需求，单位是**兆瓦/Megawatts**). 

# In[11]:

cols = df.columns
df.columns = [col.lower().replace(' ', '') for col in cols]
df = df[['timestamp', 'name', 'ptid', 'load']]


# 重新把需要的数据写回csv文件中

# In[12]:

df.to_csv('../data/nyiso/all/combined_iso.csv', index=False)


# In[13]:

df.name.unique()


# ## 建立**天气站/weather stations**的映射关系
# 
# 的数据帧里区域的名称已经有了，要建立起它们和对应的城市还有天气站之间的映射关系（简单地理解就是需要关联几个数据表用）。直接把它们放在一个`python dict/字典`里，一会儿下一个notebook会用到这个映射关系。

# In[14]:

regions = list(df.name.unique())
region_names = ['Capital', 'Central', 'Dunwoodie', 'Genese', 'Hudson Valley', 'Long Island', 'Mohawk Valley', 'Millwood', 'NYC', 'North', 'West']
cities = ['Albany', 'Syracuse', 'Yonkers', 'Rochester', 'Poughkeepsie', 'NYC', 'Utica', 'Yonkers', 'NYC', 'Plattsburgh', 'Buffalo']
weather_stations = ['kalb', 'ksyr', 'klga', 'kroc', 'kpou', 'kjfk', 'krme', 'klga', 'kjfk', 'kpbg', 'kbuf']


# In[15]:

weather_dict = dict(zip(regions, zip(weather_stations, region_names, cities)))
weather_dict


# In[17]:

import pickle as pickle
pickle.dump(weather_dict, open('weather_dict.pkl','wb'))


# ## 把数据按区域切分

# 按照12个区把数据做个切分，更细致的切分可以让天气数据更精确。同时在测试阶段，一个区一个统一的文件也会方便很多。

# In[18]:

for region in weather_dict.keys():
    subset = df[df.name == region].copy()
    filename = weather_dict[region][1].lower().replace(' ', '') + '.csv'
    subset.to_csv('../data/nyiso/all/' + filename, index=False)


# In[172]:

#其中的一个文件大概长这样


# # 输出2012年的数据用于测试

# In[24]:

print "output 2012 data for test..."
capital = pd.read_csv("../data/nyiso/all/capital.csv")
#capital[capital.timestamp < pd.to_datetime('2013-01-01')].to_csv('load2012.csv', index=False)
capital[capital.timestamp < '2013-01-01'].to_csv('load2012.csv', index=False)
csvs = []
for file in os.listdir("../data/wunderground/kalb"):
    if file.startswith("2012"):
        csvs.append(file)
print csvs
fout=open("weather2012.csv","a")

# 写入整个文件:
for line in open("../data/wunderground/kalb/"+csvs[0]):
    fout.write(line)
# 跳过头部:    
for file in csvs[1:]:
    f = open("../data/wunderground/kalb/"+file)
    f.next()
    for line in f:
         fout.write(line)
    f.close()
fout.close()


# ## 下载NYISO的预测结果
# NYISO会公布一个"day-ahead"预测数据（每次提前一天他们都会预测下一条的用电需求状况）。 如果能够比公司做的预测系统效果还要好，那妥妥的表示的模型有比较好的效果。 先把它在2014-2016年的预测结果下载下来，以便最后比对。
# 
# 网址如下: http://www.nyiso.com/public/markets_operations/market_data/custom_report/index.jsp?report=load_forecast

# In[6]:

nyiso_forecast = pd.read_csv('../data/nyiso_dayahead_forecasts/forecast_2014_2016.csv')


# In[7]:

len(nyiso_forecast)


# In[11]:

nyiso_forecast.columns = ['timestamp', 'zone', 'forecast', 'gmt']


# In[ ]:



