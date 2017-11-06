
# coding: utf-8

# ## 采集天气数据
# 这个ipython notebook主要是遍历和纽约州每个区关联的所有天气站，使用Weather Underground API同步历史上的实时天气信息。

# In[1]:

import pandas as pd
import numpy as np
import time
import random
import cPickle as pickle
import os




# In[2]:

weather_dict = pickle.load(open('weather_dict.pkl','rb'))


# In[3]:

weather_dict


# In[4]:

airports = [i[0] for i in weather_dict.values()]


# In[5]:

#去重
airports = list(set(airports))


# In[6]:

airports


# In[8]:

dates = pd.date_range(pd.to_datetime('2001-05-01'),                        pd.to_datetime('2016-03-11'), freq='D')


# In[24]:

def write_daily_weather_data(airport, dates):
    '''把2个python list（天气和日期）整合成一个CSV文件
    
    整合好的CSV文件有以下的字段:
    
    timeest | temperaturef | dewpointf | humidity | sealevelpressurein | visibilitymph | winddirection | windspeedkmh | gustspeedmph
    
        | precipitationmm | events | conditions | winddirdegrees | dateutc
    '''
    for d in dates:
        try:
            df0 = pd.read_csv('https://www.wunderground.com/history/airport/{0}/{1}/{2}/{3}/DailyHistory.html?format=1'                                 .format(airport, d.year, d.month, d.day))
            cols = df0.columns

            df0.columns = [col.lower().replace(' ','').replace('<br/>', '').replace('/','') for col in cols]
            #print df0.columns
            df0.dateutc = df0.dateutc.apply(lambda x: pd.to_datetime(x.replace('<br />', '')))

            df0.gustspeedkmh = df0.gustspeedkmh.replace('-', 0)
            df0.windspeedkmh = df0.windspeedkmh.replace('Calm', 0)
            df0.precipitationmm = df0.precipitationmm.replace('NaN', 0)
            df0.events = df0.events.replace('NaN', 0)

            filepath = '../data/wunderground/'+ airport +'/' + str(d.date()).replace('-','') + '.csv'
            print filepath
            df0.to_csv(filepath, index=False)



            t = 3
            time.sleep(t)

            if type(df0.dateutc[0]) == pd.tslib.Timestamp:
                continue
            else:
                print "Something is wrong"
                break
        except:
            print "date ",d ," can't be downloaded!"
            continue

    print "Files for %s have been written" % airport
    return


# 遍历气象站，导出天气文件

# In[ ]:

for a in airports:
    write_daily_weather_data(a, dates)


# In[25]:

dates = pd.date_range(pd.to_datetime('2012-07-03'),                        pd.to_datetime('2013-01-01'), freq='D')


# In[ ]:

write_daily_weather_data('kalb', dates)


# In[85]:


def combine_weather_data(airport):
    '''Combine the weather data for each day at an airport into one combined csv'''
    csvs = []
    for file in os.listdir("../data/wunderground/"+airport+"/"):
        if file.endswith(".csv"):
            csvs.append(file)

    fout=open("../data/wunderground/"+airport+"_all.csv","a")

    # 第一个文件完整地写进去:
    for line in open("../data/wunderground/"+airport+"/"+csvs[0]):
        fout.write(line)
    # 后续的文件，去掉头部信息:    
    for file in csvs[1:]:
        f = open("../data/wunderground/"+airport+"/"+file)
        f.next() # 跳过header
        for line in f:
             fout.write(line)
        f.close()
    fout.close()
    print "Files for %s have been combined" % airport


# In[ ]:

for a in airports:
    combine_weather_data(a)

