
# coding: utf-8

# ## 清洗合并数据
# 把所有的电能需求数据和天气数据，还有刚才做完的时间序列特征，合并在一起，组成建模数据。

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 
get_ipython().magic('matplotlib inline')


# In[2]:

weather_dict = pickle.load(open('weather_dict.pkl','rb'))


# In[3]:

weather_dict


# ### 载入2012-2015的区域数据
# 

# ## 1. 格式化时间列

# In[4]:

def format_datetime(weather, loads):
    #Format datetime columns:
    weather['date'] = weather.dateutc.apply(lambda x: pd.to_datetime(x).date())
    weather['timeest'] = weather.timeest.apply(lambda x: pd.to_datetime(x).time())
    foo = weather[['date', 'timeest']].astype(str)
    weather['timestamp'] = pd.to_datetime(foo['date'] + ' ' + foo['timeest'])
    loads['timestamp'] = loads.timestamp.apply(lambda x: pd.to_datetime(x))
    return weather, loads


# In[5]:




# ## 2. 给电能需求数据添加天气信息
# 和之前一样，对电能需求数据添加天气信息，点呢过需求数据是5分钟间隔的，我们可以用KNN去补充一部分缺失的数据。

# In[8]:

from sklearn.neighbors import NearestNeighbors

def find_nearest(group, match, groupname):
    nbrs = NearestNeighbors(1).fit(match['timestamp'].values[:, None])
    dist, ind = nbrs.kneighbors(group['timestamp'].values[:, None])

    group['nearesttime'] = match['timestamp'].values[ind.ravel()]
    return group


# ## 3. 构造特征

# 这是一个时间序列上的回归问题，需要在时间上做一些特征，可参照论文[Barta et al. 2015](http://arxiv.org/pdf/1506.06972.pdf)提到的方式，去构造细粒度的时间特征，上面那篇论文的应用场景也是用概率模型预测电价。构造的特征如下：<br>
# 
#     `dow`: day of the week (integer 0-6)
#     `doy`: day of the year (integer 0-365)
#     `day`: day of the month (integer 1-31)
#     `woy`: week of the year (integer 1-52)
#     `month`: month of the year (integer 1-12)
#     `hour`: hour of the day (integer 0-23)
#     `minute`: minute of the day (integer 0-1339)
#     
#     `t_m24`: load value from 24 hours earlier
#     `t_m48`: load value from 48 hours earlier
#     `tdif`: difference between load and t_m24

# In[9]:

#往前推n天的电能需求数据
pday = pd.Timedelta('1 day')

def get_prev_days(x, n_days):
    '''Take a datetime (x) in the 'full' dataframe, and outputs the load value n_days before that datetime'''
    try:
        lo = full[full.timestamp == x - n_days*pday].load.values[0]
    except:
        lo = full[full.timestamp == x].load.values[0]
    return lo 


# In[10]:

def add_time_features(df):
    full = df.copy()
    full['dow'] = full.timestamp.apply(lambda x: x.dayofweek)
    full['doy'] = full.timestamp.apply(lambda x: x.dayofyear)
    full['day'] = full.timestamp.apply(lambda x: x.day)
    full['month'] = full.timestamp.apply(lambda x: x.month)
    full['year'] = full.timestamp.apply(lambda x: x.year)
    full['hour'] = full.timestamp.apply(lambda x: x.hour)
    full['minute'] = full.timestamp.apply(lambda x: x.hour*60 + x.minute)

    full['t_m24'] = full.timestamp.apply(get_prev_days, args=(1,))
    full['t_m48'] = full.timestamp.apply(get_prev_days, args=(2,))
    full['tdif'] = full['load'] - full['t_m24']
    return full


# ### 遍历每一个NYS数据子集，并做同样的数据清洗和数据合并操作

# In[13]:

k = weather_dict.keys()


# In[15]:

for region in k:
    
    place = weather_dict[region][1].lower().replace(' ','')
    airport = weather_dict[region][0]

    #载入数据
    loads = pd.read_csv('../data/nyiso/all/{0}.csv'.format(place))
    weather = pd.read_csv('../data/wunderground/{0}_all.csv'.format(airport))

    #去掉无关列
    weather = weather[weather.winddirection != 'winddirection']
    
    #格式化时间列
    weather, loads = format_datetime(weather, loads)

    #用KNN补齐天气信息
    loads = find_nearest(loads,weather,'timestamp')
    full = loads.merge(weather, left_on='nearesttime', right_on='timestamp')

    #去掉无关列，重命名
    full = full[['timestamp_x', 'load', 'nearesttime', 'temperaturef',                 'dewpointf', 'humidity', 'sealevelpressurein', 'winddirection', 'windspeedkmh',                 'precipitationmm']].rename(columns={'timestamp_x': 'timestamp', 'nearesttime':'weathertime'})

    #构造特征
    full = add_time_features(full)

    #生成csv文件
    full.to_csv('full_{0}_features.csv'.format(place), index=False)


# In[17]:

#前推一定时间的数据
phour = pd.Timedelta('1 hour')

def get_prev_hours(x, n_hours):
    '''Take a datetime (x) in the 'full' dataframe, and outputs the load value n_days before that datetime'''
    try:
        lo = full[full.timestamp == x - n_hours*phour].load.values[0]
    except:
        lo = full[full.timestamp == x].load.values[0]
    return lo 


# In[ ]:

for region in k:
    place = weather_dict[region][1].lower().replace(' ','')
    airport = weather_dict[region][0]
    
    full = pd.read_csv('full_{0}_features.csv'.format(place))
    
    full['t_m1'] = full.timestamp.apply(get_prev_hours, args=(1,))
    
    full.to_csv('full_{0}_features.csv'.format(place), index=False)
    
    print ("%s done" % place)


# In[ ]:



