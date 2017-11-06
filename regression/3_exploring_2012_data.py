
# coding: utf-8

# ## 2012数据探索
#使用2种方法(Gradient Boosting regression 和 OLS回归)在2012数据上小试验一把。 

# In[408]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ### 1.导入2012子数据集

# In[249]:

loads = pd.read_csv('load2012.csv')
weather = pd.read_csv('weather2012.csv')


# ### 格式化时间列

# In[250]:

weather['date'] = weather.dateutc.apply(lambda x: pd.to_datetime(x).date())
weather['timeest'] = weather.timeest.apply(lambda x: pd.to_datetime(x).time())
foo = weather[['date', 'timeest']].astype(str)
weather['timestamp'] = pd.to_datetime(foo['date'] + ' ' + foo['timeest'])
loads['timestamp'] = loads.timestamp.apply(lambda x: pd.to_datetime(x))


# ## 2. 补充缺失的天气信息
# 天气信息的频度是小时级别的，载入的2012数据是每5分钟的间隔。下面这个函数实际上就是使用KNN去补全5分钟级别数据里的天气信息。

# In[255]:

from sklearn.neighbors import NearestNeighbors

def find_nearest(group, match, groupname):
    nbrs = NearestNeighbors(1).fit(match['timestamp'].values[:, None])
    dist, ind = nbrs.kneighbors(group['timestamp'].values[:, None])

    group['nearesttime'] = match['timestamp'].values[ind.ravel()]
    return group


# In[256]:

loads = find_nearest(loads,weather,'timestamp')


# In[258]:

full = loads.merge(weather, left_on='nearesttime', right_on='timestamp')

#去除冗余列，重命名部分列 
full = full[['timestamp_x', 'load', 'nearesttime', 'temperaturef',             'dewpointf', 'humidity', 'sealevelpressurein', 'winddirection', 'windspeedkmh',             'precipitationmm']].rename(columns={'timestamp_x': 'timestamp', 'nearesttime':'weathertime'})


# ### *导出完整数据到csv文件中*

# In[261]:

full.to_csv('full2012.csv', index=False)


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
# 

# In[362]:

#取出n天前的电力需求
pday = pd.Timedelta('1 day')

def get_prev_days(x, n_days):
    '''Take a datetime (x) in the 'full' dataframe, and outputs the load value n_days before that datetime'''
    try:
        lo = full[full.timestamp == x - n_days*pday].load.values[0]
    except:
        lo = full[full.timestamp == x].load.values[0]
    return lo 


# In[361]:

full['dow'] = full.timestamp.apply(lambda x: x.dayofweek)
full['doy'] = full.timestamp.apply(lambda x: x.dayofyear)
full['day'] = full.timestamp.apply(lambda x: x.day)
full['month'] = full.timestamp.apply(lambda x: x.month)
full['hour'] = full.timestamp.apply(lambda x: x.hour)
full['minute'] = full.timestamp.apply(lambda x: x.hour*60 + x.minute)

full['t_m24'] = full.timestamp.apply(get_prev_days, args=(1,))
full['t_m48'] = full.timestamp.apply(get_prev_days, args=(2,))
full['tdif'] = full['load'] - full['t_m24']


# In[364]:

full.to_csv('full2012_features.csv', index=False)


# ## 4. Gradient Boosting Regression

# In[366]:

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split


# In[367]:

full.columns


# In[463]:

X = full[[          'temperaturef',          'dewpointf',           'humidity',           'sealevelpressurein',          'windspeedkmh',           'precipitationmm',          'dow',          'doy',           'month',          'hour',         'minute',          't_m24',           't_m48',           'tdif'         ]]
y = full['load']


# In[464]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[457]:

gbr = GradientBoostingRegressor(loss='ls', verbose=1, warm_start=True)


# In[458]:

gbr_fitted = gbr.fit(X_train, y_train)


# In[459]:

gbr.score(X_test, y_test)


# In[460]:

gbr.score(X_train, y_train)


# ## 5. Ordinary Least Squares Regression

# In[465]:

import statsmodels.api as sm

model = sm.OLS(y,X)
results = model.fit()
results.summary()


# In[416]:

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

avg_MSE = []
alphas = np.linspace(-2, 8, 20, endpoint=False)
alphas
for alpha in alphas:
    MSE = []
    for i in range(20):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
#     model = sm.OLS(X_train, y_train)
        model = Ridge(alpha=alpha)
        model.fit(X_test, y_test)
        test_error = mean_squared_error(y_test, model.predict(X_test))
        MSE.append(test_error)
    avg_MSE.append(np.mean(MSE))

plt.figure(figsize=(6,2))
plt.xlabel('alpha', fontsize=14)
plt.ylabel('Cross Validation MSE', fontsize=11)
plt.title('alpha vs. Cross Validation MSE', fontsize=11)
plt.plot(alphas, avg_MSE)


# In[ ]:



