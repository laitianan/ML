
# coding: utf-8

# ## Gradient Boosting Machine建模
# 载入每个区的NYISO数据，合并之后用GBM建模。

# In[7]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
get_ipython().magic('matplotlib inline')


# In[8]:

weather_dict = joblib.load('weather_dict.pkl')


# In[9]:

weather_dict['N.Y.C.']


# In[10]:

region = 'N.Y.C.'
place = weather_dict[region][1].lower().replace(' ','')
full = pd.read_csv('full_{0}_features.csv'.format(place))


# In[47]:

full.head()


# In[48]:

full.dropna(inplace=True)


# In[49]:

get_ipython().magic("time full['timestamp'] = full['timestamp'].apply(lambda x: pd.to_datetime(x))")


# In[50]:

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split


# Use this list to test the model performance with different combinations of features.

# In[66]:

features = [          'temperaturef',#           'dewpointf', \
#           'humidity', \
#           'sealevelpressurein', \
#          'windspeedmph', \
#           'precipitationin',\
            'year',\
          'dow',\
          'doy', \
#           'month',\
#           'hour',\
         'minute',\
#           't_m24', \
#           't_m48', \
#           't_m1',\
         ]


# In[67]:

X_train = full[full.timestamp < pd.to_datetime('2015')][features]
X_test = full[full.timestamp >= pd.to_datetime('2015')][features]

y_train = full[full.timestamp < pd.to_datetime('2015')]['load']
y_test = full[full.timestamp >= pd.to_datetime('2015')]['load']


# In[68]:

gbr = GradientBoostingRegressor(loss='ls', n_estimators=100, max_depth=3, verbose=1, warm_start=True)


# In[69]:

gbr_fitted = gbr.fit(X_train, y_train)


# In[70]:

gbr.score(X_test, y_test)


# In[71]:

zip(features, list(gbr.feature_importances_))


# 了解你的GBM模型状态如何，画出随着estimator个数增加Error变化状况。（有兴趣可以看看early stopping）。

# In[72]:

def deviance_plot(est, X_test, y_test, ax = None, label = '', train_color='#2c7bb6', test_color = '#d7191c', alpha= 1.0, ylim = (0,1000000)):
   
    n_estimators = len(est.estimators_)
    test_dev = np.empty(n_estimators)

    for i, pred in enumerate(est.staged_predict(X_test)):
        test_dev[i] = est.loss_(y_test, pred)

    if ax is None:
        fig = plt.figure(figsize = (12,8))
        ax = plt.gca()

    ax.plot(np.arange(n_estimators) + 1, test_dev, color= test_color, label = 'Test %s' % label, linewidth=2, alpha=alpha)
    ax.plot(np.arange(n_estimators) + 1, est.train_score_, color = train_color, label= 'Train %s' % label, linewidth=2, alpha=alpha)
    ax.set_ylabel('Error')
    ax.set_xlabel('n_estimators')
    ax.set_ylim(ylim)
    return test_dev, ax

test_dev, ax = deviance_plot(gbr, X_test, y_test)
ax.legend(loc='upper right')

# add some annotations
# ax.annotate('Lowest test error', xy=(test_dev.argmin() + 1, test_dev.min() + 0.02),
#             xytext=(150, 3.5), **annotation_kw)

# ann = ax.annotate('', xy=(800, test_dev[799]),  xycoords='data',
#                   xytext=(800, est.train_score_[799]), textcoords='data',
#                   arrowprops={'arrowstyle': '<->'})
# ax.text(810, 3.5, 'train-test gap')

