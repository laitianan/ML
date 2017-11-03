
# coding: utf-8

# ## 特征工程与机器学习建模

# ### 自定义工具函数库

# In[34]:


import  pandas as pd
import numpy as np
import scipy as sp


#文件读取
def read_csv_file(f,logging=False):
    print ("============================读取数据========================",f)
    data = pd.read_csv(f)
    if logging:
        print( data.head(5))
        print( f,"  包含以下列....")
        print( data.columns.values)
        print (data.describe())
        print (data.info())
    return  data

#第一类编码
def categories_process_first_class(cate):
    cate = str(cate)
    if len(cate)==1:
        if int(cate)==0:
            return 0
    else:
        return int(cate[0])

#第2类编码
def categories_process_second_class(cate):
    cate = str(cate)
    if len(cate)<3:
        return 0
    else:
        return int(cate[1:])

#年龄处理，切段
def age_process(age):
    age = int(age)
    if age==0:
        return 0
    elif age<15:
        return 1
    elif age<25:
        return 2
    elif age<40:
        return 3
    elif age<60:
        return 4
    else:
        return 5

#省份处理
def process_province(hometown):
    hometown = str(hometown)
    province = int(hometown[0:2])
    return province

#城市处理
def process_city(hometown):
    hometown = str(hometown)
    if len(hometown)>1:
        province = int(hometown[2:])
    else:
        province = 0
    return province

#几点钟
def get_time_day(t):
    t = str(t)
    t=int(t[0:2])
    return t

#一天切成4段
def get_time_hour(t):
    t = str(t)
    t=int(t[2:4])
    if t<6:
        return 0
    elif t<12:
        return 1
    elif t<18:
        return 2
    else:
        return 3

#评估与计算logloss
def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll


# ### 特征工程+随机森林建模

# #### import 库

# In[35]:


from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


# #### 读取train_data和ad
# #### 特征工程

# In[36]:

#['label' 'clickTime' 'conversionTime' 'creativeID' 'userID' 'positionID' 'connectionType' 'telecomsOperator']
train_data = read_csv_file('./pre/train.csv',logging=True)

#['creativeID' 'adID' 'camgaignID' 'advertiserID' 'appID' 'appPlatform']
ad = read_csv_file('./pre/ad.csv',logging=True)


# In[37]:

#app
app_categories = read_csv_file('./pre/app_categories.csv',logging=True)
app_categories["app_categories_first_class"] = app_categories['appCategory'].apply(categories_process_first_class)
app_categories["app_categories_second_class"] = app_categories['appCategory'].apply(categories_process_second_class)


# In[38]:

app_categories.head()


# In[39]:

user = read_csv_file('./pre/user.csv',logging=True)


# In[40]:

user.columns


# In[41]:

user[user.age!=0].describe()


# In[42]:

import matplotlib.pyplot as plt
user.age.value_counts(ascending=False)


# In[43]:

user['residence'].head()


# In[44]:

#user
user = read_csv_file('./pre/user.csv',logging=True)
user['age_process'] = user['age'].apply(age_process)
user["hometown_province"] = user['hometown'].apply(process_province)
user["hometown_city"] = user['hometown'].apply(process_city)
user["residence_province"] = user['residence'].apply(process_province)
user["residence_city"] = user['residence'].apply(process_city)


# In[45]:

user.info()


# In[46]:

user.head()


# In[47]:

train_data.head()


# In[48]:

train_data['clickTime_day'] = train_data['clickTime'].apply(get_time_day)
train_data['clickTime_hour']= train_data['clickTime'].apply(get_time_hour)


# ### 合并数据

# In[49]:

#train data
train_data['clickTime_day'] = train_data['clickTime'].apply(get_time_day)
train_data['clickTime_hour']= train_data['clickTime'].apply(get_time_hour)
# train_data['conversionTime_day'] = train_data['conversionTime'].apply(get_time_day)
# train_data['conversionTime_hour'] = train_data['conversionTime'].apply(get_time_hour)


#test_data
test_data = read_csv_file('./pre/test.csv', True)
test_data['clickTime_day'] = test_data['clickTime'].apply(get_time_day)
test_data['clickTime_hour']= test_data['clickTime'].apply(get_time_hour)
# test_data['conversionTime_day'] = test_data['conversionTime'].apply(get_time_day)
# test_data['conversionTime_hour'] = test_data['conversionTime'].apply(get_time_hour)


train_user = pd.merge(train_data,user,on='userID')
train_user_ad = pd.merge(train_user,ad,on='creativeID')
train_user_ad_app = pd.merge(train_user_ad,app_categories,on='appID')


# In[50]:

train_user_ad_app.head()


# In[51]:

train_user_ad_app.columns


# ### 取出数据和label

# In[52]:

#特征部分
x_user_ad_app = train_user_ad_app.loc[:,['creativeID','userID','positionID',
 'connectionType','telecomsOperator','clickTime_day','clickTime_hour','age', 'gender' ,'education',
 'marriageStatus' ,'haveBaby' , 'residence' ,'age_process',
 'hometown_province', 'hometown_city','residence_province', 'residence_city',
 'adID', 'camgaignID', 'advertiserID', 'appID' ,'appPlatform' ,
 'app_categories_first_class' ,'app_categories_second_class']]

x_user_ad_app = x_user_ad_app.values
x_user_ad_app = np.array(x_user_ad_app,dtype='int32')

#标签部分
y_user_ad_app =train_user_ad_app.loc[:,['label']].values


# In[20]:

pd.DataFrame(x_user_ad_app).to_csv("yangben.csv")


# # 随机森林建模&&特征重要度排序

# In[21]:

# %matplotlib inline
# import matplotlib.pyplot as plt
# print('Plot feature importances...')
# ax = lgb.plot_importance(gbm, max_num_features=10)
# plt.show()
# 用RF 计算特征重要度

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

feat_labels = np.array(['creativeID','userID','positionID',
 'connectionType','telecomsOperator','clickTime_day','clickTime_hour','age', 'gender' ,'education',
 'marriageStatus' ,'haveBaby' , 'residence' ,'age_process',
 'hometown_province', 'hometown_city','residence_province', 'residence_city',
 'adID', 'camgaignID', 'advertiserID', 'appID' ,'appPlatform' ,
 'app_categories_first_class' ,'app_categories_second_class'])

forest = RandomForestClassifier(n_estimators=100,
                                random_state=0,
                                n_jobs=-1)

forest.fit(x_user_ad_app, y_user_ad_app.reshape(y_user_ad_app.shape[0],))
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]


# In[22]:

indices


# In[23]:

importances


# In[25]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
for f in range(x_user_ad_app.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(x_user_ad_app.shape[1]), 
        importances[indices],
        color='lightblue', 
        align='center')

plt.xticks(range(x_user_ad_app.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, x_user_ad_app.shape[1]])
plt.tight_layout()
#plt.savefig('./random_forest.png', dpi=300)
plt.show()


# ### 随机森林调参

# In[26]:

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
param_grid = {
              #'n_estimators': [100],
              'n_estimators': [10, 100, 500, 1000],
              'max_features':[0.6, 0.7, 0.8, 0.9]
             }

rf = RandomForestClassifier()
rfc = GridSearchCV(rf, param_grid, scoring = 'neg_log_loss', cv=3, n_jobs=2)
rfc.fit(x_user_ad_app, y_user_ad_app.reshape(y_user_ad_app.shape[0],))
print(rfc.best_score_)
print(rfc.best_params_)


# ### Xgboost调参

# In[ ]:

import xgboost as xgb


# In[ ]:

# import os
# import numpy as np
# from sklearn.model_selection import GridSearchCV
# import xgboost as xgb
# os.environ["OMP_NUM_THREADS"] = "8"  #并行训练
# rng = np.random.RandomState(4315)
# import warnings
# warnings.filterwarnings("ignore")

# param_grid = {
#               'max_depth': [3, 4, 5, 7, 9],
#               'n_estimators': [10, 50, 100, 400, 800, 1000, 1200],
#               'learning_rate': [0.1, 0.2, 0.3],
#               'gamma':[0, 0.2],
#               'subsample': [0.8, 1],
#               'colsample_bylevel':[0.8, 1]
#              }

# xgb_model = xgb.XGBClassifier()
# rgs = GridSearchCV(xgb_model, param_grid, n_jobs=-1)
# rgs.fit(X, y)
# print(rgs.best_score_)
# print(rgs.best_params_)


# ### 正负样本比

# In[ ]:

# positive_num = train_user_ad_app[train_user_ad_app['label']==1].values.shape[0]
# negative_num = train_user_ad_app[train_user_ad_app['label']==0].values.shape[0]

# negative_num/float(positive_num)


# **我们可以看到正负样本数量相差非常大，数据严重unbalanced**

# 我们用Bagging修正过后，处理不均衡样本的B(l)agging来进行训练和实验。

# In[27]:

from blagging import BlaggingClassifier


# In[28]:

help(BlaggingClassifier)


# In[29]:

#处理unbalanced的classifier
classifier = BlaggingClassifier(n_jobs=-1)


# In[30]:

classifier.fit(x_user_ad_app, y_user_ad_app)


# In[33]:

# classifier.predict_proba(x_test_clean)


# #### 预测

# In[56]:

test_data = pd.merge(test_data,user,on='userID')
test_user_ad = pd.merge(test_data,ad,on='creativeID')
test_user_ad_app = pd.merge(test_user_ad,app_categories,on='appID')

x_test_clean = test_user_ad_app.loc[:,['creativeID','userID','positionID',
 'connectionType','telecomsOperator','clickTime_day','clickTime_hour','age', 'gender' ,'education',
 'marriageStatus' ,'haveBaby' , 'residence' ,'age_process',
 'hometown_province', 'hometown_city','residence_province', 'residence_city',
 'adID', 'camgaignID', 'advertiserID', 'appID' ,'appPlatform' ,
 'app_categories_first_class' ,'app_categories_second_class']].values

x_test_clean = np.array(x_test_clean,dtype='int32')


##################多个模型取平均值
# result_predict_prob = []
# result_predict=[]
# for i in range(scale):
#     result_indiv = clfs[i].predict(x_test_clean)
#     result_indiv_proba = clfs[i].predict_proba(x_test_clean)[:,1]
#     result_predict.append(result_indiv)
#     result_predict_prob.append(result_indiv_proba)


# result_predict_prob = np.reshape(result_predict_prob,[-1,scale])
# result_predict = np.reshape(result_predict,[-1,scale])

# result_predict_prob = np.mean(result_predict_prob,axis=1)
# result_predict = max_count(result_predict)

result_predict_prob=classifier.predict_proba(x_test_clean)[:,1]

result_predict_prob = np.array(result_predict_prob).reshape([-1,1])


test_data['prob'] = result_predict_prob
test_data = test_data.loc[:,['instanceID','prob']]
test_data.to_csv('predict.csv',index=False)
print ("prediction done!")


# In[ ]:



