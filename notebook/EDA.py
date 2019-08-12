# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: mlpy36
#     language: python
#     name: mlpy36
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from tqdm import tqdm
from sklearn import preprocessing
import pickle
import lightgbm as lgb
from scipy import stats


df_train = pd.read_pickle('../input/train.pkl')
df_test = pd.read_pickle('../input/test.pkl')

pd.set_option('display.max_columns', 500)
df_train.head()

file_name = '/Users/zakopuro/Code/python_code/kaggle/IEEE_Fraud_Detection/model/024_lgb.sav'
lgb_model = pickle.load(open(file_name,'rb'))

lgb.plot_importance(lgb_model,max_num_features=30,figsize=(12,6))

temp = pd.concat([df_train['card1'], df_test['card1']], ignore_index=True).value_counts(dropna=False)
df_train['card1_count'] = df_train['card1'].map(temp)
df_test['card1_count'] = df_test['card1'].map(temp)

plt.scatter(df_train['C9'],df_train['C5'])

# x = df_train['C13']
# plt.hist(x)
len(df_train[(df_train['C13']==0) & (df_train['isFraud'] == 1)])/len(df_train[df_train['C13'] == 0])



plt.hist(np.log(df_train['card1_count']))

x,_ = stats.boxcox(df_train['card1_count'])
plt.hist(x)

plt.hist(np.log(df_test['card1']/df_test['card2']))
# plt.scatter(df_train[df_train['isFraud'] == 1]['card1'],df_train[df_train['isFraud'] == 1]['card2'])

plt.scatter(df_train['addr1'],df_train['isFraud'])


