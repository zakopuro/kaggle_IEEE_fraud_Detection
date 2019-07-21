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
from tqdm import tqdm
from sklearn import preprocessing
import pandas_profiling as pdf
from sklearn.model_selection import TimeSeriesSplit

# +
df_train_iden = pd.read_csv('../input/train_identity.csv')
df_train_trans = pd.read_csv('../input/train_transaction.csv')
df_test_iden = pd.read_csv('../input/test_identity.csv')
df_test_trans = pd.read_csv('../input/test_transaction.csv')

df_train = pd.merge(df_train_trans, df_train_iden, on='TransactionID', how='left')
df_test = pd.merge(df_test_trans, df_test_iden, on='TransactionID', how='left')
# -

splits = TimeSeriesSplit(n_splits=5)

index = 1
for train_index,test_index in splits.split(df_train):
    X_train = df_train['TransactionDT'][train_index]
    X_test = df_train['TransactionDT'][test_index]
    plt.subplot(510 + index)
    plt.plot(X_train)
    plt.plot([None for i in X_train]+[x for x in X_test])
    index += 1

len(df_train)/6

# +
train_start_index = 0
train_end_index = int(len(df_train)/2)
skip = int(len(df_train)/6)
horizon = int(len(df_train)/6)

SPLITS = 3
for split in range(SPLITS):
    test_start_index = train_end_index
    test_end_index = test_start_index + horizon
    
    X_train = df_train[train_start_index:train_end_index]
    X_test = df_train[test_start_index:test_end_index]
    
    train_start_index += skip
    train_end_index += skip
    print(X_train.index)
    print(X_test.index)
# -

X_train.index

X_test.index

df_train.shape

df_train = pd.read_csv('../src/make_data/data/003_train.csv',index_col='TransactionID')
df_test = pd.read_csv('../src/make_data/data/003_test.csv',index_col='TransactionID')

df_train['day']

import pickle
import xgboost as xgb

filename = '../model/003_xgb.sav'
loaded_model = pickle.load(open(filename,'rb'))
df_train = pd.read_csv('../src/make_data/data/003_train.csv',index_col='TransactionID')

fti = loaded_model.feature_importances_
train = df_train.drop('isFraud',axis=1)

# +
dict = {"feat":np.arange(0,len(train.columns)) , 'importance': np.arange(0,len(train.columns),dtype=float)}
df_feat_imp = pd.DataFrame(dict)
for i,feat in enumerate(train.columns):
    df_feat_imp['feat'][i] = feat
    df_feat_imp['importance'][i] = loaded_model.feature_importances_[i]

df_feat_imp = df_feat_imp.sort_values(by='importance',ascending=True)

# drop_list = list(df_feat_imp['feat'].head(10))
# -

df_feat_imp.to_csv('../src/make_data/data/fti_list.csv',index=False)

fti = pd.read_csv('../src/make_data/data/fti_list.csv')

fti_drop_list = list(df_feat_imp['feat'].head(30))

fti_drop_list

fti.head(30)

# +
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(18,4))

time_val = df_train['TransactionAmt'].values

sns.distplot(time_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of TransactionAmt', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

sns.distplot(np.log(time_val), ax=ax[1], color='b')
ax[1].set_title('Distribution of LOG TransactionAmt', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])

plt.show()
# -

plt.hist(np.log(df_train['TransactionAmt']))


