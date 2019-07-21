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

# +
df_train_iden = pd.read_csv('../input/train_identity.csv',index_col='TransactionID')
df_train_trans = pd.read_csv('../input/train_transaction.csv',index_col='TransactionID')
df_test_iden = pd.read_csv('../input/test_identity.csv',index_col='TransactionID')
df_test_trans = pd.read_csv('../input/test_transaction.csv',index_col='TransactionID')

df_train = pd.merge(df_train_trans, df_train_iden, on='TransactionID', how='left')
df_test = pd.merge(df_test_trans, df_test_iden, on='TransactionID', how='left')
# -

df_train_DT = pd.read_csv('../src/make_data/data/005_train.csv',index_col='TransactionID')

col_list = list(df_train.columns)

col_list

plt.hist(df_train['card2'])

# sns.countplot(y='isFraud', x="ProductCD", data=df_train)
sns.countplot(x="ProductCD",hue = "isFraud", data=df_train)

sns.countplot(x="card2",hue = "isFraud", data=df_train)

for card2 in df_train['card2'].unique():
    num_0 = len(df_train[(df_train['isFraud'] == 0) & (df_train['card2'] == card2)])
    num_1 = len(df_train[(df_train['isFraud'] == 1) & (df_train['card2'] == card2)])
    if num_0 == 0:
        num_0 = 0.01
    isf_per = num_1/num_0
    if isf_per > 0.5:
        print(card2)

card2 = [176,405,289,319]
num_0 = len(df_train[(df_train['isFraud'] == 0) & (df_train['card2'] == card2)])
num_1 = len(df_train[(df_train['isFraud'] == 1) & (df_train['card2'] == card2)])
print(num_0,num_1)

for card2 in df_train['card2'].unique():
    num_0 = len(df_train[(df_train['isFraud'] == 0) & (df_train['card2'] == card2)])
    num_1 = len(df_train[(df_train['isFraud'] == 1) & (df_train['card2'] == card2)])
#     isf_per = num_1/num_0
    if num_1 == 0:
        print(card2)
        print()


