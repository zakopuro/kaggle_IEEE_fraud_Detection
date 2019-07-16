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

# +
df_train_iden = pd.read_csv('../IEEE_Fraud_Detection/input/train_identity.csv')
df_train_trans = pd.read_csv('../IEEE_Fraud_Detection/input/train_transaction.csv')
df_test_iden = pd.read_csv('../IEEE_Fraud_Detection/input/test_identity.csv')
df_test_trans = pd.read_csv('../IEEE_Fraud_Detection/input/test_transaction.csv')

df_train = df_train_trans.merge(df_train_iden,how='left',left_index=True,right_index=True)
df_test = df_test_trans.merge(df_test_iden,how='left',left_index=True,right_index=True)
# -

df_train.shape,df_test.shape

df_train = df_train.drop('isFraud', axis=1)

for col in tqdm(df_test.columns):
    if df_train[col].dtype == 'object' or df_test[col].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_train.values) + list(df_test.values))
        df_train[col] = lbl.transform(list(df_train[col].values))
        df_test[col] = lbl.transform(list(df_test[col].values))


