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

# +
df_train_iden = pd.read_csv('../input/train_identity.csv')
df_train_trans = pd.read_csv('../input/train_transaction.csv')
df_test_iden = pd.read_csv('../input/test_identity.csv')
df_test_trans = pd.read_csv('../input/test_transaction.csv')

df_train = pd.merge(df_train_trans, df_train_iden, on='TransactionID', how='left')
df_test = pd.merge(df_test_trans, df_test_iden, on='TransactionID', how='left')
# -

pdf.ProfileReport(df_train)


