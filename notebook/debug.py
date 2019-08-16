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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from sklearn import metrics
from sklearn import preprocessing
import logging
import matplotlib.pyplot as plt
import pickle
import lightgbm as lgb

df_train = pd.read_pickle('../src/make_data/data/009_train.pkl')
df_test = pd.read_pickle('../src/make_data/data/009_train.pkl')


class Net(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256,1)
        self.fc4 = nn.Dropout(0.2)
        
    def foward(self,x):
        x = F.relu(self.fc1(x))
        x = F.batch_norm(x)
        x = F.relu(self.fc3(x))
        x = F.batch_norm(x)
        x = 



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Linear(512)
        
        
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(),lr=0.01)


def train_model(model):
    model.train()


train_model(model)

sub1 = pd.read_csv('/Users/zakopuro/Code/python_code/kaggle/IEEE_Fraud_Detection/output/031_sub_lgb.csv')
sub2 = pd.read_csv('/Users/zakopuro/Code/python_code/kaggle/IEEE_Fraud_Detection/output/033_sub_lgb.csv')
sub = pd.read_csv('/Users/zakopuro/Code/python_code/kaggle/IEEE_Fraud_Detection/input/sample_submission.csv')

sub['isFraud'] = sub1['isFraud']*0.2 + sub2['isFraud']*0.8

sub.head()

sub.to_csv('/Users/zakopuro/Code/python_code/kaggle/IEEE_Fraud_Detection/output/035_sub_lgb.csv',index=False)

train = pd.read_pickle('/Users/zakopuro/Code/python_code/kaggle/IEEE_Fraud_Detection/input/train.pkl')
test = pd.read_pickle('/Users/zakopuro/Code/python_code/kaggle/IEEE_Fraud_Detection/input/test.pkl')

pd.set_option('display.max_columns', 500)
train.head()

df = train
# df['card1'].astype(str)+'_'+df['card2'].astype(str)+'_'+df['card3'].astype(str)+'_'+df['card4'].astype(str)

make_train = pd.read_pickle('../src/make_data/data/017_train.pkl')

lgb_model = pickle.load(open('/Users/zakopuro/Code/python_code/kaggle/IEEE_Fraud_Detection/model/036_lgb.sav', 'rb'))

lgb.plot_importance(lgb_model,max_num_features=20)


