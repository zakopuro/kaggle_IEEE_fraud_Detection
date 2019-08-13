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

sub1 = pd.read_csv('/Users/zakopuro/Code/python_code/kaggle/IEEE_Fraud_Detection/output/024_sub_lgb.csv')
sub2 = pd.read_csv('/Users/zakopuro/Code/python_code/kaggle/IEEE_Fraud_Detection/output/kernel_NN.csv')
sub = pd.read_csv('/Users/zakopuro/Code/python_code/kaggle/IEEE_Fraud_Detection/input/sample_submission.csv')

sub['isFraud'] = sub1['isFraud']*0.9 + sub2['isFraud']*0.1

sub.head()

sub.to_csv('/Users/zakopuro/Code/python_code/kaggle/IEEE_Fraud_Detection/output/026_sub_lgb09+NN01.csv',index=False)


