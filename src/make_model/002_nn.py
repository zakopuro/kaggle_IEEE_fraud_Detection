import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from sklearn import metrics
from sklearn import preprocessing


class NN(nn.):
    




def main():
    df_train = pd.read_pickle('../IEEE_Fraud_Detection/src/make_data/data/009_train.pkl')
    df_test = pd.read_pickle('../IEEE_Fraud_Detection/src/make_data/data/009_test.pkl')
    target = df_train['isFraud'].copy()
    X_train = df_train.drop('isFraud',axis=1)
    X_train.drop('TransactionDT',axis=1,inplace=True)
    X_test = df_test.drop('TransactionDT',axis=1)




if __name__ == "__main__":
    main()